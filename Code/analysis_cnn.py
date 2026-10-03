import os
import gc
import math
import copy
import random
import time
from collections import OrderedDict, defaultdict, deque

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

from mqbench.prepare_by_platform import prepare_by_platform, BackendType
from mqbench.utils.state import enable_calibration, enable_quantization


# ============================================================
# CONFIG
# ============================================================
IMAGENET_ROOT = "../test_dataset"
CALIB_SIZE = 1024
SENS_SIZE = 2000
RANDOM_SEED = 69
BATCH_SIZE = 64
NUM_WORKERS = 4  # Nova warning recommends at most 2 workers
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


MIN_ISO_DROP_PP = 3.0
MIN_RESTORE_GAIN_PP = 3.0

FDR_Q = 0.05

WBIT = 4
ABIT = 4  # set to 8 for W4A8 or 4 for W4A4
KEEP_EDGE_8BIT = True

A_OBSERVER = "EMAMSEObserver"

INST_RELERR_MIN = 0.05      # >= 0.5% relative output distortion
INST_DELTA_MIN = 0.01       # >= 0.1 percentage-point local error increase
INST_GAIN_MIN = 1.15         # >= 5% local amplification
INST_ZERO_INFLATION_MIN = 0.05
INST_DEAD_RATIO_MIN = 0.05

SUPPORTED_MODELS = (
    "resnet18", "resnet34", "resnet50",
    "vgg16", "vgg19",
    "regnet_x_800mf", "mobilenet_v2",
)
MODEL_NAME = "resnet34"

INTERESTED_TYPES = (
    nn.Conv2d, nn.Linear, nn.BatchNorm2d,
    nn.ReLU, nn.ReLU6,
    nn.MaxPool2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d,
)
WEIGHTED_TYPES = (nn.Conv2d, nn.Linear)
NORM_TYPES = (nn.BatchNorm2d,)

OUTDIR = "ptq_metrics"


# ============================================================
# DATA
# ============================================================
_transform = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def build_loaders():
    dataset = datasets.ImageFolder(IMAGENET_ROOT, transform=_transform)
    rng = random.Random(RANDOM_SEED)
    all_idx = list(range(len(dataset)))
    rng.shuffle(all_idx)

    needed = CALIB_SIZE + SENS_SIZE
    if len(all_idx) < needed:
        raise RuntimeError(f"Dataset has {len(all_idx)} samples, but this script needs "
            f"CALIB_SIZE + SENS_SIZE = {needed}. Reduce CALIB_SIZE/SENS_SIZE.")


    calib_idx = all_idx[:CALIB_SIZE]
    sens_idx = all_idx[CALIB_SIZE:CALIB_SIZE + SENS_SIZE]

    assert len(set(calib_idx) & set(sens_idx)) == 0, "calib_idx and sens_idx overlap"

    calib_loader = DataLoader(
        Subset(dataset, calib_idx),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    sens_loader = DataLoader(
        Subset(dataset, sens_idx),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    return dataset, calib_loader, sens_loader, calib_idx, sens_idx


def build_model_fp32():
    builders = {
        "resnet18": lambda: models.resnet18(weights=models.ResNet18_Weights.DEFAULT),
        "resnet34": lambda: models.resnet34(weights=models.ResNet34_Weights.DEFAULT),
        "resnet50": lambda: models.resnet50(weights=models.ResNet50_Weights.DEFAULT),
        "vgg16": lambda: models.vgg16(weights=models.VGG16_Weights.DEFAULT),
        "vgg19": lambda: models.vgg19(weights=models.VGG19_Weights.DEFAULT),
        "regnet_x_800mf": lambda: models.regnet_x_800mf(weights=models.RegNet_X_800MF_Weights.DEFAULT),
        "mobilenet_v2": lambda: models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT),
    }
    if MODEL_NAME not in builders:
        raise ValueError(f"Unsupported MODEL_NAME={MODEL_NAME}. Choose from {SUPPORTED_MODELS}")
    return builders[MODEL_NAME]().eval()


def set_seed(seed=RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def validate_config():
    if WBIT != 4:
        raise ValueError(f"This analysis script is intended for W4A4/W4A8; got WBIT={WBIT}")
    if ABIT not in (4, 8):
        raise ValueError(f"ABIT must be 4 or 8; got ABIT={ABIT}")
    if MODEL_NAME not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported MODEL_NAME={MODEL_NAME}. Choose from {SUPPORTED_MODELS}")

def safe_cpu_float(x):
    return x.detach().to(dtype=torch.float32, device="cpu")


def evaluate_accuracy(model, loader, device):
    model.eval().to(device)
    correct = total = 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            pred = model(imgs).argmax(dim=1)
            correct += int((pred == labels).sum().item())
            total += labels.size(0)

    return 100.0 * correct / max(total, 1)


def evaluate_accuracy_and_correctness(model, loader, device):
    
    model.eval().to(device)
    chunks = []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            pred = model(imgs).argmax(dim=1)
            chunks.append((pred == labels).detach().cpu().numpy().astype(np.int8))

    if not chunks:
        return 0.0, np.empty(0, dtype=np.int8)

    corr = np.concatenate(chunks)
    return 100.0 * float(corr.mean()), corr


def _normal_survival(z):
    """1 - Phi(z), implemented without scipy."""
    return 0.5 * math.erfc(float(z) / math.sqrt(2.0))


def paired_effect_stats(reference_correct, candidate_correct, direction):
    
    ref = np.asarray(reference_correct, dtype=np.float64).reshape(-1)
    cand = np.asarray(candidate_correct, dtype=np.float64).reshape(-1)

    if ref.size == 0 or cand.size != ref.size:
        return {
            'effect_pp': np.nan, 'se_pp': np.nan, 'lcb95_pp': np.nan,
            'p_value': np.nan, 'n': int(min(ref.size, cand.size)),
            'positive_flips': 0, 'negative_flips': 0,
        }

    if direction == 'drop':
        d = ref - cand
    elif direction == 'gain':
        d = cand - ref
    else:
        raise ValueError("direction must be 'drop' or 'gain'")

    n = d.size
    effect_pp = 100.0 * float(d.mean())
    if n > 1:
        se_pp = 100.0 * float(d.std(ddof=1)) / math.sqrt(n)
    else:
        se_pp = float('inf')

    # One-sided 95% lower confidence bound for the raw effect.
    lcb95_pp = effect_pp - 1.6448536269514722 * se_pp if np.isfinite(se_pp) else -np.inf

    if se_pp <= 1e-15:
        p = 0.0 if effect_pp > 0.0 else 1.0
    else:
        z = effect_pp / se_pp
        p = _normal_survival(z)

    return {
        'effect_pp': effect_pp,
        'se_pp': se_pp,
        'lcb95_pp': lcb95_pp,
        'p_value': float(min(max(p, 0.0), 1.0)),
        'n': int(n),
        'positive_flips': int((d > 0).sum()),
        'negative_flips': int((d < 0).sum()),
    }


def benjamini_hochberg(p_values):
    """Return BH-FDR adjusted q-values; NaNs remain NaN."""
    p = np.asarray(p_values, dtype=np.float64)
    q = np.full(p.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(p)
    if not valid.any():
        return q

    pv = p[valid]
    order = np.argsort(pv)
    ranked = pv[order]
    m = len(ranked)
    adjusted = ranked * m / np.arange(1, m + 1, dtype=np.float64)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)

    restored = np.empty_like(adjusted)
    restored[order] = adjusted
    q[valid] = restored
    return q


def get_execution_order(model, loader, device):
    order = []
    handles = []

    def make_hook(name):
        def hook(*_):
            if name not in order:
                order.append(name)
        return hook

    for name, m in model.named_modules():
        if isinstance(m, INTERESTED_TYPES):
            handles.append(m.register_forward_hook(make_hook(name)))

    model.eval().to(device)

    with torch.no_grad():
        imgs, _ = next(iter(loader))
        _ = model(imgs[:1].to(device, non_blocking=True))

    for h in handles:
        h.remove()

    return order


def get_module_by_name(model, name):
    m = model
    for p in name.split("."):
        if not hasattr(m, p):
            return None
        m = getattr(m, p)
    return m


def align_shapes(fp, q):
    
    fp = safe_cpu_float(fp)
    q = safe_cpu_float(q)

    if fp.shape != q.shape:
        raise ValueError(f"shape mismatch: fp={tuple(fp.shape)} q={tuple(q.shape)}")

    return fp, q

def _is_fakequant_module(mod, target_name):
    cls = mod.__class__.__name__.lower()
    target_name = str(target_name).lower()
    return (
        "fakequantize" in cls
        or "fakequantizer" in cls
        or "fake_quant" in target_name
        or "activation_post_process" in target_name
    )


def _is_merge_or_branch_node(node):
    if node.op not in ("call_function", "call_method"):
        return False
    target = str(node.target).lower()
    return any(tok in target for tok in ("add", "mul", "cat", "stack", "sub"))


def inspect_mqbench_graph(quant_model, fp32_model, layer_names):
    
    info = {
        n: {
            "post_act_fq": None,
            "post_act_fqs": [],
            "weight_fq_path": None,
            "bn_folded": False,
            "folded_source_layer": None,
            "in_graph": False,
            "fq_mapping_ambiguous": False,
        }
        for n in layer_names
    }

    if not hasattr(quant_model, "graph"):
        raise RuntimeError("[mqbench] prepared model has no fx graph.")

    modules = dict(quant_model.named_modules())
    fp_modules = dict(fp32_model.named_modules())
    nodes_by_target = defaultdict(list)
    for node in quant_model.graph.nodes:
        if node.op == "call_module":
            nodes_by_target[str(node.target)].append(node)

    mapped = 0
    ambiguous = 0

    for name in layer_names:
        start_nodes = nodes_by_target.get(name, [])
        if not start_nodes:
            continue

        info[name]["in_graph"] = True
        candidates = []  # (depth, fq_name)

        for start_node in start_nodes:
            q = deque((user, 1) for user in start_node.users)
            seen = set()

            while q:
                node, depth = q.popleft()
                if node in seen or depth > 3:
                    continue
                seen.add(node)

                if _is_merge_or_branch_node(node):
                    continue

                if node.op == "call_module":
                    target = str(node.target)
                    mod = modules.get(target)

                    if mod is not None and _is_fakequant_module(mod, target):
                        candidates.append((depth, target))
                        continue

                    # Do not cross another analyzed semantic layer.
                    if mod is not None and isinstance(mod, INTERESTED_TYPES):
                        continue

                for user in node.users:
                    q.append((user, depth + 1))

        if candidates:
            min_depth = min(d for d, _ in candidates)
            nearest = []
            seen_names = set()
            for d, fq_name in candidates:
                if d == min_depth and fq_name not in seen_names:
                    nearest.append(fq_name)
                    seen_names.add(fq_name)

            if len(nearest) == 1:
                info[name]["post_act_fq"] = nearest[0]
                info[name]["post_act_fqs"] = [nearest[0]]
                mapped += 1
            else:
                info[name]["fq_mapping_ambiguous"] = True
                ambiguous += 1

        m = modules.get(name)
        if m is not None and hasattr(m, "weight_fake_quant"):
            info[name]["weight_fq_path"] = f"{name}.weight_fake_quant"

    print(
        f"[mqbench] local post-FQ mapping: {mapped}/{len(layer_names)} layers; "
        f"ambiguous={ambiguous}"
    )

   
    n_bn = 0
    n_bn_folded = 0
    last_weighted = None

    for name in layer_names:
       
        fp_m = fp_modules.get(name)

        if isinstance(fp_m, WEIGHTED_TYPES):
            last_weighted = name

        if not isinstance(fp_m, NORM_TYPES):
            continue

        n_bn += 1

        if len(nodes_by_target.get(name, [])) == 0:
            info[name]["bn_folded"] = True
            info[name]["folded_source_layer"] = last_weighted
            n_bn_folded += 1

    if n_bn > 0:
        print(f"[mqbench] BN folding: {n_bn_folded}/{n_bn} BN layers folded")
        mapped_bn = sum(
            1 for n in layer_names
            if info[n]["bn_folded"] and info[n]["folded_source_layer"] is not None
        )
        if n_bn_folded:
            print(f"[mqbench] folded BN producer mapping: {mapped_bn}/{n_bn_folded}")

    return info


def get_quant_weight(module):
    if not hasattr(module, "weight") or module.weight is None:
        return None
    if hasattr(module, "weight_fake_quant"):
        try:
            return module.weight_fake_quant(module.weight).detach()
        except Exception:
            return None
    return None


def _fq_num_levels(fq):
    """Return quantization levels for a fake-quantizer, or None if unavailable."""
    if not (hasattr(fq, "quant_min") and hasattr(fq, "quant_max")):
        return None
    try:
        return int(fq.quant_max) - int(fq.quant_min) + 1
    except Exception:
        return None


def _levels_match_bit(levels, bit):
    """Support both full 2^b and narrow-range 2^b-1 quantizers."""
    if levels is None:
        return False
    full = 1 << int(bit)
    return levels in (full, full - 1)


def _weighted_fakequant_owners(quant_model):
    """Return modules that own a real weight fake-quantizer."""
    owners = []
    for name, mod in quant_model.named_modules():
        if not name or not hasattr(mod, "weight_fake_quant"):
            continue
        weight = getattr(mod, "weight", None)
        if not isinstance(weight, torch.Tensor):
            continue
        owners.append(name)
    return owners


def _weighted_owners_in_fx_order(quant_model, owner_names):
    """Resolve weighted owners in prepared FX execution order."""
    owner_set = set(owner_names)
    order, seen = [], set()
    graph = getattr(quant_model, "graph", None)
    if graph is not None:
        for node in graph.nodes:
            if node.op != "call_module":
                continue
            target = str(node.target)
            if target in owner_set and target not in seen:
                order.append(target)
                seen.add(target)
    # Preserve any owner not represented as call_module without letting
    # registration order replace the FX order that was already recovered.
    for name in owner_names:
        if name not in seen:
            order.append(name)
            seen.add(name)
    return order


def _is_real_activation_fq_name(name, mod):
    cls = mod.__class__.__name__.lower()
    if not (("fakequantize" in cls) or ("fakequantizer" in cls)):
        return False
    if "weight_fake_quant" in name:
        return False
    if ".activation_post_process" in name:
        return False
    return True


def _activation_fqs_in_fx_order(quant_model):
    """Resolve activation fake-quantizers in the prepared FX graph order."""
    modules = dict(quant_model.named_modules())
    order, seen = [], set()
    graph = getattr(quant_model, "graph", None)
    if graph is not None:
        for node in graph.nodes:
            if node.op != "call_module":
                continue
            name = str(node.target)
            mod = modules.get(name)
            if mod is None or not _is_real_activation_fq_name(name, mod):
                continue
            if name not in seen:
                order.append(name)
                seen.add(name)
    # Fallback only for quantizers not materialized as call_module nodes.
    for name, mod in quant_model.named_modules():
        if _is_real_activation_fq_name(name, mod) and name not in seen:
            order.append(name)
            seen.add(name)
    return order


def verify_quantizer_bitwidths(quant_model):
    
    modules = dict(quant_model.named_modules())
    bad = []

    # ---- Weights.
    weighted_owners = _weighted_fakequant_owners(quant_model)
    weighted_order = _weighted_owners_in_fx_order(quant_model, weighted_owners)
    edge_w = set()
    if KEEP_EDGE_8BIT and weighted_order:
        edge_w.add(weighted_order[0])
        if len(weighted_order) > 1:
            edge_w.add(weighted_order[-1])

    checked_w = w_main = w_edge8 = 0
    for owner in weighted_owners:
        fq = getattr(modules[owner], "weight_fake_quant", None)
        levels = _fq_num_levels(fq)
        if levels is None:
            continue
        checked_w += 1
        if _levels_match_bit(levels, WBIT):
            w_main += 1
        elif KEEP_EDGE_8BIT and owner in edge_w and _levels_match_bit(levels, 8):
            w_edge8 += 1
        else:
            bad.append((f"{owner}.weight_fake_quant", "weight", levels, WBIT))

    # ---- Activations.
    activation_order = _activation_fqs_in_fx_order(quant_model)
    edge_a = set()
    if KEEP_EDGE_8BIT and ABIT < 8 and activation_order:
        edge_a.add(activation_order[0])
        if len(activation_order) > 1:
            edge_a.add(activation_order[-1])

    checked_a = a_main = a_edge8 = 0
    for name in activation_order:
        mod = modules[name]
        levels = _fq_num_levels(mod)
        if levels is None:
            continue
        checked_a += 1
        if _levels_match_bit(levels, ABIT):
            a_main += 1
        elif KEEP_EDGE_8BIT and ABIT < 8 and name in edge_a and _levels_match_bit(levels, 8):
            a_edge8 += 1
        else:
            bad.append((name, "activation", levels, ABIT))

    if WBIT < 8 and checked_w and w_main == 0:
        bad.append(("<all weighted layers>", "weight", "no requested low-bit quantizers", WBIT))
    if ABIT < 8 and checked_a and a_main == 0:
        bad.append(("<all activation FQs>", "activation", "no requested low-bit quantizers", ABIT))

    if edge_w:
        print("[mqbench] protected 8-bit weight edges: " + ", ".join(n for n in weighted_order if n in edge_w))
    if edge_a:
        print("[mqbench] protected 8-bit activation edges: " + ", ".join(n for n in activation_order if n in edge_a))

    if bad:
        preview = "; ".join(
            f"{name}:{kind} levels={levels} expected-bit={bit}"
            for name, kind, levels, bit in bad[:12])
        raise RuntimeError(f"[mqbench] genuine quantizer bit-width mismatch: {preview}")


# ============================================================
# WEIGHT METRICS (FP32 vs INT4)
# ============================================================
def _weight_metrics_one(w, suffix):
    out = {
        f"weight_cond_{suffix}": np.nan,
        f"weight_outlier_ratio_{suffix}": np.nan,
    }

    if w is None or w.ndim < 2:
        return out

    try:
        W2 = w.reshape(w.shape[0], -1)
        svals = torch.linalg.svdvals(W2)
        if svals.numel() < 2:
            out[f"weight_cond_{suffix}"] = np.nan
        else:
            sigma_min = float(svals[-1])
            sigma_max = float(svals[0])
            if sigma_min < 1e-12:
                out[f"weight_cond_{suffix}"] = float("inf")
            else:
                out[f"weight_cond_{suffix}"] = sigma_max / sigma_min
    except Exception:
        out[f"weight_cond_{suffix}"] = np.nan

    try:
        W2 = w.reshape(w.shape[0], -1).abs()
        ch_max = W2.max(dim=1).values
        ch_std = W2.std(dim=1)
        valid = ch_std > 1e-12
        if valid.any():
            ratios = ch_max[valid] / ch_std[valid]
            out[f"weight_outlier_ratio_{suffix}"] = float(ratios.max().item())
    except Exception:
        pass

    return out


def compute_weight_metrics(fp_module, q_module):
    out = {
        "weight_RelErr": np.nan,
        "weight_Cosine": np.nan,
        "weight_cond_fp32": np.nan,
        "weight_cond_int4": np.nan,
        "weight_outlier_ratio_fp32": np.nan,
        "weight_outlier_ratio_int4": np.nan,
    }

    if not hasattr(fp_module, "weight") or fp_module.weight is None:
        return out

    w_fp = safe_cpu_float(fp_module.weight)
    out.update(_weight_metrics_one(w_fp, "fp32"))

    w_q_raw = get_quant_weight(q_module)
    if w_q_raw is None:
        return out

    w_q = safe_cpu_float(w_q_raw)
    out.update(_weight_metrics_one(w_q, "int4"))

    diff = w_fp - w_q
    out["weight_RelErr"] = float(torch.norm(diff) / (torch.norm(w_fp) + 1e-12))

    a = w_fp.flatten()
    b = w_q.flatten()
    out["weight_Cosine"] = float(
        torch.dot(a, b) / (torch.norm(a) * torch.norm(b) + 1e-12)
    )

    return out


class PropAcc:
    def __init__(self, name, type_name):
        self.layer = name
        self.type = type_name

        self.in_fp_sq = 0.0
        self.in_diff_sq = 0.0

        self.out_fp_sq = 0.0
        self.out_diff_sq = 0.0
        self.out_q_sq = 0.0
        self.out_dot = 0.0

        self.n_elem = 0
        self.n_updates = 0
        self.output_only_updates = 0
        self.is_entry = False

    def mark_entry(self):
        self.is_entry = True

    def _update_output_stats(self, fp_out, q_out):
        fp_out, q_out = align_shapes(fp_out, q_out)
        self.out_fp_sq += float((fp_out * fp_out).sum().item())
        self.out_diff_sq += float(((fp_out - q_out) ** 2).sum().item())
        self.out_q_sq += float((q_out * q_out).sum().item())
        self.out_dot += float((fp_out * q_out).sum().item())
        self.n_elem += int(fp_out.numel())

    def update(self, fp_in, q_in, fp_out, q_out):
        fp_in, q_in = align_shapes(fp_in, q_in)
        self.in_fp_sq += float((fp_in * fp_in).sum().item())
        self.in_diff_sq += float(((fp_in - q_in) ** 2).sum().item())
        self._update_output_stats(fp_out, q_out)
        self.n_updates += 1

    def update_output_only(self, fp_out, q_out):
        """For folded BN: compare FP32 norm output with fused quant producer."""
        self._update_output_stats(fp_out, q_out)
        self.output_only_updates += 1

    def finalize(self):
        if self.n_elem == 0:
            return {
                "layer": self.layer,
                "type": self.type,
                "inherited_error": np.nan,
                "output_RelErr": np.nan,
                "delta_error": np.nan,
                "amplification_gain": np.nan,
                "CosineAct": np.nan,
                # "mse_output": np.nan,
                "sqnr_db": np.nan,
            }

        out_re = math.sqrt(self.out_diff_sq / max(self.out_fp_sq, 1e-12))

        if self.n_updates == 0 and self.output_only_updates > 0:
            in_re = np.nan
            delta = np.nan
            gain = np.nan
        else:
            if self.is_entry:
                in_re = 0.0
            else:
                in_re = math.sqrt(self.in_diff_sq / max(self.in_fp_sq, 1e-12))

            delta = out_re - in_re
            if in_re < 1e-8:
                gain = float("inf") if out_re > 1e-8 else 1.0
            else:
                gain = out_re / in_re

        cos = self.out_dot / max(
            math.sqrt(max(self.out_fp_sq, 1e-12)) *
            math.sqrt(max(self.out_q_sq, 1e-12)),
            1e-12,
        )

        # mse = self.out_diff_sq / max(self.n_elem, 1)

        if self.out_diff_sq < 1e-20:
            sqnr = float("inf")
        elif self.out_fp_sq < 1e-20:
            sqnr = float("-inf")
        else:
            sqnr = 10.0 * math.log10(self.out_fp_sq / self.out_diff_sq)

        return {
            "layer": self.layer,
            "type": self.type,
            "inherited_error": in_re,
            "output_RelErr": out_re,
            "delta_error": delta,
            "amplification_gain": gain,
            "CosineAct": cos,
            # "mse_output": mse,
            "sqnr_db": sqnr,
        }


class ActAcc:
    def __init__(self, name, channel_dim=1):
        self.layer = name
        self.channel_dim = channel_dim

        self.fp_n = 0
        self.q_n = 0

        self.fp_mean = 0.0
        self.fp_M2 = 0.0
        self.fp_M4 = 0.0

        self.q_mean = 0.0
        self.q_M2 = 0.0

        self.fp_zero = 0
        self.q_zero = 0

        self.q_channel_active = None

        self.shift_sum = None
        self.shift_count = 0

    @staticmethod
    def _wel4(n, mean, M2, M4, x):
        k = x.numel()
        if k == 0:
            return n, mean, M2, M4

        bm = float(x.mean())
        bv = float(x.var(unbiased=False))

        d = bm - mean
        nn_ = n + k

        new_mean = mean + d * k / nn_
        new_M2 = M2 + bv * k + d * d * n * k / nn_

        bM4 = float(((x - bm) ** 4).sum())
        new_M4 = (M4 + bM4 + (d ** 4) * n * k * (n * n - n * k + k * k) / (nn_ ** 3)
            + 6 * (d * d) * (n * n * bv * k + k * k * (M2 / max(n, 1))) / (nn_ ** 2))

        return nn_, new_mean, new_M2, new_M4

    @staticmethod
    def _wel2(n, mean, M2, x):
        k = x.numel()
        if k == 0:
            return n, mean, M2

        bm = float(x.mean())
        bv = float(x.var(unbiased=False))

        d = bm - mean
        nn_ = n + k

        new_mean = mean + d * k / nn_
        new_M2 = M2 + bv * k + d * d * n * k / nn_

        return nn_, new_mean, new_M2

    def update(self, fp_out, q_out):
        fp = fp_out.detach().to(torch.float32).cpu()
        q = q_out.detach().to(torch.float32).cpu()

        fp_flat = fp.flatten()
        q_flat = q.flatten()

        self.fp_n, self.fp_mean, self.fp_M2, self.fp_M4 = self._wel4(self.fp_n, self.fp_mean, self.fp_M2, self.fp_M4, fp_flat)

        self.q_n, self.q_mean, self.q_M2 = self._wel2(self.q_n, self.q_mean, self.q_M2, q_flat)

        self.fp_zero += int((fp == 0).sum().item())
        self.q_zero += int((q == 0).sum().item())

        if q.ndim >= 2:
            dims = tuple(d for d in range(q.ndim) if d != self.channel_dim)
            ch_active = (q.abs().amax(dim=dims) > 1e-6) if dims else (q.abs() > 1e-6)

            if self.q_channel_active is None:
                self.q_channel_active = ch_active.clone()
            else:
                m = min(self.q_channel_active.numel(), ch_active.numel())
                self.q_channel_active[:m] = self.q_channel_active[:m] | ch_active[:m]

        if fp.shape == q.shape and fp.ndim >= 2:
            diff = fp - q
            dims = tuple(d for d in range(diff.ndim) if d != self.channel_dim)

            ch_sum = diff.sum(dim=dims) if dims else diff

            ch_count = 1
            for d in dims:
                ch_count *= diff.shape[d]

            if self.shift_sum is None:
                self.shift_sum = ch_sum.clone()
            else:
                m = min(self.shift_sum.numel(), ch_sum.numel())
                self.shift_sum = self.shift_sum[:m] + ch_sum[:m]

            self.shift_count += ch_count

    def finalize(self):
        if self.fp_n == 0 or self.q_n == 0:
            return {
                "layer": self.layer,
                "variance_ratio": np.nan,
                # "act_kurtosis": np.nan,
                "zero_inflation": np.nan,
                "dead_neuron_ratio": np.nan,
                "mean_shift": np.nan,
            }

        fp_var = self.fp_M2 / max(self.fp_n, 1)
        q_var = self.q_M2 / max(self.q_n, 1)

        var_ratio = q_var / max(fp_var, 1e-12)

        # kurt = (
        #     (self.fp_M4 / self.fp_n) / (fp_var ** 2) - 3.0
        #     if self.fp_n > 0 and fp_var > 1e-12
        #     else float("nan"))

        zero_fp = self.fp_zero / max(self.fp_n, 1)
        zero_q = self.q_zero / max(self.q_n, 1)
        zero_infl = zero_q - zero_fp

        if self.q_channel_active is not None and self.q_channel_active.numel() > 0:
            dead_ratio = float(1.0 - self.q_channel_active.float().mean().item())
        else:
            dead_ratio = float("nan")

        if self.shift_sum is not None and self.shift_count > 0:
            ms = self.shift_sum / self.shift_count
            mean_shift = float(ms.abs().mean().item())
        else:
            mean_shift = float("nan")

        return {
            "layer": self.layer,
            "variance_ratio": var_ratio,
            # "act_kurtosis": kurt,
            "zero_inflation": zero_infl,
            "dead_neuron_ratio": dead_ratio,
            "mean_shift": mean_shift,
        }


def _invert_batchnorm_output_to_input(bn, y, fp_input=None):
   
    if not isinstance(bn, nn.BatchNorm2d):
        return None
    if not isinstance(y, torch.Tensor) or y.ndim < 2:
        return None

    device = y.device
    dtype = y.dtype
    c = y.shape[1]

    mean = bn.running_mean.detach().to(device=device, dtype=dtype)
    var = bn.running_var.detach().to(device=device, dtype=dtype)
    if mean.numel() != c or var.numel() != c:
        return None

    if bn.affine:
        gamma = bn.weight.detach().to(device=device, dtype=dtype)
        beta = bn.bias.detach().to(device=device, dtype=dtype)
    else:
        gamma = torch.ones_like(mean)
        beta = torch.zeros_like(mean)

    scale = gamma / torch.sqrt(var + float(bn.eps))
    shape = [1, c] + [1] * (y.ndim - 2)
    mean_v = mean.view(*shape)
    beta_v = beta.view(*shape)
    scale_v = scale.view(*shape)

    valid = scale_v.abs() > 1e-8
    safe_scale = torch.where(valid, scale_v, torch.ones_like(scale_v))
    x_hat = (y - beta_v) / safe_scale + mean_v

    if fp_input is not None and isinstance(fp_input, torch.Tensor) and fp_input.shape == y.shape:
        x_hat = torch.where(valid.expand_as(y), x_hat, fp_input.to(device=device, dtype=dtype))

    return x_hat


def analyze_propagation(fp32_model, int4_model, loader, device, mq_info, exec_order=None):
    fp32_model.eval().to(device)
    int4_model.eval().to(device)

    if exec_order is None:
        exec_order = get_execution_order(fp32_model, loader, device)

    fp_mods = dict(fp32_model.named_modules())
    q_mods = dict(int4_model.named_modules())

    exec_order = [n for n in exec_order if n in fp_mods]

    prop = OrderedDict(
        (n, PropAcc(n, fp_mods[n].__class__.__name__))
        for n in exec_order
    )
    actx = OrderedDict((n, ActAcc(n)) for n in exec_order)

    if exec_order:
        prop[exec_order[0]].mark_entry()

    fp_io = defaultdict(list)
    q_io = defaultdict(list)
    q_out_cache = defaultdict(list)
    handles = []

    def io_hook(cache, key):
        def hook(_m, inputs, output):
            if not isinstance(output, torch.Tensor):
                return
            if not inputs or not isinstance(inputs[0], torch.Tensor):
                return
            cache[key].append({
                "input": inputs[0].detach(),
                "output": output.detach(),
            })
        return hook

    def fq_hook(cache, key):
        def hook(_m, _inputs, output):
            if isinstance(output, torch.Tensor):
                cache[key].append({"output": output.detach()})
        return hook

    for name in exec_order:
        handles.append(fp_mods[name].register_forward_hook(io_hook(fp_io, name)))

        q_mod = q_mods.get(name)
        if q_mod is not None:
            handles.append(q_mod.register_forward_hook(io_hook(q_io, name)))

        fq_name = mq_info[name].get("post_act_fq")
        if fq_name is not None and fq_name in q_mods:
            handles.append(q_mods[fq_name].register_forward_hook(fq_hook(q_out_cache, name)))

    total = 0
    fp_correct = 0
    q_correct = 0
    flips = 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            fp_io.clear()
            q_io.clear()
            q_out_cache.clear()

            logits_fp = fp32_model(imgs)
            logits_q = int4_model(imgs)

            pred_fp = logits_fp.argmax(dim=1)
            pred_q = logits_q.argmax(dim=1)

            total += labels.size(0)
            fp_correct += int((pred_fp == labels).sum().item())
            q_correct += int((pred_q == labels).sum().item())
            flips += int(((pred_fp == labels) & (pred_q != labels)).sum().item())

            for name in exec_order:
                fp_list = fp_io.get(name, [])

                # --------------------------------------------------------
                # Folded normalization layer
                # --------------------------------------------------------
                if bool(mq_info[name].get("bn_folded", False)):
                    source = mq_info[name].get("folded_source_layer")
                    src_q_list = q_io.get(source, []) if source is not None else []
                    n_calls = min(len(fp_list), len(src_q_list))

                    for k in range(n_calls):
                        fp_in = fp_list[k]["input"]
                        fp_out = fp_list[k]["output"]

                        q_out = src_q_list[k]["output"]
                        q_in_equiv = _invert_batchnorm_output_to_input(
                            fp_mods[name], q_out, fp_input=fp_in
                        )

                        try:
                            if q_in_equiv is not None:
                                prop[name].update(fp_in, q_in_equiv, fp_out, q_out)
                            else:
                                prop[name].update_output_only(fp_out, q_out)
                            fa, qa = align_shapes(fp_out, q_out)
                            actx[name].update(fa, qa)
                        except ValueError:
                            continue
                    continue

                q_list = q_io.get(name, [])
                fq_list = q_out_cache.get(name, [])
                n_calls = min(len(fp_list), len(q_list))

                for k in range(n_calls):
                    fp_in = fp_list[k]["input"]
                    fp_out = fp_list[k]["output"]
                    q_in = q_list[k]["input"]

                    if k < len(fq_list):
                        q_out = fq_list[k]["output"]
                    else:
                        q_out = q_list[k]["output"]

                    try:
                        prop[name].update(fp_in, q_in, fp_out, q_out)
                        fa, qa = align_shapes(fp_out, q_out)
                        actx[name].update(fa, qa)
                    except ValueError:
                        continue

            if device.type == "cuda":
                torch.cuda.empty_cache()

    for h in handles:
        h.remove()

    df_p = pd.DataFrame([a.finalize() for a in prop.values()])
    df_a = pd.DataFrame([a.finalize() for a in actx.values()])

    w_rows = []
    for name in exec_order:
        row = {"layer": name}
        row.update(compute_weight_metrics(fp_mods[name], q_mods.get(name)))
        row["bn_folded"] = mq_info[name]["bn_folded"]
        row["folded_source_layer"] = mq_info[name].get("folded_source_layer")
        row["has_post_act_fq"] = len(mq_info[name].get("post_act_fqs", [])) > 0
        w_rows.append(row)

    df_w = pd.DataFrame(w_rows)

    df = df_p.merge(df_a, on="layer", how="left").merge(df_w, on="layer", how="left")

    exec_map = {n: i + 1 for i, n in enumerate(exec_order)}
    df["exec_order"] = df["layer"].map(exec_map)
    df = df.sort_values("exec_order").reset_index(drop=True)

    summary = {
        "fp32_acc": 100.0 * fp_correct / max(total, 1),
        "int4_acc": 100.0 * q_correct / max(total, 1),
        "acc_drop": 100.0 * (fp_correct - q_correct) / max(total, 1),
        "silent_flip_count": flips,
        "silent_flip_rate": 100.0 * flips / max(total, 1),
        "total_samples": total,
    }

    return df, summary, exec_order


# ============================================================
# ISOLATION SENSITIVITY
# ============================================================
def _quant_dequant_per_tensor_asymmetric(x: torch.Tensor, bit: int) -> torch.Tensor:
    x = x.detach()

    xmin, xmax = x.min(), x.max()

    if float((xmax - xmin).abs().item()) < 1e-12:
        return x

    qmin, qmax = 0, (1 << bit) - 1

    scale = torch.clamp(
        (xmax - xmin) / float(qmax - qmin),
        min=1e-12,
    )
    zp = torch.clamp(
        torch.round(qmin - xmin / scale),
        qmin,
        qmax,
    )

    q = torch.clamp(torch.round(x / scale + zp), qmin, qmax)

    return ((q - zp) * scale).to(x.dtype)


def _quantize_weight_per_channel_symmetric_tensor(w: torch.Tensor, bit: int) -> torch.Tensor:
    qmin = -(1 << (bit - 1))
    qmax = (1 << (bit - 1)) - 1

    if w.ndim >= 2:
        w_flat = w.reshape(w.shape[0], -1)
        max_per_ch = w_flat.abs().amax(dim=1)
        scale_per_ch = torch.clamp(max_per_ch / qmax, min=1e-12)

        view_shape = [w.shape[0]] + [1] * (w.ndim - 1)
        scale = scale_per_ch.view(*view_shape)

        q = torch.clamp(torch.round(w / scale), qmin, qmax)

        return (q * scale).to(w.dtype)

    max_abs = w.abs().max()
    scale = torch.clamp(max_abs / qmax, min=1e-12)

    q = torch.clamp(torch.round(w / scale), qmin, qmax)

    return (q * scale).to(w.dtype)


def _force_fq_quant_mode(fq):
    """Use an already-calibrated MQBench fake quantizer without updating its observer."""
    if fq is None:
        return
    if hasattr(fq, "observer_enabled"):
        try:
            fq.observer_enabled[0] = 0
        except Exception:
            pass
    if hasattr(fq, "fake_quant_enabled"):
        try:
            fq.fake_quant_enabled[0] = 1
        except Exception:
            pass


def patch_single_layer_quant_all_layers(
    fp32_model,
    quant_model,
    layer_name,
    mq_info,
    wbit=WBIT,
    abit=ABIT,
):
    fp_module = get_module_by_name(fp32_model, layer_name)
    q_module = get_module_by_name(quant_model, layer_name)

    if fp_module is None or not isinstance(fp_module, INTERESTED_TYPES):
        return (lambda: None), "unsupported"

    orig_forward = fp_module.forward
    weight_backup = None
    used_mq_weight = False
    used_fallback_weight = False

    # Weight perturbation for Conv/Linear.
    if isinstance(fp_module, WEIGHTED_TYPES) and getattr(fp_module, "weight", None) is not None:
        q_weight = get_quant_weight(q_module) if q_module is not None else None
        weight_backup = fp_module.weight.data.clone()

        if q_weight is not None and tuple(q_weight.shape) == tuple(fp_module.weight.shape):
            fp_module.weight.data.copy_(
                q_weight.to(fp_module.weight.device, fp_module.weight.dtype)
            )
            used_mq_weight = True
        else:
            fp_module.weight.data.copy_(
                _quantize_weight_per_channel_symmetric_tensor(
                    fp_module.weight.data, bit=wbit
                )
            )
            used_fallback_weight = True

    q_mods = dict(quant_model.named_modules())
    activation_fq_local = None
    fq_names = mq_info.get(layer_name, {}).get("post_act_fqs", [])

    if len(fq_names) == 1:
        activation_fq = q_mods.get(fq_names[0])
        if activation_fq is not None:
            try:
                activation_fq_local = copy.deepcopy(activation_fq).to(DEVICE).eval()
            except Exception:
                activation_fq_local = activation_fq
            _force_fq_quant_mode(activation_fq_local)

    used_mq_act = activation_fq_local is not None

    def wrapped_forward(*args, **kwargs):
        out = orig_forward(*args, **kwargs)
        if isinstance(out, torch.Tensor):
            if activation_fq_local is not None:
                out = activation_fq_local(out)
            else:
                out = _quant_dequant_per_tensor_asymmetric(out, bit=abit)
        return out

    fp_module.forward = wrapped_forward

    def restore():
        fp_module.forward = orig_forward
        if weight_backup is not None:
            fp_module.weight.data.copy_(weight_backup)

    parts = []
    if used_mq_weight:
        parts.append("mqbench-weight")
    elif used_fallback_weight:
        parts.append("fallback-weight")
    if used_mq_act:
        parts.append("mqbench-act")
    else:
        parts.append("fallback-act")

    return restore, "+".join(parts)


def isolation_sensitivity(
    fp32_model,
    quant_model,
    loader,
    device,
    exec_order,
    mq_info,
    fp32_correctness,
    wbit=WBIT,
    abit=ABIT,
) -> pd.DataFrame:
   
    fp32_model.eval().to(device)
    quant_model.eval().to(device)

    col_name = f"iso_sens_W{wbit}A{abit}"
    rows = []

    for i, name in enumerate(exec_order, 1):
        restore, source = patch_single_layer_quant_all_layers(
            fp32_model,
            quant_model,
            name,
            mq_info,
            wbit=wbit,
            abit=abit,
        )

        if source == "unsupported":
           
            rows.append({
                "layer": name,
                col_name: np.nan,
                "iso_se_pp": np.nan,
                "iso_lcb95_pp": np.nan,
                "iso_p_value": np.nan,
                "iso_positive_flips": np.nan,
                "iso_negative_flips": np.nan,
                "iso_quant_source": source,
            })
            print(f"     iso {i:3d}/{len(exec_order)} {name:<55} drop=N/A (unsupported module)")
            continue

        try:
            _, layer_corr = evaluate_accuracy_and_correctness(fp32_model, loader, device)
        finally:
            restore()

        stats = paired_effect_stats(
            fp32_correctness, layer_corr, direction="drop"
        )
        drop = stats["effect_pp"]

        rows.append({
            "layer": name,
            col_name: drop,
            "iso_se_pp": stats["se_pp"],
            "iso_lcb95_pp": stats["lcb95_pp"],
            "iso_p_value": stats["p_value"],
            "iso_positive_flips": stats["positive_flips"],
            "iso_negative_flips": stats["negative_flips"],
            "iso_quant_source": source,
        })
        print(
            f"     iso {i:3d}/{len(exec_order)} {name:<55} "
            f"drop={drop:.4f}  LCB95={stats['lcb95_pp']:.4f}  source={source}"
        )

        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    out = pd.DataFrame(rows)
    out["iso_q_value"] = benjamini_hochberg(
        pd.to_numeric(out["iso_p_value"], errors="coerce").to_numpy()
    )
    return out

def list_all_fakequants(int4_model):
    return [
        m for m in int4_model.modules()
        if "FakeQuantize" in m.__class__.__name__
        or "FakeQuantizer" in m.__class__.__name__]


def toggle_fq(fq, enable):
    if fq is None:
        return

    if hasattr(fq, "fake_quant_enabled"):
        try:
            fq.fake_quant_enabled[0] = 1 if enable else 0
            return
        except Exception:
            pass

    if enable and hasattr(fq, "enable_fake_quant"):
        fq.enable_fake_quant()
    elif not enable and hasattr(fq, "disable_fake_quant"):
        fq.disable_fake_quant()


def set_all_fq(int4_model, enable):
    for fq in list_all_fakequants(int4_model):
        toggle_fq(fq, enable)


def get_layer_fqs(int4_model, layer_name, mq_info):
    q_mods = dict(int4_model.named_modules())

    m = q_mods.get(layer_name)
    w_fq = getattr(m, "weight_fake_quant", None) if m is not None else None

    a_fqs = []
    for a_fq_name in mq_info[layer_name].get("post_act_fqs", []):
        a_fq = q_mods.get(a_fq_name)
        if a_fq is not None:
            a_fqs.append(a_fq)

    return w_fq, a_fqs


def toggle_many_fq(fqs, enable):
    for fq in fqs:
        toggle_fq(fq, enable)


def restoration_sensitivity(
    int4_model,
    int4_acc,
    int4_correctness,
    loader,
    device,
    exec_order,
    mq_info,
):
    """Weight/activation restoration with paired statistical support.

    The old implementation also evaluated weight+activation restored together
    for every layer. That third pass is not used in flagging, so it is removed
    to reduce runtime.
    """
    set_all_fq(int4_model, True)
    rows = []

    for i, name in enumerate(exec_order, 1):
        w_fq, a_fqs = get_layer_fqs(int4_model, name, mq_info)

        row = {
            "layer": name,
            "restore_gain_weight": np.nan,
            "restore_gain_weight_se_pp": np.nan,
            "restore_gain_weight_lcb95_pp": np.nan,
            "restore_gain_weight_p_value": np.nan,
            "restore_gain_activation": np.nan,
            "restore_gain_activation_se_pp": np.nan,
            "restore_gain_activation_lcb95_pp": np.nan,
            "restore_gain_activation_p_value": np.nan,
        }

        if w_fq is not None:
            toggle_fq(w_fq, False)
            try:
                _, corr = evaluate_accuracy_and_correctness(int4_model, loader, device)
            finally:
                toggle_fq(w_fq, True)

            s = paired_effect_stats(int4_correctness, corr, direction="gain")
            row.update({
                "restore_gain_weight": s["effect_pp"],
                "restore_gain_weight_se_pp": s["se_pp"],
                "restore_gain_weight_lcb95_pp": s["lcb95_pp"],
                "restore_gain_weight_p_value": s["p_value"],
            })

        if len(a_fqs) > 0:
            toggle_many_fq(a_fqs, False)
            try:
                _, corr = evaluate_accuracy_and_correctness(int4_model, loader, device)
            finally:
                toggle_many_fq(a_fqs, True)

            s = paired_effect_stats(int4_correctness, corr, direction="gain")
            row.update({
                "restore_gain_activation": s["effect_pp"],
                "restore_gain_activation_se_pp": s["se_pp"],
                "restore_gain_activation_lcb95_pp": s["lcb95_pp"],
                "restore_gain_activation_p_value": s["p_value"],
            })

        rows.append(row)
        print(f"     restore {i:3d}/{len(exec_order)} {name:<55} "
            f"W={row['restore_gain_weight'] if pd.notna(row['restore_gain_weight']) else 'N/A'}  "
            f"A={row['restore_gain_activation'] if pd.notna(row['restore_gain_activation']) else 'N/A'}")

        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    out = pd.DataFrame(rows)
    out["restore_gain_weight_q_value"] = benjamini_hochberg(pd.to_numeric(out["restore_gain_weight_p_value"], errors="coerce").to_numpy())
    out["restore_gain_activation_q_value"] = benjamini_hochberg(pd.to_numeric(out["restore_gain_activation_p_value"], errors="coerce").to_numpy())

    return out


# ============================================================
# PERFORMANCE-GROUNDED FLAGGING
# ============================================================
def sensitivity_column():
    return f"iso_sens_W{WBIT}A{ABIT}"


def flag_layers(df, iso_col):
    
    out = df.copy()

    weighted = ("Conv2d", "Linear")
    norm = ("BatchNorm2d",)
    activ = ("ReLU", "ReLU6", "MaxPool2d", "AvgPool2d", "AdaptiveAvgPool2d")

    relerr = pd.to_numeric(out["output_RelErr"], errors="coerce")
    delta = pd.to_numeric(out["delta_error"], errors="coerce")
    gain = pd.to_numeric(out["amplification_gain"], errors="coerce")
    zero_inf = pd.to_numeric(out.get("zero_inflation", np.nan), errors="coerce")
    dead = pd.to_numeric(out.get("dead_neuron_ratio", np.nan), errors="coerce")

    def mechanism_row(r):
        t = r["type"]
        flags = []

        d = r.get("delta_error", np.nan)
        g = r.get("amplification_gain", np.nan)

        if pd.notna(d) and d > 0:
            flags.append("injection")
        if pd.notna(g) and np.isfinite(g) and g > 1:
            flags.append("amplification")

        if t in weighted:
            if (
                pd.notna(r.get("weight_cond_fp32"))
                and pd.notna(r.get("weight_cond_int4"))
                and np.isfinite(r["weight_cond_fp32"])
                and r["weight_cond_fp32"] > 0
                and (
                    np.isinf(r["weight_cond_int4"])
                    or r["weight_cond_int4"] > 2 * r["weight_cond_fp32"]
                )
            ):
                flags.append("conditioning-broken")

        if t in norm:
            if not bool(r.get("bn_folded", False)) and bool(r.get("has_post_act_fq", False)):
                flags.append("bn-unfolded")

        if t in activ:
            if pd.notna(r.get("dead_neuron_ratio")) and r["dead_neuron_ratio"] > 0:
                flags.append("dead-channels")
            if pd.notna(r.get("zero_inflation")) and r["zero_inflation"] > 0:
                flags.append("zero-inflation")

        return flags

    mech = out.apply(mechanism_row, axis=1)
    out["mechanism_flags"] = mech.apply(lambda lst: "+".join(lst) if lst else "none")
    out["has_mechanism"] = mech.apply(lambda lst: len(lst) > 0)

    # --------------------------------------------------------
    # 1) Paired task-performance evidence.
    # --------------------------------------------------------
    iso_effect = pd.to_numeric(out[iso_col], errors="coerce")
    iso_q = pd.to_numeric(out["iso_q_value"], errors="coerce")
    w_effect = pd.to_numeric(out["restore_gain_weight"], errors="coerce")
    w_q = pd.to_numeric(out["restore_gain_weight_q_value"], errors="coerce")
    a_effect = pd.to_numeric(out["restore_gain_activation"], errors="coerce")
    a_q = pd.to_numeric(out["restore_gain_activation_q_value"], errors="coerce")

    is_fragile = (iso_effect.ge(MIN_ISO_DROP_PP) & iso_q.le(FDR_Q)).fillna(False)
    is_w_contrib = (w_effect.ge(MIN_RESTORE_GAIN_PP) & w_q.le(FDR_Q)).fillna(False)
    is_a_contrib = (a_effect.ge(MIN_RESTORE_GAIN_PP) & a_q.le(FDR_Q)).fillna(False)

    supported_iso = iso_effect.where(iso_q.le(FDR_Q), np.nan)
    supported_w = w_effect.where(w_q.le(FDR_Q), np.nan)
    supported_a = a_effect.where(a_q.le(FDR_Q), np.nan)

    task_effect = pd.concat(
        [supported_iso, supported_w, supported_a], axis=1
    ).max(axis=1, skipna=True)
    has_task_evidence = task_effect.ge(MIN_ISO_DROP_PP).fillna(False)

    # --------------------------------------------------------
    # 2) All-layer numerical-instability evidence.
    # --------------------------------------------------------
    conditioning_broken = pd.Series(False, index=out.index)
    is_weighted = out["type"].isin(weighted)
    valid_cond = (
        is_weighted
        & pd.to_numeric(out["weight_cond_fp32"], errors="coerce").gt(0)
    )
    c0 = pd.to_numeric(out["weight_cond_fp32"], errors="coerce")
    c1 = pd.to_numeric(out["weight_cond_int4"], errors="coerce")
    conditioning_broken = (
        valid_cond
        & (
            np.isinf(c1)
            | (c1 > 2.0 * c0)
        )
    ).fillna(False)

    meaningful_output_error = relerr.ge(INST_RELERR_MIN).fillna(False)
    local_injection = delta.ge(INST_DELTA_MIN).fillna(False)
    local_amplification = gain.ge(INST_GAIN_MIN).fillna(False)
    structural_activation = (
        zero_inf.ge(INST_ZERO_INFLATION_MIN).fillna(False)
        | dead.ge(INST_DEAD_RATIO_MIN).fillna(False)
    )

    corroborating_signal = (
        local_injection
        | local_amplification
        | structural_activation
        | conditioning_broken)

    strong_instability = meaningful_output_error & corroborating_signal

    out["inst_meaningful_output_error"] = meaningful_output_error
    out["inst_local_injection"] = local_injection
    out["inst_local_amplification"] = local_amplification
    out["inst_structural_activation"] = structural_activation
    out["inst_conditioning_broken"] = conditioning_broken
    out["has_strong_instability"] = strong_instability

    out["is_fragile"] = is_fragile
    out["is_weight_contributor"] = is_w_contrib
    out["is_activation_contributor"] = is_a_contrib
    out["task_effect_pp"] = task_effect
    out["has_task_evidence"] = has_task_evidence
    out["has_causal_evidence"] = has_task_evidence

    is_bn = out["type"].eq("BatchNorm2d")
    out["has_bn_propagation_evidence"] = strong_instability & is_bn
    out["has_non_bn_propagation_evidence"] = strong_instability & ~is_bn
    out["has_propagation_evidence"] = strong_instability


    out["is_flagged"] = has_task_evidence | strong_instability

    out["is_collective_failure"] = out["has_mechanism"] & ~out["is_flagged"]
    return out


def assign_layer_role(df):
    out = df.copy()

    def role(r):
        g = r["amplification_gain"]
        d = r["delta_error"]
        inh = r["inherited_error"]

        if pd.notna(g) and not np.isfinite(g):
            return "injector" if (pd.notna(d) and d > 0) else "stable"
        if pd.notna(g) and g > 1.0 and pd.notna(d) and d > 0:
            return "amplifier"
        if pd.notna(g) and g > 1.0 and pd.notna(d) and d <= 0:
            return "pure_amplifier"
        if pd.notna(g) and abs(g - 1.0) <= 0.05:
            return "propagator"
        if pd.notna(g) and g < 1.0 and pd.notna(inh) and inh > 0.01:
            return "damper"
        if pd.notna(inh) and inh <= 0.01 and pd.notna(d) and d > 0:
            return "injector"
        return "stable"

    out["layer_role"] = out.apply(role, axis=1)

    candidates = out[
        out["is_flagged"]
        & out["mechanism_flags"].str.contains("amplification|injection", regex=True, na=False)]

    out["is_origin"] = False

    if len(candidates):
        out.loc[candidates["exec_order"].idxmin(), "is_origin"] = True

    return out


def classify_failure_mode(df):
    out = df.copy()

    weighted = ("Conv2d", "Linear")
    norm = ("BatchNorm2d",)
    activ = ("ReLU", "ReLU6", "MaxPool2d", "AvgPool2d", "AdaptiveAvgPool2d")

    flagged = out[out["is_flagged"]]

    if len(flagged) == 0:
        out["failure_mode"] = "stable"
        return out

    def med(col):
        s = flagged[col].replace([np.inf, -np.inf], np.nan).dropna()
        s = s[s.abs() > 1e-12]
        return float(s.median()) if len(s) else float("inf")

    thr = {
        "weight_RelErr": med("weight_RelErr"),
        "weight_outlier_ratio_int4": med("weight_outlier_ratio_int4"),
        "zero_inflation": med("zero_inflation"),
        "mean_shift": med("mean_shift"),
    }

    def route(r):
        if not r["is_flagged"]:
            return "stable"

        t = r["type"]
        modes = []

        if t in weighted:
            if pd.notna(r["weight_RelErr"]) and r["weight_RelErr"] > thr["weight_RelErr"]:
                modes.append("weight-limited")

            if (
                pd.notna(r["weight_outlier_ratio_int4"])
                and np.isfinite(r["weight_outlier_ratio_int4"])
                and r["weight_outlier_ratio_int4"] > thr["weight_outlier_ratio_int4"]
            ):
                modes.append("weight-outlier")

            if "conditioning-broken" in str(r["mechanism_flags"]):
                modes.append("ill-conditioned")

            if pd.notna(r["zero_inflation"]) and r["zero_inflation"] > thr["zero_inflation"]:
                modes.append("activation-outlier")

            if pd.notna(r["mean_shift"]) and r["mean_shift"] > thr["mean_shift"]:
                modes.append("bias-correctable")

        elif t in norm:
            if "bn-unfolded" in str(r["mechanism_flags"]):
                modes.append("bn-unfolded")

            if "amplification" in str(r["mechanism_flags"]):
                modes.append("inherits-from-upstream")

        elif t in activ:
            if "dead-channels" in str(r["mechanism_flags"]):
                modes.append("dead-channels")

            if pd.notna(r["zero_inflation"]) and r["zero_inflation"] > thr["zero_inflation"]:
                modes.append("activation-clipping")

            if pd.notna(r["output_RelErr"]) and r["output_RelErr"] > 0.95:
                modes.append("signal-collapse")

        return "+".join(modes) if modes else "mixed"

    out["failure_mode"] = out.apply(route, axis=1)

    return out


# ============================================================
# CSV WRITER
# ============================================================
def df_to_csv_with_labels(df, path):
    weight_cols = [
        "weight_RelErr", "weight_Cosine",
        "weight_cond_fp32", "weight_cond_int4",
        "weight_outlier_ratio_fp32", "weight_outlier_ratio_int4",
    ]

    no_weight_types = {
        "BatchNorm2d",
        "ReLU", "ReLU6",
        "MaxPool2d", "AvgPool2d", "AdaptiveAvgPool2d",
    }

    out = df.copy()

    for col in weight_cols:
        if col not in out.columns:
            continue

        new_vals = []

        for _, row in out[[col, "type"]].iterrows():
            v = row[col]
            t = row["type"]

            if pd.isna(v):
                new_vals.append("N/A" if t in no_weight_types else "NaN")
            elif np.isinf(v):
                new_vals.append("inf")
            else:
                new_vals.append(v)

        out[col] = new_vals

    out.to_csv(path, index=False)


# ============================================================
# MAIN
# ============================================================
def main():
    validate_config()
    set_seed(RANDOM_SEED)
    print(f"[init] device={DEVICE}  calib={CALIB_SIZE}  sens={SENS_SIZE}")
    print(f"[init] model={MODEL_NAME}  bits=W{WBIT}A{ABIT}  a_observer={A_OBSERVER}")

    dataset, calib_loader, sens_loader, calib_idx, sens_idx = build_loaders()

    print("[1] FP32 model ...")
    fp32 = build_model_fp32().to(DEVICE)

    print("[2] MQBench quantized model ...")
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    quant_setup_start = time.perf_counter()

    qm = build_model_fp32()

    cfg = {
        "extra_qconfig_dict": {
            "w_observer": "MinMaxObserver",
            "a_observer": A_OBSERVER,
            "w_fakequantize": "FixedFakeQuantize",
            "a_fakequantize": "FixedFakeQuantize",
            "w_qscheme": {
                "bit": WBIT,
                "symmetry": True,
                "per_channel": True,
                "pot_scale": False,
            },
            "a_qscheme": {
                "bit": ABIT,
                "symmetry": False,
                "per_channel": False,
                "pot_scale": False,
            },
        }
    }

    int4 = prepare_by_platform(qm, BackendType.Academic, cfg).to(DEVICE).eval()

    print("[3] Calibrating ...")
    enable_calibration(int4)

    with torch.no_grad():
        for imgs, _ in calib_loader:
            _ = int4(imgs.to(DEVICE, non_blocking=True))

    enable_quantization(int4)
    verify_quantizer_bitwidths(int4)

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    quant_setup_time_s = time.perf_counter() - quant_setup_start

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    analysis_start = time.perf_counter()

    print(f"     quantized-model setup/calibration (excluded): {quant_setup_time_s:.2f} s")

    print("[4] MQBench graph inspection ...")
    
    exec_order = get_execution_order(fp32, calib_loader, DEVICE)
    fp_mods = dict(fp32.named_modules())
    exec_order = [n for n in exec_order if n in fp_mods]

    mq_info = inspect_mqbench_graph(int4, fp32, exec_order)

    print("[5] Propagation + activation analysis ...")
    df, summary, exec_order = analyze_propagation(fp32, int4, calib_loader, DEVICE, mq_info, exec_order=exec_order)

    print(f"     calib: fp32={summary['fp32_acc']:.2f}  "
        f"int4={summary['int4_acc']:.2f}  "
        f"drop={summary['acc_drop']:.2f}")

    print(f"[6] {SENS_SIZE}-image paired accuracy baselines ...")
    fp32_acc_sens, fp32_corr_sens = evaluate_accuracy_and_correctness(fp32, sens_loader, DEVICE)
    int4_acc_sens, int4_corr_sens = evaluate_accuracy_and_correctness(int4, sens_loader, DEVICE)
    drop = fp32_acc_sens - int4_acc_sens

    print(f"     fp32={fp32_acc_sens:.2f}  "
        f"int4={int4_acc_sens:.2f}  "
        f"drop={drop:.2f}")
    print(f"     actionable thresholds: iso_drop>={MIN_ISO_DROP_PP:.2f} pp, "
        f"restore_gain>={MIN_RESTORE_GAIN_PP:.2f} pp, BH-FDR q<={FDR_Q:.2f}")
    print(f"     numerical-instability floors: relerr>={INST_RELERR_MIN:.4f} AND "
        f"(delta>={INST_DELTA_MIN:.4f} OR gain>={INST_GAIN_MIN:.2f} OR "
        f"zero/dead>={INST_ZERO_INFLATION_MIN:.2f})")

    if int4_acc_sens < 5.0 and fp32_acc_sens > 20.0:
        print("[diagnostic] Very large PTQ degradation for this architecture/configuration. "
            "The analysis will continue; low W4A4 accuracy is not treated as a code failure.")

    iso_col = f"iso_sens_W{WBIT}A{ABIT}"

    print(f"[7] All-FP32-layer iso_sens (W{WBIT}A{ABIT}); BN/ReLU/Pool are NEVER excluded by MQBench fusion ...")
    df_iso = isolation_sensitivity(
        fp32, int4, sens_loader, DEVICE, exec_order, mq_info,
        fp32_correctness=fp32_corr_sens, wbit=WBIT, abit=ABIT
    )

    iso_lookup = df_iso.set_index("layer")[iso_col]
    must_have_iso = [
        n for n in exec_order
        if isinstance(fp_mods[n], (
            nn.BatchNorm2d, nn.ReLU, nn.ReLU6,
            nn.MaxPool2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d
        ))
    ]
    missing_iso = [
        n for n in must_have_iso
        if n not in iso_lookup.index or pd.isna(iso_lookup.loc[n])
    ]
    if missing_iso:
        raise RuntimeError("[sanity] BN/ReLU/Pool layers lost from isolated sensitivity: "+ ", ".join(missing_iso[:20]))

    print( f"     iso coverage: {len(df_iso)}/{len(exec_order)} FP32 layers; "
        f"BN/ReLU/Pool coverage={len(must_have_iso) - len(missing_iso)}/{len(must_have_iso)}")

    print("[8] restoration gain (weight / activation) with paired statistics ...")
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    restoration_start = time.perf_counter()

    df_res = restoration_sensitivity(
        int4, int4_acc_sens, int4_corr_sens, sens_loader, DEVICE, exec_order, mq_info
    )

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    restoration_gain_time_s = time.perf_counter() - restoration_start

    df = df.merge(df_iso, on="layer", how="left").merge(df_res, on="layer", how="left")

    print("[9] Flagging (paired task evidence + corroborated all-layer numerical instability) ...")
    df = flag_layers(df, iso_col)
    df = assign_layer_role(df)
    df = classify_failure_mode(df)

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    analysis_total_time_s = time.perf_counter() - analysis_start
    analysis_core_time_s = max(0.0, analysis_total_time_s - restoration_gain_time_s)

    n_flagged = int(df["is_flagged"].sum())
    n_mech = int(df["has_mechanism"].sum())
    n_fragile = int(df["is_fragile"].sum())
    n_w = int(df["is_weight_contributor"].sum())
    n_a = int(df["is_activation_contributor"].sum())
    n_task = int(df["has_task_evidence"].sum())
    n_bn_prop = int(df["has_bn_propagation_evidence"].sum())
    n_non_bn_prop = int(df["has_non_bn_propagation_evidence"].sum())
    n_prop = int(df["has_propagation_evidence"].sum())
    n_strong = int(df["has_strong_instability"].sum())
    n_coll = int(df["is_collective_failure"].sum())

    print(f"     fragile: {n_fragile}  "
        f"weight-contrib: {n_w}  "
        f"activation-contrib: {n_a}  "
        f"unique task-evidence: {n_task}")
    print(f"     BN strong-instability: {n_bn_prop}  "
        f"non-BN strong-instability: {n_non_bn_prop}  "
        f"unique strong-instability: {n_strong}  "
        f"collective-only: {n_coll}")

    print(f"     mechanism (any): {n_mech}/{len(df)}  "
        f"is_flagged: {n_flagged}/{len(df)}")

    print(f"     analysis core time (excluding restoration): {analysis_core_time_s:.2f} s")
    print(f"     restoration gain time: {restoration_gain_time_s:.2f} s")
    print(f"     total analysis/data collection time: {analysis_total_time_s:.2f} s")
    print(f"     quantized-model setup time (excluded): {quant_setup_time_s:.2f} s")

    if drop > 5.0 and n_flagged == 0:
        print(f"[diagnostic] acc_drop={drop:.2f}% but zero layers passed the paired "
            "practical+statistical selection rule. CSV is still written for inspection.")

    os.makedirs(OUTDIR, exist_ok=True)

    main_csv = os.path.join(OUTDIR, f"instability_{MODEL_NAME}_W{WBIT}A{ABIT}.csv")
    summary_csv = os.path.join(OUTDIR, f"model_summary_{MODEL_NAME}_W{WBIT}A{ABIT}.csv")

    df_to_csv_with_labels(df, main_csv)

    pd.DataFrame([{
        "model": MODEL_NAME,
        "wbit": WBIT,
        "abit": ABIT,
        "calib_size": CALIB_SIZE,
        "sens_size": SENS_SIZE,
        "seed": RANDOM_SEED,
        "min_iso_drop_pp": MIN_ISO_DROP_PP,
        "min_restore_gain_pp": MIN_RESTORE_GAIN_PP,
        "inst_relerr_min": INST_RELERR_MIN,
        "inst_delta_min": INST_DELTA_MIN,
        "inst_gain_min": INST_GAIN_MIN,
        "inst_zero_inflation_min": INST_ZERO_INFLATION_MIN,
        "inst_dead_ratio_min": INST_DEAD_RATIO_MIN,
        "fdr_q": FDR_Q,
        "fp32_acc_sens": fp32_acc_sens,
        "int4_acc_sens": int4_acc_sens,
        "acc_drop_sens": drop,
        "n_fragile": n_fragile,
        "n_weight_contributor": n_w,
        "n_activation_contributor": n_a,
        "n_task_evidence": n_task,
        "n_bn_propagation": n_bn_prop,
        "n_non_bn_propagation": n_non_bn_prop,
        "n_propagation": n_prop,
        "n_collective_failure": n_coll,
        "n_with_mechanism": n_mech,
        "n_strong_instability": n_strong,
        "n_flagged": n_flagged,
        "n_layers": len(df),
        "quant_setup_time_s_excluded": quant_setup_time_s,
        "analysis_core_time_s_excluding_restoration": analysis_core_time_s,
        "restoration_gain_time_s": restoration_gain_time_s,
        "analysis_total_time_s": analysis_total_time_s,
    }]).to_csv(summary_csv, index=False)


if __name__ == "__main__":
    main()
