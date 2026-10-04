import os
import gc
import argparse
import copy
import time
import random
import threading
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models

from mqbench.prepare_by_platform import prepare_by_platform, BackendType
from mqbench.utils.state import enable_calibration, enable_quantization
from mqbench.advanced_ptq import ptq_reconstruction

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


# ============================================================
# Configuration
# ============================================================
SEED = 69
BATCH_SIZE = 32
NUM_WORKERS = 0
NUM_CALIB = 1024
NUM_EVAL = 5000
FULL_EVAL = True
REPAIR_FIT_SAMPLES = 512

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGENET_ROOT = "./test_dataset"
OUTDIR = "quantdir_classification_results"


# ============================================================
# Model selection 
# ============================================================
MODEL_NAME = "resnet18"  # default; can be overridden with --model

SUPPORTED_MODELS = {
    "resnet18":      (models.resnet18,      models.ResNet18_Weights.IMAGENET1K_V1),
    "resnet34":      (models.resnet34,      models.ResNet34_Weights.IMAGENET1K_V1),
    "resnet50":      (models.resnet50,      models.ResNet50_Weights.IMAGENET1K_V1),
    "vgg16":         (models.vgg16,         models.VGG16_Weights.IMAGENET1K_V1),
    "vgg19":         (models.vgg19,         models.VGG19_Weights.IMAGENET1K_V1),
    "mobilenet_v2":  (models.mobilenet_v2,  models.MobileNet_V2_Weights.IMAGENET1K_V1),
    "regnet_x_800mf":(models.regnet_x_800mf,models.RegNet_X_800MF_Weights.IMAGENET1K_V1),
}
MODEL_ALIASES = {
    "mobilenetv2": "mobilenet_v2",
    "mobilenet-v2": "mobilenet_v2",
    "regnetx800mf": "regnet_x_800mf",
    "regnet-x-800mf": "regnet_x_800mf",
}


def canonical_model_name(name):
    key = name.strip().lower()
    key = MODEL_ALIASES.get(key, key)
    if key not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported MODEL_NAME={name!r}. Supported: {list(SUPPORTED_MODELS)}")
    return key


def get_model_builder_and_weights(name=MODEL_NAME):
    key = canonical_model_name(name)
    return key, *SUPPORTED_MODELS[key]


def build_fp32_model(name=MODEL_NAME, device=None):
    key, builder, weights = get_model_builder_and_weights(name)
    model = builder(weights=weights).eval()
    if device is not None:
        model = model.to(device)
    return model


def get_imagenet_transform(name=MODEL_NAME):
  
    _, _, weights = get_model_builder_and_weights(name)
    return weights.transforms()

WEIGHT_BITS = 4
ACTIVATION_BITS = 8  

MODEL_NAME = canonical_model_name(MODEL_NAME)
ANALYSIS_CSV = f"./ptq_metrics/instability_{MODEL_NAME}_W{WEIGHT_BITS}A{ACTIVATION_BITS}.csv"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run QuantDIR-selected repair for an ImageNet model under W4A4 or W4A8 quantization."))
    parser.add_argument(
        "--model",
        required=True,
        help=(
            "Model name: resnet18, resnet34, resnet50, mobilenet_v2, "
            "vgg16, vgg19, or regnet_x_800mf."
        ),
    )

    quant_group = parser.add_mutually_exclusive_group(required=True)
    quant_group.add_argument(
        "--w4a4",
        action="store_const",
        const="w4a4",
        dest="quant_mode",
        help="Use 4-bit weights and 4-bit activations.",)
    quant_group.add_argument(
        "--w4a8",
        action="store_const",
        const="w4a8",
        dest="quant_mode",
        help="Use 4-bit weights and 8-bit activations.",)
    quant_group.add_argument(
        "--quant",
        choices=("w4a4", "w4a8"),
        dest="quant_mode",
        help="Quantization mode; equivalent to --w4a4 or --w4a8.",)
    return parser.parse_args()


def configure_experiment(model_name, quant_mode):
    """Apply CLI selections before loaders/models/CSV paths are created."""
    global MODEL_NAME, WEIGHT_BITS, ACTIVATION_BITS, ANALYSIS_CSV

    MODEL_NAME = canonical_model_name(model_name)
    WEIGHT_BITS = 4

    mode = str(quant_mode).strip().lower()
    if mode == "w4a4":
        ACTIVATION_BITS = 4
    elif mode == "w4a8":
        ACTIVATION_BITS = 8
    else:
        raise ValueError(f"Unsupported quantization mode: {quant_mode!r}")

    ANALYSIS_CSV = (f"./ptq_metrics/instability_{MODEL_NAME}_"f"W{WEIGHT_BITS}A{ACTIVATION_BITS}.csv")

MAX_PRODUCERS_PER_FLAGGED = 2

# Selected MQBench  settings.
GUIDED_ROUND_MAX_COUNT = 2000
GUIDED_ROUND_WARM_UP = 0.20
GUIDED_ROUND_REG_WEIGHT = 0.05
GUIDED_ROUND_B_RANGE = [20, 2]
GUIDED_ROUND_MODE = "learned_hard_sigmoid"
GUIDED_ROUND_PROB = 0.5
GUIDED_KEEP_GPU = True
GUIDED_CALIB_BATCHES = 16
GUIDED_LEARN_ACT_SCALE = True
GUIDED_ACT_SCALE_LR = 4.0e-5


# ============================================================
# Whole-repair memory tracking
# ============================================================
def _cpu_process_tree_rss_mb():
    """Current RSS of this process plus active DataLoader children."""
    if not HAS_PSUTIL:
        return 0.0

    try:
        process = psutil.Process()
        total = process.memory_info().rss
        for child in process.children(recursive=True):
            try:
                total += child.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return total / 1024**2
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return 0.0


class RepairMemoryTracker:
    """Track the highest memory usage over the complete repair pipeline."""

    def __init__(self, device=DEVICE, sample_interval=0.05):
        self.device = device
        self.sample_interval = max(float(sample_interval), 0.01)
        self._stop = threading.Event()
        self._thread = None
        self._cpu_highest_mb = 0.0

    def _sample_cpu(self):
        while not self._stop.is_set():
            self._cpu_highest_mb = max(self._cpu_highest_mb, _cpu_process_tree_rss_mb())
            self._stop.wait(self.sample_interval)

    def start(self):
        gc.collect()
        self._stop.clear()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        else:
            if not HAS_PSUTIL:
                raise RuntimeError(
                    "CPU memory tracking requires psutil. Install it with: pip install psutil")
            self._cpu_highest_mb = _cpu_process_tree_rss_mb()
            self._thread = threading.Thread(target=self._sample_cpu, daemon=True)
            self._thread.start()

        return self

    def stop(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            highest = torch.cuda.max_memory_allocated(self.device) / 1024**2
            return {
                "device": "GPU",
                "highest_memory_mb": highest,
            }

        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

        current_mb = _cpu_process_tree_rss_mb()
        self._cpu_highest_mb = max(self._cpu_highest_mb, current_mb)
        return {
            "device": "CPU",
            "highest_memory_mb": self._cpu_highest_mb,
        }


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# Data
# ============================================================

def get_loaders(num_calib=NUM_CALIB, num_eval=NUM_EVAL, full_eval=FULL_EVAL):
    """Create deterministic calibration, repair-fit, and evaluation loaders."""
    transform = get_imagenet_transform(MODEL_NAME)
    dataset = datasets.ImageFolder(root=IMAGENET_ROOT, transform=transform)
    indices = list(range(len(dataset)))
    random.Random(SEED).shuffle(indices)

    if len(indices) <= num_calib:
        raise RuntimeError("Dataset must contain more images than NUM_CALIB.")

    calib_idx = indices[:num_calib]
    remaining = indices[num_calib:]
    eval_idx = remaining if full_eval else remaining[:min(num_eval, len(remaining))]

    repair_fit_idx = calib_idx[:min(REPAIR_FIT_SAMPLES, len(calib_idx))]

    common = dict(
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )
    calib_loader = DataLoader(Subset(dataset, calib_idx), **common)
    repair_fit_loader = DataLoader(Subset(dataset, repair_fit_idx), **common)
    eval_loader = DataLoader(Subset(dataset, eval_idx), **common)
    return calib_loader, repair_fit_loader, eval_loader


@torch.no_grad()
def evaluate(model, loader, device, max_batches=None):
    model.eval()
    correct = total = 0
    for i, (x, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        pred = model(x).argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return 100.0 * correct / max(total, 1)




# ============================================================
# Build calibrated Quantized model
# ============================================================
def get_quant_config():
    return {
        "extra_qconfig_dict": {
            "w_observer": "MinMaxObserver",
            "a_observer": "EMAMSEObserver",
            "w_fakequantize": "AdaRoundFakeQuantize",
            "a_fakequantize": "FixedFakeQuantize",
            "w_qscheme": {
                "bit": WEIGHT_BITS,
                "symmetry": True,
                "per_channel": True,
                "pot_scale": False,
            },
            "a_qscheme": {
                "bit": ACTIVATION_BITS,
                "symmetry": False,
                "per_channel": False,
                "pot_scale": False,
            },
        },
        "module_qconfig_dict": {
            "conv1": {
                "w_qscheme": {
                    "bit": 8,
                    "symmetry": True,
                    "per_channel": True,
                    "pot_scale": False,
                }
            },
            "fc": {
                "w_qscheme": {
                    "bit": 8,
                    "symmetry": True,
                    "per_channel": False,
                    "pot_scale": False,
                }
            },
        },
    }


def build_calibrated_int4(fp32_ref, calib_loader, enable_quant=True):
   
    extra_config = get_quant_config()
    fp_for_q = copy.deepcopy(fp32_ref).eval()
    qm = prepare_by_platform(fp_for_q, BackendType.Academic, extra_config).to(DEVICE)
    enable_calibration(qm)
    with torch.no_grad():
        for x, _ in calib_loader:
            qm(x.to(DEVICE, non_blocking=True))
    if enable_quant:
        enable_quantization(qm)
    return qm

def _fq_bits(module):
    qmin = getattr(module, "quant_min", None)
    qmax = getattr(module, "quant_max", None)
    if qmin is None or qmax is None:
        return None
    levels = int(qmax) - int(qmin) + 1
    if levels <= 0:
        return None
    return int(round(np.log2(levels)))


def print_quantizer_inventory(model, expected_w=4, expected_a=4):
    
    weight = {}
    act = {}
    unexpected = []

    for name, module in model.named_modules():
        cls = module.__class__.__name__
        if "FakeQuantize" not in cls and "FakeQuantizer" not in cls:
            continue

        bits = _fq_bits(module)
        if bits is None:
            continue

        if "weight_fake_quant" in name:
            weight[bits] = weight.get(bits, 0) + 1
            if bits not in {int(expected_w), 8}:
                unexpected.append(f"{name}:{bits}b")
        else:
            act[bits] = act.get(bits, 0) + 1
            if bits not in {int(expected_a), 8}:
                unexpected.append(f"{name}:{bits}b")

    if expected_w != 8 and weight.get(expected_w, 0) == 0:
        raise RuntimeError(f"No interior W{expected_w} weight fake-quantizers found.")
    if expected_a != 8 and act.get(expected_a, 0) == 0:
        raise RuntimeError(f"No interior A{expected_a} activation fake-quantizers found.")
    if unexpected:
        raise RuntimeError(
            "Unexpected fake-quant bit widths: " + ", ".join(unexpected[:12]))


# ============================================================
# Flagged-layer selection
# ============================================================
def load_flagged_layers(analysis_csv):
    if not os.path.exists(analysis_csv):
        raise FileNotFoundError(f"Analysis CSV not found: {analysis_csv}")
    df = pd.read_csv(analysis_csv)
    if "exec_order" in df.columns:
        df["exec_order"] = pd.to_numeric(df["exec_order"], errors="coerce")
        df = df.sort_values("exec_order")
    elif "idx" in df.columns:
        df = df.sort_values("idx")

    if "is_flagged" in df.columns:
        flag = df["is_flagged"]
        if flag.dtype == object:
            flag = flag.astype(str).str.lower().isin(["true", "1", "yes"])
        flagged = df.loc[flag.astype(bool), "layer"].dropna().tolist()
    elif "instability_score" in df.columns:
        flagged = df.loc[df["instability_score"] >= 2, "layer"].dropna().tolist()
    else:
        raise KeyError(
            "Analysis CSV must contain 'is_flagged' (preferred) or 'instability_score'.")
    return list(dict.fromkeys(flagged))


def load_sensitivity_scores(analysis_csv):
    
    df = pd.read_csv(analysis_csv)

    quant_tag = f"W{WEIGHT_BITS}A{ACTIVATION_BITS}"
    candidates = [
        f"sensitivity_{quant_tag}",
        f"iso_sens_{quant_tag}",]

    sens_col = next((c for c in candidates if c in df.columns), None)
    if sens_col is None:
        return {}, None

    scores = {}
    for _, row in df.iterrows():
        layer = row.get("layer")
        value = pd.to_numeric(row.get(sens_col), errors="coerce")
        if pd.isna(layer) or pd.isna(value):
            continue
        scores[str(layer)] = max(float(value), 0.0)

    return scores, sens_col


def unwrap_corrected_module(module):
    """Sequential(original, QuantDIR-correction) -> original."""
    if isinstance(module, nn.Sequential) and len(module) > 0:
        return module[0]
    return module


def is_weight_bearing_module(module):
    module = unwrap_corrected_module(module)
    return (
        isinstance(module, (nn.Conv2d, nn.Linear))
        and getattr(module, "weight", None) is not None)


def _fx_nodes(value):
    """Recursively collect torch.fx.Node objects from args/kwargs."""
    nodes = []
    if isinstance(value, torch.fx.Node):
        nodes.append(value)
    elif isinstance(value, (tuple, list)):
        for item in value:
            nodes.extend(_fx_nodes(item))
    elif isinstance(value, dict):
        for item in value.values():
            nodes.extend(_fx_nodes(item))
    return nodes


def _find_graph_node(model, module_name):
    graph = getattr(model, "graph", None)
    if graph is None:
        return None
    for node in graph.nodes:
        if node.op == "call_module" and str(node.target) == module_name:
            return node
    return None


def find_upstream_weighted_producers_fx(model, layer_name):
    
    modules = dict(model.named_modules())

    layer_module = modules.get(layer_name)
    if layer_module is not None and is_weight_bearing_module(layer_module):
        return [layer_name]

    start_node = _find_graph_node(model, layer_name)
    if start_node is None:
        return []

    frontier = _fx_nodes(start_node.args) + _fx_nodes(start_node.kwargs)
    visited = set()

    while frontier:
        next_frontier = []
        found = []

        for node in frontier:
            if node in visited:
                continue
            visited.add(node)

            if node.op == "call_module":
                target = str(node.target)
                module = modules.get(target)
                if module is not None and is_weight_bearing_module(module):
                    found.append(target)
                    continue

            next_frontier.extend(_fx_nodes(node.args))
            next_frontier.extend(_fx_nodes(node.kwargs))

        if found:
            return list(dict.fromkeys(found))

        frontier = next_frontier

    return []

def map_flagged_layers_to_repair_targets(model, flagged_layers, sensitivity_scores):
    
    pairs = []
    mapping_rows = []
    seen_producers = set()

    for flagged_layer in flagged_layers:
        producers = find_upstream_weighted_producers_fx(model, flagged_layer)
        producers = sorted(
            producers,
            key=lambda name: sensitivity_scores.get(name, 0.0),
            reverse=True,
        )[:MAX_PRODUCERS_PER_FLAGGED]

        if not producers:
            mapping_rows.append({
                "flagged_layer": flagged_layer,
                "repair_target": "SKIP_NO_FX_WEIGHTED_PRODUCER",
            })
            continue

        for producer in producers:
            mapping_rows.append({
                "flagged_layer": flagged_layer,
                "repair_target": producer,
            })
            if producer not in seen_producers:
                pairs.append((flagged_layer, producer))
                seen_producers.add(producer)

    return pairs, mapping_rows

# ============================================================
# Instability-guided repair
# ============================================================
class GuidedReconstructionConfig(SimpleNamespace):
    """
    MQBench uses both attribute access (config.max_count) and, for the
    exclusion path, mapping-style access (config["exclude_node"]).
    """
    def __contains__(self, key):
        return key in self.__dict__

    def __getitem__(self, key):
        return self.__dict__[key]


def _is_roundable_weight_module(module):
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        return True
    return (
        hasattr(module, "weight")
        and hasattr(module, "weight_fake_quant")
        and isinstance(getattr(module, "weight", None), torch.Tensor))

def collect_guided_stage5_inputs(loader, max_batches=GUIDED_CALIB_BATCHES):
    
    batches = []
    with torch.no_grad():
        for batch_idx, (images, _labels) in enumerate(loader):
            if max_batches is not None and batch_idx >= int(max_batches):
                break
            batches.append(images.contiguous().cpu())
    if not batches:
        raise RuntimeError("Received no reconstruction batches.")
    return batches


def build_guided_exclusion_list(prepared_model, repair_targets):
    
    if not hasattr(prepared_model, "graph"):
        raise TypeError("Expected the MQBench Academic prepared model to be a torch.fx.GraphModule." )

    modules = dict(prepared_model.named_modules())
    requested = set(repair_targets)

    target_to_node_name = {}
    weighted_nodes = []

    for node in prepared_model.graph.nodes:
        if node.op != "call_module":
            continue
        module = modules.get(str(node.target))
        if module is None or not _is_roundable_weight_module(module):
            continue
        weighted_nodes.append((str(node.target), str(node.name)))
        target_to_node_name[str(node.target)] = str(node.name)

    selected = [t for t in repair_targets if t in target_to_node_name]
    missing = [t for t in repair_targets if t not in target_to_node_name]

    if missing:
        print("WARNING: selected producers absent from prepared FX graph:")
        for name in missing:
            print(f"  {name}")

    if not selected:
        raise RuntimeError(
            "None of the QuantDIR repair targets are roundable nodes in the ")

    selected_set = set(selected)
    excluded = [
        node_name
        for target, node_name in weighted_nodes
        if target not in selected_set]

    return selected, excluded


def run_guided_rounding_reconstruction(
    calibrated_template_cpu,
    repair_targets,
    repair_fit_loader,
):
    repair_targets = list(dict.fromkeys(repair_targets))
    if not repair_targets:
        return None, [], 0.0


    prepared = copy.deepcopy(calibrated_template_cpu).to(DEVICE).eval()

    selected, excluded = build_guided_exclusion_list(
        prepared,
        repair_targets,
    )

    cali_data = collect_guided_stage5_inputs(
        repair_fit_loader,
        max_batches=GUIDED_CALIB_BATCHES,)

    cfg_kwargs = dict(
        pattern="layer",
        warm_up=GUIDED_ROUND_WARM_UP,
        weight=GUIDED_ROUND_REG_WEIGHT,
        max_count=GUIDED_ROUND_MAX_COUNT,
        b_range=list(GUIDED_ROUND_B_RANGE),
        keep_gpu=GUIDED_KEEP_GPU,
        round_mode=GUIDED_ROUND_MODE,
        prob=float(GUIDED_ROUND_PROB),

        exclude_node_prefix=True,
        exclude_node=excluded,
    )

    if GUIDED_LEARN_ACT_SCALE:
        cfg_kwargs["scale_lr"] = GUIDED_ACT_SCALE_LR

    cfg = GuidedReconstructionConfig(**cfg_kwargs)

    for target in selected:
        print(f"  target: {target}")

    if DEVICE.type == "cuda":
        torch.cuda.synchronize(DEVICE)
    t0 = time.perf_counter()

    reconstructed = ptq_reconstruction(prepared,cali_data, cfg,).to(DEVICE)
    enable_quantization(reconstructed)

    if DEVICE.type == "cuda":
        torch.cuda.synchronize(DEVICE)
    elapsed = time.perf_counter() - t0

    del prepared, cali_data
    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    return reconstructed, selected, elapsed

def main():
    set_seed(SEED)
    os.makedirs(OUTDIR, exist_ok=True)

    quant_label = f"W{WEIGHT_BITS}A{ACTIVATION_BITS}"
    quant_label_lower = quant_label.lower()

    print(f"[config] model={MODEL_NAME}  quant={quant_label}  "
        f"analysis_csv={ANALYSIS_CSV}")

    calib_loader, repair_fit_loader, eval_loader = get_loaders()

    # ------------------------------------------------------------
    # FP32 reference
    # ------------------------------------------------------------
    fp32 = build_fp32_model(MODEL_NAME, DEVICE)
    fp32_acc = evaluate(fp32, eval_loader, DEVICE)

    recon_template = build_calibrated_int4(
        fp32,
        calib_loader,
        enable_quant=False,)

    # Exact same calibrated state for the selected W4A4/W4A8 baseline.
    int4_baseline = copy.deepcopy(recon_template).to(DEVICE).eval()
    enable_quantization(int4_baseline)

    print_quantizer_inventory(
        int4_baseline,
        expected_w=WEIGHT_BITS,
        expected_a=ACTIVATION_BITS,)

    int4_acc = evaluate(
        int4_baseline,
        eval_loader,
        DEVICE,)

    print(f"[baseline] FP32={fp32_acc:.2f}%  " f"W{WEIGHT_BITS}A{ACTIVATION_BITS}={int4_acc:.2f}%")
    
    flagged = load_flagged_layers(ANALYSIS_CSV)
    sensitivity_scores, sensitivity_column = load_sensitivity_scores(ANALYSIS_CSV)

    repair_pairs, target_mapping = map_flagged_layers_to_repair_targets(
        int4_baseline,
        flagged,
        sensitivity_scores,)

    repair_targets = list(dict.fromkeys(producer for _observation, producer in repair_pairs))

    if not repair_targets:
        raise RuntimeError(
            "No valid weighted producer could be mapped from the flagged layers." )

    int4_baseline = int4_baseline.cpu().eval()
    recon_template = recon_template.cpu().eval()

    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize(DEVICE)

    repair_memory_tracker = RepairMemoryTracker().start()
    if DEVICE.type == "cuda":
        torch.cuda.synchronize(DEVICE)
    repair_start = time.perf_counter()

    repaired_model, _, _ = (
        run_guided_rounding_reconstruction(
            recon_template,
            repair_targets,
            repair_fit_loader,
        )
    )

    if repaired_model is None:
        raise RuntimeError("Selected flagged-layer returned no model.")

    if DEVICE.type == "cuda":
        torch.cuda.synchronize(DEVICE)
    repair_time_sec = time.perf_counter() - repair_start
    repair_memory = repair_memory_tracker.stop()

    repaired_acc = evaluate(
        repaired_model,
        eval_loader,
        DEVICE,)

    result = {
        "model": MODEL_NAME,
        "fp32_accuracy": fp32_acc,
        "quant_config": quant_label,
        "quantized_accuracy": int4_acc,
        "repaired_accuracy": repaired_acc,
        "flagged_observations": len(flagged),
        "repair_time_sec": repair_time_sec,
        "highest_repair_memory_mb": repair_memory["highest_memory_mb"],
    }

    print("\n=== SELECTED FLAGGED-LAYER REPAIR ===")
    print(f"Model                         : {MODEL_NAME}")
    print(f"FP32 accuracy                 : {fp32_acc:.2f}%")
    print(f"{quant_label} accuracy".ljust(31) + f": {int4_acc:.2f}%")
    print(f"Repaired accuracy             : {repaired_acc:.2f}%")
    print(f"Repair time                   : {repair_time_sec:.2f} s")
    print(
        f"Highest repair memory ({repair_memory['device']})    : "
        f"{repair_memory['highest_memory_mb']:.1f} MB")

    out = os.path.join(OUTDIR, f"quantdir_{MODEL_NAME}_{quant_label_lower}_summary.csv",)
    pd.DataFrame([result]).to_csv(out, index=False)
    

if __name__ == "__main__":
    args = parse_args()
    configure_experiment(args.model, args.quant_mode)
    main()
