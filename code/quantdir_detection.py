import argparse
import copy
import gc
import json
import os
import random
import threading
import time
from collections import deque
from types import SimpleNamespace

import numpy as np
import pandas as pd

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import detection as det_models
from torchvision.ops.misc import FrozenBatchNorm2d

from mqbench.prepare_by_platform import prepare_by_platform, BackendType
from mqbench.utils.state import enable_calibration, enable_quantization
from mqbench.advanced_ptq import ptq_reconstruction

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    from pycocotools.cocoeval import COCOeval
except ImportError as exc:
    raise ImportError("pycocotools is required: pip install pycocotools") from exc


SEED = 73
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_MODEL = "ssd300_vgg16"
DEFAULT_COCO_ROOT = "./coco"
DEFAULT_ANALYSIS_DIR = "./ptq_metrics_detection"
DEFAULT_WEIGHT_BITS = 4
DEFAULT_ACTIVATION_BITS = 8
NUM_CALIB = 512
NUM_EVAL = 5000
DATA_BATCH_SIZE = 1
NUM_WORKERS = 0


SUPPORTED_MODELS = {
    "ssd300_vgg16": (
        det_models.ssd300_vgg16,
        det_models.SSD300_VGG16_Weights.COCO_V1,
        (300, 300),
        "SSD300/VGG-16",
    ),
    "retinanet_resnet50_fpn": (
        det_models.retinanet_resnet50_fpn,
        det_models.RetinaNet_ResNet50_FPN_Weights.COCO_V1,
        (800, 800),
        "RetinaNet/R50-FPN",
    ),
    "fasterrcnn_resnet50_fpn": (
        det_models.fasterrcnn_resnet50_fpn,
        det_models.FasterRCNN_ResNet50_FPN_Weights.COCO_V1,
        (800, 800),
        "Faster R-CNN/R50-FPN",
    ),
}

KEEP_SCOPE_EDGES_8BIT = True
GUIDED_RECON_PATTERN = "layer"
GUIDED_ROUND_MAX_COUNT = 1200
GUIDED_ROUND_WARM_UP = 0.20
GUIDED_ROUND_REG_WEIGHT = 0.01
GUIDED_ROUND_B_RANGE = [20, 2]
GUIDED_ROUND_MODE = "learned_hard_sigmoid"
GUIDED_ROUND_PROB = 0.8
GUIDED_KEEP_GPU = True
GUIDED_CALIB_SAMPLES = 16   # number of captured scope-input batches, as in original repair code
GUIDED_LEARN_ACT_SCALE = True
GUIDED_ACT_SCALE_LR = 4.0e-5
MAX_PRODUCERS_PER_FLAG = 2

PROFILE_INFERENCE = True
INFERENCE_BATCH_SIZE = 1
INFERENCE_WARMUP = 10
INFERENCE_ITERS = 50

class GuidedConfig(SimpleNamespace):
    def __contains__(self, key):
        return key in self.__dict__

    def __getitem__(self, key):
        return self.__dict__[key]


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)




# -----------------------------------------------------------------------------
# Memory tracking
# -----------------------------------------------------------------------------
def _cpu_rss_mb():
    if not HAS_PSUTIL:
        return 0.0
    try:
        return psutil.Process(os.getpid()).memory_info().rss / 1024**2
    except Exception:
        return 0.0


class RepairMemoryTracker:
    def __init__(self, device=DEVICE, poll_interval=0.05):
        self.device = device
        self.poll_interval = poll_interval
        self._thread = None
        self._stop = None
        self._start = 0.0
        self._cpu_peak = 0.0

    def start(self):
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize(self.device)
            self._start = torch.cuda.memory_allocated(self.device) / 1024**2
            torch.cuda.reset_peak_memory_stats(self.device)
        else:
            self._start = _cpu_rss_mb()
            self._cpu_peak = self._start
            self._stop = threading.Event()

            def poll():
                while not self._stop.wait(self.poll_interval):
                    self._cpu_peak = max(self._cpu_peak, _cpu_rss_mb())

            self._thread = threading.Thread(target=poll, daemon=True)
            self._thread.start()
        return self

    def stop(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            peak = torch.cuda.max_memory_allocated(self.device) / 1024**2
            end = torch.cuda.memory_allocated(self.device) / 1024**2
            return {
                "device": "GPU",
                "peak_memory_mb": peak,
                "additional_peak_memory_mb": max(0.0, peak - self._start),
                "final_allocated_memory_mb": end,
            }
        if self._stop is not None:
            self._stop.set()
        if self._thread is not None:
            self._thread.join()
        end = _cpu_rss_mb()
        self._cpu_peak = max(self._cpu_peak, end)
        return {
            "device": "CPU",
            "peak_memory_mb": self._cpu_peak,
            "additional_peak_memory_mb": max(0.0, self._cpu_peak - self._start),
            "final_allocated_memory_mb": end,
        }


# -----------------------------------------------------------------------------
# COCO val2017 only
# -----------------------------------------------------------------------------
class CocoDetectionWithImageId(datasets.CocoDetection):
    def __getitem__(self, index):
        image, annotations = super().__getitem__(index)
        return image, {"image_id": int(self.ids[index]), "annotations": annotations}


def detection_collate(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)


def _first_existing(candidates, label):
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"Could not find {label}. Tried: {candidates}")


def _sample_filename(annotation_file):
    with open(annotation_file, "r", encoding="utf-8") as f:
        payload = json.load(f)
    for item in payload.get("images", []):
        if item.get("file_name"):
            return str(item["file_name"])
    return None


def resolve_coco_layout(coco_root):
    root = os.path.abspath(os.path.expanduser(coco_root))
    ann = _first_existing(
        [
            os.path.join(root, "annotations", "instances_val2017.json"),
            os.path.join(root, "instances_val2017.json"),
        ],
        "COCO val2017 annotations",
    )
    sample = _sample_filename(ann)
    candidates = [
        os.path.join(root, "val2017"),
        os.path.join(root, "images", "val2017"),
        os.path.join(root, "val"),
        os.path.join(root, "images", "val"),
    ]
    image_dir = None
    for path in candidates:
        if os.path.isdir(path) and (sample is None or os.path.isfile(os.path.join(path, sample))):
            image_dir = path
            break
    if image_dir is None:
        raise FileNotFoundError(
            f"Could not find val2017 images containing {sample!r}. Tried: {candidates}"
        )
    return root, image_dir, ann


def _make_loader(dataset):
    return DataLoader(
        dataset,
        batch_size=DATA_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
        collate_fn=detection_collate,
    )


def build_coco_calib_eval_loaders(coco_root, num_calib=NUM_CALIB, num_eval=NUM_EVAL, seed=SEED):
    root, image_dir, ann = resolve_coco_layout(coco_root)
    print(f"[COCO] root        : {root}")
    print(f"[COCO] val images  : {image_dir}")
    print(f"[COCO] val ann     : {ann}")

    full = CocoDetectionWithImageId(image_dir, ann, transform=transforms.ToTensor())
    n = len(full)
    if num_calib <= 0 or num_calib >= n:
        raise ValueError(f"num_calib={num_calib} invalid for val2017 size={n}")

    order = list(range(n))
    random.Random(seed).shuffle(order)
    calib_idx = order[:num_calib]
    remaining = order[num_calib:]
    eval_idx = remaining if num_eval is None else remaining[: min(int(num_eval), len(remaining))]

    print(f"[COCO] calibration : {len(calib_idx)} images from val2017 (labels unused)")
    print(f"[COCO] evaluation  : {len(eval_idx)} disjoint images from val2017")
    return _make_loader(Subset(full, calib_idx)), _make_loader(Subset(full, eval_idx))


def get_underlying_coco(dataset):
    while isinstance(dataset, Subset):
        dataset = dataset.dataset
    return dataset.coco


# -----------------------------------------------------------------------------
# Detector/scope utilities
# -----------------------------------------------------------------------------
def build_detector(model_name, device=None):
    builder, weights, _hw, _label = SUPPORTED_MODELS[model_name]
    model = builder(weights=weights).eval()
    if device is not None:
        model = model.to(device)
    return model


def quantized_scope_for_model(model_name):
    if model_name == "ssd300_vgg16":
        return "backbone.features"
    if model_name in {"retinanet_resnet50_fpn", "fasterrcnn_resnet50_fpn"}:
        return "backbone.body"
    raise ValueError(model_name)


def get_nested_module(model, path):
    module = model
    for part in path.split("."):
        module = getattr(module, part)
    return module


def set_nested_module(model, path, new_module):
    parts = path.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


class _CapturedScopeInput(RuntimeError):
    pass


def replace_frozen_bn_with_bn(module):
    """In-place exact eval-mode replacement of TorchVision FrozenBatchNorm2d."""
    replaced = 0
    for child_name, child in list(module.named_children()):
        if isinstance(child, FrozenBatchNorm2d):
            n = int(child.weight.numel())
            bn = nn.BatchNorm2d(
                n,
                eps=float(child.eps),
                momentum=0.1,
                affine=True,
                track_running_stats=True,
            )
            with torch.no_grad():
                bn.weight.copy_(child.weight.detach())
                bn.bias.copy_(child.bias.detach())
                bn.running_mean.copy_(child.running_mean.detach())
                bn.running_var.copy_(child.running_var.detach())
                bn.num_batches_tracked.zero_()
            bn.weight.requires_grad_(False)
            bn.bias.requires_grad_(False)
            bn.eval()
            setattr(module, child_name, bn)
            replaced += 1
        else:
            replaced += replace_frozen_bn_with_bn(child)
    return replaced


def make_guided_mqbench_scope(fp32_scope, weight_bits, activation_bits):
    scope = copy.deepcopy(fp32_scope).cpu().eval()
    before = sum(isinstance(m, FrozenBatchNorm2d) for m in scope.modules())
    replaced = replace_frozen_bn_with_bn(scope)
    after = sum(isinstance(m, FrozenBatchNorm2d) for m in scope.modules())
    print(f"[prep] FrozenBatchNorm2d: before={before}, replaced={replaced}, after={after}")
    if after:
        raise RuntimeError("FrozenBatchNorm2d remained inside MQBench scope after conversion.")

    prepared = prepare_by_platform(
        scope,
        BackendType.Academic,
        get_guided_qconfig(scope, weight_bits, activation_bits),
    ).to(DEVICE).eval()
    return prepared


@torch.no_grad()
def stream_calibrate_scope_and_collect_reconstruction_inputs(
    detector,
    loader,
    prepared_scope,
    scope_path,
    device,
    num_calib_samples,
    num_recon_samples,
):
    scope = get_nested_module(detector, scope_path)
    retained_cpu = []
    calibrated = 0
    enable_calibration(prepared_scope)

    def pre_hook(_module, inputs):
        nonlocal calibrated
        x = inputs[0]
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected Tensor input at {scope_path}, got {type(x)}")
        prepared_scope(x)
        if len(retained_cpu) < int(num_recon_samples):
            retained_cpu.append(x.detach().cpu().contiguous())
        calibrated += int(x.shape[0])
        raise _CapturedScopeInput()

    handle = scope.register_forward_pre_hook(pre_hook)
    detector.eval()
    try:
        for images, _targets in loader:
            images = [img.to(device, non_blocking=True) for img in images]
            try:
                detector(images)
            except _CapturedScopeInput:
                pass
            del images
            if calibrated >= int(num_calib_samples):
                break
            if device.type == "cuda" and calibrated % 16 == 0:
                torch.cuda.empty_cache()
    finally:
        handle.remove()

    if calibrated == 0 or not retained_cpu:
        raise RuntimeError(f"No calibration/repair input captured at {scope_path}")
    return retained_cpu


def detector_with_scope(fp32_detector_cpu, scope_path, replacement_scope):
    detector = copy.deepcopy(fp32_detector_cpu).eval()
    set_nested_module(detector, scope_path, replacement_scope)
    return detector


# -----------------------------------------------------------------------------
# COCO evaluation
# -----------------------------------------------------------------------------
@torch.no_grad()
def evaluate_coco_bbox(model, loader, device, print_summary=True):
    model.eval()
    results = []
    image_ids = []
    for images, targets in loader:
        images = [img.to(device, non_blocking=True) for img in images]
        outputs = model(images)
        for output, target in zip(outputs, targets):
            image_id = int(target["image_id"])
            image_ids.append(image_id)
            boxes = output["boxes"].detach().cpu()
            scores = output["scores"].detach().cpu()
            labels = output["labels"].detach().cpu()
            if boxes.numel() == 0:
                continue
            boxes_xywh = boxes.clone()
            boxes_xywh[:, 2] = boxes[:, 2] - boxes[:, 0]
            boxes_xywh[:, 3] = boxes[:, 3] - boxes[:, 1]
            for box, score, label in zip(boxes_xywh, scores, labels):
                results.append({
                    "image_id": image_id,
                    "category_id": int(label.item()),
                    "bbox": [float(v) for v in box.tolist()],
                    "score": float(score.item()),
                })
        del images, outputs

    if not results:
        return {"AP": 0.0, "AP50": 0.0, "AP75": 0.0}
    coco_gt = get_underlying_coco(loader.dataset)
    coco_dt = coco_gt.loadRes(results)
    evaluator = COCOeval(coco_gt, coco_dt, iouType="bbox")
    evaluator.params.imgIds = sorted(set(image_ids))
    evaluator.evaluate()
    evaluator.accumulate()
    if print_summary:
        evaluator.summarize()
    return {
        "AP": 100.0 * float(evaluator.stats[0]),
        "AP50": 100.0 * float(evaluator.stats[1]),
        "AP75": 100.0 * float(evaluator.stats[2]),
    }


# -----------------------------------------------------------------------------
# Quantization recipe: original selected-rounding style, generalized to scope
# -----------------------------------------------------------------------------
def first_last_weight_module_names(scope):
    weighted = [
        name for name, module in scope.named_modules()
        if name and isinstance(module, (nn.Conv2d, nn.Linear))
    ]
    if not weighted:
        return None, None
    return weighted[0], weighted[-1]


def get_guided_qconfig(scope, weight_bits, activation_bits):
    config = {
        "extra_qconfig_dict": {
            "w_observer": "MinMaxObserver",
            "a_observer": "EMAMSEObserver",
            "w_fakequantize": "AdaRoundFakeQuantize",
            "a_fakequantize": "FixedFakeQuantize",
            "w_qscheme": {
                "bit": int(weight_bits),
                "symmetry": True,
                "per_channel": True,
                "pot_scale": False,
            },
            "a_qscheme": {
                "bit": int(activation_bits),
                "symmetry": False,
                "per_channel": False,
                "pot_scale": False,
            },
        },
        "module_qconfig_dict": {},
    }

    if KEEP_SCOPE_EDGES_8BIT:
        first_name, last_name = first_last_weight_module_names(scope)
        for name in [first_name, last_name]:
            if not name or name in config["module_qconfig_dict"]:
                continue
            module = dict(scope.named_modules())[name]
            config["module_qconfig_dict"][name] = {
                "w_qscheme": {
                    "bit": 8,
                    "symmetry": True,
                    "per_channel": isinstance(module, nn.Conv2d),
                    "pot_scale": False,
                }
            }
        if first_name or last_name:
            print(f"[qconfig] W8 edge weights in scope: first={first_name}, last={last_name}")
    return config




# -----------------------------------------------------------------------------
# CSV loading and scope-relative observation normalization
# -----------------------------------------------------------------------------
def detect_name_column(df):
    for col in ("module_name", "layer", "layer_name", "module", "name", "node_name"):
        if col in df.columns:
            return col
    raise KeyError(
        "Could not find observation-name column. Expected one of "
        "module_name/layer/layer_name/module/name/node_name; "
        f"available={list(df.columns)}"
    )


def flag_series(df):
    if "is_flagged" in df.columns:
        x = df["is_flagged"]
        if x.dtype == object:
            x = x.astype(str).str.strip().str.lower().isin(["true", "1", "yes", "y"])
        else:
            x = x.fillna(False).astype(bool)
        return x, "is_flagged"
    if "instability_score" in df.columns:
        x = pd.to_numeric(df["instability_score"], errors="coerce").fillna(0.0) >= 2.0
        return x, "instability_score>=2"
    raise KeyError("CSV must contain is_flagged or instability_score")


def normalize_to_scope(raw_name, scope_path, available_names):
    raw = str(raw_name).strip()
    prefixes_to_remove = [
        "module.model.",
        "model.",
        "module.",
    ]
    variants = [raw]
    for p in prefixes_to_remove:
        if raw.startswith(p):
            variants.append(raw[len(p):])

    expanded = list(variants)
    for name in variants:
        # exact analysis scope prefix -> relative MQBench scope name
        if name == scope_path:
            expanded.append("")
        if name.startswith(scope_path + "."):
            expanded.append(name[len(scope_path) + 1:])

        # common detector wrappers
        if name.startswith("backbone."):
            expanded.append(name[len("backbone."):])
        if scope_path == "backbone.body" and name.startswith("body."):
            expanded.append(name[len("body."):])
        if scope_path == "backbone.features" and name.startswith("features."):
            expanded.append(name[len("features."):])

    # For older ResNet CSVs that omit body.
    if scope_path == "backbone.body":
        for name in list(expanded):
            if name.startswith("backbone.body."):
                expanded.append(name[len("backbone.body."):])
            if name.startswith("layer") or name.startswith("conv1") or name.startswith("bn1") or name.startswith("relu"):
                expanded.append(name)

    seen = set()
    for cand in expanded:
        cand = cand.strip(".")
        if cand in seen:
            continue
        seen.add(cand)
        if cand in available_names:
            return cand
    return None


def load_flagged_observations(csv_path, scope_path, available_names):
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Analysis CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    name_col = detect_name_column(df)
    flags, flag_source = flag_series(df)

    if "exec_order" in df.columns:
        df = df.assign(_flag=flags)
        df["exec_order"] = pd.to_numeric(df["exec_order"], errors="coerce")
        df = df.sort_values("exec_order", kind="stable")
        flags = df["_flag"]
    elif "idx" in df.columns:
        df = df.assign(_flag=flags)
        df["idx"] = pd.to_numeric(df["idx"], errors="coerce")
        df = df.sort_values("idx", kind="stable")
        flags = df["_flag"]

    raw_flagged = df.loc[flags.astype(bool), name_col].dropna().astype(str).tolist()
    raw_flagged = list(dict.fromkeys(raw_flagged))
    resolved, unresolved = [], []
    for raw in raw_flagged:
        name = normalize_to_scope(raw, scope_path, available_names)
        if name is None:
            unresolved.append(raw)
        else:
            resolved.append(name)

    print(f"[CSV] analysis file       : {csv_path}")
    print(f"[CSV] observation column : {name_col}")
    print(f"[CSV] flag source        : {flag_source}")
    print(f"[CSV] flagged rows       : {len(raw_flagged)}")
    return raw_flagged, list(dict.fromkeys(resolved)), unresolved, df, name_col


def load_sensitivity_scores(df, name_col, scope_path, available_names, weight_bits, activation_bits):
    candidates = [
        f"sensitivity_W{weight_bits}A{activation_bits}",
        f"iso_sens_W{weight_bits}A{activation_bits}",
        "sensitivity_W4A4", "iso_sens_W4A4",
        "sensitivity_W4A8", "iso_sens_W4A8",
    ]
    sens_col = next((c for c in candidates if c in df.columns), None)
    if sens_col is None:
        return {}, None
    scores = {}
    for _, row in df.iterrows():
        raw = row.get(name_col)
        val = pd.to_numeric(row.get(sens_col), errors="coerce")
        if pd.isna(raw) or pd.isna(val):
            continue
        name = normalize_to_scope(str(raw), scope_path, available_names)
        if name is not None:
            scores[name] = max(0.0, float(val))
    return scores, sens_col


# -----------------------------------------------------------------------------
# FX observation -> nearest weighted producer mapping
# -----------------------------------------------------------------------------
def _is_roundable_weight_module(module):
    return (
        isinstance(module, (nn.Conv2d, nn.Linear))
        or (
            hasattr(module, "weight")
            and hasattr(module, "weight_fake_quant")
            and isinstance(getattr(module, "weight", None), torch.Tensor)
        )
    )


def _fx_nodes(value):
    if isinstance(value, torch.fx.Node):
        return [value]
    if isinstance(value, (tuple, list)):
        out = []
        for item in value:
            out.extend(_fx_nodes(item))
        return out
    if isinstance(value, dict):
        out = []
        for item in value.values():
            out.extend(_fx_nodes(item))
        return out
    return []


def _find_graph_node(model, module_name):
    for node in model.graph.nodes:
        if node.op == "call_module" and str(node.target) == module_name:
            return node
    return None


def find_upstream_weighted_producers_fx(model, observation_name):
    modules = dict(model.named_modules())
    direct = modules.get(observation_name)
    if direct is not None and _is_roundable_weight_module(direct):
        return [observation_name]

    start = _find_graph_node(model, observation_name)
    if start is None:
        return []

    frontier = _fx_nodes(start.args) + _fx_nodes(start.kwargs)
    visited = set()
    while frontier:
        next_frontier = []
        found = []
        for node in frontier:
            if node in visited:
                continue
            visited.add(node)
            if node.op == "call_module":
                module = modules.get(str(node.target))
                if module is not None and _is_roundable_weight_module(module):
                    found.append(str(node.target))
                    continue
            next_frontier.extend(_fx_nodes(node.args))
            next_frontier.extend(_fx_nodes(node.kwargs))
        if found:
            return list(dict.fromkeys(found))
        frontier = next_frontier
    return []


def map_flags_to_targets(model, flagged, sensitivity_scores):
    pairs = []
    rows = []
    seen_producers = set()
    for obs in flagged:
        producers = find_upstream_weighted_producers_fx(model, obs)
        if sensitivity_scores:
            producers = sorted(producers, key=lambda p: sensitivity_scores.get(p, 0.0), reverse=True)
        producers = producers[:MAX_PRODUCERS_PER_FLAG]
        if not producers:
            rows.append({"flagged_layer": obs, "repair_target": None})
            continue
        for producer in producers:
            rows.append({"flagged_layer": obs, "repair_target": producer})
            if producer not in seen_producers:
                pairs.append((obs, producer))
                seen_producers.add(producer)
    return pairs, rows


# -----------------------------------------------------------------------------
# Selected layer-wise repair
# -----------------------------------------------------------------------------
def build_guided_exclusion_list(prepared_model, repair_targets):
    modules = dict(prepared_model.named_modules())
    weighted_nodes = []
    target_to_node = {}
    for node in prepared_model.graph.nodes:
        if node.op != "call_module":
            continue
        module = modules.get(str(node.target))
        if module is None or not _is_roundable_weight_module(module):
            continue
        weighted_nodes.append((str(node.target), str(node.name)))
        target_to_node[str(node.target)] = str(node.name)

    selected = [t for t in repair_targets if t in target_to_node]
    missing = [t for t in repair_targets if t not in target_to_node]
    if missing:
        print("[repair] WARNING: mapped producers absent from prepared FX graph:")
        for name in missing:
            print(f"  {name}")
    if not selected:
        raise RuntimeError("No mapped producer is reconstructable in prepared scope")

    selected_set = set(selected)
    excluded = [node_name for target, node_name in weighted_nodes if target not in selected_set]
    return selected, excluded


def run_guided_reconstruction(calibrated_template_cpu, repair_targets, scope_inputs):
    prepared = copy.deepcopy(calibrated_template_cpu).to(DEVICE).eval()
    selected, excluded = build_guided_exclusion_list(prepared, repair_targets)
    cali_cpu = scope_inputs[: min(len(scope_inputs), int(GUIDED_CALIB_SAMPLES))]

    kwargs = dict(
        pattern=GUIDED_RECON_PATTERN,
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
        kwargs["scale_lr"] = GUIDED_ACT_SCALE_LR
    cfg = GuidedConfig(**kwargs)

    reconstructed = ptq_reconstruction(prepared, list(cali_cpu), cfg).to(DEVICE)
    enable_quantization(reconstructed)
    del prepared
    return reconstructed, selected


# -----------------------------------------------------------------------------
# Isolated inference profile
# -----------------------------------------------------------------------------
@torch.no_grad()
def profile_detection_inference_gpu(model, device, image_hw, batch_size=1, warmup=10, iters=50):
    if device.type != "cuda":
        return None
    model.cpu().eval()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    before_load = torch.cuda.memory_allocated(device) / 1024**2
    model.to(device).eval()
    torch.cuda.synchronize(device)
    loaded = torch.cuda.memory_allocated(device) / 1024**2

    h, w = image_hw
    images = [torch.rand(3, h, w, device=device) for _ in range(int(batch_size))]
    for _ in range(int(warmup)):
        out = model(images)
        del out
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter()
    out = None
    for _ in range(int(iters)):
        out = model(images)
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - t0
    peak = torch.cuda.max_memory_allocated(device) / 1024**2

    del out, images
    model.cpu()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    return {
        "resident_model_delta_mb": max(0.0, loaded - before_load),
        "peak_inference_mb": peak,
        "latency_ms_per_batch": 1000.0 * elapsed / max(int(iters), 1),
    }


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def default_analysis_csv(args):
    return os.path.join(
        DEFAULT_ANALYSIS_DIR,
        f"instability_{args.model}_W{args.weight_bits}A{args.activation_bits}.csv",
    )


def main(args):
    model_name = args.model
    scope_path = quantized_scope_for_model(model_name)
    analysis_csv = default_analysis_csv(args)
    _builder, _weights, profile_hw, display_name = SUPPORTED_MODELS[model_name]
    quant_label = f"W{args.weight_bits}A{args.activation_bits}"

    # Fail fast on obvious wrong-model CSVs.
    base = os.path.basename(analysis_csv).lower()
    other_models = [m for m in SUPPORTED_MODELS if m != model_name]
    if any(m in base for m in other_models):
        raise ValueError(f"CSV/model mismatch: --model={model_name}, CSV={analysis_csv}")

    print("=" * 96)
    print(f"QUANTDIR SELECTED LAYER REPAIR: {display_name}")

    calib_loader, eval_loader = build_coco_calib_eval_loaders(
        DEFAULT_COCO_ROOT, NUM_CALIB, NUM_EVAL, SEED)

    fp32 = build_detector(model_name, DEVICE)
    print("\n FP32 COCO evaluation")
    fp32_metrics = evaluate_coco_bbox(fp32, eval_loader, DEVICE)

    fp32_scope = get_nested_module(fp32, scope_path)
   
    print("\n MQBench prepare + STREAMED calibration of exact scope")
    recon_template = make_guided_mqbench_scope(
        fp32_scope, args.weight_bits, args.activation_bits)

    scope_inputs = stream_calibrate_scope_and_collect_reconstruction_inputs(
        fp32, calib_loader, recon_template, scope_path, DEVICE,
        args.num_calib, GUIDED_CALIB_SAMPLES,)

    fp32 = fp32.cpu().eval()
    recon_template = recon_template.cpu().eval()
    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize(DEVICE)

    baseline_scope = copy.deepcopy(recon_template).to(DEVICE).eval()
    enable_quantization(baseline_scope)
    baseline_detector = detector_with_scope(fp32, scope_path, baseline_scope).to(DEVICE).eval()

    print(f"\n {quant_label} Baseline COCO evaluation")
    baseline_metrics = evaluate_coco_bbox(baseline_detector, eval_loader, DEVICE)

    available_names = set(dict(baseline_scope.named_modules()).keys())
    raw_flagged, flagged, unresolved, df, name_col = load_flagged_observations(analysis_csv, scope_path, available_names)

    sensitivity_scores, sensitivity_column = load_sensitivity_scores(
        df, name_col, scope_path, available_names, args.weight_bits, args.activation_bits)

    pairs, mapping_rows = map_flags_to_targets(baseline_scope, flagged, sensitivity_scores)
    repair_targets = list(dict.fromkeys(producer for _obs, producer in pairs))

    if unresolved:
        print(f"[selection] unresolved flags={len(unresolved)}")
        for name in unresolved[:30]:
            print(f"  unresolved: {name}")
    if raw_flagged and not flagged:
        raise RuntimeError(
            f"All {len(raw_flagged)} flagged observations failed to resolve inside {scope_path}. "
            "This usually means the CSV belongs to a different model/scope.")

    for row in mapping_rows:
        print(f"  {str(row['flagged_layer']):<48} -> {row['repair_target']}")

    if not repair_targets:
        raise RuntimeError("No valid weighted producer could be mapped from the flagged observations")

    baseline_detector.cpu()
    baseline_scope.cpu()
    recon_template.cpu()
    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize(DEVICE)

    print("\n Offline selected layer-wise repair")
    tracker = RepairMemoryTracker(DEVICE).start()
    if DEVICE.type == "cuda":
        torch.cuda.synchronize(DEVICE)
    t0 = time.perf_counter()

    repaired_scope, reconstructed_targets = run_guided_reconstruction(recon_template, repair_targets, scope_inputs)

    if DEVICE.type == "cuda":
        torch.cuda.synchronize(DEVICE)
    repair_time = time.perf_counter() - t0
    repair_memory = tracker.stop()

    repaired_scope = repaired_scope.cpu().eval()
    repaired_detector = detector_with_scope(fp32, scope_path, repaired_scope).to(DEVICE).eval()

    repaired_metrics = evaluate_coco_bbox(repaired_detector, eval_loader, DEVICE)

    if PROFILE_INFERENCE and DEVICE.type == "cuda":
        repaired_detector.cpu()
        baseline_detector.cpu()
        repaired_scope.cpu()
        recon_template.cpu()
        fp32.cpu()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize(DEVICE)

        # new change
        fp32_infer = profile_detection_inference_gpu(
            fp32, DEVICE, profile_hw,
            INFERENCE_BATCH_SIZE, INFERENCE_WARMUP, INFERENCE_ITERS)

        baseline_infer = profile_detection_inference_gpu(
            baseline_detector, DEVICE, profile_hw,
            INFERENCE_BATCH_SIZE, INFERENCE_WARMUP, INFERENCE_ITERS)

        repaired_infer = profile_detection_inference_gpu(
            repaired_detector, DEVICE, profile_hw,
            INFERENCE_BATCH_SIZE, INFERENCE_WARMUP, INFERENCE_ITERS)

        print("\n=== ISOLATED INFERENCE PROFILE ===")
        print(f"FP32 resident model GPU        : {fp32_infer['resident_model_delta_mb']:.1f} MB")
        print(f"FP32 highest inference GPU        : {fp32_infer['peak_inference_mb']:.1f} MB")
        print(f"FP32 inference time            : {fp32_infer['latency_ms_per_batch']:.3f} ms/batch")
        print(f"{quant_label} resident model GPU       : {baseline_infer['resident_model_delta_mb']:.1f} MB")
        print(f"Repaired resident model GPU   : {repaired_infer['resident_model_delta_mb']:.1f} MB")
        print(f"{quant_label} highest inference GPU       : {baseline_infer['peak_inference_mb']:.1f} MB")
        print(f"Repaired highest inference GPU   : {repaired_infer['peak_inference_mb']:.1f} MB")
        print(f"Inference peak overhead       : {repaired_infer['peak_inference_mb'] - baseline_infer['peak_inference_mb']:+.1f} MB")
        print(f"{quant_label} inference time           : {baseline_infer['latency_ms_per_batch']:.3f} ms/batch")
        print(f"Repaired inference time       : {repaired_infer['latency_ms_per_batch']:.3f} ms/batch")
        print(f"Inference-time overhead       : {repaired_infer['latency_ms_per_batch'] - baseline_infer['latency_ms_per_batch']:+.3f} ms/batch")

    print("\n=== SELECTED FLAGGED-LAYER REPAIR ===")
    print(f"Model                          : {display_name}")
    print(f"Quantized scope                : {scope_path}")
    print(f"FP32 AP                        : {fp32_metrics['AP']:.2f}")
    print(f"{quant_label} AP                        : {baseline_metrics['AP']:.2f}")
    print(f"Repaired AP                    : {repaired_metrics['AP']:.2f}")
    print(f"Repair time                    : {repair_time:.2f} s")
    print(f"Highest repair memory ({repair_memory['device']})        : {repair_memory['peak_memory_mb']:.1f} MB")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=DEFAULT_MODEL, choices=list(SUPPORTED_MODELS))
    p.add_argument("--coco-root", default=DEFAULT_COCO_ROOT)
    p.add_argument("--analysis-csv", default=None)
    p.add_argument("--weight-bits", type=int, default=DEFAULT_WEIGHT_BITS)
    p.add_argument("--activation-bits", type=int, default=DEFAULT_ACTIVATION_BITS)

    return p.parse_args()

if __name__ == "__main__":
    main(parse_args())
