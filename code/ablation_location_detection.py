import os
import argparse
import gc
import copy
import time
import random
from collections import OrderedDict
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets
from torchvision.transforms import functional as TF
from torchvision.models import detection as det_models
from torchvision.ops.misc import FrozenBatchNorm2d
from mqbench.prepare_by_platform import prepare_by_platform, BackendType
from mqbench.utils.state import enable_calibration, enable_quantization
from mqbench.advanced_ptq import ptq_reconstruction
try:
    from pycocotools.cocoeval import COCOeval
except ImportError:
    COCOeval = None

# ============================================================
# Configuration: edit these values directly
# ============================================================
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "ssd300_vgg16"
W_BITS = 4
A_BITS = 8
STRATEGY = "random"                 # "early", "random", or "late"
COCO_ROOT = "./coco"
NUM_CALIB = 512
NUM_EVAL = 5000
CALIB_BATCH_SIZE = 4
EVAL_BATCH_SIZE = 2

ANALYSIS_CSV = f"./ptq_metrics_detection/instability_{MODEL_NAME}_W{W_BITS}A{A_BITS}.csv"
OUTDIR = "ablation_early_random_late_detection_results"

EARLY_MIN_RES = 28.0
LATE_MAX_RES = 14.0
POSITION_REFERENCE_SIZE = 224.0
RECON_BATCHES = 16                
MAX_COUNT = 1000
KEEP_GPU = True
ADAROUND_CONFIG = dict(pattern="layer", scale_lr=4.0e-5, warm_up=0.2, weight=0.01, max_count=MAX_COUNT, b_range=[20, 2], keep_gpu=KEEP_GPU, round_mode="learned_hard_sigmoid", prob=1.0)

SUPPORTED_MODELS = {
    "ssd300_vgg16": (det_models.ssd300_vgg16, det_models.SSD300_VGG16_Weights.COCO_V1, (300, 300), "SSD300/VGG-16"),
    "retinanet_resnet50_fpn": (det_models.retinanet_resnet50_fpn, det_models.RetinaNet_ResNet50_FPN_Weights.COCO_V1, (800, 800), "RetinaNet/R50-FPN"),
    "fasterrcnn_resnet50_fpn": (det_models.fasterrcnn_resnet50_fpn, det_models.FasterRCNN_ResNet50_FPN_Weights.COCO_V1, (800, 800), "Faster R-CNN/R50-FPN"),
}

QUANT_SCOPE = {
    "ssd300_vgg16": "backbone",
    "retinanet_resnet50_fpn": "backbone.body",
    "fasterrcnn_resnet50_fpn": "backbone.body",
}

class DictNamespace(SimpleNamespace):
    def __contains__(self, key):
        return key in self.__dict__
    def __getitem__(self, key):
        return self.__dict__[key]

class CocoImageOnly(Dataset):
    def __init__(self, image_root, ann_file):
        self.base = datasets.CocoDetection(root=image_root, annFile=ann_file)
        self.ids = list(self.base.ids)
        self.coco = self.base.coco
    def __len__(self):
        return len(self.base)
    def __getitem__(self, index):
        image, _ = self.base[index]
        return TF.to_tensor(image), int(self.ids[index])

def collate_detection(batch):
    images, image_ids = zip(*batch)
    return list(images), list(image_ids)

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_module_by_path(root, path):
    module = root
    for part in path.split("."):
        module = getattr(module, part)
    return module

def set_module_by_path(root, path, replacement):
    parts = path.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], replacement)

class SSDNormalizeScale(nn.Module):
    """FX-safe version of SSD's functional L2-normalization and scale multiply."""
    def __init__(self, scale_weight):
        super().__init__()
        self.scale_weight = nn.Parameter(scale_weight.detach().clone())
    def forward(self, x):
        return self.scale_weight.view(1, -1, 1, 1) * F.normalize(x)

class SSDBackboneFXSafe(nn.Module):
    """Preserve TorchVision SSD backbone behavior while hiding its functional mul from FX."""
    
    def __init__(self, backbone):
        super().__init__()
        self.features = backbone.features
        self.normalize_scale = SSDNormalizeScale(backbone.scale_weight)
        self.extra = backbone.extra
    
    def forward(self, x):
        x = self.features(x)
        output = [self.normalize_scale(x)]
        for block in self.extra:
            x = block(x)
            output.append(x)
        return OrderedDict((str(i), value) for i, value in enumerate(output))

def get_quant_config(w_bits, a_bits):
    return {
        
        "leaf_module": [FrozenBatchNorm2d, SSDNormalizeScale],
        "extra_qconfig_dict": {
            "w_observer": "MinMaxObserver",
            "a_observer": "EMAMinMaxObserver",
            "w_fakequantize": "AdaRoundFakeQuantize",
            "a_fakequantize": "FixedFakeQuantize",
            "w_qscheme": {"bit": int(w_bits), "symmetry": True, "per_channel": True, "pot_scale": False},
            "a_qscheme": {"bit": int(a_bits), "symmetry": False, "per_channel": False, "pot_scale": False},
        },
    }

def load_detector(model_name):
    builder, weights, _, display_name = SUPPORTED_MODELS[model_name]
    detector = builder(weights=weights).eval()
    if model_name == "ssd300_vgg16":
        detector.backbone = SSDBackboneFXSafe(detector.backbone).eval()
    return detector, display_name

def build_coco_loaders(coco_root, num_calib=NUM_CALIB, num_eval=NUM_EVAL, calib_batch_size=CALIB_BATCH_SIZE, eval_batch_size=EVAL_BATCH_SIZE):
    image_root = os.path.join(coco_root, "val2017")
    ann_file = os.path.join(coco_root, "annotations", "instances_val2017.json")
    if not os.path.isdir(image_root):
        raise FileNotFoundError(f"COCO image directory not found: {image_root}")
    if not os.path.isfile(ann_file):
        raise FileNotFoundError(f"COCO annotation file not found: {ann_file}")
    dataset = CocoImageOnly(image_root, ann_file)
    indices = list(range(len(dataset)))
    random.Random(SEED).shuffle(indices)
    calib_idx = indices[:min(int(num_calib), len(indices))]
    eval_idx = list(range(len(dataset))) if num_eval is None or int(num_eval) <= 0 or int(num_eval) >= len(dataset) else list(range(int(num_eval)))
    
    calib_loader = DataLoader(Subset(dataset, calib_idx), batch_size=int(calib_batch_size), shuffle=False, num_workers=2, pin_memory=torch.cuda.is_available(), collate_fn=collate_detection)
    eval_loader = DataLoader(Subset(dataset, eval_idx), batch_size=int(eval_batch_size), shuffle=False, num_workers=2, pin_memory=torch.cuda.is_available(), collate_fn=collate_detection)
    return dataset, calib_loader, eval_loader

def is_weight_bearing_module(module):
    if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
        return hasattr(module, "weight") and module.weight is not None
    if hasattr(module, "weight_fake_quant") and hasattr(module, "weight"):
        return module.weight is not None
    return False

def localize_scope_name(full_name, scope_prefix):
    name = str(full_name)
    prefix = scope_prefix + "."
    return name[len(prefix):] if name.startswith(prefix) else name

def load_analysis_df(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Analysis CSV not found: {csv_path}")
    df = pd.read_csv(csv_path).copy()
    if "is_flagged" not in df.columns:
        raise ValueError(f"Analysis CSV is missing 'is_flagged'. Available columns: {df.columns.tolist()}")
    if "module_name" in df.columns:
        name_col = "module_name"
    elif "layer" in df.columns:
        name_col = "layer"
    else:
        raise ValueError(f"Analysis CSV needs 'module_name' or 'layer'. Available columns: {df.columns.tolist()}")
    if df["is_flagged"].dtype == object:
        df["is_flagged"] = df["is_flagged"].astype(str).str.lower().isin(["true", "1", "yes"])
    if "exec_order" in df.columns:
        df["exec_order"] = pd.to_numeric(df["exec_order"], errors="coerce")
        df = df.sort_values(["exec_order"], kind="stable", na_position="last").reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)
    df["_analysis_order"] = np.arange(len(df))
    df["layer"] = df[name_col].astype(str)
    print(f"[CSV] using module-name column: {name_col}")
    return df

@torch.no_grad()
def get_transformed_backbone_input(detector, images):
    detector = detector.to(DEVICE).eval()
    images = [image.to(DEVICE, non_blocking=True) for image in images]
    image_list, _ = detector.transform(images, None)
    return image_list.tensors

@torch.no_grad()
def profile_scope_execution(fp32_scope, sample_input):
    """Profile weighted backbone layers and normalize feature sizes to a 224-input reference."""
    fp32_scope = fp32_scope.to(DEVICE).eval()
    module_names = {module: name for name, module in fp32_scope.named_modules()}
    weighted_rows = []
    all_events = []
    handles = []
    call_index = {"value": 0}
    input_side = float(max(sample_input.shape[-2], sample_input.shape[-1]))
    def make_hook(module):
        def hook(_module, _inputs, output):
            name = module_names[module]
            idx = call_index["value"]
            call_index["value"] += 1
            all_events.append((idx, name, is_weight_bearing_module(module)))
            if not is_weight_bearing_module(module):
                return
            h = w = None
            if torch.is_tensor(output) and output.ndim >= 4:
                h, w = int(output.shape[-2]), int(output.shape[-1])
            effective_res = None if h is None else max(h, w) * POSITION_REFERENCE_SIZE / input_side
            weighted_rows.append({"layer": name, "exec_order": idx, "module_type": type(module).__name__, "output_h": h, "output_w": w, "effective_resolution": effective_res, "is_linear": isinstance(module, nn.Linear)})
        return hook
    for module in fp32_scope.modules():
        if len(list(module.children())) == 0:
            handles.append(module.register_forward_hook(make_hook(module)))
    fp32_scope(sample_input.to(DEVICE, non_blocking=True))
    for handle in handles:
        handle.remove()
    seen = set()
    unique_rows = []
    for row in sorted(weighted_rows, key=lambda x: x["exec_order"]):
        if row["layer"] not in seen:
            seen.add(row["layer"])
            unique_rows.append(row)
    return unique_rows, all_events

def map_flagged_to_weighted_producers(analysis_df, fp32_scope, scope_prefix, execution_events):
    """Map in-scope flagged observations to the nearest preceding weighted backbone producer."""
    modules = dict(fp32_scope.named_modules())
    weighted_names = {name for name, module in modules.items() if name and is_weight_bearing_module(module)}
    preceding_weight = {}
    latest_weight = None
    for _, name, is_weight in sorted(execution_events, key=lambda x: x[0]):
        if is_weight:
            latest_weight = name
        if name not in preceding_weight:
            preceding_weight[name] = latest_weight
    localized_analysis = []
    for _, row in analysis_df.iterrows():
        full_name = str(row["layer"])
        in_scope = full_name.startswith(scope_prefix + ".")
        localized_analysis.append((full_name, localize_scope_name(full_name, scope_prefix), in_scope))
    rows = []
    for pos, row in analysis_df.iterrows():
        if not bool(row["is_flagged"]):
            continue
        flagged = str(row["layer"])
        in_scope = flagged.startswith(scope_prefix + ".")
        local_name = localize_scope_name(flagged, scope_prefix)
        mapped = None
        reason = None
        if not in_scope:
            reason = "FLAG_OUTSIDE_QUANT_SCOPE"
        elif local_name in weighted_names:
            mapped = local_name
            reason = "DIRECT_WEIGHT"
        else:
            for prior_pos in range(pos, -1, -1):
                _, candidate_local, candidate_in_scope = localized_analysis[prior_pos]
                if candidate_in_scope and candidate_local in weighted_names:
                    mapped = candidate_local
                    reason = "CSV_PRECEDING_WEIGHT"
                    break
            if mapped is None and local_name in preceding_weight and preceding_weight[local_name] is not None:
                mapped = preceding_weight[local_name]
                reason = "RUNTIME_PRECEDING_WEIGHT"
        rows.append({"flagged_layer": flagged, "local_flagged_layer": local_name, "mapped_weight_layer": mapped, "mapping_reason": reason or "NO_WEIGHT_PRODUCER_FOUND"})
    return rows

@torch.no_grad()
def build_calibrated_quant_scope(fp32_scope, detector, calib_loader, w_bits, a_bits, recon_batches=RECON_BATCHES):
    prepared = prepare_by_platform(copy.deepcopy(fp32_scope).eval(), BackendType.Academic, get_quant_config(w_bits, a_bits)).to(DEVICE)
    enable_calibration(prepared)
    reconstruction_inputs = []
    total_images = 0
    detector = detector.to(DEVICE).eval()
    for batch_idx, (images, _) in enumerate(calib_loader):
        backbone_input = get_transformed_backbone_input(detector, images)
        prepared(backbone_input)
        total_images += int(backbone_input.shape[0])
        if recon_batches is None or batch_idx < int(recon_batches):
            reconstruction_inputs.append(backbone_input.detach().cpu().contiguous())
    if total_images == 0:
        raise RuntimeError("No calibration images were processed.")
    if not reconstruction_inputs:
        raise RuntimeError("No reconstruction inputs were retained.")
    return prepared, reconstruction_inputs, total_images

def get_reconstructable_weighted_targets(prepared_scope):
    if not hasattr(prepared_scope, "graph"):
        raise TypeError("Expected MQBench Academic prepare_by_platform to return a GraphModule.")
    modules = dict(prepared_scope.named_modules())
    targets = []
    for node in prepared_scope.graph.nodes:
        if node.op != "call_module":
            continue
        target = str(node.target)
        module = modules.get(target)
        if module is not None and is_weight_bearing_module(module):
            targets.append(target)
    return list(dict.fromkeys(targets))

def select_positional_targets(profile_rows, reconstructable_targets, strategy, target_count):
    reconstructable = set(reconstructable_targets)
    rows = [row for row in profile_rows if row["layer"] in reconstructable]
    rows = sorted(rows, key=lambda x: x["exec_order"])
    early_rows = [row for row in rows if row["effective_resolution"] is not None and row["effective_resolution"] >= EARLY_MIN_RES]
    late_rows = [row for row in rows if row["is_linear"] or (row["effective_resolution"] is not None and row["effective_resolution"] <= LATE_MAX_RES)]
    if strategy == "early":
        selected_rows = early_rows[:min(target_count, len(early_rows))]
    elif strategy == "late":
        n = min(target_count, len(late_rows))
        selected_rows = late_rows[-n:] if n > 0 else []
    elif strategy == "random":
        rng = random.Random(SEED)
        n = min(target_count, len(rows))
        selected_names = set(rng.sample([row["layer"] for row in rows], n))
        selected_rows = [row for row in rows if row["layer"] in selected_names]
    else:
        raise ValueError("STRATEGY must be 'early', 'random', or 'late'.")
    for row in selected_rows:
        row["strategy"] = strategy
    return [row["layer"] for row in selected_rows], selected_rows, len(early_rows), len(late_rows), len(rows)

def build_exclusion_list(prepared_scope, selected_targets):
    modules = dict(prepared_scope.named_modules())
    requested = set(selected_targets)
    actual_selected = []
    excluded_node_names = []
    for node in prepared_scope.graph.nodes:
        if node.op != "call_module":
            continue
        target = str(node.target)
        module = modules.get(target)
        if module is None or not is_weight_bearing_module(module):
            continue
        if target in requested:
            actual_selected.append(target)
        else:
            excluded_node_names.append(str(node.name))
    actual_selected = list(dict.fromkeys(actual_selected))
    excluded_node_names = list(dict.fromkeys(excluded_node_names))
    missing = [name for name in selected_targets if name not in set(actual_selected)]
    if missing:
        print("[selection] WARNING: selected targets not found as weighted FX nodes:")
        for name in missing:
            print(f"  {name}")
    if not actual_selected:
        raise RuntimeError("No selected positional weighted layer is reconstructable.")
    return actual_selected, excluded_node_names

def collect_reconstruction_inputs(transformed_calib, max_batches=RECON_BATCHES):
    selected = transformed_calib if max_batches is None else transformed_calib[:int(max_batches)]
    data = [batch.contiguous().cpu() for batch in selected]
    if not data:
        raise RuntimeError("No reconstruction inputs were collected.")
    return data

def run_selected_adaround(calibrated_scope_cpu, transformed_calib, selected_targets, config, recon_batches=RECON_BATCHES):
    model = copy.deepcopy(calibrated_scope_cpu).to(DEVICE).eval()
    actual_selected, excluded_nodes = build_exclusion_list(model, selected_targets)
    cali_data = collect_reconstruction_inputs(transformed_calib, max_batches=recon_batches)
    cfg_dict = dict(config)
    cfg_dict["pattern"] = "layer"
    cfg_dict["exclude_node_prefix"] = True
    cfg_dict["exclude_node"] = excluded_nodes
    cfg = DictNamespace(**cfg_dict)
    print(f"\n[{STRATEGY.upper()} positional AdaRound] targets={len(actual_selected)} max_count={cfg.max_count} prob={cfg.prob} keep_gpu={cfg.keep_gpu} recon_batches={len(cali_data)}")
    for name in actual_selected:
        print(f"  target: {name}")
    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize(DEVICE)
        start_alloc = torch.cuda.memory_allocated(DEVICE) / 1024**2
        torch.cuda.reset_peak_memory_stats(DEVICE)
    else:
        start_alloc = 0.0
    start = time.perf_counter()
    repaired = ptq_reconstruction(model, cali_data, cfg).to(DEVICE)
    enable_quantization(repaired)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize(DEVICE)
    repair_time_sec = time.perf_counter() - start
    if DEVICE.type == "cuda":
        peak_mb = torch.cuda.max_memory_allocated(DEVICE) / 1024**2
        additional_peak_mb = max(0.0, peak_mb - start_alloc)
    else:
        peak_mb = float("nan")
        additional_peak_mb = float("nan")
    del cali_data
    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return repaired, actual_selected, repair_time_sec, peak_mb, additional_peak_mb

def install_quant_scope(detector, scope_path, quant_scope):
    detector = detector.cpu().eval()
    set_module_by_path(detector, scope_path, quant_scope.cpu().eval())
    return detector

def _box_xyxy_to_xywh(box):
    x1, y1, x2, y2 = [float(v) for v in box]
    return [x1, y1, x2 - x1, y2 - y1]

@torch.no_grad()
def evaluate_coco_map(detector, loader, coco_gt, device):
    if COCOeval is None:
        raise ImportError("pycocotools is required for COCO mAP evaluation. Install it with: pip install pycocotools")
    detector = detector.to(device).eval()
    results = []
    evaluated_image_ids = []
    
    for batch_idx, (images, image_ids) in enumerate(loader, start=1):
        images = [img.to(device, non_blocking=True) for img in images]
        outputs = detector(images)
        for image_id, output in zip(image_ids, outputs):
            evaluated_image_ids.append(int(image_id))
            boxes = output.get("boxes", torch.empty((0, 4))).detach().cpu()
            scores = output.get("scores", torch.empty((0,))).detach().cpu()
            labels = output.get("labels", torch.empty((0,), dtype=torch.long)).detach().cpu()
            for box, score, label in zip(boxes, scores, labels):
                results.append({"image_id": int(image_id), "category_id": int(label.item()), "bbox": _box_xyxy_to_xywh(box.tolist()), "score": float(score.item())})
        if batch_idx % 100 == 0:
            print(f"    evaluated {len(evaluated_image_ids)} images")
    if not results:
        return 0.0
    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.params.imgIds = evaluated_image_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return 100.0 * float(coco_eval.stats[0])

def main():
    global MODEL_NAME, W_BITS, A_BITS, STRATEGY, ANALYSIS_CSV
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=sorted(SUPPORTED_MODELS), default=MODEL_NAME)
    parser.add_argument("--location", choices=["early", "random", "late"], default=STRATEGY)
    parser.add_argument("--weight-bits", type=int, default=W_BITS)
    parser.add_argument("--activation-bits", type=int, default=A_BITS)
    args = parser.parse_args()
    MODEL_NAME = args.model
    STRATEGY = args.location
    W_BITS = args.weight_bits
    A_BITS = args.activation_bits
    
    ANALYSIS_CSV = f"./ptq_metrics_detection_up/instability_{MODEL_NAME}_W{W_BITS}A{A_BITS}.csv"
    
    if MODEL_NAME not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported MODEL_NAME={MODEL_NAME!r}. Choose from: {sorted(SUPPORTED_MODELS)}")
    if STRATEGY not in {"early", "random", "late"}:
        raise ValueError("STRATEGY must be 'early', 'random', or 'late'.")
    if W_BITS != 4 or A_BITS not in (4, 8):
        raise ValueError("This experiment expects W_BITS=4 and A_BITS in {4, 8}.")
    
    set_seed(SEED)
    os.makedirs(OUTDIR, exist_ok=True)
    scope_path = QUANT_SCOPE[MODEL_NAME]
    analysis_csv = ANALYSIS_CSV or default_analysis_csv(MODEL_NAME, W_BITS, A_BITS)
    recon_batches = None if int(RECON_BATCHES) == 0 else int(RECON_BATCHES)
    config = dict(ADAROUND_CONFIG)
    config["max_count"] = int(MAX_COUNT)
    config["keep_gpu"] = bool(KEEP_GPU)

    dataset, calib_loader, eval_loader = build_coco_loaders(COCO_ROOT)
    
    print("\n Loading FP32 detector and profiling backbone positions")
    fp32_detector, display_name = load_detector(MODEL_NAME)
    fp32_detector = fp32_detector.to(DEVICE).eval()
    fp32_scope = copy.deepcopy(get_module_by_path(fp32_detector, scope_path)).to(DEVICE).eval()
    sample_images, _ = next(iter(calib_loader))
    sample_backbone_input = get_transformed_backbone_input(fp32_detector, sample_images[:1])
    profile_rows, execution_events = profile_scope_execution(fp32_scope, sample_backbone_input)
    
    print("\n Loading flags and mapping them to weighted producers")
    analysis_df = load_analysis_df(analysis_csv)
    flagged_count = int(analysis_df["is_flagged"].sum())
    mapped_rows = map_flagged_to_weighted_producers(analysis_df, fp32_scope, scope_path, execution_events)
    mapped_unique = list(dict.fromkeys([row["mapped_weight_layer"] for row in mapped_rows if row["mapped_weight_layer"]]))
    mapped_in_scope_count = sum(row["mapped_weight_layer"] is not None for row in mapped_rows)
    
    calibrated_scope, reconstruction_inputs, n_calib_processed = build_calibrated_quant_scope(fp32_scope, fp32_detector, calib_loader, W_BITS, A_BITS, recon_batches=recon_batches)
    reconstructable_targets = get_reconstructable_weighted_targets(calibrated_scope)
    selected_targets, selected_rows, early_available, late_available, total_weighted = select_positional_targets(profile_rows, reconstructable_targets, STRATEGY, flagged_count)
    
    for row in selected_rows:
        res = "Linear" if row["is_linear"] else f"{row['effective_resolution']:.2f}"
        print(f"  {row['layer']:<45} effective_res_224={res}")
    
    if not selected_targets:
        raise RuntimeError(f"No {STRATEGY} weighted layers are available for reconstruction.")
    fp32_detector = fp32_detector.cpu().eval()
    fp32_scope = fp32_scope.cpu().eval()
    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    print("\n Evaluating FP32 detector")
    fp32_map = evaluate_coco_map(copy.deepcopy(fp32_detector), eval_loader, dataset.coco, DEVICE)
    print(f"    FP32 mAP={fp32_map:.2f}")
 
    baseline_scope = copy.deepcopy(calibrated_scope).to(DEVICE).eval()
    enable_quantization(baseline_scope)
    baseline_detector = install_quant_scope(copy.deepcopy(fp32_detector), scope_path, baseline_scope)
    baseline_map = evaluate_coco_map(baseline_detector, eval_loader, dataset.coco, DEVICE)
    
    print(f"    W{W_BITS}A{A_BITS} mAP={baseline_map:.2f}")
    del baseline_detector, baseline_scope
    calibrated_scope = calibrated_scope.cpu().eval()
    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    
    print(f"\nRunning {STRATEGY} positional AdaRound")
    repaired_scope, actual_targets, repair_time_sec, peak_repair_memory_mb, additional_peak_repair_memory_mb = run_selected_adaround(calibrated_scope, reconstruction_inputs, selected_targets, config, recon_batches=recon_batches)
    
    print("\n Evaluating repaired detector")
    repaired_detector = install_quant_scope(copy.deepcopy(fp32_detector), scope_path, repaired_scope)
    repaired_map = evaluate_coco_map(repaired_detector, eval_loader, dataset.coco, DEVICE)
    print(f"    repaired mAP={repaired_map:.2f}")

    print("\n=== POSITIONAL ABLATION DETECTION RESULTS ===")
    print(f"Model                         : {display_name}")
    print(f"Configuration                 : W{W_BITS}A{A_BITS}")
    print(f"Strategy                      : {STRATEGY}")
    print(f"Quantization scope            : {scope_path}")
    print(f"FP32 mAP                      : {fp32_map:.2f}")
    print(f"Quantized baseline mAP        : {baseline_map:.2f}")
    print(f"Repaired mAP                  : {repaired_map:.2f}")
    print(f"Repair time                   : {repair_time_sec:.2f} s")
    
    if DEVICE.type == "cuda":
        print(f"Highest repair GPU memory        : {peak_repair_memory_mb:.1f} MB")
       
    quant_tag = f"W{W_BITS}A{A_BITS}"
    stem = f"{MODEL_NAME}_{quant_tag}_{STRATEGY}_positional_adaround"
    
    summary = {
        "model": MODEL_NAME,
        "display_name": display_name,
        "configuration": quant_tag,
        "strategy": STRATEGY,
        "quantization_scope": scope_path,
        "fp32_map": fp32_map,
        "quantized_baseline_map": baseline_map,
        "repaired_map": repaired_map,
        "repair_time_sec": repair_time_sec,
        "highest_repair_memory_mb": peak_repair_memory_mb,

    }

  
if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    main()
