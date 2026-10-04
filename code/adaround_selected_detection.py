import os
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
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets
from torchvision.transforms import functional as TF
from torchvision.models import detection as det_models

from mqbench.prepare_by_platform import prepare_by_platform, BackendType
from mqbench.utils.state import enable_calibration, enable_quantization
from mqbench.advanced_ptq import ptq_reconstruction
from torchvision.ops.misc import FrozenBatchNorm2d

try:
    from pycocotools.cocoeval import COCOeval
except ImportError:
    COCOeval = None


# ============================================================
# Configuration
# ============================================================
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Select experiment here; no command-line parser is used.
MODEL_NAME = "fasterrcnn_resnet50_fpn"
W_BITS = 4
A_BITS = 8

COCO_ROOT = "./coco"
NUM_CALIB = 512
NUM_EVAL = 5000
CALIB_BATCH_SIZE = 4
EVAL_BATCH_SIZE = 2
RECON_BATCHES = 16          # set to 0 to use all calibration batches
MAX_COUNT = 1000
KEEP_GPU = True

# ANALYSIS_DIR = "ptq_metrics_detection_up"
ANALYSIS_CSV = f"./ptq_metrics_detection/instability_{MODEL_NAME}_W{W_BITS}A{A_BITS}.csv"       
OUTDIR = "selected_unstable_adaround_detection_results"

ADAROUND_CONFIG = dict(
    pattern="layer",
    scale_lr=4.0e-5,
    warm_up=0.2,
    weight=0.01,
    max_count=MAX_COUNT,
    b_range=[20, 2],
    keep_gpu=KEEP_GPU,
    round_mode="learned_hard_sigmoid",
    prob=1.0,
)


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


QUANT_SCOPE = {
    "ssd300_vgg16": "backbone.features",
    "retinanet_resnet50_fpn": "backbone.body",
    "fasterrcnn_resnet50_fpn": "backbone.body",
}


class DictNamespace(SimpleNamespace):
    def __contains__(self, key):
        return key in self.__dict__

    def __getitem__(self, key):
        return self.__dict__[key]


class CocoImageOnly(Dataset):
    """COCO val2017 images plus image IDs; targets are not needed for inference."""

    def __init__(self, image_root, ann_file):
        self.base = datasets.CocoDetection(root=image_root, annFile=ann_file)
        self.ids = list(self.base.ids)
        self.coco = self.base.coco

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        image, _ = self.base[index]
        image = TF.to_tensor(image)  # [0, 1], CxHxW; detector applies its own transform
        image_id = int(self.ids[index])
        return image, image_id


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


def get_quant_config(w_bits, a_bits):
    return {
        "leaf_module": [FrozenBatchNorm2d],

        "extra_qconfig_dict": {
            "w_observer": "MinMaxObserver",
            "a_observer": "EMAMinMaxObserver",
            "w_fakequantize": "AdaRoundFakeQuantize",
            "a_fakequantize": "FixedFakeQuantize",

            "w_qscheme": {
                "bit": int(w_bits),
                "symmetry": True,
                "per_channel": True,
                "pot_scale": False,
            },

            "a_qscheme": {
                "bit": int(a_bits),
                "symmetry": False,
                "per_channel": False,
                "pot_scale": False,
            },
        },
    }


def load_detector(model_key):
    builder, weights, _, display_name = SUPPORTED_MODELS[model_key]
    model = builder(weights=weights).eval()
    return model, display_name


def build_coco_loaders(
    coco_root,
    num_calib=NUM_CALIB,
    num_eval=NUM_EVAL,
    calib_batch_size=CALIB_BATCH_SIZE,
    eval_batch_size=EVAL_BATCH_SIZE,
):
    image_root = os.path.join(coco_root, "val2017")
    ann_file = os.path.join(coco_root, "annotations", "instances_val2017.json")

    if not os.path.isdir(image_root):
        raise FileNotFoundError(f"COCO image directory not found: {image_root}")
    if not os.path.isfile(ann_file):
        raise FileNotFoundError(f"COCO annotation file not found: {ann_file}")

    dataset = CocoImageOnly(image_root, ann_file)
    n = len(dataset)

    indices = list(range(n))
    random.Random(SEED).shuffle(indices)

    num_calib = min(int(num_calib), n)
    calib_idx = indices[:num_calib]

    if num_eval is None or int(num_eval) <= 0 or int(num_eval) >= n:
        eval_idx = list(range(n))
    else:
        eval_idx = list(range(int(num_eval)))

    calib_loader = DataLoader(
        Subset(dataset, calib_idx),
        batch_size=int(calib_batch_size),
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_detection,
    )

    eval_loader = DataLoader(
        Subset(dataset, eval_idx),
        batch_size=int(eval_batch_size),
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_detection,
    )

    return dataset, calib_loader, eval_loader


@torch.no_grad()
def build_calibrated_quant_scope(
    fp32_scope,
    detector,
    calib_loader,
    w_bits,
    a_bits,
    recon_batches=RECON_BATCHES,
):
    
    prepared = prepare_by_platform(copy.deepcopy(fp32_scope).eval(), BackendType.Academic, get_quant_config(w_bits, a_bits)).to(DEVICE)

    enable_calibration(prepared)
    recon_inputs = []
    total_images = 0

    detector.eval()
    for batch_idx, (images, _) in enumerate(calib_loader):
        images = [img.to(DEVICE, non_blocking=True) for img in images]
        image_list, _ = detector.transform(images, None)
        backbone_input = image_list.tensors

        prepared(backbone_input)
        total_images += int(backbone_input.shape[0])

        if recon_batches is None or batch_idx < int(recon_batches):
            recon_inputs.append(backbone_input.detach().cpu().contiguous())

    if total_images == 0:
        raise RuntimeError("No calibration images were processed.")
    if not recon_inputs:
        raise RuntimeError("No reconstruction inputs were retained.")

    return prepared, recon_inputs, total_images


def is_weight_bearing_module(module):
    # FP32 modules.
    if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
        return hasattr(module, "weight") and module.weight is not None

    if hasattr(module, "weight_fake_quant") and hasattr(module, "weight"):
        return module.weight is not None

    return False


def load_flagged_layers(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Analysis CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required = {"module_name", "is_flagged"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Analysis CSV is missing columns: {sorted(missing)}")

    if df["is_flagged"].dtype == object:
        df["is_flagged"] = (
            df["is_flagged"]
            .astype(str)
            .str.lower()
            .isin(["true", "1", "yes"])
        )

    if "exec_order" in df.columns:
        df["exec_order"] = pd.to_numeric(df["exec_order"], errors="coerce")
        df = df.sort_values("exec_order")

    return df.loc[df["is_flagged"], "module_name"].astype(str).tolist()


def localize_scope_name(flagged_name, scope_prefix):

    name = str(flagged_name)
    prefix = scope_prefix + "."
    if name.startswith(prefix):
        return name[len(prefix):]
    return name


def get_direct_flagged_adaround_targets(prepared_scope, flagged_layers, scope_prefix):
    """
    Direct selected-layer AdaRound for detection.

    A flag is selected only when the exact flagged module, after removing the
    quantization-scope prefix, exists in the prepared scope and is itself a
    weight-bearing Conv/Linear-like module.

    NO upstream producer mapping is performed.  Flagged BN/ReLU/pool/residual,
    FPN/head locations, and other non-weight locations are skipped.
    """
    modules = dict(prepared_scope.named_modules())
    targets = []
    selected_rows = []
    skipped_rows = []

    for flagged_name in flagged_layers:
        local_name = localize_scope_name(flagged_name, scope_prefix)
        module = modules.get(local_name)

        if module is None:
            skipped_rows.append({
                "flagged_layer": flagged_name,
                "local_scope_name": local_name,
                "reason": "LAYER_NOT_FOUND_IN_QUANT_SCOPE",
            })
            continue

        if not is_weight_bearing_module(module):
            skipped_rows.append({
                "flagged_layer": flagged_name,
                "local_scope_name": local_name,
                "reason": "FLAGGED_BUT_NON_WEIGHT",
            })
            continue

        targets.append(local_name)
        selected_rows.append({
            "flagged_layer": flagged_name,
            "selected_local_target": local_name,
        })

    targets = list(dict.fromkeys(targets))

    # Keep one selected-row entry per unique target in execution order.
    seen = set()
    unique_selected_rows = []
    for row in selected_rows:
        target = row["selected_local_target"]
        if target not in seen:
            seen.add(target)
            unique_selected_rows.append(row)

    return targets, unique_selected_rows, skipped_rows


def build_exclusion_list(prepared_scope, selected_targets):
    """Exclude every weighted FX node except the directly selected targets."""
    if not hasattr(prepared_scope, "graph"):
        raise TypeError(
            "Expected MQBench Academic prepare_by_platform to return a GraphModule."
        )

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
        raise RuntimeError("No directly flagged weighted layer is reconstructable.")

    return actual_selected, excluded_node_names


def collect_reconstruction_inputs(transformed_calib, max_batches=RECON_BATCHES):
    if max_batches is None:
        selected = transformed_calib
    else:
        selected = transformed_calib[: int(max_batches)]

    data = [batch.contiguous().cpu() for batch in selected]
    if not data:
        raise RuntimeError("No reconstruction inputs were collected.")
    return data


def run_selected_unstable_adaround(
    calibrated_scope_cpu,
    transformed_calib,
    selected_targets,
    config,
    recon_batches=RECON_BATCHES,
):
    selected_targets = list(dict.fromkeys(selected_targets))
    if not selected_targets:
        raise RuntimeError("No directly flagged weighted layers were selected.")

    model = copy.deepcopy(calibrated_scope_cpu).to(DEVICE).eval()

    actual_selected, excluded_nodes = build_exclusion_list(model, selected_targets)
    cali_data = collect_reconstruction_inputs(transformed_calib, max_batches=recon_batches)

    cfg_dict = dict(config)
    cfg_dict["pattern"] = "layer"
    cfg_dict["exclude_node_prefix"] = True
    cfg_dict["exclude_node"] = excluded_nodes
    cfg = DictNamespace(**cfg_dict)

    print(
        f"\n[Selected AdaRound: direct flagged weighted layers only] "
        f"targets={len(actual_selected)} "
        f"max_count={cfg.max_count} "
        f"prob={cfg.prob} "
        f"keep_gpu={cfg.keep_gpu} "
        f"recon_batches={len(cali_data)}"
    )
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
def evaluate_coco_map(detector, loader, coco_gt, device, max_dets_per_image=None):
    if COCOeval is None:
        raise ImportError(
            "pycocotools is required for COCO mAP evaluation. "
            "Install it with: pip install pycocotools"
        )

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

            if max_dets_per_image is not None and len(scores) > int(max_dets_per_image):
                order = torch.argsort(scores, descending=True)[: int(max_dets_per_image)]
                boxes = boxes[order]
                scores = scores[order]
                labels = labels[order]

            for box, score, label in zip(boxes, scores, labels):
                results.append({
                    "image_id": int(image_id),
                    "category_id": int(label.item()),
                    "bbox": _box_xyxy_to_xywh(box.tolist()),
                    "score": float(score.item()),
                })

        if batch_idx % 100 == 0:
            print(f"    evaluated {len(evaluated_image_ids)} images")

    if not results:
        print("[eval] No detections produced; returning 0.0 mAP")
        return 0.0

    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.params.imgIds = evaluated_image_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # stats[0] = AP @[ IoU=0.50:0.95 | area=all | maxDets=100 ]
    return 100.0 * float(coco_eval.stats[0])


def main():
    if MODEL_NAME not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported MODEL_NAME={MODEL_NAME!r}. Choose from: {sorted(SUPPORTED_MODELS)}")
    if A_BITS not in (4, 8):
        raise ValueError("A_BITS must be 4 or 8.")

    set_seed(SEED)
    os.makedirs(OUTDIR, exist_ok=True)

    model_key = MODEL_NAME
    scope_path = QUANT_SCOPE[model_key]
    analysis_csv = ANALYSIS_CSV or default_analysis_csv(model_key, W_BITS, A_BITS)
    recon_batches = None if int(RECON_BATCHES) == 0 else int(RECON_BATCHES)

    config = dict(ADAROUND_CONFIG)
    config["max_count"] = int(MAX_COUNT)
    config["keep_gpu"] = bool(KEEP_GPU)

    dataset, calib_loader, eval_loader = build_coco_loaders(COCO_ROOT, num_calib=NUM_CALIB, num_eval=NUM_EVAL, calib_batch_size=CALIB_BATCH_SIZE, eval_batch_size=EVAL_BATCH_SIZE)

    print("\n Loading FP32 detector")
    fp32_detector, display_name = load_detector(model_key)
    fp32_detector = fp32_detector.to(DEVICE).eval()

    print("\n Preparing/calibrating quantized backbone scope")
    fp32_scope = copy.deepcopy(get_module_by_path(fp32_detector, scope_path)).cpu().eval()
    calibrated_scope, reconstruction_inputs, n_calib_processed = build_calibrated_quant_scope(fp32_scope, fp32_detector, calib_loader, W_BITS, A_BITS, recon_batches=recon_batches)
    n_recon_images = sum(int(batch.shape[0]) for batch in reconstruction_inputs)


    fp32_detector = fp32_detector.cpu().eval()
    fp32_scope = fp32_scope.cpu().eval()
    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    print("\n Loading QuantDIR flagged locations")
    flagged_layers = load_flagged_layers(analysis_csv)
    repair_targets, selected_rows, skipped_rows = get_direct_flagged_adaround_targets(calibrated_scope, flagged_layers, scope_path)

    if not repair_targets:
        raise RuntimeError("No directly flagged weighted layer is available for AdaRound. Because mapping is disabled, BN/ReLU/pool/residual flags are intentionally skipped.")

    print("\n Evaluating FP32 detector")
    fp32_map = evaluate_coco_map(copy.deepcopy(fp32_detector), eval_loader, dataset.coco, DEVICE)
    print(f"    FP32 mAP={fp32_map:.2f}")

    print(f"\n Evaluating W{W_BITS}A{A_BITS} baseline")
    baseline_scope = copy.deepcopy(calibrated_scope).to(DEVICE).eval()
    enable_quantization(baseline_scope)
    baseline_detector = install_quant_scope(copy.deepcopy(fp32_detector), scope_path, baseline_scope)
    baseline_map = evaluate_coco_map(baseline_detector, eval_loader, dataset.coco, DEVICE)
    print(f"    W{W_BITS}A{A_BITS} mAP={baseline_map:.2f}")

    del baseline_detector, baseline_scope
    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    print("\n Running selected-layer AdaRound")
    calibrated_scope_cpu = calibrated_scope.cpu().eval()
    del calibrated_scope
    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    repaired_scope, actual_targets, repair_time_sec, peak_repair_memory_mb, additional_peak_repair_memory_mb = run_selected_unstable_adaround(calibrated_scope_cpu, reconstruction_inputs, repair_targets, config, recon_batches=recon_batches)

    print("\n Evaluating repaired detector")
    repaired_detector = install_quant_scope(copy.deepcopy(fp32_detector), scope_path, repaired_scope)
    repaired_map = evaluate_coco_map(repaired_detector, eval_loader, dataset.coco, DEVICE)
    print(f"    repaired mAP={repaired_map:.2f}")

    print("\n=== SELECTED-UNSTABLE ADAROUND DETECTION RESULTS ===")
    print(f"Model                         : {display_name}")
    print(f"Configuration                 : W{W_BITS}A{A_BITS}")
    print(f"Quantization scope            : {scope_path}")
    print(f"FP32 mAP                      : {fp32_map:.2f}")
    print(f"Quantized baseline mAP        : {baseline_map:.2f}")
    print(f"Repaired mAP                  : {repaired_map:.2f}")
  
    quant_tag = f"W{W_BITS}A{A_BITS}"
    stem = f"{model_key}_{quant_tag}_selected_unstable_adaround"
    summary = {
        "model": model_key,
        "display_name": display_name,
        "configuration": quant_tag,
        "fp32_map": fp32_map,
        "quantized_baseline_map": baseline_map,
        "repaired_map": repaired_map,
    }

if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    main()
