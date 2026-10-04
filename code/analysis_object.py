import argparse
import contextlib
import gc
import io
import os
import random
import re
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CocoDetection
from torchvision.models import detection as det_models
# MQBench
from mqbench.prepare_by_platform import prepare_by_platform, BackendType
from mqbench.utils.state import enable_calibration, enable_quantization
try:
    from pycocotools.cocoeval import COCOeval
except Exception as exc:
    COCOeval = None
    _COCO_IMPORT_ERROR = exc
else:
    _COCO_IMPORT_ERROR = None

# ============================================================
# CONFIGURATION
# ============================================================
SEED = 69
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SUPPORTED_MODELS = (
    "retinanet_resnet50_fpn",
    "fasterrcnn_resnet50_fpn",
    "ssd300_vgg16",
    "ssdlite320_mobilenet_v3_large",)

MODEL_NAME = "fasterrcnn_resnet50_fpn"  # default model
def parse_args():
    parser = argparse.ArgumentParser(
        description="Object-detection PTQ numerical-instability analysis."
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=SUPPORTED_MODELS,
        help="Object detection model to analyze.",
    )
    parser.add_argument(
        "--a-bit",
        type=int,
        required=True,
        choices=[4, 8],
        dest="a_bit",
        help="Activation bit-width.",
    )
    return parser.parse_args()

def configure_from_args(args):
    global MODEL_NAME, WBIT, ABIT
    MODEL_NAME = str(args.model).lower()
    WBIT = 4
    ABIT = int(args.a_bit)

COCO_ROOT = "./coco"
COCO_IMAGE_DIR = os.path.join(COCO_ROOT, "val2017")
COCO_ANN_FILE = os.path.join(COCO_ROOT, "annotations", "instances_val2017.json")

WBIT = 4
ABIT = 8
MIN_ISO_DROP_AP = 3.0
INST_RELERR_MIN = 0.05
INST_GAIN_MIN = 1.15
INST_DEAD_RATIO_MIN = 0.05
NUM_CALIB = 512
BATCH_SIZE = 2
NUM_WORKERS = 2
PROPAGATION_BATCHES = 5
DEAD_NEURON_BATCHES = 10
SENSITIVITY_NUM_IMAGES: Optional[int] = 5000
BASELINE_MAP_MAX_BATCHES: Optional[int] = None
OUTDIR = "ptq_metrics_detection"

def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
class COCODetectionTensor(CocoDetection):
    """COCO dataset returning [0,1] float tensors plus image id."""
    def __getitem__(self, index):
        img, anns = super().__getitem__(index)
        image_id = int(self.ids[index])
        img = TF.to_tensor(img)
        target = {
            "image_id": torch.tensor(image_id, dtype=torch.int64),
            "annotations": anns,
        }
        return img, target

def collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)

def images_to_device(images: Sequence[torch.Tensor], device: torch.device) -> List[torch.Tensor]:
    return [img.to(device, non_blocking=True) for img in images]

def unwrap_coco_dataset(dataset):
    while isinstance(dataset, Subset):
        dataset = dataset.dataset
    return dataset

def make_loaders() -> Tuple[DataLoader, DataLoader, DataLoader]:
    if not os.path.isdir(COCO_IMAGE_DIR):
        raise FileNotFoundError(f"COCO image directory not found: {COCO_IMAGE_DIR}")
    if not os.path.isfile(COCO_ANN_FILE):
        raise FileNotFoundError(f"COCO annotation file not found: {COCO_ANN_FILE}")
    dataset = COCODetectionTensor(COCO_IMAGE_DIR, COCO_ANN_FILE)
    n = len(dataset)
    if n == 0:
        raise RuntimeError("COCO dataset is empty")
    # Keep deterministic ordering, analogous to range(...) in the CNN script.
    calib_n = min(NUM_CALIB, n)
    calib_idx = list(range(calib_n))
    eval_idx = list(range(n))
    if SENSITIVITY_NUM_IMAGES is None:
        sens_idx = eval_idx
    else:
        sens_n = min(int(SENSITIVITY_NUM_IMAGES), n)
        # Prefer images after calibration when possible.
        start = calib_n if calib_n + sens_n <= n else 0
        sens_idx = list(range(start, start + sens_n))
    common = dict(
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn,
        persistent_workers=(NUM_WORKERS > 0),)
    calib_loader = DataLoader(Subset(dataset, calib_idx), **common)
    eval_loader = DataLoader(Subset(dataset, eval_idx), **common)
    sensitivity_loader = DataLoader(Subset(dataset, sens_idx), **common)
    return calib_loader, eval_loader, sensitivity_loader

def model_spec(model_name: str) -> Dict[str, object]:
    specs = {
        "retinanet_resnet50_fpn": {
            "constructor": det_models.retinanet_resnet50_fpn,
            "weights": det_models.RetinaNet_ResNet50_FPN_Weights.COCO_V1,
            "scope": "backbone.body",
        },
        "fasterrcnn_resnet50_fpn": {
            "constructor": det_models.fasterrcnn_resnet50_fpn,
            "weights": det_models.FasterRCNN_ResNet50_FPN_Weights.COCO_V1,
            "scope": "backbone.body",
        },
        "ssd300_vgg16": {
            "constructor": det_models.ssd300_vgg16,
            "weights": det_models.SSD300_VGG16_Weights.COCO_V1,
            "scope": "backbone",
        },
        "ssdlite320_mobilenet_v3_large": {
            "constructor": det_models.ssdlite320_mobilenet_v3_large,
            "weights": det_models.SSDLite320_MobileNet_V3_Large_Weights.COCO_V1,
            "scope": "backbone",
        },
    }
    if model_name not in specs:
        raise ValueError(f"Unsupported MODEL_NAME={model_name}. Supported: {list(specs)}")
    return specs[model_name]

def build_fp32_model(model_name: str = MODEL_NAME) -> nn.Module:
    spec = model_spec(model_name)
    constructor = spec["constructor"]
    weights = spec["weights"]
    try:
        model = constructor(weights=weights)
    except TypeError:
        # Compatibility fallback for older torchvision.
        model = constructor(pretrained=True)
    return model.eval()

def get_submodule_by_path(model: nn.Module, path: str) -> nn.Module:
    cur = model
    for part in path.split("."):
        cur = getattr(cur, part)
    return cur

def set_submodule_by_path(model: nn.Module, path: str, new_module: nn.Module) -> None:
    parts = path.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)

def in_quant_scope(name: str, scope: str) -> bool:
    return name == scope or name.startswith(scope + ".")

def get_mqbench_qconfig() -> dict:
   
    return {
        "extra_qconfig_dict": {
            "w_observer": "MinMaxObserver",
            "a_observer": "EMAMinMaxObserver",
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

def build_quantized_model(calib_loader: DataLoader) -> nn.Module:
    """Build complete detector but FX-prepare only its traceable CNN scope."""
    q_model = build_fp32_model(MODEL_NAME)
    scope_path = str(model_spec(MODEL_NAME)["scope"])
    scope_module = get_submodule_by_path(q_model, scope_path)
  
    prepared_scope = prepare_by_platform(scope_module,BackendType.Academic,get_mqbench_qconfig(),)
    set_submodule_by_path(q_model, scope_path, prepared_scope)
    q_model = q_model.to(DEVICE).eval()
  
    enable_calibration(q_model)
    with torch.no_grad():
        for bi, (images, _) in enumerate(calib_loader):
            images = images_to_device(images, DEVICE)
            _ = q_model(images)
    enable_quantization(q_model)
    return q_model

@torch.no_grad()
def evaluate_coco_map(
    model: nn.Module,
    loader: DataLoader,
    max_batches: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, float]:
    if COCOeval is None:
        raise RuntimeError(f"pycocotools is required: {_COCO_IMPORT_ERROR}")
    model.eval()
    device = next(model.parameters()).device
    root_dataset = unwrap_coco_dataset(loader.dataset)
    coco_gt = root_dataset.coco
    results = []
    img_ids = []
    for bi, (images, targets) in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        images = images_to_device(images, device)
        outputs = model(images)
        for output, target in zip(outputs, targets):
            image_id = int(target["image_id"].item())
            img_ids.append(image_id)
            boxes = output.get("boxes", torch.empty((0, 4), device=device)).detach().cpu()
            scores = output.get("scores", torch.empty((0,), device=device)).detach().cpu()
            labels = output.get("labels", torch.empty((0,), dtype=torch.int64, device=device)).detach().cpu()
            for box, score, label in zip(boxes, scores, labels):
                x1, y1, x2, y2 = box.tolist()
                results.append({
                    "image_id": image_id,
                    "category_id": int(label.item()),
                    "bbox": [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)],
                    "score": float(score.item()),
                })
    if not img_ids:
        return {"mAP": float("nan"), "mAP50": float("nan"), "mAP75": float("nan")}
    if not results:
        return {"mAP": 0.0, "mAP50": 0.0, "mAP75": 0.0}
    sink = io.StringIO()
    stream = contextlib.nullcontext() if verbose else contextlib.redirect_stdout(sink)
    with stream:
        coco_dt = coco_gt.loadRes(results)
        evaluator = COCOeval(coco_gt, coco_dt, "bbox")
        evaluator.params.imgIds = sorted(set(img_ids))
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    return {
        "mAP": 100.0 * float(evaluator.stats[0]),
        "mAP50": 100.0 * float(evaluator.stats[1]),
        "mAP75": 100.0 * float(evaluator.stats[2]),
    }

# ============================================================
# WEIGHT / ACTIVATION HELPERS -- same metrics as CNN analysis
# ============================================================
def flatten_for_svd(w: torch.Tensor) -> np.ndarray:
    arr = w.detach().cpu().numpy()
    if arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], -1)
    return arr

def cond_number(w: torch.Tensor) -> float:
    arr = flatten_for_svd(w)
    if arr.size == 0:
        return np.nan
    try:
        return float(np.linalg.cond(arr))
    except Exception:
        return float("nan")
    
def variance(w: torch.Tensor) -> float:
    return float(torch.var(w.detach().float()).cpu().item())

def cosine_similarity(w_fp: torch.Tensor, w_q: torch.Tensor) -> float:
    a = w_fp.detach().cpu().numpy().ravel().astype(np.float32, copy=False)
    b = w_q.detach().cpu().numpy().ravel().astype(np.float32, copy=False)
    n = min(a.size, b.size)
    if n == 0:
        return float("nan")
    a, b = a[:n], b[:n]
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

def _quant_dequant_weight_per_channel(
    w: torch.Tensor,
    bit: int = 4,
    symmetric: bool = False,
) -> torch.Tensor:
    """Per-output-channel Q/DQ for Conv/Linear weights."""
    x = w.detach()
    if x.ndim < 2:
        return _quant_dequant_tensor(x, bit=bit, symmetric=symmetric)
    reduce_dims = tuple(range(1, x.ndim))
    shape = [x.shape[0]] + [1] * (x.ndim - 1)
    if symmetric:
        qmax = (1 << (bit - 1)) - 1
        qmin = -(1 << (bit - 1))
        maxv = x.abs().amax(dim=reduce_dims, keepdim=True)
        scale = torch.clamp(maxv / max(qmax, 1), min=1e-12)
        q = torch.clamp(torch.round(x / scale), qmin, qmax)
        return (q * scale).to(x.dtype)
    qmin, qmax = 0, (1 << bit) - 1
    xmin = x.amin(dim=reduce_dims, keepdim=True)
    xmax = x.amax(dim=reduce_dims, keepdim=True)
    scale = torch.clamp((xmax - xmin) / float(qmax - qmin), min=1e-12)
    zp = torch.clamp(torch.round(qmin - xmin / scale), qmin, qmax)
    q = torch.clamp(torch.round(x / scale + zp), qmin, qmax)
    return ((q - zp) * scale).reshape_as(x).to(x.dtype)

def get_quant_weight(m: nn.Module, bit: int = WBIT) -> torch.Tensor:
    if hasattr(m, "weight_fake_quant"):
        try:
            return m.weight_fake_quant(m.weight)
        except Exception:
            pass
    return _quant_dequant_weight_per_channel(m.weight, bit=bit, symmetric=True)

def rel_error(fp: torch.Tensor, q: torch.Tensor) -> float:
    return float((torch.norm(fp - q) / (torch.norm(fp) + 1e-12)).item())

def snr(fp: torch.Tensor, q: torch.Tensor) -> float:
    s_pow = torch.var(fp).item()
    n_pow = torch.mean((fp - q) ** 2).item()
    return 10.0 * np.log10(max(s_pow, 1e-12) / max(n_pow, 1e-12))

def kl_div(fp: torch.Tensor, q: torch.Tensor, bins: int = 256, eps: float = 1e-8) -> float:
    fp_np = fp.detach().to(dtype=torch.float32, device="cpu").reshape(-1).numpy()
    q_np = q.detach().to(dtype=torch.float32, device="cpu").reshape(-1).numpy()
    n = min(fp_np.size, q_np.size)
    if n == 0:
        return float("nan")
    fp_np, q_np = fp_np[:n], q_np[:n]
    lo = float(min(fp_np.min(), q_np.min()))
    hi = float(max(fp_np.max(), q_np.max()))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return float("nan")
    hist_fp, edges = np.histogram(fp_np, bins=bins, range=(lo, hi))
    hist_q, _ = np.histogram(q_np, bins=edges)
    p = hist_fp.astype(np.float64) + eps
    qh = hist_q.astype(np.float64) + eps
    p /= p.sum()
    qh /= qh.sum()
    val = float(np.sum(p * (np.log(p) - np.log(qh))))
    return max(val, 0.0)

def mse_error(fp: torch.Tensor, q: torch.Tensor) -> float:
    a = fp.detach().cpu().float().reshape(-1)
    b = q.detach().cpu().float().reshape(-1)
    n = min(a.numel(), b.numel())
    if n == 0:
        return float("nan")
    diff = a[:n] - b[:n]
    return float((diff * diff).mean().item())

def sign_flip_rate(fp: torch.Tensor, q: torch.Tensor) -> float:
    return float((fp.sign() != q.sign()).float().mean().item())

def magnitude_loss(fp: torch.Tensor, q: torch.Tensor, tau: Optional[float] = None) -> float:
    fp_abs = fp.detach().abs()
    q_abs = q.detach().abs()
    if tau is None:
        med = torch.median(fp_abs)
        tau = 1e-6 if (not torch.isfinite(med) or med == 0) else float(med) * 1e-3
    fp_sig = fp_abs > tau
    q_zero = q_abs <= tau
    denom = fp_sig.sum().item()
    if denom == 0:
        return 0.0
    return float((fp_sig & q_zero).sum().item() / denom)

def _quant_dequant_tensor(x: torch.Tensor, bit: int = 8, symmetric: bool = False) -> torch.Tensor:
    if not isinstance(x, torch.Tensor):
        return x
    if x.numel() == 0:
        return x
    if symmetric:
        qmax = (1 << (bit - 1)) - 1
        qmin = -(1 << (bit - 1))
        maxv = x.detach().abs().max()
        scale = torch.clamp(maxv / max(qmax, 1), min=1e-12)
        q = torch.clamp(torch.round(x / scale), qmin, qmax)
        return (q * scale).to(x.dtype)
    qmin, qmax = 0, (1 << bit) - 1
    xmin = x.detach().min()
    xmax = x.detach().max()
    if float((xmax - xmin).abs().item()) < 1e-12:
        return x
    scale = torch.clamp((xmax - xmin) / float(qmax - qmin), min=1e-12)
    zp = torch.clamp(torch.round(qmin - xmin / scale), qmin, qmax)
    q = torch.clamp(torch.round(x / scale + zp), qmin, qmax)
    return ((q - zp) * scale).to(x.dtype)

INTERESTED_TYPES = (nn.Conv2d, nn.Linear, nn.BatchNorm2d, nn.LayerNorm, nn.ReLU, nn.ReLU6, nn.MaxPool2d,nn.AvgPool2d,nn.AdaptiveAvgPool2d,)

NONPARAMETRIC_SENS_TYPES = (nn.BatchNorm2d,nn.LayerNorm,nn.ReLU,nn.ReLU6,nn.MaxPool2d,nn.AvgPool2d,nn.AdaptiveAvgPool2d,)
def collect_execution_order(model: nn.Module, loader: DataLoader, scope: str) -> List[str]:
    order: List[str] = []
    seen = set()
    handles = []
    def make_hook(name: str):
        def hook(_m, _i, _o):
            if name not in seen:
                order.append(name)
                seen.add(name)
        return hook
    for name, module in model.named_modules():
        if in_quant_scope(name, scope) and isinstance(module, INTERESTED_TYPES):
            handles.append(module.register_forward_hook(make_hook(name)))
    model.eval()
    device = next(model.parameters()).device
    try:
        images, _ = next(iter(loader))
        images = images_to_device(images[:1], device)
        with torch.no_grad():
            _ = model(images)
    finally:
        for h in handles:
            h.remove()
    return order

# ============================================================
# LAYER STRUCTURAL ANALYSIS
# ============================================================

@torch.no_grad()
def layer_analysis_detection(
    fp32_model: nn.Module,
    quant_model: nn.Module,
    calib_loader: DataLoader,
    num_batches: int = DEAD_NEURON_BATCHES,
    device: Optional[torch.device] = None,
) -> pd.DataFrame:
    if device is None:
        device = DEVICE
   
    fp32_model.eval().to(device)
    quant_model.eval().to(device)
    scope = str(model_spec(MODEL_NAME)["scope"])
    execution_order = collect_execution_order(fp32_model, calib_loader, scope)
  
    fp_modules = dict(fp32_model.named_modules())
    q_modules = dict(quant_model.named_modules())
    relu_zero = defaultdict(float)
    relu_total = defaultdict(int)
    handles = []
    def make_relu_hook(name: str):
        def hook(_m, _i, out):
            if isinstance(out, torch.Tensor):
                x = out.detach()
                relu_zero[name] += float((x == 0).sum().item())
                relu_total[name] += int(x.numel())
        return hook
    for name in execution_order:
        module = q_modules.get(name)
        if isinstance(module, (nn.ReLU, nn.ReLU6)):
            handles.append(module.register_forward_hook(make_relu_hook(name)))
    try:
        for bidx, (images, _) in enumerate(calib_loader):
            if bidx >= num_batches:
                break
            images = images_to_device(images, device)
            _ = quant_model(images)
    finally:
        for h in handles:
            h.remove()
    rows = []
    for idx, name in enumerate(execution_order, start=1):
        if name not in fp_modules or name not in q_modules:
            continue
        m_fp = fp_modules[name]
        m_q = q_modules[name]
        cond32 = cond4 = var32 = var4 = cosine_sim = np.nan
        dead_neuron = np.nan
        if isinstance(m_fp, (nn.Conv2d, nn.Linear)) and hasattr(m_fp, "weight"):
            w_fp = m_fp.weight
            w_q = get_quant_weight(m_q, bit=WBIT)
            var32 = variance(w_fp)
            var4 = variance(w_q)
            cond32 = cond_number(w_fp)
            cond4 = cond_number(w_q)
            cosine_sim = cosine_similarity(w_fp, w_q)
        elif isinstance(m_fp, (nn.BatchNorm2d, nn.LayerNorm)) and getattr(m_fp, "weight", None) is not None:
          
            w_fp = m_fp.weight
            w_q = _quant_dequant_tensor(w_fp, bit=WBIT, symmetric=False)
            var32 = variance(w_fp)
            var4 = variance(w_q)
            cosine_sim = cosine_similarity(w_fp, w_q)
            if w_fp.ndim != 1:
                cond32 = cond_number(w_fp)
                cond4 = cond_number(w_q)
        elif isinstance(m_fp, (nn.ReLU, nn.ReLU6)):
            total = relu_total.get(name, 0)
            if total > 0:
                dead_neuron = relu_zero[name] / total
        rows.append({
            "idx": idx,
            "layer": name,
            "type": m_fp.__class__.__name__,
            "cond_32": cond32,
            "cond_4": cond4,
            "var_32": var32,
            "var_4": var4,
            "cosine_sim": cosine_sim,
            "dead_neuron_ratio": dead_neuron,
        })
    df = pd.DataFrame(rows)
    return df

# ============================================================
# PROPAGATION ANALYSIS
# ============================================================
def _reduce_activation(out: torch.Tensor) -> Optional[torch.Tensor]:
    if not isinstance(out, torch.Tensor) or out.numel() == 0:
        return None
    x = out.detach().float()
    if x.ndim == 4:
        x = x.mean(dim=(2, 3))
    elif x.ndim == 2:
        pass
    elif x.ndim == 1:
        x = x.unsqueeze(0)
    elif x.ndim > 2:
        x = x.flatten(start_dim=1)
    else:
        x = x.reshape(1, -1)
    return x.cpu()

@torch.no_grad()
def layer_analysis_propagation(
    fp32_model: nn.Module,
    quant_model: nn.Module,
    loader: DataLoader,
    num_batches: int = PROPAGATION_BATCHES,
    device: Optional[torch.device] = None,
) -> pd.DataFrame:
    if device is None:
        device = DEVICE
  
    fp32_model.eval().to(device)
    quant_model.eval().to(device)
    scope = str(model_spec(MODEL_NAME)["scope"])
    execution_order = collect_execution_order(fp32_model, loader, scope)
    fp_modules = dict(fp32_model.named_modules())
    q_modules = dict(quant_model.named_modules())
    common_names = [n for n in execution_order if n in fp_modules and n in q_modules]
    fp_outs: Dict[str, List[torch.Tensor]] = defaultdict(list)
    q_outs: Dict[str, List[torch.Tensor]] = defaultdict(list)
    handles = []
    def make_hook(store, name):
        def hook(_m, _i, out):
            reduced = _reduce_activation(out)
            if reduced is not None:
                store[name].append(reduced)
        return hook
    for name in common_names:
        handles.append(fp_modules[name].register_forward_hook(make_hook(fp_outs, name)))
        handles.append(q_modules[name].register_forward_hook(make_hook(q_outs, name)))
    try:
        for batch_idx, (images, _) in enumerate(loader):
            if batch_idx >= num_batches:
                break
            images = images_to_device(images, device)
            _ = fp32_model(images)
            _ = quant_model(images)
    finally:
        for h in handles:
            h.remove()
    rows = []
    prev_relerr = None
    eps = 1e-12
    for idx, name in enumerate(common_names, start=1):
        a_list = fp_outs.get(name, [])
        b_list = q_outs.get(name, [])
        if not a_list or not b_list:
            continue
        
        fp_parts = []
        q_parts = []
        for a, b in zip(a_list, b_list):
            if a.ndim != 2 or b.ndim != 2:
                continue
            n = min(a.shape[0], b.shape[0])
            c = min(a.shape[1], b.shape[1])
            if n == 0 or c == 0:
                continue
            fp_parts.append(a[:n, :c])
            q_parts.append(b[:n, :c])
        if not fp_parts:
            continue
        fp_acts = torch.cat(fp_parts, dim=0)
        q_acts = torch.cat(q_parts, dim=0)
        relerr = rel_error(fp_acts, q_acts)
        snr_val = snr(fp_acts, q_acts)
        kl_val = kl_div(fp_acts, q_acts)
        mse_val = mse_error(fp_acts, q_acts)
        mag_loss = magnitude_loss(fp_acts, q_acts)
        sfr = sign_flip_rate(fp_acts, q_acts)
        inherited_error = float("nan") if prev_relerr is None else float(prev_relerr)
        error_delta = relerr - (prev_relerr if prev_relerr is not None else 0.0)
        if prev_relerr is None:
            error_propagation = float("nan")
        elif prev_relerr > eps:
            error_propagation = relerr / (prev_relerr + eps)
        else:
            error_propagation = 1.0
        rows.append({
            "idx": idx,
            "layer": name,
            "type": fp_modules[name].__class__.__name__,
            "SNR": snr_val,
            "KL": kl_val,
            "SFR": sfr,
            "magnitude_loss": mag_loss,
            "RelErr": relerr,
            "mse_error": mse_val,
            "inherited_error": inherited_error,
            "error_delta": error_delta,
            "error_propagation": error_propagation,
        })
        prev_relerr = relerr
    del fp_outs, q_outs
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return pd.DataFrame(rows)

@torch.no_grad()
def layerwise_quantization_sensitivity_all(
    model_fp32: nn.Module,
    dataloader: DataLoader,
    target_layers: Sequence[str],
    wbit: int = WBIT,
    abit: int = ABIT,
    include_nonparametric: bool = True,
) -> pd.DataFrame:
    
    model_fp32.eval().to(DEVICE)
    modules = dict(model_fp32.named_modules())
    target_col = "isolation_map_drop_pp"
    base = evaluate_coco_map(model_fp32, dataloader, verbose=False)["mAP"]

    rows = []
    supported_count = 0
    for li, name in enumerate(target_layers, start=1):
        module = modules.get(name)
        if module is None:
            rows.append({"layer": name, target_col: np.nan})
            continue
        is_weighted = isinstance(module, (nn.Conv2d, nn.Linear)) and getattr(module, "weight", None) is not None
        is_nonparam = include_nonparametric and isinstance(module, NONPARAMETRIC_SENS_TYPES)
        if not (is_weighted or is_nonparam):
            rows.append({"layer": name, target_col: np.nan})
            continue
        supported_count += 1
        backup_weight = module.weight.detach().clone() if is_weighted else None
        def activation_qdq_hook(_m, _i, out):
            if isinstance(out, torch.Tensor):
                return _quant_dequant_tensor(out, bit=abit, symmetric=False)
            return out
        handle = None
        try:
            if is_weighted:
                q_weight = _quant_dequant_weight_per_channel(
                    module.weight,
                    bit=wbit,
                    symmetric=True,
                )
                module.weight.copy_(q_weight)
            handle = module.register_forward_hook(activation_qdq_hook)
            ap_q = evaluate_coco_map(model_fp32, dataloader, verbose=False)["mAP"]
            drop = float(base - ap_q)
            rows.append({"layer": name, target_col: drop})
           
        except Exception as exc:
            rows.append({"layer": name, target_col: np.nan})

        finally:
            if handle is not None:
                handle.remove()
            if backup_weight is not None:
                module.weight.copy_(backup_weight)
                del backup_weight
        gc.collect()
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    return pd.DataFrame(rows).sort_values(target_col, ascending=False, na_position="last").reset_index(drop=True)

def flag_layers(df, sensitivity_col):
    out = df.copy()
    relerr = pd.to_numeric(out.get("RelErr", np.nan), errors="coerce")
    gain = pd.to_numeric(out.get("error_propagation", np.nan), errors="coerce")
    dead = pd.to_numeric(out.get("dead_neuron_ratio", np.nan), errors="coerce")
    cond_fp32 = pd.to_numeric(out.get("cond_32", np.nan), errors="coerce")
    cond_int4 = pd.to_numeric(out.get("cond_4", np.nan), errors="coerce")
    sensitivity = pd.to_numeric(out[sensitivity_col], errors="coerce")
    has_task_evidence = sensitivity.ge(MIN_ISO_DROP_AP).fillna(False)
    meaningful_output_error = relerr.ge(INST_RELERR_MIN).fillna(False)
    local_amplification = gain.ge(INST_GAIN_MIN).fillna(False)
    dead_channel_flag = dead.ge(INST_DEAD_RATIO_MIN).fillna(False)
    conditioning_broken = (
        cond_fp32.gt(0)
        & (np.isinf(cond_int4) | (cond_int4 > 2.0 * cond_fp32))
    ).fillna(False)
    strong_instability = (
        meaningful_output_error
        | local_amplification
        | dead_channel_flag
        | conditioning_broken
    )
    out["inst_meaningful_output_error"] = meaningful_output_error
    out["inst_local_amplification"] = local_amplification
    out["inst_dead_channels"] = dead_channel_flag
    out["inst_conditioning_broken"] = conditioning_broken
    out["has_strong_instability"] = strong_instability
    out["has_task_evidence"] = has_task_evidence
    out["is_flagged"] = has_task_evidence & strong_instability
    return out

def main() -> None:
    set_seed(SEED)
    os.makedirs(OUTDIR, exist_ok=True)
    calib_loader, eval_loader, sensitivity_loader = make_loaders()
    
    # --------------------------------------------------------
    # 1) FP32 baseline
    # --------------------------------------------------------
   
    fp32_model = build_fp32_model(MODEL_NAME).to(DEVICE).eval()
    fp32_map = evaluate_coco_map(
        fp32_model,
        eval_loader,
        max_batches=BASELINE_MAP_MAX_BATCHES,
        verbose=True,)
    
  print(f"FP32 bbox AP={fp32_map['mAP']:.2f}, "
        f"AP50={fp32_map['mAP50']:.2f}, AP75={fp32_map['mAP75']:.2f}")
    
    quant_model = build_quantized_model(calib_loader)
    quant_map = evaluate_coco_map(
        quant_model,
        eval_loader,
        max_batches=BASELINE_MAP_MAX_BATCHES,
        verbose=True,
    )
    print(f"W{WBIT}A{ABIT} bbox AP={quant_map['mAP']:.2f}, "f"AP50={quant_map['mAP50']:.2f}, AP75={quant_map['mAP75']:.2f}")
    print(f"mAP drop: {fp32_map['mAP'] - quant_map['mAP']:.2f}")
    
    df_detection = layer_analysis_detection(
        fp32_model,
        quant_model,
        calib_loader,
        num_batches=DEAD_NEURON_BATCHES,
        device=DEVICE,)

    print("\n Running propagation analysis...")
    df_propagation = layer_analysis_propagation(
        fp32_model,
        quant_model,
        sensitivity_loader,
        num_batches=PROPAGATION_BATCHES,
        device=DEVICE,)
    
    df_det = df_detection.copy()
    df_prop = df_propagation.copy()
    df_det["layer"] = df_det["layer"].apply(lambda s: re.sub(r"_dup\d+$", "", s))
    df_det = df_det.sort_values("idx").drop_duplicates(subset=["layer"], keep="first")
    prop_keys = {"layer", "idx", "type"}
    prop_cols = ["layer"] + [c for c in df_prop.columns if c not in prop_keys]
    df_all = df_det.merge(df_prop[prop_cols], on="layer", how="left")
    df_all = df_all.sort_values("idx").reset_index(drop=True)
    df_all["idx"] = np.arange(1, len(df_all) + 1)

    df_sens = layerwise_quantization_sensitivity_all(
        model_fp32=fp32_model,
        dataloader=sensitivity_loader,
        target_layers=df_all["layer"].tolist(),
        wbit=WBIT,
        abit=ABIT,
        include_nonparametric=True,)
  
    df_all = df_all.merge(df_sens, on="layer", how="left")
   
    sensitivity_col = "isolation_map_drop_pp"
    df_all = flag_layers(df_all, sensitivity_col)
    
    combined_path = os.path.join(OUTDIR, f"instability_{MODEL_NAME}_W{WBIT}A{ABIT}.csv",)
  
    df_all.to_csv(combined_path, index=False)
    print("\n=== ANALYSIS SUMMARY ===")
    print(f"Model                    : {MODEL_NAME}")
    print(f"Quantized scope          : {model_spec(MODEL_NAME)['scope']}")
    print(f"FP32 bbox AP             : {fp32_map['mAP']:.2f}")
    print(f"W{WBIT}A{ABIT} bbox AP           : {quant_map['mAP']:.2f}")
    print(f"mAP drop             : {fp32_map['mAP'] - quant_map['mAP']:.2f}")
   
if __name__ == "__main__":
    args = parse_args()
    configure_from_args(args)
    torch.multiprocessing.set_start_method("spawn", force=True)
    main()
