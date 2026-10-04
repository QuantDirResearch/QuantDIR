import os
import gc
import copy
import time
import random
import argparse
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models

from mqbench.prepare_by_platform import prepare_by_platform, BackendType
from mqbench.utils.state import enable_calibration, enable_quantization
from mqbench.advanced_ptq import ptq_reconstruction


# ============================================================
# Configuration: edit these values directly
# ============================================================
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "resnet18"            # "resnet18", "resnet34", "resnet50", "mobilenet_v2", "vgg16", "vgg19", or "regnet_x_800mf"
W_BITS = 4
A_BITS = 4
STRATEGY = "late"                 # "early", "random", or "late"
IMAGENET_ROOT = "../test_dataset"
NUM_CALIB = 1024
NUM_EVAL = 50000
BATCH_SIZE = 32
FULL_EVAL = True

ANALYSIS_DIR = "ptq_metrics"
ANALYSIS_CSV = None                
OUTDIR = "ablation_early_random_late_classification_results"

EARLY_MIN_RES = 28.0
LATE_MAX_RES = 14.0
POSITION_REFERENCE_SIZE = 224.0
RECON_BATCHES = 16                  
MAX_COUNT = 1000
KEEP_GPU = True
ADAROUND_CONFIG = dict(pattern="layer", scale_lr=4.0e-5, warm_up=0.2, weight=0.01, max_count=MAX_COUNT, b_range=[20, 2], keep_gpu=KEEP_GPU, round_mode="learned_hard_sigmoid", prob=1.0)

SUPPORTED_MODELS = {
    "resnet18": (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1, "ResNet-18"),
    "resnet34": (models.resnet34, models.ResNet34_Weights.IMAGENET1K_V1, "ResNet-34"),
    "resnet50": (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V1, "ResNet-50"),
    "mobilenet_v2": (models.mobilenet_v2, models.MobileNet_V2_Weights.IMAGENET1K_V1, "MobileNetV2"),
    "vgg16": (models.vgg16, models.VGG16_Weights.IMAGENET1K_V1, "VGG-16"),
    "vgg19": (models.vgg19, models.VGG19_Weights.IMAGENET1K_V1, "VGG-19"),
    "regnet_x_800mf": (models.regnet_x_800mf, models.RegNet_X_800MF_Weights.IMAGENET1K_V1, "RegNetX-800MF"),
}


class DictNamespace(SimpleNamespace):
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


def get_quant_config(w_bits, a_bits):
    return {
        "extra_qconfig_dict": {
            "w_observer": "MinMaxObserver",
            "a_observer": "EMAMinMaxObserver",
            "w_fakequantize": "AdaRoundFakeQuantize",
            "a_fakequantize": "FixedFakeQuantize",
            "w_qscheme": {"bit": int(w_bits), "symmetry": True, "per_channel": True, "pot_scale": False},
            "a_qscheme": {"bit": int(a_bits), "symmetry": False, "per_channel": False, "pot_scale": False},
        },
    }


def load_model(model_name):
    builder, weights, display_name = SUPPORTED_MODELS[model_name]
    return builder(weights=weights).eval(), display_name


def get_loaders(num_calib=NUM_CALIB, num_eval=NUM_EVAL, full_eval=FULL_EVAL):
    transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    dataset = datasets.ImageFolder(root=IMAGENET_ROOT, transform=transform)
    indices = list(range(len(dataset)))
    random.Random(SEED).shuffle(indices)
    calib_idx = indices[:min(int(num_calib), len(indices))]
    eval_idx = list(range(len(dataset))) if full_eval else indices[len(calib_idx):len(calib_idx) + int(num_eval)]
    calib_loader = DataLoader(Subset(dataset, calib_idx), batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=torch.cuda.is_available())
    eval_loader = DataLoader(Subset(dataset, eval_idx), batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=torch.cuda.is_available())
    return calib_loader, eval_loader
@torch.no_grad()


def evaluate(model, loader, device):
    model = model.to(device).eval()
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.numel()
    return 100.0 * correct / max(total, 1)


def is_weight_bearing_module(module):
    if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
        return hasattr(module, "weight") and module.weight is not None
    if hasattr(module, "weight_fake_quant") and hasattr(module, "weight"):
        return module.weight is not None
    return False


def load_analysis_df(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Analysis CSV not found: {csv_path}")
    df = pd.read_csv(csv_path).copy()
    required = {"layer", "is_flagged"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Analysis CSV is missing columns: {sorted(missing)}")
    if df["is_flagged"].dtype == object:
        df["is_flagged"] = df["is_flagged"].astype(str).str.lower().isin(["true", "1", "yes"])
    if "exec_order" in df.columns:
        df["exec_order"] = pd.to_numeric(df["exec_order"], errors="coerce")
        df = df.sort_values(["exec_order"], kind="stable", na_position="last").reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)
    df["_analysis_order"] = np.arange(len(df))
    df["layer"] = df["layer"].astype(str)
    return df
@torch.no_grad()


def profile_model_execution(model, sample_input):
    """Record execution order and 224-reference output resolution for weighted modules."""
    model = model.to(DEVICE).eval()
    module_names = {module: name for name, module in model.named_modules()}
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
    for module in model.modules():
        if len(list(module.children())) == 0:
            handles.append(module.register_forward_hook(make_hook(module)))
    model(sample_input.to(DEVICE, non_blocking=True))
    for handle in handles:
        handle.remove()
    seen = set()
    unique_rows = []
    for row in sorted(weighted_rows, key=lambda x: x["exec_order"]):
        if row["layer"] not in seen:
            seen.add(row["layer"])
            unique_rows.append(row)
    return unique_rows, all_events


def map_flagged_to_weighted_producers(analysis_df, fp32_model, execution_events):
    """Map each flagged observation to itself if weighted, otherwise to the nearest preceding weighted producer."""
    modules = dict(fp32_model.named_modules())
    weighted_names = {name for name, module in modules.items() if name and is_weight_bearing_module(module)}
    event_positions = {}
    preceding_weight = {}
    latest_weight = None
    for idx, name, is_weight in sorted(execution_events, key=lambda x: x[0]):
        if is_weight:
            latest_weight = name
        if name not in event_positions:
            event_positions[name] = idx
            preceding_weight[name] = latest_weight
    rows = []
    for pos, row in analysis_df.iterrows():
        if not bool(row["is_flagged"]):
            continue
        flagged = str(row["layer"])
        mapped = None
        reason = None
        if flagged in weighted_names:
            mapped = flagged
            reason = "DIRECT_WEIGHT"
        else:
            prior = analysis_df.iloc[:pos + 1]
            for candidate in reversed(prior["layer"].astype(str).tolist()):
                if candidate in weighted_names:
                    mapped = candidate
                    reason = "CSV_PRECEDING_WEIGHT"
                    break
            if mapped is None and flagged in preceding_weight and preceding_weight[flagged] is not None:
                mapped = preceding_weight[flagged]
                reason = "RUNTIME_PRECEDING_WEIGHT"
        rows.append({"flagged_layer": flagged, "mapped_weight_layer": mapped, "mapping_reason": reason or "NO_WEIGHT_PRODUCER_FOUND"})
    return rows


def build_calibrated_model(fp32_model, calib_loader, w_bits, a_bits):
    qmodel = prepare_by_platform(copy.deepcopy(fp32_model).eval(), BackendType.Academic, get_quant_config(w_bits, a_bits)).to(DEVICE)
    enable_calibration(qmodel)
    with torch.no_grad():
        for images, _ in calib_loader:
            qmodel(images.to(DEVICE, non_blocking=True))
    return qmodel


def get_reconstructable_weighted_targets(prepared_model):
    if not hasattr(prepared_model, "graph"):
        raise TypeError("Expected MQBench Academic prepare_by_platform to return a GraphModule.")
    modules = dict(prepared_model.named_modules())
    targets = []
    for node in prepared_model.graph.nodes:
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


def build_exclusion_list(prepared_model, selected_targets):
    modules = dict(prepared_model.named_modules())
    requested = set(selected_targets)
    actual_selected = []
    excluded_node_names = []
    for node in prepared_model.graph.nodes:
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


def collect_reconstruction_inputs(loader, max_batches=RECON_BATCHES):
    data = []
    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(loader):
            if max_batches is not None and batch_idx >= int(max_batches):
                break
            data.append(images.contiguous().cpu())
    if not data:
        raise RuntimeError("No reconstruction inputs were collected.")
    return data


def run_selected_adaround(calibrated_template_cpu, calib_loader, selected_targets, config, recon_batches=RECON_BATCHES):
    model = copy.deepcopy(calibrated_template_cpu).to(DEVICE).eval()
    actual_selected, excluded_nodes = build_exclusion_list(model, selected_targets)
    cali_data = collect_reconstruction_inputs(calib_loader, max_batches=recon_batches)
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


def default_analysis_csv(model_name, w_bits, a_bits):
    return os.path.join(ANALYSIS_DIR, f"instability_{model_name}_W{int(w_bits)}A{int(a_bits)}.csv")


def main():
    if MODEL_NAME not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported MODEL_NAME={MODEL_NAME!r}. Choose from: {sorted(SUPPORTED_MODELS)}")
    if STRATEGY not in {"early", "random", "late"}:
        raise ValueError("STRATEGY must be 'early', 'random', or 'late'.")
    if W_BITS != 4 or A_BITS not in (4, 8):
        raise ValueError("This experiment expects W_BITS=4 and A_BITS in {4, 8}.")
    
    set_seed(SEED)
    os.makedirs(OUTDIR, exist_ok=True)
    analysis_csv = ANALYSIS_CSV or default_analysis_csv(MODEL_NAME, W_BITS, A_BITS)
    recon_batches = None if int(RECON_BATCHES) == 0 else int(RECON_BATCHES)
    config = dict(ADAROUND_CONFIG)
    config["max_count"] = int(MAX_COUNT)
    config["keep_gpu"] = bool(KEEP_GPU)
    
    calib_loader, eval_loader = get_loaders()
    
    print("\n Loading FP32 model")
    
    fp32_model, display_name = load_model(MODEL_NAME)
    fp32_model = fp32_model.to(DEVICE).eval()
    sample_images, _ = next(iter(calib_loader))
    profile_rows, execution_events = profile_model_execution(fp32_model, sample_images[:1].to(DEVICE))
    
    print("\n Loading flags and mapping them to weighted producers")
    analysis_df = load_analysis_df(analysis_csv)
    flagged_count = int(analysis_df["is_flagged"].sum())
    mapped_rows = map_flagged_to_weighted_producers(analysis_df, fp32_model, execution_events)
    mapped_unique = list(dict.fromkeys([row["mapped_weight_layer"] for row in mapped_rows if row["mapped_weight_layer"]]))
    
    calibrated_template = build_calibrated_model(fp32_model, calib_loader, W_BITS, A_BITS)
    reconstructable_targets = get_reconstructable_weighted_targets(calibrated_template)
    selected_targets, selected_rows, early_available, late_available, total_weighted = select_positional_targets(profile_rows, reconstructable_targets, STRATEGY, flagged_count)
    if not selected_targets:
        raise RuntimeError(f"No {STRATEGY} weighted layers are available for reconstruction.")

    fp32_acc = evaluate(fp32_model, eval_loader, DEVICE)
    baseline = copy.deepcopy(calibrated_template).to(DEVICE).eval()
    enable_quantization(baseline)
    baseline_acc = evaluate(baseline, eval_loader, DEVICE)
    print(f"    W{W_BITS}A{A_BITS} accuracy={baseline_acc:.2f}%")
    fp32_model = fp32_model.cpu().eval()
    baseline = baseline.cpu().eval()
    calibrated_template = calibrated_template.cpu().eval()
    del baseline
    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    print(f"\n Running {STRATEGY} positional AdaRound")
    repaired, actual_targets, repair_time_sec, peak_repair_memory_mb, additional_peak_repair_memory_mb = run_selected_adaround(calibrated_template, calib_loader, selected_targets, config, recon_batches=recon_batches)
    print("\n Evaluating repaired model")
    repaired_acc = evaluate(repaired, eval_loader, DEVICE)
    
    print(f"    repaired accuracy={repaired_acc:.2f}%")
    print("\n=== POSITIONAL ABLATION RESULTS ===")
    print(f"Model                         : {display_name}")
    print(f"Configuration                 : W{W_BITS}A{A_BITS}")
    print(f"Strategy                      : {STRATEGY}")
    print(f"FP32 accuracy                 : {fp32_acc:.2f}%")
    print(f"Quantized baseline accuracy   : {baseline_acc:.2f}%")
    print(f"Repaired accuracy             : {repaired_acc:.2f}%")
    print(f"Repair time                   : {repair_time_sec:.2f} s")
    
    if DEVICE.type == "cuda":
        print(f"Highest repair GPU memory        : {peak_repair_memory_mb:.1f} MB")
    
    quant_tag = f"W{W_BITS}A{A_BITS}"
    stem = f"{MODEL_NAME}_{quant_tag}_{STRATEGY}_positional_adaround"
    
    summary = {
        "model": MODEL_NAME,
        "configuration": quant_tag,
        "strategy": STRATEGY,
        "fp32_accuracy": fp32_acc,
        "quantized_baseline_accuracy": baseline_acc,
        "repaired_accuracy": repaired_acc,
        "repair_time_sec": repair_time_sec,
        "highest_repair_memory_mb": peak_repair_memory_mb,
    }
   


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=sorted(SUPPORTED_MODELS.keys()), default=MODEL_NAME)
    parser.add_argument("--location", choices=["early", "random", "late"], default=STRATEGY)
    parser.add_argument("--weight-bits", type=int, default=W_BITS)
    parser.add_argument("--activation-bits", type=int, default=A_BITS)
    args = parser.parse_args()
    MODEL_NAME = args.model
    STRATEGY = args.location
    W_BITS = args.weight_bits
    A_BITS = args.activation_bits
    torch.multiprocessing.set_start_method("spawn", force=True)
    main()
