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


SEED = 42
BATCH_SIZE = 32
NUM_CALIB = 1024
NUM_EVAL = 50000

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGENET_ROOT = "./test_dataset"

weight_bits = 4
activation_bits = 8
model = "resnet18"

ANALYSIS_CSV = "ptq_metrics/instability_{model}_W{weight_bits}A{activation_bits}.csv"
OUTDIR = "selected_unstable_adaround_results"

ADAROUND_CONFIG = dict(
    pattern="layer",
    scale_lr=4.0e-5,
    warm_up=0.2,
    weight=0.01,
    max_count=1000,
    b_range=[20, 2],
    keep_gpu=True,
    round_mode="learned_hard_sigmoid",
    prob=1.0,
)

# 16 * 32 = 512 reconstruction images.
# Set --recon-batches 0 to use all 1024 calibration images.
RECON_BATCHES = 16


class DictNamespace(SimpleNamespace):
    def __contains__(self, key):
        return key in self.__dict__

    def __getitem__(self, key):
        return self.__dict__[key]


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_loaders(num_calib=NUM_CALIB, num_eval=NUM_EVAL, full_eval=False):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        ),
    ])

    dataset = datasets.ImageFolder(
        root=IMAGENET_ROOT,
        transform=transform,
    )

    indices = list(range(len(dataset)))
    random.Random(SEED).shuffle(indices)

    calib_idx = indices[:num_calib]
    if full_eval:
        eval_idx = list(range(len(dataset)))
    else:
        eval_idx = indices[num_calib:num_calib + num_eval]

    calib_loader = DataLoader(
        Subset(dataset, calib_idx),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )
    eval_loader = DataLoader(
        Subset(dataset, eval_idx),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )
    return calib_loader, eval_loader


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.numel()
    return 100.0 * correct / max(total, 1)


def get_w4a4_config(weight_bits=4, activation_bits=4):
    return {
        "extra_qconfig_dict": {
            "w_observer": "MinMaxObserver",
            "a_observer": "EMAMinMaxObserver",
            "w_fakequantize": "AdaRoundFakeQuantize",
            "a_fakequantize": "FixedFakeQuantize",
            "w_qscheme": {
                "bit": weight_bits,
                "symmetry": True,
                "per_channel": True,
                "pot_scale": False,
            },
            "a_qscheme": {
                "bit": activation_bits,
                "symmetry": False,
                "per_channel": False,
                "pot_scale": False,
            },
        },
    }


def build_calibrated_w4a4(
    fp32_model, calib_loader, enable_quant=True, weight_bits=4, activation_bits=4
):
    qmodel = prepare_by_platform(
        copy.deepcopy(fp32_model).eval(),
        BackendType.Academic,
        get_w4a4_config(weight_bits, activation_bits),
    ).to(DEVICE)

    enable_calibration(qmodel)
    with torch.no_grad():
        for images, _ in calib_loader:
            qmodel(images.to(DEVICE, non_blocking=True))

    if enable_quant:
        enable_quantization(qmodel)
    return qmodel


def load_flagged_layers(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Analysis CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required = {"layer", "is_flagged"}
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

    return df.loc[df["is_flagged"], "layer"].tolist()


def is_weight_bearing_module(module):
    return (
        isinstance(module, (nn.Conv2d, nn.Linear))
        and hasattr(module, "weight")
        and module.weight is not None
    )


def get_direct_flagged_adaround_targets(model, flagged_layers):
    modules = dict(model.named_modules())
    targets = []
    skipped = []

    for name in flagged_layers:
        module = modules.get(name)
        if module is None:
            skipped.append({
                "flagged_layer": name,
                "reason": "LAYER_NOT_FOUND",
            })
            continue

        if is_weight_bearing_module(module):
            targets.append(name)
        else:
            skipped.append({
                "flagged_layer": name,
                "reason": "FLAGGED_BUT_NON_WEIGHT",
            })

    targets = list(dict.fromkeys(targets))
    return targets, skipped


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


def build_exclusion_list(prepared_model, selected_targets):
    """
    Restrict MQBench reconstruction to selected flagged producers by excluding
    every other weighted FX node.
    """
    if not hasattr(prepared_model, "graph"):
        raise TypeError(
            "Expected MQBench Academic prepare_by_platform to return a GraphModule."
        )

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

    missing = [
        name for name in selected_targets
        if name not in set(actual_selected)
    ]
    if missing:
        print("[selection] WARNING: mapped targets not found as weighted FX nodes:")
        for name in missing:
            print(f"  {name}")

    if not actual_selected:
        raise RuntimeError("No selected flagged producer is reconstructable.")

    return actual_selected, excluded_node_names


def run_selected_unstable_adaround(
    calibrated_template_cpu,
    calib_loader,
    selected_targets,
    config,
    recon_batches=RECON_BATCHES,
):
    selected_targets = list(dict.fromkeys(selected_targets))
    if not selected_targets:
        raise RuntimeError("No unstable weighted producers were selected.")

    model = copy.deepcopy(calibrated_template_cpu).to(DEVICE).eval()

    actual_selected, excluded_nodes = build_exclusion_list(
        model,
        selected_targets,
    )

    cali_data = collect_reconstruction_inputs(
        calib_loader,
        max_batches=recon_batches,
    )

    cfg_dict = dict(config)
    cfg_dict["pattern"] = "layer"
    cfg_dict["exclude_node_prefix"] = True
    cfg_dict["exclude_node"] = excluded_nodes
    cfg = DictNamespace(**cfg_dict)

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

    repaired = ptq_reconstruction(
        model,
        cali_data,
        cfg,
    ).to(DEVICE)

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-csv", default=ANALYSIS_CSV)
    parser.add_argument("--model", default="vgg19", choices=sorted(models.list_models()))
    parser.add_argument("--weight-bits", type=int, choices=(4,), default=4)
    parser.add_argument("--activation-bits", type=int, choices=(4, 8), default=4)
    parser.add_argument("--full-eval", action="store_true")
    parser.add_argument(
        "--recon-batches",
        type=int,
        default=RECON_BATCHES,
        help="Selected-AdaRound reconstruction batches; use 0 for all calibration batches.",
    )
    args = parser.parse_args()

    set_seed(SEED)
    os.makedirs(OUTDIR, exist_ok=True)

    config = dict(ADAROUND_CONFIG)
    recon_batches = None if int(args.recon_batches) == 0 else int(args.recon_batches)

    calib_loader, eval_loader = get_loaders(full_eval=args.full_eval)

    fp32 = models.get_model(args.model, weights="DEFAULT").to(DEVICE).eval()
    fp32_acc = evaluate(fp32, eval_loader, DEVICE)
    print(f"    accuracy={fp32_acc:.2f}%")

    print(f"\n[2] Calibrating W{args.weight_bits}A{args.activation_bits} template")
    calibrated_template = build_calibrated_w4a4(
        fp32,
        calib_loader,
        enable_quant=False,
        weight_bits=args.weight_bits,
        activation_bits=args.activation_bits,
    )

    baseline = copy.deepcopy(calibrated_template).to(DEVICE).eval()
    enable_quantization(baseline)
    baseline_acc = evaluate(baseline, eval_loader, DEVICE)
    print(f"    W4A4 accuracy={baseline_acc:.2f}%")
    flagged_layers = load_flagged_layers(args.analysis_csv)

    repair_targets, skipped_flagged = get_direct_flagged_adaround_targets(
        baseline, flagged_layers, )

    if not repair_targets:
        raise RuntimeError(
            "No directly flagged Conv/Linear layer is available for AdaRound.")

    # Free models not needed during reconstruction to reduce GPU memory.
    fp32 = fp32.cpu().eval()
    baseline = baseline.cpu().eval()
    calibrated_template = calibrated_template.cpu().eval()

    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize(DEVICE)

    (
        repaired,
        actual_targets,
        repair_time_sec,
        peak_repair_memory_mb,
        additional_peak_repair_memory_mb,
    ) = run_selected_unstable_adaround(
        calibrated_template,
        calib_loader,
        repair_targets,
        config,
        recon_batches=recon_batches,
    )

    # Final held-out evaluation is outside repair timing.
    repaired_acc = evaluate(repaired, eval_loader, DEVICE)

    print("\n=== SELECTED-UNSTABLE ADAROUND RESULTS ===")
    print(f"FP32 accuracy                 : {fp32_acc:.2f}%")
    print(f"W4A4 accuracy                 : {baseline_acc:.2f}%")
    print(f"Repaired accuracy             : {repaired_acc:.2f}%")
  
    if DEVICE.type == "cuda":
        print(f"Peak repair GPU memory        : {peak_repair_memory_mb:.1f} MB")
        print(
            f"Additional peak repair memory : "
            f"{additional_peak_repair_memory_mb:.1f} MB"
        )

    summary = {
        "model": args.model,
        "method": "AdaRound on directly QuantDIR-flagged layers only",
        "fp32_accuracy": fp32_acc,
        "w4a4_accuracy": baseline_acc,
        "repaired_accuracy": repaired_acc,
    }

if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    main()
