"""Evaluate native observation consistency and optional fused-cube quality."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from modeles_non_lineaires.fusion_metrics import (
    metric_dict,
    spatial_total_variation,
    spectral_smoothness,
)
from modeles_non_lineaires.native_fusion_data import (
    SENSOR_ORDER,
    NativePatchDataset,
    load_native_archive,
    model_from_data,
    move_batch,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "training" / "andorra_20180118_native_toa.npz"


def average(records: list[dict[str, float]]) -> dict[str, float]:
    return {key: float(np.mean([record[key] for record in records])) for key in records[0]}


def reference_metrics(reference_path: Path, fused_path: Path) -> dict[str, float]:
    with np.load(reference_path, allow_pickle=False) as archive:
        key = "fused" if "fused" in archive.files else archive.files[0]
        reference = archive[key].astype(np.float32)
    with np.load(fused_path, allow_pickle=False) as archive:
        fused = archive["fused"].astype(np.float32)
    if reference.shape != fused.shape:
        raise ValueError(f"Reference {reference.shape} and fusion {fused.shape} differ")
    pred = torch.from_numpy(fused).unsqueeze(0)
    target = torch.from_numpy(reference).unsqueeze(0)
    return metric_dict(pred, target, ratio=1.0, data_range=1.5)


def markdown_report(report: dict) -> str:
    lines = ["# Native sparse Tucker fusion evaluation", "",
             f"Checkpoint: `{report['checkpoint']}`", "",
             "## Native observation consistency", "",
             "| Sensor | RMSE | PSNR (dB) | SAM (deg) | ERGAS | SSIM | UIQI |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for sensor, values in report["native_consistency"].items():
        lines.append(
            f"| {sensor} | {values['rmse']:.6f} | {values['psnr_db']:.3f} | "
            f"{values['sam_deg']:.3f} | {values['ergas']:.3f} | "
            f"{values['ssim']:.5f} | {values['uiqi']:.5f} |"
        )
    lines.extend(["", "## Unsupervised diagnostics", ""])
    for key, value in report["diagnostics"].items():
        lines.append(f"- {key}: {value:.8f}")
    if "fused_reference" in report:
        lines.extend(["", "## Fused reference metrics", ""])
        for key, value in report["fused_reference"].items():
            lines.append(f"- {key}: {value:.8f}")
    lines.extend(["", "The native metrics compare each decoded observation with its original "
                  "sensor raster on the same grid; they are not a substitute for fused-cube "
                  "ground truth.", ""])
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--split", choices=("train", "validation", "all"),
                        default="validation")
    parser.add_argument("--reference", type=Path,
                        help="Optional NPZ ground-truth fused cube (bands, height, width)")
    parser.add_argument("--fused", type=Path,
                        help="Exported fused NPZ, required together with --reference")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    training_args = checkpoint.get("arguments", {})
    patch = int(training_args.get("patch_10m", 120))
    stride = int(training_args.get("stride_10m", 120))
    ranks = tuple(training_args.get("ranks", (12, 12, 16)))
    data = load_native_archive(args.data)
    model = model_from_data(
        data,
        ranks=ranks,
        width=int(training_args.get("width", 96)),
        heads=int(training_args.get("heads", 4)),
        attention_layers=int(training_args.get("attention_layers", 3)),
        residual_scale=float(training_args.get("residual_scale", 0.05)),
        shrink_init=float(training_args.get("shrink_init", 0.02)),
        shrink_max=float(training_args.get("shrink_max", 0.25)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    dataset = NativePatchDataset(data, patch, stride, args.split)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=device.type == "cuda",
                        persistent_workers=args.workers > 0)
    records = {sensor: [] for sensor in SENSOR_ORDER}
    smoothness, total_variation, core_density = [], [], []
    with torch.no_grad():
        for raw_batch in loader:
            batch = move_batch(raw_batch, device)
            output = model(batch["inputs"], batch["coordinates"])
            for sensor in SENSOR_ORDER:
                records[sensor].append(metric_dict(
                    output["native"][sensor], batch["targets"][sensor], ratio=1.0,
                    data_range=1.5, mask=batch["masks"][sensor]
                ))
            smoothness.append(float(spectral_smoothness(output["fused"])))
            total_variation.append(float(spatial_total_variation(output["fused"])))
            core_density.append(float((output["core"].abs() > 1e-4).float().mean()))

    report = {
        "checkpoint": str(args.checkpoint.resolve()),
        "data": str(args.data.resolve()),
        "device": str(device),
        "split": args.split,
        "patches": len(dataset),
        "native_consistency": {sensor: average(values) for sensor, values in records.items()},
        "diagnostics": {
            "spectral_second_difference_l1": float(np.mean(smoothness)),
            "spatial_total_variation_l1": float(np.mean(total_variation)),
            "sparse_core_active_fraction_gt_1e-4": float(np.mean(core_density)),
        },
        "interpretation": "native metrics are same-grid observation consistency; no resizing",
    }
    if bool(args.reference) != bool(args.fused):
        raise ValueError("--reference and --fused must be passed together")
    if args.reference:
        report["fused_reference"] = reference_metrics(args.reference, args.fused)
    output = args.output or args.checkpoint.parent / "evaluation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    output.with_suffix(".md").write_text(markdown_report(report), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
