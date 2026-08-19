"""Train the native-grid multimodal sparse Tucker autoencoder.

The task is self-supervised: the fused high-spatial/high-spectral cube is
never used as a target. It is constrained through the native S2/OLCI
observation operators, their SRFs/PSFs, and sparse Tucker regularisation.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from modeles_non_lineaires.fusion_metrics import sam, spectral_smoothness
from modeles_non_lineaires.native_fusion_data import (
    SENSOR_ORDER,
    NativePatchDataset,
    load_native_archive,
    model_from_data,
    model_metadata,
    move_batch,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "training" / "andorra_20180118_native_toa.npz"
DEFAULT_OUTPUT = ROOT / "outputs" / "native_sparse_tucker_andorra"
OBSERVATION_WEIGHTS = {"s2_10": 1.8, "s2_20": 1.2, "s2_60": 0.6, "olci": 1.8}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def masked_charbonnier(prediction: torch.Tensor, target: torch.Tensor,
                       mask: torch.Tensor, epsilon: float = 1e-3) -> torch.Tensor:
    valid = mask.expand_as(target)
    error = torch.sqrt((prediction - target).square() + epsilon ** 2)
    return error[valid].mean()


def loss_terms(output: dict, batch: dict, model: torch.nn.Module,
               epoch: int, warmup_epochs: int, weights: dict[str, float]) -> dict[str, torch.Tensor]:
    observations = []
    terms: dict[str, torch.Tensor] = {}
    for sensor in SENSOR_ORDER:
        value = masked_charbonnier(output["native"][sensor], batch["targets"][sensor],
                                   batch["masks"][sensor])
        terms[f"reconstruction_{sensor}"] = value
        observations.append(weights[sensor] * value)
    reconstruction = torch.stack(observations).sum() / sum(weights.values())
    terms["reconstruction"] = reconstruction

    terms["sam_olci"] = sam(output["native"]["olci"], batch["targets"]["olci"],
                             batch["masks"]["olci"])
    terms["core_l1"] = output["core"].abs().mean()
    terms["dictionary"] = model.dictionary_penalty()
    terms["residual"] = torch.stack([
        value.square().mean() for value in output["residuals"].values()
    ] + [output["fused_residual"].square().mean()]).mean()
    terms["spectral_smoothness"] = spectral_smoothness(output["fused"])

    regularisation_ramp = min(1.0, (epoch + 1) / max(warmup_epochs, 1))
    total = reconstruction
    total = total + weights["sam"] * terms["sam_olci"]
    total = total + regularisation_ramp * (
        weights["core"] * terms["core_l1"]
        + weights["dictionary"] * terms["dictionary"]
        + weights["residual"] * terms["residual"]
        + weights["spectral"] * terms["spectral_smoothness"]
    )
    terms["total"] = total
    return terms


def run_epoch(model: torch.nn.Module, loader: DataLoader, device: torch.device,
              epoch: int, warmup_epochs: int, optimizer=None, scaler=None,
              max_grad_norm: float = 1.0,
              loss_weights: dict[str, float] | None = None) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    accumulated: dict[str, float] = {}
    batches = 0
    for raw_batch in loader:
        batch = move_batch(raw_batch, device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            amp_context = (torch.amp.autocast("cuda") if device.type == "cuda"
                           else nullcontext())
            with amp_context:
                output = model(batch["inputs"], batch["coordinates"])
                terms = loss_terms(output, batch, model, epoch, warmup_epochs, loss_weights)
            if training:
                scaler.scale(terms["total"]).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
        for name, value in terms.items():
            accumulated[name] = accumulated.get(name, 0.0) + float(value.detach())
        batches += 1
    return {name: value / max(batches, 1) for name, value in accumulated.items()}


@torch.no_grad()
def export_full_fusion(model: torch.nn.Module, data: dict[str, np.ndarray],
                       device: torch.device, output_path: Path, patch_10m: int,
                       batch_size: int, workers: int) -> None:
    dataset = NativePatchDataset(data, patch_10m=patch_10m, stride_10m=patch_10m,
                                 split="all")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers,
                        pin_memory=device.type == "cuda")
    bands = int(data["olci"].shape[0])
    height, width = data["s2_10"].shape[-2:]
    fused_sum = np.zeros((bands, height, width), dtype=np.float32)
    counts = np.zeros((1, height, width), dtype=np.float32)
    model.eval()
    for raw_batch in loader:
        origins = raw_batch["origin_10m"].numpy()
        batch = move_batch(raw_batch, device)
        prediction = model(batch["inputs"], batch["coordinates"])["fused"].float().cpu().numpy()
        for sample, (row, col) in zip(prediction, origins):
            row, col = int(row), int(col)
            fused_sum[:, row:row + patch_10m, col:col + patch_10m] += sample
            counts[:, row:row + patch_10m, col:col + patch_10m] += 1
    fused = fused_sum / np.maximum(counts, 1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, fused=fused.astype(np.float16),
                        wavelength_nm=data["olci_wavelength_nm"].astype(np.float32))


def save_checkpoint(path: Path, model: torch.nn.Module, optimizer, scheduler,
                    epoch: int, best_validation: float, best_epoch: int, args: argparse.Namespace,
                    history: list[dict]) -> None:
    serializable_arguments = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "epoch": epoch,
        "best_validation": best_validation,
        "best_epoch": best_epoch,
        "arguments": serializable_arguments,
        "model_metadata": model_metadata(model),
        "history": history,
    }, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--patch-10m", type=int, default=120)
    parser.add_argument("--stride-10m", type=int, default=120)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1.5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=int, default=20)
    parser.add_argument("--ranks", type=int, nargs=3, default=(20, 20, 16))
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--attention-layers", type=int, default=3)
    parser.add_argument("--residual-scale", type=float, default=0.10)
    parser.add_argument("--shrink-init", type=float, default=0.05)
    parser.add_argument("--shrink-max", type=float, default=0.35)
    parser.add_argument("--s2-10-weight", type=float, default=OBSERVATION_WEIGHTS["s2_10"])
    parser.add_argument("--s2-20-weight", type=float, default=OBSERVATION_WEIGHTS["s2_20"])
    parser.add_argument("--s2-60-weight", type=float, default=OBSERVATION_WEIGHTS["s2_60"])
    parser.add_argument("--olci-weight", type=float, default=OBSERVATION_WEIGHTS["olci"])
    parser.add_argument("--sam-weight", type=float, default=2e-4)
    parser.add_argument("--core-weight", type=float, default=5e-3)
    parser.add_argument("--dictionary-weight", type=float, default=1e-3)
    parser.add_argument("--residual-weight", type=float, default=1e-3)
    parser.add_argument("--spectral-weight", type=float, default=5e-4)
    parser.add_argument("--early-stopping-patience", type=int, default=30)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--no-export-fused", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_native_archive(args.data)
    loss_weights = {
        "s2_10": args.s2_10_weight, "s2_20": args.s2_20_weight,
        "s2_60": args.s2_60_weight, "olci": args.olci_weight,
        "sam": args.sam_weight, "core": args.core_weight,
        "dictionary": args.dictionary_weight, "residual": args.residual_weight,
        "spectral": args.spectral_weight,
    }
    train_data = NativePatchDataset(data, args.patch_10m, args.stride_10m, "train")
    validation_data = NativePatchDataset(data, args.patch_10m, args.stride_10m,
                                         "validation")
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=device.type == "cuda",
                              persistent_workers=args.workers > 0, generator=generator)
    validation_loader = DataLoader(validation_data, batch_size=args.batch_size, shuffle=False,
                                   num_workers=args.workers, pin_memory=device.type == "cuda",
                                   persistent_workers=args.workers > 0)
    model = model_from_data(data, tuple(args.ranks), args.width, args.heads,
                            args.attention_layers, residual_scale=args.residual_scale,
                            shrink_init=args.shrink_init,
                            shrink_max=args.shrink_max).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs,
                                                            eta_min=args.learning_rate / 20)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    start_epoch, best_validation, best_epoch, history = 0, math.inf, 0, []
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_validation = float(checkpoint["best_validation"])
        history = list(checkpoint.get("history", []))
        best_epoch = int(checkpoint.get("best_epoch", 0))
        if best_epoch == 0 and history:
            best_epoch = min(history, key=lambda item: item["validation"]["total"])["epoch"]

    run_info = {
        "device": str(device),
        "torch_version": torch.__version__,
        "train_patches": len(train_data),
        "validation_patches": len(validation_data),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "data": str(args.data.resolve()),
        "no_observed_image_resampling": True,
        "loss_weights": loss_weights,
    }
    print(json.dumps(run_info), flush=True)
    (args.output / "run_config.json").write_text(
        json.dumps({**run_info, "arguments": vars(args)}, indent=2, default=str),
        encoding="utf-8",
    )
    started = time.time()
    for epoch in range(start_epoch, args.epochs):
        train_metrics = run_epoch(model, train_loader, device, epoch, args.warmup_epochs,
                                  optimizer, scaler, loss_weights=loss_weights)
        validation_metrics = run_epoch(model, validation_loader, device, epoch,
                                       args.warmup_epochs, loss_weights=loss_weights)
        scheduler.step()
        record = {
            "epoch": epoch + 1,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train": train_metrics,
            "validation": validation_metrics,
            "elapsed_minutes": (time.time() - started) / 60,
        }
        history.append(record)
        print(json.dumps(record), flush=True)
        if validation_metrics["total"] < best_validation - args.early_stopping_min_delta:
            best_validation = validation_metrics["total"]
            best_epoch = epoch + 1
            save_checkpoint(args.output / "best_checkpoint.pt", model, optimizer, scheduler,
                            epoch, best_validation, best_epoch, args, history)
        save_checkpoint(args.output / "last_checkpoint.pt", model, optimizer, scheduler,
                        epoch, best_validation, best_epoch, args, history)
        (args.output / "history.json").write_text(json.dumps(history, indent=2),
                                                   encoding="utf-8")
        if epoch + 1 - best_epoch >= args.early_stopping_patience:
            print(json.dumps({"status": "early_stopping", "epoch": epoch + 1,
                              "best_epoch": best_epoch,
                              "best_validation": best_validation}), flush=True)
            break

    best = torch.load(args.output / "best_checkpoint.pt", map_location=device,
                      weights_only=False)
    model.load_state_dict(best["model_state"])
    if not args.no_export_fused:
        export_full_fusion(model, data, device, args.output / "fused_cube_float16.npz",
                           args.patch_10m, args.batch_size, args.workers)
    print(json.dumps({"status": "complete", "best_epoch": best_epoch,
                      "best_validation": best_validation,
                      "output": str(args.output.resolve())}), flush=True)


if __name__ == "__main__":
    main()
