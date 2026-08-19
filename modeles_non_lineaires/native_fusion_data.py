"""Dataset utilities for co-located rasters kept on their native grids."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from modeles_non_lineaires.native_sparse_tucker_ae import NativeSparseTuckerAE, SensorSpec


SENSOR_ORDER = ("s2_10", "s2_20", "s2_60", "olci")
TOKEN_STRIDES = {"s2_10": 10, "s2_20": 5, "s2_60": 2, "olci": 1}


def load_native_archive(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    required = {
        *(SENSOR_ORDER),
        *(f"mask_{name}" for name in SENSOR_ORDER),
        *(f"coord_y_{name}" for name in SENSOR_ORDER),
        *(f"coord_x_{name}" for name in SENSOR_ORDER),
        *(f"response_{name}" for name in SENSOR_ORDER),
        *(f"recon_indices_{name}" for name in SENSOR_ORDER),
    }
    missing = required.difference(data)
    if missing:
        raise KeyError(f"Missing arrays in {path}: {sorted(missing)}")
    return data


def sensor_specs(data: dict[str, np.ndarray]) -> list[SensorSpec]:
    return [
        SensorSpec(
            name=name,
            input_channels=int(data[name].shape[0]),
            reconstructed_channels=int(data[f"response_{name}"].shape[0]),
            token_stride=TOKEN_STRIDES[name],
        )
        for name in SENSOR_ORDER
    ]


def model_from_data(
    data: dict[str, np.ndarray],
    ranks: tuple[int, int, int] = (12, 12, 16),
    width: int = 96,
    heads: int = 4,
    attention_layers: int = 3,
    output_max: float = 1.5,
    residual_scale: float = 0.05,
    shrink_init: float = 2e-2,
    shrink_max: float = 2.5e-1,
) -> NativeSparseTuckerAE:
    responses = {name: data[f"response_{name}"] for name in SENSOR_ORDER}
    psfs = {name: data[f"psf_{name}"] for name in SENSOR_ORDER if f"psf_{name}" in data}
    model = NativeSparseTuckerAE(
        sensors=sensor_specs(data),
        spectral_responses=responses,
        spatial_psfs=psfs,
        target_bands=int(data["olci"].shape[0]),
        ranks=ranks,
        width=width,
        heads=heads,
        attention_layers=attention_layers,
        output_max=output_max,
        anchor_sensor="s2_10",
        residual_scale=residual_scale,
        shrink_init=shrink_init,
        shrink_max=shrink_max,
    )
    # Initialise each radiometric bias from its native observation mean. This
    # keeps the first forward pass in the physical reflectance range while all
    # spatial/spectral structure still has to be learned.
    with torch.no_grad():
        for name in SENSOR_ORDER:
            indices = data[f"recon_indices_{name}"].astype(np.int64)
            means = data[name][indices].mean(axis=(1, 2)) / output_max
            probabilities = np.clip(means, 1e-3, 1 - 1e-3)
            logits = np.log(probabilities / (1 - probabilities)).astype(np.float32)
            model.output_biases[name].copy_(torch.from_numpy(logits)[None, :, None, None])
        fused_probability = float(np.clip(data["olci"].mean() / output_max, 1e-3, 1 - 1e-3))
        model.fusion_bias.fill_(float(np.log(fused_probability / (1 - fused_probability))))
    return model


def model_metadata(model: NativeSparseTuckerAE) -> dict:
    return {
        "sensors": [asdict(model.sensor_specs[name]) for name in model.sensor_names],
        "ranks": list(model.ranks),
        "target_bands": model.target_bands,
        "output_max": model.output_max,
        "anchor_sensor": model.anchor_sensor,
        "residual_scale": model.residual_scale,
        "shrink_max": model.shrink.maximum,
    }


class NativePatchDataset(Dataset):
    """Map one geographic patch to exact integer windows on every native grid."""

    def __init__(
        self,
        data: dict[str, np.ndarray],
        patch_10m: int = 120,
        stride_10m: int = 60,
        split: str = "train",
        validation_modulo: int = 5,
    ):
        if split not in {"train", "validation", "all"}:
            raise ValueError(f"Invalid split: {split}")
        self.data = data
        self.patch_10m = int(patch_10m)
        self.stride_10m = int(stride_10m)
        self.split = split
        height, width = data["s2_10"].shape[-2:]
        if self.patch_10m > min(height, width):
            raise ValueError("patch_10m is larger than the scene")
        if self.patch_10m % 30 or self.stride_10m % 30:
            raise ValueError("patch_10m and stride_10m must be divisible by 30")
        if split != "all" and self.stride_10m < self.patch_10m:
            raise ValueError("Training/validation stride must be >= patch size to avoid leakage")

        candidates = []
        for row_index, row in enumerate(range(0, height - self.patch_10m + 1,
                                               self.stride_10m)):
            for col_index, col in enumerate(range(0, width - self.patch_10m + 1,
                                                   self.stride_10m)):
                is_validation = (row_index + 2 * col_index) % validation_modulo == 0
                if split == "all" or (split == "validation" and is_validation) or (
                        split == "train" and not is_validation):
                    candidates.append((row, col))
        if not candidates:
            raise ValueError(f"No patches generated for split={split}")
        self.windows = candidates
        self.base_height = height
        self.base_width = width

    def __len__(self) -> int:
        return len(self.windows)

    def _window(self, sensor: str, row_10m: int, col_10m: int) -> tuple[int, int, int, int]:
        height, width = self.data[sensor].shape[-2:]
        row = row_10m * height // self.base_height
        col = col_10m * width // self.base_width
        patch_h = self.patch_10m * height // self.base_height
        patch_w = self.patch_10m * width // self.base_width
        return row, row + patch_h, col, col + patch_w

    def __getitem__(self, index: int) -> dict:
        row_10m, col_10m = self.windows[index]
        inputs, targets, masks, coordinates = {}, {}, {}, {}
        for sensor in SENSOR_ORDER:
            row0, row1, col0, col1 = self._window(sensor, row_10m, col_10m)
            image = np.ascontiguousarray(self.data[sensor][:, row0:row1, col0:col1])
            indices = self.data[f"recon_indices_{sensor}"].astype(np.int64)
            mask = np.ascontiguousarray(self.data[f"mask_{sensor}"][:, row0:row1, col0:col1])
            coord_y = np.ascontiguousarray(self.data[f"coord_y_{sensor}"][row0:row1])
            coord_x = np.ascontiguousarray(self.data[f"coord_x_{sensor}"][col0:col1])
            inputs[sensor] = torch.from_numpy(image).float()
            targets[sensor] = torch.from_numpy(image[indices]).float()
            masks[sensor] = torch.from_numpy(mask.astype(np.bool_))
            coordinates[sensor] = (torch.from_numpy(coord_y).float(),
                                   torch.from_numpy(coord_x).float())
        return {
            "inputs": inputs,
            "targets": targets,
            "masks": masks,
            "coordinates": coordinates,
            "origin_10m": torch.tensor((row_10m, col_10m), dtype=torch.int64),
        }


def move_batch(batch: dict, device: torch.device) -> dict:
    return {
        "inputs": {key: value.to(device, non_blocking=True)
                   for key, value in batch["inputs"].items()},
        "targets": {key: value.to(device, non_blocking=True)
                    for key, value in batch["targets"].items()},
        "masks": {key: value.to(device, non_blocking=True)
                  for key, value in batch["masks"].items()},
        "coordinates": {
            key: (value[0].to(device, non_blocking=True), value[1].to(device, non_blocking=True))
            for key, value in batch["coordinates"].items()
        },
        "origin_10m": batch["origin_10m"],
    }
