"""Native-grid multimodal sparse Tucker autoencoder.

Every sensor is encoded on its own grid. Cross-attention produces one shared
sparse Tucker core, and continuous mode dictionaries render predictions
directly on each sensor grid. No observed image is resized to another grid.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class SensorSpec:
    name: str
    input_channels: int
    reconstructed_channels: int
    token_stride: int


class SoftShrink(nn.Module):
    def __init__(self, channels: int, init: float = 2e-2, maximum: float = 2.5e-1):
        super().__init__()
        self.maximum = float(maximum)
        probability = float(np.clip(init / maximum, 1e-5, 1 - 1e-5))
        logit = np.log(probability / (1 - probability))
        self.theta = nn.Parameter(torch.full((1, 1, 1, channels), float(logit)))
        self.enabled = True

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return value
        threshold = self.maximum * torch.sigmoid(self.theta)
        return torch.sign(value) * F.relu(value.abs() - threshold)


class SensorTokenEncoder(nn.Module):
    """Produce coordinate-aware tokens without aligning sensor rasters."""

    def __init__(self, channels: int, width: int, token_stride: int):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(channels, width // 2, 3, padding=1),
            nn.GroupNorm(4, width // 2),
            nn.GELU(),
            nn.Conv2d(width // 2, width, token_stride, stride=token_stride),
            nn.GroupNorm(8, width),
            nn.GELU(),
        )
        self.position = nn.Sequential(nn.Linear(2, width), nn.GELU(), nn.Linear(width, width))

    def forward(self, image: torch.Tensor, sensor_embedding: torch.Tensor,
                coordinates: tuple[torch.Tensor, torch.Tensor] | None = None) -> torch.Tensor:
        feature = self.stem(image)
        batch, channels, height, width = feature.shape
        if coordinates is None:
            yy = (torch.arange(image.shape[-2], device=image.device, dtype=image.dtype) + 0.5) / image.shape[-2]
            xx = (torch.arange(image.shape[-1], device=image.device, dtype=image.dtype) + 0.5) / image.shape[-1]
            yy = yy.unsqueeze(0).expand(batch, -1)
            xx = xx.unsqueeze(0).expand(batch, -1)
        else:
            yy, xx = coordinates
        grid_y = yy[:, :, None].expand(-1, -1, xx.shape[1])
        grid_x = xx[:, None, :].expand(-1, yy.shape[1], -1)
        coordinate_map = torch.stack((grid_y, grid_x), dim=1)
        stride = self.stem[3].stride[0]
        coordinate_map = F.avg_pool2d(coordinate_map, stride, stride)
        position = self.position(coordinate_map.permute(0, 2, 3, 1).reshape(batch, -1, 2))
        tokens = feature.flatten(2).transpose(1, 2)
        return tokens + position + sensor_embedding.reshape(1, 1, channels)


class CrossAttentionBlock(nn.Module):
    def __init__(self, width: int, heads: int, expansion: int = 4):
        super().__init__()
        self.q_norm = nn.LayerNorm(width)
        self.kv_norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.ff_norm = nn.LayerNorm(width)
        self.ff = nn.Sequential(
            nn.Linear(width, expansion * width), nn.GELU(),
            nn.Linear(expansion * width, width),
        )

    def forward(self, query: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        update, _ = self.attention(self.q_norm(query), self.kv_norm(tokens),
                                   self.kv_norm(tokens), need_weights=False)
        query = query + update
        return query + self.ff(self.ff_norm(query))


class CoordinateBasis(nn.Module):
    def __init__(self, rank: int, width: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, width), nn.SiLU(), nn.Linear(width, width), nn.SiLU(),
            nn.Linear(width, rank),
        )

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        values = self.net((2 * coordinates - 1).unsqueeze(-1))
        return F.normalize(values, dim=-1, eps=1e-8)


class NativeSparseTuckerAE(nn.Module):
    def __init__(
        self,
        sensors: list[SensorSpec],
        spectral_responses: dict[str, np.ndarray | torch.Tensor],
        spatial_psfs: dict[str, np.ndarray | torch.Tensor] | None,
        target_bands: int,
        ranks: tuple[int, int, int] = (12, 12, 16),
        width: int = 96,
        heads: int = 4,
        attention_layers: int = 3,
        output_max: float = 1.5,
        anchor_sensor: str | None = None,
        residual_scale: float = 0.05,
        shrink_init: float = 2e-2,
        shrink_max: float = 2.5e-1,
    ):
        super().__init__()
        self.sensor_specs = {sensor.name: sensor for sensor in sensors}
        self.sensor_names = tuple(sensor.name for sensor in sensors)
        self.ranks = tuple(ranks)
        self.target_bands = int(target_bands)
        self.output_max = float(output_max)
        self.residual_scale = float(residual_scale)
        self.anchor_sensor = anchor_sensor or sensors[0].name
        if self.anchor_sensor not in self.sensor_specs:
            raise ValueError(f"Unknown anchor sensor: {self.anchor_sensor}")

        self.encoders = nn.ModuleDict({
            sensor.name: SensorTokenEncoder(sensor.input_channels, width, sensor.token_stride)
            for sensor in sensors
        })
        self.sensor_embeddings = nn.ParameterDict({
            sensor.name: nn.Parameter(torch.randn(width) * 0.02) for sensor in sensors
        })

        rank_x, rank_y, rank_s = self.ranks
        self.core_query = nn.Parameter(torch.randn(rank_x * rank_y, width) * 0.02)
        self.core_position = nn.Sequential(
            nn.Linear(2, width), nn.GELU(), nn.Linear(width, width)
        )
        self.attention = nn.ModuleList([
            CrossAttentionBlock(width, heads) for _ in range(attention_layers)
        ])
        self.core_projection = nn.Linear(width, rank_s)
        self.shrink = SoftShrink(rank_s, init=shrink_init, maximum=shrink_max)

        self.row_basis = CoordinateBasis(rank_x)
        self.column_basis = CoordinateBasis(rank_y)
        spectral = torch.from_numpy(self._dct_dictionary(target_bands, rank_s)).float()
        self.spectral_dictionary = nn.Parameter(spectral)

        for name, response in spectral_responses.items():
            matrix = torch.as_tensor(response, dtype=torch.float32)
            if matrix.ndim != 2 or matrix.shape[1] != target_bands:
                raise ValueError(f"Invalid response for {name}: {tuple(matrix.shape)}")
            self.register_buffer(f"response_{name}", matrix)

        spatial_psfs = spatial_psfs or {}
        for sensor in sensors:
            psf = spatial_psfs.get(sensor.name)
            if psf is None:
                kernels = torch.ones(sensor.reconstructed_channels, 1, 1)
            else:
                kernels = torch.as_tensor(psf, dtype=torch.float32)
                if kernels.ndim != 3 or kernels.shape[0] != sensor.reconstructed_channels:
                    raise ValueError(f"Invalid PSF for {sensor.name}: {tuple(kernels.shape)}")
                if kernels.shape[-2] % 2 != 1 or kernels.shape[-1] % 2 != 1:
                    raise ValueError(f"PSF kernels for {sensor.name} must have odd dimensions")
            kernels = kernels / (kernels.sum(dim=(-2, -1), keepdim=True) + 1e-12)
            self.register_buffer(f"psf_{sensor.name}", kernels[:, None])

        initial_logit = float(np.log(0.1 / 0.9))
        self.output_biases = nn.ParameterDict({
            sensor.name: nn.Parameter(torch.full((1, sensor.reconstructed_channels, 1, 1),
                                                  initial_logit))
            for sensor in sensors
        })
        self.fusion_bias = nn.Parameter(torch.full((1, target_bands, 1, 1), initial_logit))

        self.refiners = nn.ModuleDict()
        for sensor in sensors:
            channels = sensor.reconstructed_channels
            self.refiners[sensor.name] = nn.Sequential(
                nn.Conv2d(channels + sensor.input_channels, max(16, 2 * channels),
                          3, padding=1),
                nn.GELU(),
                nn.Conv2d(max(16, 2 * channels), channels, 3, padding=1),
            )
        anchor_channels = self.sensor_specs[self.anchor_sensor].input_channels
        self.fusion_refiner = nn.Sequential(
            nn.Conv2d(target_bands + anchor_channels, 2 * target_bands, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(2 * target_bands, target_bands, 3, padding=1),
        )
        for refiner in [*self.refiners.values(), self.fusion_refiner]:
            nn.init.zeros_(refiner[-1].weight)
            nn.init.zeros_(refiner[-1].bias)

    @staticmethod
    def _dct_dictionary(length: int, rank: int) -> np.ndarray:
        positions = np.arange(length, dtype=np.float64)[:, None]
        frequencies = np.arange(rank, dtype=np.float64)[None, :]
        dictionary = np.cos(np.pi * (positions + 0.5) * frequencies / max(length, 1))
        dictionary[:, 0] /= np.sqrt(2)
        dictionary *= np.sqrt(2 / max(length, 1))
        return dictionary.astype(np.float32)

    def _encode_core(self, inputs: dict[str, torch.Tensor],
                     coordinates: dict[str, tuple[torch.Tensor, torch.Tensor]] | None = None
                     ) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = []
        for name in self.sensor_names:
            if name in inputs:
                sensor_coordinates = None if coordinates is None else coordinates.get(name)
                tokens.append(self.encoders[name](inputs[name], self.sensor_embeddings[name],
                                                  sensor_coordinates))
        if not tokens:
            raise ValueError("At least one sensor input is required")
        tokens = torch.cat(tokens, dim=1)
        batch = tokens.shape[0]

        rank_x, rank_y, rank_s = self.ranks
        yy = (torch.arange(rank_x, device=tokens.device, dtype=tokens.dtype) + 0.5) / rank_x
        xx = (torch.arange(rank_y, device=tokens.device, dtype=tokens.dtype) + 0.5) / rank_y
        grid_y, grid_x = torch.meshgrid(2 * yy - 1, 2 * xx - 1, indexing="ij")
        coordinates = torch.stack((grid_y, grid_x), dim=-1).reshape(-1, 2)
        query = self.core_query + self.core_position(coordinates)
        query = query.unsqueeze(0).expand(batch, -1, -1)
        for block in self.attention:
            query = block(query, tokens)
        pre_core = self.core_projection(query).reshape(batch, rank_x, rank_y, rank_s)
        return self.shrink(pre_core), pre_core

    def _spectral_factor(self, sensor: str) -> torch.Tensor:
        dictionary = self.spectral_dictionary
        dictionary = dictionary / (dictionary.square().sum(dim=0, keepdim=True).sqrt() + 1e-8)
        response = getattr(self, f"response_{sensor}")
        return response @ dictionary

    def _apply_psf(self, image: torch.Tensor, sensor: str) -> torch.Tensor:
        kernel = getattr(self, f"psf_{sensor}")
        padding = (kernel.shape[-2] // 2, kernel.shape[-1] // 2)
        return F.conv2d(image, kernel, padding=padding, groups=image.shape[1])

    def _render(self, core: torch.Tensor, sensor: str, input_image: torch.Tensor,
                coordinates: tuple[torch.Tensor, torch.Tensor] | None = None
                ) -> tuple[torch.Tensor, torch.Tensor]:
        height, width = input_image.shape[-2:]
        if coordinates is None:
            row_coordinates = ((torch.arange(height, device=core.device, dtype=core.dtype) + 0.5) / height)
            column_coordinates = ((torch.arange(width, device=core.device, dtype=core.dtype) + 0.5) / width)
            row_coordinates = row_coordinates.unsqueeze(0).expand(core.shape[0], -1)
            column_coordinates = column_coordinates.unsqueeze(0).expand(core.shape[0], -1)
        else:
            row_coordinates, column_coordinates = coordinates
        row = self.row_basis(row_coordinates)
        column = self.column_basis(column_coordinates)
        spectral = self._spectral_factor(sensor)
        linear = torch.einsum("bijk,bhi,bwj,ck->bchw", core, row, column, spectral)
        base = self.output_max * torch.sigmoid(linear + self.output_biases[sensor])
        residual = self.refiners[sensor](torch.cat((base, input_image), dim=1))
        corrected = torch.clamp(base + self.residual_scale * torch.tanh(residual),
                                0.0, self.output_max)
        prediction = self._apply_psf(corrected, sensor)
        return prediction, residual

    def forward(self, inputs: dict[str, torch.Tensor],
                coordinates: dict[str, tuple[torch.Tensor, torch.Tensor]] | None = None
                ) -> dict[str, torch.Tensor | dict]:
        core, pre_core = self._encode_core(inputs, coordinates)
        native, residuals = {}, {}
        for name, image in inputs.items():
            sensor_coordinates = None if coordinates is None else coordinates.get(name)
            prediction, residual = self._render(core, name, image, sensor_coordinates)
            native[name] = prediction
            residuals[name] = residual

        if self.anchor_sensor not in inputs:
            raise ValueError(f"Fusion requires anchor input '{self.anchor_sensor}'")
        anchor = inputs[self.anchor_sensor]
        rank_x, rank_y, _ = self.ranks
        anchor_coordinates = None if coordinates is None else coordinates.get(self.anchor_sensor)
        if anchor_coordinates is None:
            row_coordinates = ((torch.arange(anchor.shape[-2], device=core.device, dtype=core.dtype) + 0.5)
                               / anchor.shape[-2]).unsqueeze(0).expand(core.shape[0], -1)
            column_coordinates = ((torch.arange(anchor.shape[-1], device=core.device, dtype=core.dtype) + 0.5)
                                  / anchor.shape[-1]).unsqueeze(0).expand(core.shape[0], -1)
        else:
            row_coordinates, column_coordinates = anchor_coordinates
        row = self.row_basis(row_coordinates)
        column = self.column_basis(column_coordinates)
        spectral = self.spectral_dictionary
        spectral = spectral / (spectral.square().sum(dim=0, keepdim=True).sqrt() + 1e-8)
        linear_fused = torch.einsum("bijk,bhi,bwj,ck->bchw", core, row, column, spectral)
        fused_base = self.output_max * torch.sigmoid(linear_fused + self.fusion_bias)
        fused_residual = self.fusion_refiner(torch.cat((fused_base, anchor), dim=1))
        fused = torch.clamp(fused_base + self.residual_scale * torch.tanh(fused_residual),
                            0.0, self.output_max)
        return {
            "fused": fused,
            "native": native,
            "core": core,
            "pre_core": pre_core,
            "residuals": residuals,
            "fused_residual": fused_residual,
        }

    def dictionary_penalty(self) -> torch.Tensor:
        dictionary = self.spectral_dictionary
        gram = dictionary.T @ dictionary
        identity = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
        return (gram - identity).square().mean()
