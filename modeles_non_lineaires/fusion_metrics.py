"""Full-reference and observation-consistency metrics for spectral cubes."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _masked_vectors(prediction: torch.Tensor, target: torch.Tensor,
                    mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    if mask is None:
        return prediction.reshape(-1), target.reshape(-1)
    valid = mask.expand_as(prediction).bool()
    return prediction[valid], target[valid]


def rmse(prediction: torch.Tensor, target: torch.Tensor,
         mask: torch.Tensor | None = None) -> torch.Tensor:
    pred, true = _masked_vectors(prediction, target, mask)
    return torch.sqrt(torch.mean((pred - true).square()) + 1e-12)


def psnr(prediction: torch.Tensor, target: torch.Tensor,
         data_range: float = 1.5, mask: torch.Tensor | None = None) -> torch.Tensor:
    value = rmse(prediction, target, mask)
    return 20 * torch.log10(torch.as_tensor(data_range, device=value.device) / value)


def sam(prediction: torch.Tensor, target: torch.Tensor,
        mask: torch.Tensor | None = None) -> torch.Tensor:
    pred = prediction.movedim(1, -1).reshape(-1, prediction.shape[1])
    true = target.movedim(1, -1).reshape(-1, target.shape[1])
    cosine = (pred * true).sum(1) / (pred.norm(dim=1) * true.norm(dim=1) + 1e-8)
    angles = torch.acos(cosine.clamp(-1 + 1e-7, 1 - 1e-7)) * (180 / math.pi)
    if mask is not None:
        valid = mask[:, 0].reshape(-1).bool()
        angles = angles[valid]
    return angles.mean()


def uiqi(prediction: torch.Tensor, target: torch.Tensor,
         mask: torch.Tensor | None = None) -> torch.Tensor:
    pred, true = _masked_vectors(prediction, target, mask)
    mean_p, mean_t = pred.mean(), true.mean()
    var_p = (pred - mean_p).square().mean()
    var_t = (true - mean_t).square().mean()
    covariance = ((pred - mean_p) * (true - mean_t)).mean()
    return ((2 * mean_p * mean_t + 1e-8) * (2 * covariance + 1e-8)
            / ((mean_p.square() + mean_t.square() + 1e-8) * (var_p + var_t + 1e-8)))


def ssim(prediction: torch.Tensor, target: torch.Tensor,
         data_range: float = 1.5, window: int = 7) -> torch.Tensor:
    padding = window // 2
    mean_p = F.avg_pool2d(prediction, window, 1, padding)
    mean_t = F.avg_pool2d(target, window, 1, padding)
    var_p = F.avg_pool2d(prediction.square(), window, 1, padding) - mean_p.square()
    var_t = F.avg_pool2d(target.square(), window, 1, padding) - mean_t.square()
    covariance = F.avg_pool2d(prediction * target, window, 1, padding) - mean_p * mean_t
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    score = ((2 * mean_p * mean_t + c1) * (2 * covariance + c2)
             / ((mean_p.square() + mean_t.square() + c1) * (var_p + var_t + c2)))
    return score.mean()


def ergas(prediction: torch.Tensor, target: torch.Tensor, ratio: float,
          mask: torch.Tensor | None = None) -> torch.Tensor:
    errors = []
    for band in range(target.shape[1]):
        pred_band = prediction[:, band:band + 1]
        true_band = target[:, band:band + 1]
        band_rmse = rmse(pred_band, true_band, mask)
        if mask is None:
            mean = true_band.mean()
        else:
            mean = true_band[mask.expand_as(true_band).bool()].mean()
        errors.append((band_rmse / (mean.abs() + 1e-8)).square())
    return 100 / ratio * torch.sqrt(torch.stack(errors).mean())


def spectral_smoothness(cube: torch.Tensor) -> torch.Tensor:
    if cube.shape[1] < 3:
        return torch.zeros((), device=cube.device, dtype=cube.dtype)
    second = cube[:, 2:] - 2 * cube[:, 1:-1] + cube[:, :-2]
    return second.abs().mean()


def spatial_total_variation(cube: torch.Tensor) -> torch.Tensor:
    vertical = (cube[..., 1:, :] - cube[..., :-1, :]).abs().mean()
    horizontal = (cube[..., :, 1:] - cube[..., :, :-1]).abs().mean()
    return vertical + horizontal


@torch.no_grad()
def metric_dict(prediction: torch.Tensor, target: torch.Tensor,
                ratio: float = 1.0, data_range: float = 1.5,
                mask: torch.Tensor | None = None) -> dict[str, float]:
    return {
        "rmse": float(rmse(prediction, target, mask)),
        "psnr_db": float(psnr(prediction, target, data_range, mask)),
        "sam_deg": float(sam(prediction, target, mask)),
        "ergas": float(ergas(prediction, target, ratio, mask)),
        "ssim": float(ssim(prediction, target, data_range)),
        "uiqi": float(uiqi(prediction, target, mask)),
    }

