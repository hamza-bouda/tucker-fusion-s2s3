"""Prepare official Sentinel-2/3 response files for PyTorch training.

Inputs are the unmodified ESA/Copernicus downloads stored under
``data/instrument_response/raw``. Outputs are compact NumPy archives under
``data/instrument_response/processed``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from netCDF4 import Dataset


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW = ROOT / "data" / "instrument_response" / "raw"
DEFAULT_OUT = ROOT / "data" / "instrument_response" / "processed"
S2_BANDS = ("B01", "B02", "B03", "B04", "B05", "B06", "B07",
            "B08", "B8A", "B09", "B10", "B11", "B12")

SOURCES = {
    "s2_srf": "https://sentiwiki.copernicus.eu/web/s2-documents",
    "s2_psf": "https://sentiwiki.copernicus.eu/web/s2-mission",
    "s3_srf": "https://sentiwiki.copernicus.eu/web/s3-olci-instrument",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    denom = values.sum(axis=1, keepdims=True)
    if np.any(np.abs(denom) < 1e-12):
        raise ValueError("A response row has a zero integral")
    return values / denom


def prepare_s2_srf(raw: Path, out: Path) -> list[dict]:
    workbook = raw / "Sentinel-2_SRF_v5.xlsx"
    records = []
    for unit in ("S2A", "S2B"):
        sheet = pd.read_excel(workbook, sheet_name=f"Spectral Responses ({unit})")
        wavelengths = sheet.iloc[:, 0].to_numpy(dtype=np.float64)
        response = sheet.iloc[:, 1:14].to_numpy(dtype=np.float64).T
        response = normalize_rows(np.clip(response, 0.0, None))
        target = out / f"{unit.lower()}_srf_1nm.npz"
        np.savez_compressed(
            target,
            wavelength_nm=wavelengths,
            response=response.astype(np.float32),
            bands=np.asarray(S2_BANDS),
            unit=unit,
        )
        records.append({"file": str(target.relative_to(ROOT)), "sha256": sha256(target),
                        "shape": list(response.shape), "source": SOURCES["s2_srf"]})
    return records


def native_psf(oversampled: np.ndarray, factor: int = 5) -> np.ndarray:
    """Integrate an oversampled PSF into bins centred on native pixels."""
    if oversampled.ndim != 2 or oversampled.shape[0] != oversampled.shape[1]:
        raise ValueError(f"Expected a square PSF, got {oversampled.shape}")
    centre = (oversampled.shape[0] - 1) / 2
    coords = np.rint((np.arange(oversampled.shape[0]) - centre) / factor).astype(int)
    radius = int(np.max(np.abs(coords)))
    result = np.zeros((2 * radius + 1, 2 * radius + 1), dtype=np.float64)
    for iy, by in enumerate(coords + radius):
        for ix, bx in enumerate(coords + radius):
            result[by, bx] += oversampled[iy, ix]
    result /= result.sum()
    return result


def prepare_s2_psf(raw: Path, out: Path) -> list[dict]:
    records = []
    for unit in ("S2A", "S2B"):
        folder = raw / f"{unit}_PSF" / f"{unit}_PSF"
        raw_stack, native_stack, bands = [], [], []
        for band in S2_BANDS:
            if band == "B10":  # B10 does not observe the ground: no official PSF.
                continue
            matrix = np.loadtxt(folder / f"{unit}_PSF_{band}.csv", delimiter=",")
            matrix = matrix.astype(np.float64)
            matrix /= matrix.sum()
            raw_stack.append(matrix)
            native_stack.append(native_psf(matrix))
            bands.append(band)
        target = out / f"{unit.lower()}_psf.npz"
        np.savez_compressed(
            target,
            psf_oversampled=np.stack(raw_stack).astype(np.float32),
            psf_native=np.stack(native_stack).astype(np.float32),
            bands=np.asarray(bands),
            oversampling_factor=np.int32(5),
            centre_index=np.int32(16),
            unit=unit,
        )
        records.append({"file": str(target.relative_to(ROOT)), "sha256": sha256(target),
                        "raw_shape": list(np.stack(raw_stack).shape),
                        "native_shape": list(np.stack(native_stack).shape),
                        "source": SOURCES["s2_psf"]})
    return records


def prepare_olci_srf(raw: Path, out: Path) -> list[dict]:
    records = []
    for unit in ("S3A", "S3B"):
        source = raw / f"{unit}_OLCI_mean_SRF.nc4"
        with Dataset(source) as dataset:
            response = np.asarray(dataset["mean_spectral_response_function"][:], dtype=np.float64)
            wavelengths = np.asarray(
                dataset["mean_spectral_response_function_wavelength"][:], dtype=np.float64)
            response = normalize_rows(np.clip(response, 0.0, None))
            centre = np.asarray(dataset["srf_centre_wavelength"][:], dtype=np.float64)
            fwhm = np.asarray(dataset["bandwidth_fwhm"][:], dtype=np.float64)
        target = out / f"{unit.lower()}_olci_mean_srf.npz"
        np.savez_compressed(
            target,
            wavelength_nm=wavelengths.astype(np.float32),
            response=response.astype(np.float32),
            centre_wavelength_nm=centre.astype(np.float32),
            fwhm_nm=fwhm.astype(np.float32),
            bands=np.asarray([f"Oa{i:02d}" for i in range(1, 22)]),
            unit=unit,
        )
        records.append({"file": str(target.relative_to(ROOT)), "sha256": sha256(target),
                        "shape": list(response.shape), "source": SOURCES["s3_srf"]})
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    products = []
    products += prepare_s2_srf(args.raw, args.out)
    products += prepare_s2_psf(args.raw, args.out)
    products += prepare_olci_srf(args.raw, args.out)
    manifest = {
        "description": "Official sensor responses prepared for native-grid training",
        "products": products,
        "notes": [
            "S2 PSFs are L1B responses, sampled 5 times per native pixel.",
            "psf_native is obtained by integrating samples into nearest native-pixel bins.",
            "B10 is retained in the S2 SRF but has no ground PSF and should not be reconstructed.",
            "Use the response file matching the spacecraft unit in the source product name.",
        ],
    }
    manifest_path = args.out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Prepared {len(products)} response archives in {args.out}")


if __name__ == "__main__":
    main()
