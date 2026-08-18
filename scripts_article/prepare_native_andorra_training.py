"""Build a compact, native-grid S2A/OLCI-A training scene for Andorra.

Sentinel-2 L1C digital numbers and OLCI L1B radiances are converted to TOA
reflectance. The native 10/20/60/300 m grids are cropped but never resized.
"""

from __future__ import annotations

import argparse
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from netCDF4 import Dataset
from PIL import Image
from pyproj import Transformer


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NATIVE = ROOT / "data" / "native" / "andorra" / "20180118"
DEFAULT_RESPONSE = ROOT / "data" / "instrument_response" / "processed"
DEFAULT_OUTPUT = ROOT / "data" / "training" / "andorra_20180118_native_toa.npz"

S2_GROUPS = {
    "s2_10": (10, ("B02", "B03", "B04", "B08")),
    "s2_20": (20, ("B05", "B06", "B07", "B8A", "B11", "B12")),
    "s2_60": (60, ("B01", "B09")),
}
RECONSTRUCTED_BANDS = {
    "s2_10": ("B02", "B03", "B04", "B08"),
    "s2_20": ("B05", "B06", "B07", "B8A"),
    "s2_60": ("B01", "B09"),
}


def find_one(root: Path, pattern: str) -> Path:
    matches = list(root.glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one match for {pattern}, found {len(matches)}")
    return matches[0]


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def s2_radiometry(product: Path) -> tuple[float, dict[int, float]]:
    metadata = product / "MTD_MSIL1C.xml"
    root = ET.parse(metadata).getroot()
    quantification = None
    offsets: dict[int, float] = {}
    for element in root.iter():
        name = local_name(element.tag)
        if name == "QUANTIFICATION_VALUE":
            quantification = float(element.text)
        elif name == "RADIO_ADD_OFFSET":
            offsets[int(element.attrib["band_id"])] = float(element.text)
    if quantification is None:
        raise ValueError("QUANTIFICATION_VALUE missing from S2 metadata")
    return quantification, offsets


def read_s2_crop(path: Path, box: tuple[int, int, int, int], quantification: float,
                 offset: float) -> tuple[np.ndarray, np.ndarray]:
    Image.MAX_IMAGE_PIXELS = None
    with Image.open(path) as image:
        digital = np.asarray(image.crop(box), dtype=np.float32)
    valid = digital > 0
    reflectance = (digital + offset) / quantification
    reflectance = np.clip(reflectance, 0.0, 1.5)
    return reflectance, valid


def crop_s2(product: Path, start_10m: int, size_10m: int) -> tuple[dict, dict, dict]:
    quantification, offsets = s2_radiometry(product)
    band_ids = {"B01": 0, "B02": 1, "B03": 2, "B04": 3, "B05": 4, "B06": 5,
                "B07": 6, "B08": 7, "B8A": 8, "B09": 9, "B10": 10,
                "B11": 11, "B12": 12}
    arrays, masks, coordinates = {}, {}, {}
    ulx, uly = 300000.0, 4800000.0  # Tile 31TCH metadata, EPSG:32631.
    for sensor, (resolution, bands) in S2_GROUPS.items():
        ratio = resolution // 10
        start = start_10m // ratio
        size = size_10m // ratio
        box = (start, start, start + size, start + size)
        channels, validity = [], []
        for band in bands:
            path = find_one(product, f"GRANULE/*/IMG_DATA/*_{band}.jp2")
            values, valid = read_s2_crop(path, box, quantification,
                                         offsets.get(band_ids[band], 0.0))
            channels.append(values)
            validity.append(valid)
        arrays[sensor] = np.stack(channels).astype(np.float32)
        masks[sensor] = np.logical_and.reduce(validity)[None]
        x = ulx + (np.arange(start, start + size) + 0.5) * resolution
        y = uly - (np.arange(start, start + size) + 0.5) * resolution
        coordinates[sensor] = (y, x)
    return arrays, masks, coordinates


def olci_crop_indices(product: Path, centre_lon: float, centre_lat: float,
                      size: int) -> tuple[int, int, int, int]:
    with Dataset(product / "geo_coordinates.nc") as dataset:
        latitude = np.asarray(dataset["latitude"][:], dtype=np.float64)
        longitude = np.asarray(dataset["longitude"][:], dtype=np.float64)
    distance = (latitude - centre_lat) ** 2 + ((longitude - centre_lon)
                                              * math.cos(math.radians(centre_lat))) ** 2
    row, column = np.unravel_index(np.nanargmin(distance), distance.shape)
    row0 = int(np.clip(row - size // 2, 0, latitude.shape[0] - size))
    col0 = int(np.clip(column - size // 2, 0, latitude.shape[1] - size))
    return row0, row0 + size, col0, col0 + size


def read_olci_toa(product: Path, box: tuple[int, int, int, int]
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    row0, row1, col0, col1 = box
    with Dataset(product / "geo_coordinates.nc") as dataset:
        latitude = np.asarray(dataset["latitude"][row0:row1, col0:col1], dtype=np.float64)
        longitude = np.asarray(dataset["longitude"][row0:row1, col0:col1], dtype=np.float64)
    with Dataset(product / "instrument_data.nc") as dataset:
        detector = np.asarray(dataset["detector_index"][row0:row1, col0:col1], dtype=np.int64)
        solar_flux = np.asarray(dataset["solar_flux"][:], dtype=np.float64)
        lambda0 = np.asarray(dataset["lambda0"][:], dtype=np.float64)
    with Dataset(product / "tie_geometries.nc") as dataset:
        tie_sza = np.asarray(dataset["SZA"][row0:row1], dtype=np.float64)
    tie_columns = np.linspace(0, 4864, tie_sza.shape[1])
    sample_columns = np.arange(col0, col1)
    sza = np.stack([np.interp(sample_columns, tie_columns, row) for row in tie_sza])
    cosine = np.cos(np.deg2rad(sza))

    valid_detector = (detector >= 0) & (detector < solar_flux.shape[1]) & (cosine > 0.05)
    safe_detector = np.clip(detector, 0, solar_flux.shape[1] - 1)
    reflectances, valid_bands, centres = [], [], []
    for band in range(21):
        with Dataset(product / f"Oa{band + 1:02d}_radiance.nc") as dataset:
            radiance = dataset[f"Oa{band + 1:02d}_radiance"][row0:row1, col0:col1]
            band_mask = ~np.ma.getmaskarray(radiance)
            radiance = np.asarray(np.ma.filled(radiance, 0.0), dtype=np.float64)
        irradiance = solar_flux[band, safe_detector]
        toa = math.pi * radiance / (irradiance * cosine + 1e-8)
        reflectances.append(np.clip(toa, 0.0, 1.5))
        valid_bands.append(band_mask & valid_detector & np.isfinite(toa))
        centres.append(float(np.median(lambda0[band, safe_detector[valid_detector]])))
    mask = np.logical_and.reduce(valid_bands)[None]
    return np.stack(reflectances).astype(np.float32), mask, latitude, longitude, np.asarray(centres)


def hat_basis(sample_wavelengths: np.ndarray, centres: np.ndarray) -> np.ndarray:
    basis = np.zeros((len(sample_wavelengths), len(centres)), dtype=np.float64)
    for index in range(len(centres)):
        values = np.zeros_like(sample_wavelengths, dtype=np.float64)
        if index > 0:
            left = (sample_wavelengths >= centres[index - 1]) & (sample_wavelengths <= centres[index])
            values[left] = ((sample_wavelengths[left] - centres[index - 1])
                            / (centres[index] - centres[index - 1]))
        if index < len(centres) - 1:
            right = (sample_wavelengths >= centres[index]) & (sample_wavelengths <= centres[index + 1])
            values[right] = ((centres[index + 1] - sample_wavelengths[right])
                             / (centres[index + 1] - centres[index]))
        if index == 0:
            values[sample_wavelengths <= centres[0]] = 1.0
        if index == len(centres) - 1:
            values[sample_wavelengths >= centres[-1]] = 1.0
        basis[:, index] = values
    return basis


def spectral_response_matrices(response_dir: Path, centres: np.ndarray) -> dict[str, np.ndarray]:
    archive = np.load(response_dir / "s2a_srf_1nm.npz")
    wavelength = archive["wavelength_nm"].astype(np.float64)
    srf = archive["response"].astype(np.float64)
    bands = [str(value) for value in archive["bands"]]
    projection = srf @ hat_basis(wavelength, centres)
    projection /= projection.sum(axis=1, keepdims=True) + 1e-12
    lookup = {band: projection[index] for index, band in enumerate(bands)}
    return {
        sensor: np.stack([lookup[band] for band in RECONSTRUCTED_BANDS[sensor]]).astype(np.float32)
        for sensor in RECONSTRUCTED_BANDS
    }


def spatial_psf_kernels(response_dir: Path) -> dict[str, np.ndarray]:
    """Select the official S2A native-grid PSFs for reconstructed bands."""
    archive = np.load(response_dir / "s2a_psf.npz")
    psf = archive["psf_native"].astype(np.float32)
    lookup = {str(band): psf[index] for index, band in enumerate(archive["bands"])}
    return {
        sensor: np.stack([lookup[band] for band in RECONSTRUCTED_BANDS[sensor]])
        for sensor in RECONSTRUCTED_BANDS
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native", type=Path, default=DEFAULT_NATIVE)
    parser.add_argument("--responses", type=Path, default=DEFAULT_RESPONSE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--start-10m", type=int, default=8760)
    parser.add_argument("--size-10m", type=int, default=1200)
    args = parser.parse_args()
    if args.start_10m % 6 or args.size_10m % 300:
        raise ValueError("start-10m must be divisible by 6 and size-10m by 300")

    s2 = find_one(args.native, "S2A_MSIL1C_*.SAFE")
    olci = find_one(args.native, "S3A_OL_1_EFR_*.SEN3")
    arrays, masks, coordinates = crop_s2(s2, args.start_10m, args.size_10m)

    y10, x10 = coordinates["s2_10"]
    transformer = Transformer.from_crs(32631, 4326, always_xy=True)
    centre_lon, centre_lat = transformer.transform(float(x10.mean()), float(y10.mean()))
    # The requested size is expressed in 10 m pixels: 30 such pixels span one
    # native OLCI 300 m sample.
    olci_size = args.size_10m // 30
    olci_box = olci_crop_indices(olci, centre_lon, centre_lat, olci_size)
    olci_cube, olci_mask, olci_lat, olci_lon, centres = read_olci_toa(olci, olci_box)
    arrays["olci"] = olci_cube
    masks["olci"] = olci_mask

    to_utm = Transformer.from_crs(4326, 32631, always_xy=True)
    olci_x, olci_y = to_utm.transform(olci_lon, olci_lat)
    coordinates["olci"] = (np.mean(olci_y, axis=1), np.mean(olci_x, axis=0))

    xmin = float(x10.min() - 5)
    xmax = float(x10.max() + 5)
    ymin = float(y10.min() - 5)
    ymax = float(y10.max() + 5)
    payload: dict[str, np.ndarray] = {}
    for sensor, array in arrays.items():
        payload[sensor] = array
        payload[f"mask_{sensor}"] = masks[sensor].astype(np.uint8)
        y, x = coordinates[sensor]
        payload[f"coord_y_{sensor}"] = ((ymax - y) / (ymax - ymin)).astype(np.float32)
        payload[f"coord_x_{sensor}"] = ((x - xmin) / (xmax - xmin)).astype(np.float32)
    responses = spectral_response_matrices(args.responses, centres)
    responses["olci"] = np.eye(21, dtype=np.float32)
    for sensor, matrix in responses.items():
        payload[f"response_{sensor}"] = matrix
    psfs = spatial_psf_kernels(args.responses)
    psfs["olci"] = np.ones((21, 1, 1), dtype=np.float32)
    for sensor, kernels in psfs.items():
        payload[f"psf_{sensor}"] = kernels
    payload["olci_wavelength_nm"] = centres.astype(np.float32)
    payload["recon_indices_s2_10"] = np.arange(4, dtype=np.int64)
    payload["recon_indices_s2_20"] = np.arange(4, dtype=np.int64)
    payload["recon_indices_s2_60"] = np.arange(2, dtype=np.int64)
    payload["recon_indices_olci"] = np.arange(21, dtype=np.int64)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **payload)
    summary = {
        "output": str(args.output),
        "s2_product": s2.name,
        "olci_product": olci.name,
        "olci_crop": list(olci_box),
        "shapes": {key: list(value.shape) for key, value in arrays.items()},
        "valid_fraction": {key: float(masks[key].mean()) for key in masks},
        "radiometry": "TOA reflectance, clipped to [0, 1.5]",
        "resampling": "none; crop only on each native grid",
        "psf": "official S2A native PSF; OLCI identity on its native grid",
    }
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
