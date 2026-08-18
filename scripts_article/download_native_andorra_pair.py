"""Download the native Sentinel-2A/OLCI-A pair selected for Andorra.

The two acquisitions are separated by about 49 minutes on 2018-01-18. The
script never stores credentials: define CDSE_USERNAME and CDSE_PASSWORD in
the process environment before running it.
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import shutil
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "data" / "native" / "andorra" / "20180118"
TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/"
    "protocol/openid-connect/token"
)
DOWNLOAD_URL = "https://download.dataspace.copernicus.eu/odata/v1/Products({})/$value"

PRODUCTS = (
    {
        "sensor": "S2A_MSI_L1C",
        "id": "cf14019f-ecad-4183-bda7-09fb72981f2d",
        "name": "S2A_MSIL1C_20180118T104351_N0500_R008_T31TCH_20230802T041909.SAFE",
        "acquisition_utc": "2018-01-18T10:43:51Z",
        "catalogue_bytes": 382447045,
    },
    {
        "sensor": "S2A_MSI_L2A",
        "id": "be4c7f80-eb40-4565-be01-dc6623ab8639",
        "name": "S2A_MSIL2A_20180118T104351_N0500_R008_T31TCH_20230802T205745.SAFE",
        "acquisition_utc": "2018-01-18T10:43:51Z",
        "catalogue_bytes": 566277129,
    },
    {
        "sensor": "S3A_OLCI_L1_EFR",
        "id": "ff2bcad9-f3e3-3211-8b4c-da479853a14e",
        "name": "S3A_OL_1_EFR____20180118T095416_20180118T095716_20240530T092508_0180_027_022_2160_MAR_R_NT_004.SEN3",
        "acquisition_utc": "2018-01-18T09:54:16Z",
        "catalogue_bytes": 863825133,
    },
    {
        "sensor": "S3A_OLCI_L2_LFR",
        "id": "bf66b658-fc0a-11ef-8be2-fa163e6e2f6d",
        "name": "S3A_OL_2_LFR____20180118T095416_20180118T095716_20250308T104643_0180_027_022_2160_ESA_R_NT_003.SEN3",
        "acquisition_utc": "2018-01-18T09:54:16Z",
        "catalogue_bytes": 127391246,
    },
)


def access_token() -> str:
    username = os.environ.get("CDSE_USERNAME") or input("CDSE username: ").strip()
    password = os.environ.get("CDSE_PASSWORD") or getpass.getpass("CDSE password: ")
    if not username or not password:
        raise RuntimeError("Copernicus Data Space credentials are required")
    payload = urllib.parse.urlencode({
        "client_id": "cdse-public",
        "grant_type": "password",
        "username": username,
        "password": password,
    }).encode("utf-8")
    request = urllib.request.Request(TOKEN_URL, data=payload, method="POST")
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)["access_token"]


def download(product: dict, destination: Path, token: str) -> Path:
    archive = destination / f"{product['name']}.zip"
    partial = archive.with_suffix(archive.suffix + ".part")
    if archive.exists() and archive.stat().st_size > 0:
        print(f"Already present: {archive.name}")
        return archive
    request = urllib.request.Request(
        DOWNLOAD_URL.format(product["id"]),
        headers={"Authorization": f"Bearer {token}"},
    )
    with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as stream:
        shutil.copyfileobj(response, stream, length=1024 * 1024)
    partial.replace(archive)
    print(f"Downloaded {archive.name} ({archive.stat().st_size / 2**20:.1f} MiB)")
    return archive


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--list-only", action="store_true")
    args = parser.parse_args()

    print(json.dumps(PRODUCTS, indent=2))
    if args.list_only:
        return
    args.output.mkdir(parents=True, exist_ok=True)
    token = access_token()
    archives = [download(product, args.output, token) for product in PRODUCTS]
    if args.extract:
        for archive in archives:
            with zipfile.ZipFile(archive) as zipped:
                zipped.extractall(args.output)
            print(f"Extracted {archive.name}")


if __name__ == "__main__":
    main()
