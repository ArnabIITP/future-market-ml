"""Download the licensed FI-2010 limit order book benchmark from Fairdata."""
from __future__ import annotations

import hashlib
from pathlib import Path
import zipfile

import requests


DATASET_ID = "73eb48d7-4dbc-4a10-a52a-da745b47a649"
METAX = "https://metax.fairdata.fi/v3"
ETSIN = "https://etsin.fairdata.fi/api/v3/download"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def download(output: Path | None = None, benchmark_output: Path | None = None) -> Path:
    output = output or Path("data/market_microstructure/raw/FI-2010-fairdata-package.zip")
    benchmark_output = benchmark_output or Path("data/market_microstructure/raw/BenchmarkDatasets.zip")
    output.parent.mkdir(parents=True, exist_ok=True)
    benchmark_output.parent.mkdir(parents=True, exist_ok=True)
    metadata = requests.get(f"{METAX}/datasets/{DATASET_ID}", timeout=30)
    metadata.raise_for_status()
    dataset = metadata.json()
    license_url = dataset["access_rights"]["license"][0].get("custom_url", "")
    if "creativecommons.org/licenses/by/4.0" not in license_url:
        raise RuntimeError(f"Unexpected FI-2010 license metadata: {license_url}")
    file_list = requests.get(f"{METAX}/datasets/{DATASET_ID}/files", timeout=30)
    file_list.raise_for_status()
    file_info = file_list.json()["results"][0]
    expected_file_checksum = file_info["checksum"].removeprefix("sha256:")
    if benchmark_output.exists() and benchmark_output.stat().st_size == file_info["size"]:
        if _sha256(benchmark_output) == expected_file_checksum:
            print(f"Verified benchmark already present: {benchmark_output}")
            return benchmark_output

    package_response = requests.get(f"{ETSIN}/requests", params={"cr_id": DATASET_ID}, timeout=30)
    package_response.raise_for_status()
    package = package_response.json()
    if package["status"] != "SUCCESS":
        raise RuntimeError(f"Fairdata package is not ready: {package['status']}")
    auth_response = requests.post(
        f"{ETSIN}/authorize",
        json={"cr_id": DATASET_ID, "package": package["package"]},
        timeout=30,
    )
    auth_response.raise_for_status()
    signed_url = auth_response.json()["url"]

    partial = output.with_suffix(output.suffix + ".part")
    digest = hashlib.sha256()
    received = 0
    with requests.get(signed_url, stream=True, timeout=(60, 180)) as response:
        response.raise_for_status()
        total = int(response.headers.get("Content-Length", package["size"]))
        with partial.open("wb") as stream:
            for chunk in response.iter_content(chunk_size=64 * 1024):
                if not chunk:
                    continue
                stream.write(chunk)
                digest.update(chunk)
                received += len(chunk)
                if received % (128 * 1024 * 1024) < len(chunk):
                    stream.flush()
                    print(f"Downloaded {received / 1e9:.2f} / {total / 1e9:.2f} GB")
    if received != package["size"]:
        raise RuntimeError(f"Downloaded {received} bytes; Fairdata expected {package['size']}.")
    if digest.hexdigest() != package["checksum"].removeprefix("sha256:"):
        raise RuntimeError("The downloaded Fairdata package failed its published SHA-256 check.")
    partial.replace(output)
    print(f"Verified package SHA-256 {digest.hexdigest()}")

    # Fairdata wraps the published benchmark ZIP in a second package ZIP.
    expected_inner_path = "published/BenchmarkDatasets/BenchmarkDatasets.zip"
    with zipfile.ZipFile(output) as package_zip:
        if expected_inner_path not in package_zip.namelist():
            raise RuntimeError(f"Fairdata package did not contain {expected_inner_path}.")
        partial_inner = benchmark_output.with_suffix(benchmark_output.suffix + ".part")
        inner_digest = hashlib.sha256()
        written = 0
        with package_zip.open(expected_inner_path) as source, partial_inner.open("wb") as destination:
            while chunk := source.read(1024 * 1024):
                destination.write(chunk)
                inner_digest.update(chunk)
                written += len(chunk)
    if written != file_info["size"]:
        raise RuntimeError(f"Extracted {written} bytes; Fairdata metadata expected {file_info['size']}.")
    if inner_digest.hexdigest() != expected_file_checksum:
        raise RuntimeError("The extracted benchmark failed its published file SHA-256 check.")
    partial_inner.replace(benchmark_output)
    print(f"Verified dataset file SHA-256 {inner_digest.hexdigest()}")
    print(f"Official dataset file: {file_info['filename']} ({file_info['size']} bytes)")
    return benchmark_output


if __name__ == "__main__":
    download()
