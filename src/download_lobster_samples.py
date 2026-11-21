"""Download LOBSTER's publicly posted five-level demonstration files.

The source samples are kept outside Git. The official sample page is linked in
the README and report; the archive hashes below make local reruns reproducible.
"""
from __future__ import annotations

import hashlib
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath


SAMPLE_SHA256 = {
    "AMZN": "23290152d0a588359ac59a969d385748112de5733065c81f5ba3ae9d770f74db",
    "AAPL": "46419dd0d6106be5fcf43600ccd08b2baa5d4cfb76284f1c58773537a3adac50",
    "GOOG": "a1146b74a3951cc34a8571c75940b7fd498445e48949123cae88a29a4e8b5441",
    "INTC": "2c76234e46906591598b893c5bab853f67e956c1520171b36f7df644dc35c96e",
    "MSFT": "94e65c07786b8d54712fe9a12aee29d51f418d402e96b6e21b4af1d0bd44131d",
}
BASE_URL = "https://php.lobsterdata.com/info/sample/LOBSTER_SampleFile_{ticker}_2012-06-21_5.zip"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_samples(root: Path) -> dict[str, str]:
    """Fetch and safely extract the five official sample archives."""
    raw_dir = root / "data" / "market_microstructure" / "raw"
    sample_root = raw_dir / "lobster_samples"
    raw_dir.mkdir(parents=True, exist_ok=True)
    hashes: dict[str, str] = {}
    for ticker, expected in SAMPLE_SHA256.items():
        archive = raw_dir / f"LOBSTER_SampleFile_{ticker}_2012-06-21_5.zip"
        if not archive.exists() or sha256(archive) != expected:
            request = urllib.request.Request(BASE_URL.format(ticker=ticker), headers={"User-Agent": "FUTURE-research/1.0"})
            with urllib.request.urlopen(request, timeout=60) as response, archive.open("wb") as output:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
        actual = sha256(archive)
        if actual != expected:
            archive.unlink(missing_ok=True)
            raise ValueError(f"SHA-256 mismatch for {archive.name}: {actual}")
        target = sample_root / ticker
        target.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as bundle:
            for member in bundle.infolist():
                relative = PurePosixPath(member.filename)
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError(f"Unsafe archive member: {member.filename}")
                destination = target.joinpath(*relative.parts)
                if member.is_dir():
                    destination.mkdir(parents=True, exist_ok=True)
                else:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with bundle.open(member) as source, destination.open("wb") as output:
                        while chunk := source.read(1024 * 1024):
                            output.write(chunk)
        hashes[ticker] = actual
    return hashes


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    for symbol, digest in download_samples(project_root).items():
        print(f"{symbol}: {digest}")
