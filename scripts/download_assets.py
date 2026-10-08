"""Download the versioned demo catalog and model weights (standard library only)."""

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile
import time
import urllib.error
import urllib.request
import zipfile


def download(url, output, expected_hash):
    for attempt in range(3):
        try:
            digest = hashlib.sha256()
            request = urllib.request.Request(url, headers={"User-Agent": "Guess-Hackathon"})
            with urllib.request.urlopen(request, timeout=120) as response, output.open("wb") as target:
                while chunk := response.read(1024 * 1024):
                    target.write(chunk)
                    digest.update(chunk)
            if digest.hexdigest() != expected_hash:
                raise ValueError("Asset checksum mismatch; refusing to extract the download.")
            return
        except (OSError, urllib.error.URLError):
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=root)
    parser.add_argument("--manifest", type=Path, default=root / "runtime-assets.json")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    destination = args.destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {manifest['version']} demo assets...", flush=True)
    with tempfile.TemporaryDirectory(prefix="guess-assets-") as temp:
        archive = Path(temp) / "assets.zip"
        download(manifest["url"], archive, manifest["sha256"])
        with zipfile.ZipFile(archive) as bundle:
            for entry in bundle.infolist():
                path = PurePosixPath(entry.filename)
                if path.is_absolute() or ".." in path.parts or "\\" in entry.filename or ":" in entry.filename:
                    raise ValueError("Unsafe archive path")
                if path.parts[0] not in {"data", "pretrained", "artifacts", "cache"}:
                    raise ValueError("Unexpected asset directory")
                if ((entry.external_attr >> 16) & 0o170000) == 0o120000:
                    raise ValueError("Archive symlinks are not supported")
                target = destination.joinpath(*path.parts)
                if not target.resolve().is_relative_to(destination):
                    raise ValueError("Asset path escapes destination")
            for entry in bundle.infolist():
                target = destination / entry.filename
                if entry.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                partial = target.with_name(target.name + ".download")
                try:
                    with bundle.open(entry) as source, partial.open("wb") as output:
                        shutil.copyfileobj(source, output)
                    os.replace(partial, target)
                finally:
                    partial.unlink(missing_ok=True)
    print(f"Assets ready in {destination}", flush=True)


if __name__ == "__main__":
    main()
