#!/usr/bin/env python3
"""
Download MNIST and Fashion-MNIST IDX files (gzip) for use with `python-mnist`.

Output layout:
  data/mnist/mnist/
  data/mnist/fashion-mnist/

The input nodes in this project are configured to look in these locations.
"""

from __future__ import annotations

from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError
import sys


DATASETS = {
    "mnist": {
        "base_urls": [
            "https://storage.googleapis.com/cvdf-datasets/mnist",
        ],
        "files": [
            "train-images-idx3-ubyte.gz",
            "train-labels-idx1-ubyte.gz",
            "t10k-images-idx3-ubyte.gz",
            "t10k-labels-idx1-ubyte.gz",
        ],
    },
    "fashion-mnist": {
        "base_urls": [
            "https://fashion-mnist.s3-website.eu-central-1.amazonaws.com",
            "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion",
            "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion",
        ],
        "files": [
            "train-images-idx3-ubyte.gz",
            "train-labels-idx1-ubyte.gz",
            "t10k-images-idx3-ubyte.gz",
            "t10k-labels-idx1-ubyte.gz",
        ],
    },
}


def download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = Request(url, headers={"User-Agent": "MiniCortex dataset downloader"})
    with urlopen(req) as resp, dest.open("wb") as f:
        while True:
            chunk = resp.read(1024 * 64)
            if not chunk:
                break
            f.write(chunk)


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    target_root = repo_root / "data" / "mnist"
    print(f"Target root: {target_root}")

    failures = []
    for dataset_name, spec in DATASETS.items():
        dataset_dir = target_root / dataset_name
        print(f"\n[{dataset_name}] -> {dataset_dir}")
        dataset_dir.mkdir(parents=True, exist_ok=True)

        for filename in spec["files"]:
            dest = dataset_dir / filename

            if dest.exists() and dest.stat().st_size > 0:
                print(f"  skip  {filename} (exists)")
                continue

            print(f"  fetch {filename}")
            last_error = None
            for base_url in spec["base_urls"]:
                url = f"{base_url}/{filename}"
                try:
                    download_file(url, dest)
                    last_error = None
                    break
                except (HTTPError, URLError, OSError) as exc:
                    last_error = exc
                    print(f"  fail  {filename} @ {base_url}: {exc}")
                    if dest.exists():
                        try:
                            dest.unlink()
                        except OSError:
                            pass
            if last_error is not None:
                failures.append((filename, str(last_error)))

    if failures:
        print("\nSome downloads failed:", file=sys.stderr)
        for filename, err in failures:
            print(f"  - {filename}: {err}", file=sys.stderr)
        return 1

    print("\nDone.")
    print("MNIST directory: data/mnist/mnist")
    print("Fashion-MNIST directory: data/mnist/fashion-mnist")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
