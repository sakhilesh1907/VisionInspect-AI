"""
scripts/download_mvtec.py

Downloads the official MVTec Anomaly Detection (MVTec AD) dataset archive (4.9 GB).
Leaves the file in its original .tar.xz format without unzipping/extracting.

Usage:
    python scripts/download_mvtec.py
    python scripts/download_mvtec.py --extract   # only if you want to extract later
"""

import sys
import tarfile
import urllib.request
import click
from pathlib import Path
from tqdm import tqdm

FULL_DATASET_URL = "https://huggingface.co/datasets/micguida1/mvtech_anomaly_detection/resolve/main/mvtec_anomaly_detection.tar.xz"


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_dataset_tar(target_dir: Path, extract: bool = False):
    target_dir.mkdir(parents=True, exist_ok=True)
    archive_path = target_dir / "mvtec_anomaly_detection.tar.xz"

    print("=" * 70)
    print(" 📦 MVTec AD Dataset Downloader (4.9 GB)")
    print(f" Target File: {archive_path.resolve()}")
    print("=" * 70)

    # 1. Download archive
    if archive_path.exists() and archive_path.stat().st_size > 4.5 * 1024 * 1024 * 1024:
        print(f"✅ Archive already exists at {archive_path} ({archive_path.stat().st_size / (1024**3):.2f} GB).")
    else:
        print(f"\nDownloading 4.9 GB .tar.xz file from Hugging Face...")
        print(f"URL: {FULL_DATASET_URL}\n")
        opener = urllib.request.build_opener()
        opener.addheaders = [("User-Agent", "Mozilla/5.0")]
        urllib.request.install_opener(opener)

        with DownloadProgressBar(unit="B", unit_scale=True, miniters=1, desc="Downloading mvtec_anomaly_detection.tar.xz") as t:
            urllib.request.urlretrieve(FULL_DATASET_URL, filename=archive_path, reporthook=t.update_to)

        print("\n✅ Download complete!")
        print(f"File saved in .tar.xz format at:\n👉 {archive_path.resolve()}")
        print(f"Size: {archive_path.stat().st_size / (1024**3):.2f} GB")

    # 2. Extract only if requested
    if extract:
        print(f"\nExtracting archive to {target_dir}...")
        with tarfile.open(archive_path, "r:xz") as tar:
            tar.extractall(path=target_dir)
        print("✅ Extraction complete!")
    else:
        print("\n🔒 Kept in .tar.xz format (not unzipped) as requested.")


@click.command()
@click.option("--target-dir", default="data/raw/mvtec", help="Directory where .tar.xz will be saved")
@click.option("--extract", is_flag=True, help="Optional: extract files after download (default is NO)")
def main(target_dir, extract):
    out_path = Path(target_dir)
    download_dataset_tar(out_path, extract=extract)


if __name__ == "__main__":
    main()
