"""
Download RSNA Pneumonia Detection Challenge data from Kaggle.

Usage:
    python data/download.py --data-dir ./rsna_data
"""
import argparse
import os
import subprocess
import sys
import zipfile


def download_rsna(data_dir: str):
    """Download RSNA Pneumonia Detection Challenge dataset."""
    os.makedirs(data_dir, exist_ok=True)

    print("=" * 60)
    print("Downloading RSNA Pneumonia Detection Challenge data...")
    print("=" * 60)
    print()
    print("Prerequisites:")
    print("  1. Install kaggle: pip install kaggle")
    print("  2. Set API token (env variable or kaggle.json)")
    print("  3. Accept competition rules on Kaggle")
    print()

    try:
        subprocess.run(
            [
                "kaggle", "competitions", "download",
                "-c", "rsna-pneumonia-detection-challenge",
                "-p", data_dir
            ],
            check=True
        )
    except FileNotFoundError:
        print("ERROR: kaggle CLI not found. Install with: pip install kaggle")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Download failed: {e}")
        print("Make sure you've accepted the competition rules on Kaggle.")
        sys.exit(1)

    # ✅ FIXED UNZIP (WORKS ON WINDOWS)
    zip_files = [f for f in os.listdir(data_dir) if f.endswith(".zip")]

    if not zip_files:
        print("No zip files found. Skipping extraction.")
        return

    for zf in zip_files:
        zpath = os.path.join(data_dir, zf)
        print(f"Extracting {zf}...")

        try:
            with zipfile.ZipFile(zpath, "r") as zip_ref:
                zip_ref.extractall(data_dir)
            os.remove(zpath)
        except Exception as e:
            print(f"ERROR extracting {zf}: {e}")
            sys.exit(1)

    print()
    print(f"Data downloaded to: {data_dir}")
    print("Contents:")
    for item in sorted(os.listdir(data_dir)):
        path = os.path.join(data_dir, item)
        if os.path.isdir(path):
            count = len(os.listdir(path))
            print(f"  {item}/ ({count} files)")
        else:
            size_mb = os.path.getsize(path) / 1e6
            print(f"  {item} ({size_mb:.1f} MB)")


def download_nih_chestxray(data_dir: str):
    """Download NIH ChestX-ray14 dataset for domain-adaptive pretraining."""
    os.makedirs(data_dir, exist_ok=True)

    print("=" * 60)
    print("Downloading NIH ChestX-ray14 dataset...")
    print("=" * 60)
    print()
    print("NOTE: This dataset is ~42GB. Ensure sufficient disk space.")
    print()

    try:
        subprocess.run(
            [
                "kaggle", "datasets", "download",
                "-d", "nih-chest-xrays/data",
                "-p", data_dir
            ],
            check=True
        )

        # Extract NIH dataset as well
        zip_files = [f for f in os.listdir(data_dir) if f.endswith(".zip")]

        for zf in zip_files:
            zpath = os.path.join(data_dir, zf)
            print(f"Extracting {zf}...")

            with zipfile.ZipFile(zpath, "r") as zip_ref:
                zip_ref.extractall(data_dir)
            os.remove(zpath)

    except Exception as e:
        print(f"Download failed: {e}")
        print("You can skip this step if not running domain-adaptive pretraining.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download datasets")
    parser.add_argument("--data-dir", type=str, default="./rsna_data")
    parser.add_argument("--download-nih", action="store_true",
                        help="Also download NIH ChestX-ray14 for pretraining")
    parser.add_argument("--nih-dir", type=str, default="./nih_chestxray")
    args = parser.parse_args()

    download_rsna(args.data_dir)

    if args.download_nih:
        download_nih_chestxray(args.nih_dir)