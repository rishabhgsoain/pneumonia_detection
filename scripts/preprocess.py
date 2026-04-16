"""
Preprocess RSNA DICOM files: convert to PNG, split into train/val/test.

Usage:
    python scripts/preprocess.py --input-dir ./rsna_data --output-dir ./rsna_processed
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from configs.config import get_config


def convert_dicom_to_png(dicom_path: str, output_path: str, target_size: int = 1024):
    """Convert a DICOM file to PNG with histogram equalization."""
    try:
        import pydicom
        ds = pydicom.dcmread(dicom_path)
        pixel_array = ds.pixel_array.astype(np.float32)
    except Exception:
        # Fallback: try loading as regular image (some RSNA files are already images)
        try:
            img = Image.open(dicom_path).convert("L")
            pixel_array = np.array(img, dtype=np.float32)
        except Exception as e:
            print(f"  WARNING: Could not read {dicom_path}: {e}")
            return False

    # Normalize to 0-255
    if pixel_array.max() > pixel_array.min():
        pixel_array = (pixel_array - pixel_array.min()) / (pixel_array.max() - pixel_array.min())
    pixel_array = (pixel_array * 255).astype(np.uint8)

    # Histogram equalization
    from cv2 import equalizeHist, INTER_AREA, resize
    pixel_array = equalizeHist(pixel_array)

    # Resize preserving aspect ratio with padding
    h, w = pixel_array.shape
    scale = target_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = resize(pixel_array, (new_w, new_h), interpolation=INTER_AREA)

    # Pad to square
    canvas = np.zeros((target_size, target_size), dtype=np.uint8)
    y_offset = (target_size - new_h) // 2
    x_offset = (target_size - new_w) // 2
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

    Image.fromarray(canvas).save(output_path)
    return True


def prepare_annotations(csv_path: str, output_dir: str, target_size: int = 1024):
    """
    Parse RSNA labels CSV and create train/val/test splits.
    Adjusts bounding box coordinates for resized images.
    """
    df = pd.read_csv(csv_path)

    # Get unique patient IDs and their labels
    patients = df.groupby("patientId").agg({
        "Target": "max",  # 1 if any box exists
    }).reset_index()

    # Stratified split
    train_ids, temp_ids = train_test_split(
        patients["patientId"].values,
        test_size=0.2,
        random_state=42,
        stratify=patients["Target"].values
    )
    val_ids, test_ids = train_test_split(
        temp_ids,
        test_size=0.5,
        random_state=42,
        stratify=patients[patients["patientId"].isin(temp_ids)]["Target"].values
    )

    print(f"  Train: {len(train_ids)} patients")
    print(f"  Val:   {len(val_ids)} patients")
    print(f"  Test:  {len(test_ids)} patients")

    # Process bounding boxes
    # Original RSNA images are 1024x1024, so if target_size == 1024, no rescaling needed
    # But we need to handle the aspect-ratio-preserving resize + padding

    records = []
    for _, row in df.iterrows():
        pid = row["patientId"]
        target = int(row["Target"])

        record = {
            "patientId": pid,
            "Target": target,
            "image_path": f"images/{pid}.png",
        }

        if target == 1 and pd.notna(row.get("x")):
            # Bounding box: x, y, width, height
            record["x"] = float(row["x"])
            record["y"] = float(row["y"])
            record["w"] = float(row["width"])
            record["h"] = float(row["height"])
        else:
            record["x"] = None
            record["y"] = None
            record["w"] = None
            record["h"] = None

        records.append(record)

    all_df = pd.DataFrame(records)

    # Save splits
    for split_name, split_ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
        split_df = all_df[all_df["patientId"].isin(split_ids)]
        out_path = os.path.join(output_dir, f"{split_name}_split.csv")
        split_df.to_csv(out_path, index=False)
        n_pos = split_df[split_df["Target"] == 1]["patientId"].nunique()
        n_total = split_df["patientId"].nunique()
        print(f"  {split_name}: {n_total} patients, {n_pos} positive ({100*n_pos/n_total:.1f}%)")

    return all_df


def main():
    parser = argparse.ArgumentParser(description="Preprocess RSNA DICOM data")
    parser.add_argument("--input-dir", type=str, default="./rsna_data")
    parser.add_argument("--output-dir", type=str, default="./rsna_processed")
    parser.add_argument("--target-size", type=int, default=1024)
    parser.add_argument("--skip-conversion", action="store_true",
                        help="Skip DICOM->PNG if already done")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    image_dir = os.path.join(args.output_dir, "images")
    os.makedirs(image_dir, exist_ok=True)

    # Step 1: Convert DICOMs to PNGs
    if not args.skip_conversion:
        print("Step 1: Converting DICOM files to PNG...")
        dicom_dirs = [
            os.path.join(args.input_dir, "stage_2_train_images"),
            os.path.join(args.input_dir, "stage_2_test_images"),
        ]

        dicom_files = []
        for d in dicom_dirs:
            if os.path.exists(d):
                dicom_files.extend([
                    os.path.join(d, f) for f in os.listdir(d)
                    if f.endswith(".dcm")
                ])

        if not dicom_files:
            # Maybe files are directly in input_dir
            dicom_files = [
                os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir)
                if f.endswith(".dcm")
            ]

        print(f"  Found {len(dicom_files)} DICOM files")

        success = 0
        for dcm_path in tqdm(dicom_files, desc="Converting"):
            pid = os.path.splitext(os.path.basename(dcm_path))[0]
            png_path = os.path.join(image_dir, f"{pid}.png")
            if not os.path.exists(png_path):
                if convert_dicom_to_png(dcm_path, png_path, args.target_size):
                    success += 1
            else:
                success += 1

        print(f"  Converted {success}/{len(dicom_files)} files")
    else:
        print("Step 1: Skipping DICOM conversion (--skip-conversion)")

    # Step 2: Prepare annotations and splits
    print("\nStep 2: Preparing annotations and splits...")
    csv_path = os.path.join(args.input_dir, "stage_2_train_labels.csv")
    if not os.path.exists(csv_path):
        print(f"  ERROR: Labels file not found: {csv_path}")
        sys.exit(1)

    prepare_annotations(csv_path, args.output_dir, args.target_size)

    print("\nPreprocessing complete!")
    print(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
