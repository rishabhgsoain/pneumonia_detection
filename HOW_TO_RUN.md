# 🫁 Pneumonia Detection — How to Run

**AMS 563: Medical Image Analysis | Final Project**  
Multi-Scale Transformer-Based Pneumonia Detection and Localization in Chest X-Rays

---

## 📋 Table of Contents
1. [Project Overview](#-project-overview)
2. [Requirements](#-requirements)
3. [Dataset Setup](#-dataset-setup)
4. [Running on Google Colab (Recommended)](#-running-on-google-colab-recommended)
5. [Running Locally](#-running-locally)
6. [Experiment Results](#-experiment-results)
7. [Project Structure](#-project-structure)
8. [Troubleshooting](#-troubleshooting)

---

## 🔍 Project Overview

This project compares CNN-based (Faster R-CNN) and transformer-based (DETR, Multi-Scale DETR) architectures for pneumonia localization on the RSNA Pneumonia Detection Challenge dataset.

| Experiment | Model | mAP@0.5 |
|---|---|---|
| E1 | Faster R-CNN + ResNet-50-FPN | **0.3462** 🏆 |
| E2 | Vanilla DETR (from scratch) | 0.0000 |
| E4 | Multi-Scale DETR (FPN backbone) | 0.0000 |

> **Key Finding:** Faster R-CNN significantly outperforms DETR-based models under limited compute (30 epochs). DETR requires 300+ epochs to converge, making it impractical for small medical datasets without large-scale pretraining.

---

## 📦 Requirements

### Python Dependencies
```bash
pip install torch torchvision
pip install pydicom albumentations opencv-python
pip install scikit-learn tqdm timm pycocotools scipy
pip install kaggle transformers
```

### Hardware
| Environment | Minimum | Recommended |
|---|---|---|
| **GPU VRAM** | 8 GB | 15 GB (T4) |
| **RAM** | 12 GB | 16 GB |
| **Disk** | 20 GB | 50 GB |

---

## 📥 Dataset Setup

The project uses the **RSNA Pneumonia Detection Challenge** dataset from Kaggle.

### Option A — Automatic (Kaggle API)

1. Get your Kaggle API key from: https://www.kaggle.com/settings/account  
   → Scroll to **Legacy API Credentials** → **Create Legacy API Key**

2. Place the downloaded `kaggle.json` at `~/.kaggle/kaggle.json`:
```bash
mkdir -p ~/.kaggle
cp ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

3. Download the dataset:
```bash
kaggle competitions download -c rsna-pneumonia-detection-challenge -p data/
cd data && unzip rsna-pneumonia-detection-challenge.zip
```

### Option B — Manual Download

1. Go to: https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/data
2. Download all files and extract into `data/rsna-pneumonia-detection-challenge/`

### Expected Data Structure
```
data/
└── rsna-pneumonia-detection-challenge/
    ├── stage_2_train_images/        # 26,684 DICOM files
    ├── stage_2_test_images/         # Test DICOM files
    ├── stage_2_train_labels.csv     # Bounding box annotations
    └── stage_2_detailed_class_info.csv
```

---

## ☁️ Running on Google Colab (Recommended)

This is the **easiest and fastest** way to run the project. Colab provides a free T4 GPU.

### Step 1 — Open the Notebook
1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Click **File → Upload notebook**
3. Upload `Pneumonia_Detection_Colab.ipynb`

### Step 2 — Enable GPU
```
Runtime → Change runtime type → T4 GPU → Save
```

### Step 3 — Run Cells in Order

| Cell | Description | Time |
|---|---|---|
| ✅ Step 1 | Check GPU | ~5 sec |
| ✅ Step 2 | Install dependencies | ~1 min |
| ✅ Step 3 | Mount Google Drive (saves checkpoints) | ~10 sec |
| ✅ Step 4 | **Paste your Kaggle API key** | ~10 sec |
| ✅ Step 5 | Download RSNA dataset (~3.96 GB) | ~10 min |
| ✅ Step 6 | Write project code files | ~1 min |
| ✅ Step 7 | Preprocess DICOMs → PNGs (512×512) | ~15 min |
| ✅ Step 8 | Build dataloaders & verify | ~1 min |
| ✅ **PATCH** | Apply bbox fix + augmentation fix | ~5 sec |
| ✅ Step 9 | Run experiments E1–E4 | ~8–10 hrs |
| ✅ Step 10 | Results comparison table & plots | ~1 min |

### Step 4 — Fill in Kaggle Credentials (Step 4 cell)
```python
kaggle_creds = {
    'username': 'YOUR_KAGGLE_USERNAME',
    'key': 'YOUR_LEGACY_API_KEY'   # from kaggle.json
}
```
To get your key: `cat ~/.kaggle/kaggle.json`

### ⚠️ Important: Apply the Patch Cell Before Experiments

Before running any experiment, add and run this patch (fixes bbox coordinate scaling):

```python
# The RSNA CSV stores bounding boxes in 1024×1024 pixel space
# but images are resized to 512×512 — this patch fixes the mismatch.
# Also replaces deprecated ShiftScaleRotate with Affine.
# → Paste the full patch from the project documentation
```

> See `PATCH_CELL.py` in the project root for the complete patch code.

### After Runtime Disconnection

Colab resets `/content/` on disconnect. To recover:

1. Re-run **Steps 1–3** (fast, ~2 min)
2. Re-run **Step 4** (Kaggle key)
3. Re-run **Step 5** — OR restore from Drive if you saved PNGs:
```python
# Restore preprocessed PNGs from Drive (skips 15-min DICOM conversion)
import shutil, os
DRIVE_IMGS = '/content/drive/MyDrive/AMS563_Pneumonia/rsna_processed'
if os.path.exists(DRIVE_IMGS):
    shutil.copytree(DRIVE_IMGS, '/content/rsna_processed')
    print("✅ Restored from Drive!")
```
4. Re-run **Step 6** (write code files)
5. Re-run **PATCH cell**
6. **Skip completed experiments** — results are saved in Drive

---

## 💻 Running Locally

> ⚠️ Local training is slow without a GPU. Recommended only for testing/debugging.

### Step 1 — Clone and Install
```bash
cd "AMS 563 Medical Image Analysis/Project/pneumonia-detection"
pip install -r requirements.txt
```

### Step 2 — Configure Paths
Edit `configs/config.py`:
```python
raw_data_dir: str = 'data/rsna-pneumonia-detection-challenge'
processed_dir: str = 'data/rsna_processed'
image_size: int = 512   # reduce to 256 if low on RAM
```

### Step 3 — Preprocess Data
```bash
python scripts/preprocess.py
```
This converts DICOMs to PNGs and creates train/val/test split CSVs.

### Step 4 — Run Individual Experiments
```bash
# E1: Faster R-CNN baseline
python scripts/train.py --model faster_rcnn --experiment E1_faster_rcnn --epochs 30

# E2: Vanilla DETR
python scripts/train.py --model detr --experiment E2_detr_vanilla --epochs 30

# E4: Multi-Scale DETR
python scripts/train.py --model detr_multiscale --experiment E4_detr_multiscale --epochs 30
```

### Step 5 — Run All Experiments
```bash
bash scripts/run_all_experiments.sh
```

---

## 📊 Experiment Results

Results are saved to `outputs/<experiment_name>/`:

| File | Contents |
|---|---|
| `best_model.pth` | Best model weights (highest val mAP@0.5) |
| `checkpoint_epoch_N.pth` | Periodic checkpoints every 5 epochs |
| `history.json` | Per-epoch train loss + validation metrics |

### Loading a Saved Model
```python
import torch
from models.faster_rcnn import FasterRCNNWrapper
from configs.config import get_config

cfg = get_config()
model = FasterRCNNWrapper(cfg.model, pretrained=False)
model.load_state_dict(torch.load('outputs/E1_faster_rcnn/best_model.pth'))
model.eval()
```

### Metrics Reported
- `mAP@0.5` — Main detection metric (IoU threshold = 0.5)
- `mAP@0.5:0.95` — COCO-standard averaged across IoU thresholds
- `Precision@0.5` — Fraction of detections that are correct
- `Recall@0.5` — Fraction of ground truth boxes detected
- `F1@0.5` — Harmonic mean of precision and recall

---

## 🗂️ Project Structure

```
pneumonia-detection/
├── configs/
│   └── config.py              # Central config (image size, LR, epochs, etc.)
├── data/
│   ├── dataset.py             # RSNAPneumoniaDataset class + DataLoader builder
│   ├── augmentations.py       # Standard & class-aware augmentation pipelines
│   └── download.py            # Kaggle dataset downloader
├── models/
│   ├── faster_rcnn.py         # E1: Faster R-CNN with ResNet-50-FPN
│   ├── detr_baseline.py       # E2: Vanilla DETR with ResNet-50 backbone
│   └── detr_multiscale.py     # E4: Multi-Scale DETR with FPN backbone
├── utils/
│   ├── engine.py              # Trainer class (train loop, validation, checkpointing)
│   ├── losses.py              # DETRLoss (focal + L1 + GIoU + Hungarian matching)
│   └── metrics.py             # mAP, F1, precision, recall, per-size AP
├── scripts/
│   ├── preprocess.py          # DICOM → PNG conversion + train/val/test split
│   ├── train.py               # Main training script
│   └── run_all_experiments.sh # Bash script to run all experiments sequentially
├── Pneumonia_Detection_Colab.ipynb   # ← Main notebook for Colab
├── requirements.txt
├── README.md
└── HOW_TO_RUN.md              # ← This file
```

---

## 🔧 Troubleshooting

### `ValueError: Expected x_min ... to be in range [0.0, 1.0]`
**Cause:** Bounding box coordinates are in 1024×1024 original space but images are 512×512.  
**Fix:** Apply the PATCH cell before running experiments. The fix scales boxes by `image_size / 1024.0`.

### `mAP@0.5 = 0.0000` for DETR experiments
**Cause:** DETR requires 300–500 epochs to converge (original paper). 30 epochs is insufficient.  
**Fix:** This is expected. Use Faster R-CNN (E1) as your primary model. For transformer results, use a pretrained HuggingFace DETR (`facebook/detr-resnet-50`) with fine-tuning.

### `ShiftScaleRotate` / `Affine` warnings from albumentations
**Cause:** API change in newer albumentations versions.  
**Fix:** Apply the PATCH cell which replaces `ShiftScaleRotate` with `A.Affine`.

### Colab disconnects during training
**Fix:** 
- Keep the browser tab open (don't close it)
- Checkpoints are saved to Google Drive every 5 epochs
- On reconnect, re-run Steps 1–8, skip completed experiments

### `CUDA out of memory`
**Fix:** Reduce batch size in config:
```python
cfg.train.batch_size = 4   # default is 8
```

### Dataset download fails (Kaggle API error)
**Fix:** Ensure you accepted the competition rules at:  
https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/rules

---

## 📧 Contact

For questions about this project, refer to the course materials for **AMS 563 — Medical Image Analysis**.

---

*Last updated: April 2026*
