# Multi-Scale Attention-Guided Pneumonia Detection in Chest X-Rays

## Project Structure
```
pneumonia_detection/
├── configs/
│   └── config.py              # All hyperparameters and paths
├── data/
│   ├── dataset.py             # RSNA dataset class + preprocessing
│   ├── augmentations.py       # Class-aware augmentation pipeline
│   └── download.py            # Data download helper
├── models/
│   ├── faster_rcnn.py         # Baseline: Faster R-CNN
│   ├── detr_baseline.py       # Baseline: Vanilla DETR
│   ├── detr_multiscale.py     # Proposed: Multi-scale DETR
│   └── domain_pretrain.py     # Domain-adaptive MAE pretraining
├── utils/
│   ├── engine.py              # Training and evaluation loops
│   ├── metrics.py             # mAP, FROC, per-size AP
│   ├── visualization.py       # Bbox plotting, attention maps, error analysis
│   └── losses.py              # Focal loss, Hungarian matching
├── scripts/
│   ├── preprocess.py          # DICOM -> PNG conversion
│   ├── train.py               # Main training script
│   ├── evaluate.py            # Evaluation script
│   ├── pretrain_mae.py        # Self-supervised pretraining
│   └── run_all_experiments.sh # Run all experiments E1-E7
├── outputs/                   # Checkpoints, logs, figures
├── requirements.txt
└── README.md
```


## Dataset

This project uses the [RSNA Pneumonia Detection Challenge dataset](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/data).

**How to download:**
1. Go to the [Kaggle competition page](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/data).
2. Accept the competition rules and download the dataset files.
3. Place the downloaded files in the `rsna_data/` directory.

*Note: The dataset is not included in this repository due to size and licensing restrictions.*

## Setup

```bash
# 1. Create environment
conda create -n pneumonia python=3.10 -y
conda activate pneumonia

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download RSNA data (requires Kaggle API key)
python data/download.py --data-dir ./rsna_data

# 4. Preprocess DICOM files
python scripts/preprocess.py --input-dir ./rsna_data --output-dir ./rsna_processed

# 5. Run all experiments
bash scripts/run_all_experiments.sh
```

## Experiments
| ID | Command | Description |
|----|---------|-------------|
| E1 | `python scripts/train.py --model faster_rcnn` | Faster R-CNN baseline |
| E2 | `python scripts/train.py --model detr` | Vanilla DETR baseline |
| E3 | `python scripts/train.py --model detr --domain-pretrain` | DETR + domain pretraining |
| E4 | `python scripts/train.py --model detr_multiscale` | DETR + multi-scale fusion |
| E5 | `python scripts/train.py --model detr --augment class_aware` | DETR + class-aware aug |
| E6 | `python scripts/train.py --model detr_multiscale --domain-pretrain --augment class_aware` | Full pipeline |
| E7 | `python scripts/evaluate.py --analyze-errors` | Error analysis |
