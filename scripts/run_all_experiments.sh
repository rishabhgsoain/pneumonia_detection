#!/bin/bash
# =============================================================================
# Run All Experiments (E1-E7)
# Multi-Scale Attention-Guided Pneumonia Detection
# =============================================================================
set -e

echo "============================================================"
echo "  Pneumonia Detection: Full Experiment Pipeline"
echo "============================================================"
echo ""

# Configuration
DATA_DIR="./rsna_data"
PROCESSED_DIR="./rsna_processed"
DEVICE="cuda"
EPOCHS=50
BATCH_SIZE=4

# -------------------------------------------------------
# Step 0: Data Preparation (run once)
# -------------------------------------------------------
echo "[Step 0] Preparing data..."
if [ ! -d "$PROCESSED_DIR/images" ]; then
    python scripts/preprocess.py \
        --input-dir $DATA_DIR \
        --output-dir $PROCESSED_DIR
else
    echo "  Data already preprocessed, skipping."
fi
echo ""

# -------------------------------------------------------
# Step 0b: Domain-Adaptive Pretraining (optional, for E3/E6)
# -------------------------------------------------------
PRETRAIN_PATH="./outputs/mae_pretrained.pth"
if [ ! -f "$PRETRAIN_PATH" ] && [ -d "./nih_chestxray/images" ]; then
    echo "[Step 0b] Running MAE pretraining on unlabeled chest X-rays..."
    python scripts/pretrain_mae.py \
        --image-dir ./nih_chestxray/images \
        --output-path $PRETRAIN_PATH \
        --epochs 100 \
        --batch-size 32
    echo ""
else
    echo "[Step 0b] Skipping MAE pretraining (already done or no NIH data)."
fi
echo ""

# -------------------------------------------------------
# E1: Faster R-CNN Baseline
# -------------------------------------------------------
echo "============================================================"
echo "[E1] Training Faster R-CNN baseline..."
echo "============================================================"
python scripts/train.py \
    --model faster_rcnn \
    --experiment E1_faster_rcnn \
    --augment standard \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --device $DEVICE

python scripts/evaluate.py \
    --model faster_rcnn \
    --checkpoint outputs/E1_faster_rcnn/best_model.pth \
    --experiment E1_faster_rcnn \
    --analyze-errors \
    --device $DEVICE
echo ""

# -------------------------------------------------------
# E2: Vanilla DETR Baseline
# -------------------------------------------------------
echo "============================================================"
echo "[E2] Training vanilla DETR baseline..."
echo "============================================================"
python scripts/train.py \
    --model detr \
    --experiment E2_detr_vanilla \
    --augment standard \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --device $DEVICE

python scripts/evaluate.py \
    --model detr \
    --checkpoint outputs/E2_detr_vanilla/best_model.pth \
    --experiment E2_detr_vanilla \
    --analyze-errors \
    --device $DEVICE
echo ""

# -------------------------------------------------------
# E3: DETR + Domain-Adaptive Pretraining
# -------------------------------------------------------
echo "============================================================"
echo "[E3] Training DETR with domain-adaptive pretraining..."
echo "============================================================"
python scripts/train.py \
    --model detr \
    --experiment E3_detr_pretrained \
    --augment standard \
    --domain-pretrain \
    --pretrain-path $PRETRAIN_PATH \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --device $DEVICE

python scripts/evaluate.py \
    --model detr \
    --checkpoint outputs/E3_detr_pretrained/best_model.pth \
    --experiment E3_detr_pretrained \
    --analyze-errors \
    --device $DEVICE
echo ""

# -------------------------------------------------------
# E4: Multi-Scale DETR
# -------------------------------------------------------
echo "============================================================"
echo "[E4] Training multi-scale DETR..."
echo "============================================================"
python scripts/train.py \
    --model detr_multiscale \
    --experiment E4_detr_multiscale \
    --augment standard \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --device $DEVICE

python scripts/evaluate.py \
    --model detr_multiscale \
    --checkpoint outputs/E4_detr_multiscale/best_model.pth \
    --experiment E4_detr_multiscale \
    --analyze-errors \
    --device $DEVICE
echo ""

# -------------------------------------------------------
# E5: DETR + Class-Aware Augmentation
# -------------------------------------------------------
echo "============================================================"
echo "[E5] Training DETR with class-aware augmentation..."
echo "============================================================"
python scripts/train.py \
    --model detr \
    --experiment E5_detr_augmented \
    --augment class_aware \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --device $DEVICE

python scripts/evaluate.py \
    --model detr \
    --checkpoint outputs/E5_detr_augmented/best_model.pth \
    --experiment E5_detr_augmented \
    --analyze-errors \
    --device $DEVICE
echo ""

# -------------------------------------------------------
# E6: Full Pipeline (all components)
# -------------------------------------------------------
echo "============================================================"
echo "[E6] Training full pipeline..."
echo "============================================================"
python scripts/train.py \
    --model detr_multiscale \
    --experiment E6_full_pipeline \
    --augment class_aware \
    --domain-pretrain \
    --pretrain-path $PRETRAIN_PATH \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --device $DEVICE

python scripts/evaluate.py \
    --model detr_multiscale \
    --checkpoint outputs/E6_full_pipeline/best_model.pth \
    --experiment E6_full_pipeline \
    --analyze-errors \
    --device $DEVICE
echo ""

# -------------------------------------------------------
# E7: Cross-Experiment Comparison
# -------------------------------------------------------
echo "============================================================"
echo "[E7] Generating comparison across all experiments..."
echo "============================================================"
python scripts/evaluate.py --compare-all
echo ""

echo "============================================================"
echo "  All experiments complete!"
echo "  Results saved in ./outputs/"
echo "============================================================"
