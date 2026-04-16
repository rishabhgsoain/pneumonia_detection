"""
Evaluation script: load trained model, evaluate on test set, generate visualizations.

Usage:
    # Evaluate single model
    python scripts/evaluate.py --model detr_multiscale --checkpoint outputs/E6_full_pipeline/best_model.pth

    # Compare all experiments
    python scripts/evaluate.py --compare-all

    # Full error analysis
    python scripts/evaluate.py --model detr_multiscale --checkpoint outputs/E6_full_pipeline/best_model.pth --analyze-errors
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from configs.config import get_config
from data.dataset import build_dataloaders
from data.augmentations import build_augmentation
from utils.metrics import evaluate_detections
from utils.visualization import (
    visualize_predictions, visualize_attention_maps,
    plot_metrics_comparison, plot_error_analysis, _denormalize
)


def load_model(model_name, checkpoint_path, cfg, device):
    """Load model from checkpoint."""
    from scripts.train import build_model
    model = build_model(model_name, cfg)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model


@torch.no_grad()
def evaluate_model(model, test_loader, cfg, device, model_name):
    """Run evaluation on test set and return predictions + metrics."""
    all_predictions = []
    all_targets = []
    all_images = []
    inference_times = []

    for images, targets in tqdm(test_loader, desc="Evaluating"):
        images = images.to(device)

        start = torch.cuda.Event(enable_timing=True) if device == "cuda" else None
        end = torch.cuda.Event(enable_timing=True) if device == "cuda" else None

        if start:
            start.record()

        if model_name == "faster_rcnn":
            predictions = model(images)
        elif hasattr(model, 'predict'):
            predictions = model.predict(images, score_threshold=cfg.model.score_threshold)
        else:
            outputs = model(images)
            from utils.engine import _decode_detr_outputs
            predictions = _decode_detr_outputs(outputs, images.shape, cfg.model.score_threshold)

        if end:
            end.record()
            torch.cuda.synchronize()
            inference_times.append(start.elapsed_time(end))

        for pred in predictions:
            all_predictions.append({k: v.cpu() for k, v in pred.items()})
        for tgt in targets:
            all_targets.append({k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in tgt.items()})
        all_images.append(images.cpu())

    metrics = evaluate_detections(all_predictions, all_targets, iou_thresholds=cfg.eval.iou_thresholds)

    if inference_times:
        metrics["avg_inference_ms"] = np.mean(inference_times)
        metrics["fps"] = 1000.0 / np.mean(inference_times)

    return all_predictions, all_targets, all_images, metrics


def run_full_evaluation(args):
    """Evaluate a single model with full analysis."""
    cfg = get_config(experiment_name=args.experiment or "eval")
    device = args.device if torch.cuda.is_available() else "cpu"

    # Load model
    model = load_model(args.model, args.checkpoint, cfg, device)

    # Build test loader
    val_transform = build_augmentation("none", cfg.data.image_size)
    _, _, test_loader = build_dataloaders(cfg, val_transform=val_transform)

    # Evaluate
    predictions, targets, images, metrics = evaluate_model(
        model, test_loader, cfg, device, args.model
    )

    # Print results
    print(f"\n{'='*60}")
    print(f"Test Results: {args.model}")
    print(f"{'='*60}")
    for key, value in sorted(metrics.items()):
        if isinstance(value, float):
            print(f"  {key:30s}: {value:.4f}")
    print()

    # Save results
    save_dir = os.path.join("outputs", args.experiment or "eval")
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, "test_metrics.json"), "w") as f:
        json.dump({k: float(v) for k, v in metrics.items()}, f, indent=2)

    # Visualize sample predictions
    print("Generating visualizations...")
    vis_dir = os.path.join(save_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    n_vis = min(args.num_vis, len(predictions))
    all_images_flat = torch.cat(images, dim=0) if images else torch.tensor([])

    for i in range(n_vis):
        if i >= len(all_images_flat):
            break
        img_np = _denormalize(all_images_flat[i])
        visualize_predictions(
            img_np, predictions[i], targets[i],
            save_path=os.path.join(vis_dir, f"pred_{i:04d}.png"),
            title=f"Sample {i} | mAP@0.5={metrics['mAP@0.5']:.3f}"
        )

    # Error analysis
    if args.analyze_errors:
        print("Running error analysis...")
        error_dir = os.path.join(save_dir, "error_analysis")
        plot_error_analysis(predictions, targets, error_dir)

        # Attention maps (DETR models only)
        if args.model in ["detr", "detr_multiscale"] and len(all_images_flat) > 0:
            print("Generating attention maps...")
            attn_dir = os.path.join(save_dir, "attention_maps")
            visualize_attention_maps(
                model, all_images_flat[:10].to(device), attn_dir
            )

    print(f"\nAll results saved to: {save_dir}")
    return metrics


def compare_all_experiments(args):
    """Compare metrics across all completed experiments."""
    output_base = "./outputs"
    results = {}

    for exp_dir in sorted(os.listdir(output_base)):
        metrics_path = os.path.join(output_base, exp_dir, "test_metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                results[exp_dir] = json.load(f)

    if not results:
        print("No experiment results found. Run evaluation for each experiment first.")
        return

    print(f"\n{'='*80}")
    print("Experiment Comparison")
    print(f"{'='*80}")

    # Print table
    header = f"{'Experiment':30s} {'mAP@0.5':>10s} {'mAP@0.5:0.95':>14s} {'Prec':>8s} {'Recall':>8s} {'F1':>8s}"
    print(header)
    print("-" * len(header))
    for exp, metrics in results.items():
        print(f"{exp:30s} "
              f"{metrics.get('mAP@0.5', 0):.4f}     "
              f"{metrics.get('mAP@0.5:0.95', 0):.4f}         "
              f"{metrics.get('precision@0.5', 0):.4f}   "
              f"{metrics.get('recall@0.5', 0):.4f}   "
              f"{metrics.get('f1@0.5', 0):.4f}")

    # Generate comparison plot
    plot_path = os.path.join(output_base, "comparison_chart.png")
    plot_metrics_comparison(results, plot_path)
    print(f"\nComparison chart saved to: {plot_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate detection models")
    parser.add_argument("--model", type=str,
                        choices=["faster_rcnn", "detr", "detr_multiscale"])
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--experiment", type=str)
    parser.add_argument("--analyze-errors", action="store_true")
    parser.add_argument("--compare-all", action="store_true")
    parser.add_argument("--num-vis", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    if args.compare_all:
        compare_all_experiments(args)
    elif args.model and args.checkpoint:
        run_full_evaluation(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
