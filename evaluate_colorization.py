#!/usr/bin/env python3
"""
Evaluate trained colorization model and visualize results.

Uses Needle operations and Python math instead of numpy.
"""

import sys
import os
import pickle
import json
import math

sys.path.append('./python')

import needle as ndl
import needle.nn as nn
from needle import ops
from needle.data import DataLoader
from needle.data.datasets import CIFAR10Dataset, ColorizationDataset


def compute_psnr(pred: ndl.Tensor, target: ndl.Tensor, max_val: float = 1.0) -> float:
    """Compute Peak Signal-to-Noise Ratio using Needle operations."""
    diff = pred - target
    squared_diff = diff * diff
    
    numel = 1
    for dim in squared_diff.shape:
        numel *= dim
    
    mse_tensor = ops.summation(squared_diff) / numel
    mse = float(mse_tensor.numpy().flatten()[0])
    
    if mse == 0:
        return float('inf')
    return 20 * math.log10(max_val / math.sqrt(mse))


def compute_ssim_simple(pred: ndl.Tensor, target: ndl.Tensor) -> float:
    """Compute simplified SSIM using Needle operations."""
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    # Flatten
    numel = 1
    for dim in pred.shape:
        numel *= dim
    
    pred_flat = ops.reshape(pred, (numel,))
    target_flat = ops.reshape(target, (numel,))
    
    mu1 = float((ops.summation(pred_flat) / numel).numpy().flatten()[0])
    mu2 = float((ops.summation(target_flat) / numel).numpy().flatten()[0])
    
    pred_c = pred_flat - mu1
    target_c = target_flat - mu2
    
    sigma1_sq = float((ops.summation(pred_c * pred_c) / numel).numpy().flatten()[0])
    sigma2_sq = float((ops.summation(target_c * target_c) / numel).numpy().flatten()[0])
    sigma12 = float((ops.summation(pred_c * target_c) / numel).numpy().flatten()[0])
    
    ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
           ((mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim


def compute_l1(pred: ndl.Tensor, target: ndl.Tensor) -> float:
    """Compute L1 error using Needle operations."""
    diff = pred - target
    abs_diff = ops.power_scalar(diff * diff, 0.5)
    
    numel = 1
    for dim in abs_diff.shape:
        numel *= dim
    
    return float((ops.summation(abs_diff) / numel).numpy().flatten()[0])


def lab_to_rgb_python(L_2d, ab, H, W):
    """Convert L and ab channels to RGB using pure Python.
    Returns nested list [H][W][3].
    """
    rgb_result = []
    for i in range(H):
        row = []
        for j in range(W):
            l = float(L_2d[i, j]) * 100.0
            a = float(ab[0, i, j]) * 128.0
            b = float(ab[1, i, j]) * 128.0
            
            # Lab to XYZ
            fy = (l + 16) / 116
            fx = a / 500 + fy
            fz = fy - b / 200
            
            delta = 6/29
            
            def f_inv(t):
                if t > delta:
                    return t ** 3
                else:
                    return 3 * delta**2 * (t - 4/29)
            
            X = 0.95047 * f_inv(fx)
            Y = 1.00000 * f_inv(fy)
            Z = 1.08883 * f_inv(fz)
            
            # XYZ to RGB
            R = 3.2406 * X - 1.5372 * Y - 0.4986 * Z
            G = -0.9689 * X + 1.8758 * Y + 0.0415 * Z
            B = 0.0557 * X - 0.2040 * Y + 1.0570 * Z
            
            # Clip
            R = max(0.0, min(1.0, R))
            G = max(0.0, min(1.0, G))
            B = max(0.0, min(1.0, B))
            
            row.append([R, G, B])
        rgb_result.append(row)
    return rgb_result


def load_model(checkpoint_path, device):
    """Load trained model from checkpoint."""
    from needle.nn import ColorizationNet
    
    model = ColorizationNet(device=device, dtype="float32")
    
    with open(checkpoint_path, 'rb') as f:
        checkpoint = pickle.load(f)
    
    # Restore parameters
    params = model.parameters()
    for param, value in zip(params, checkpoint['param_values']):
        param.cached_data = ndl.Tensor(value, device=device, dtype="float32").cached_data
    
    print(f"Model loaded from: {checkpoint_path}")
    print(f"  Training epoch: {checkpoint['epoch'] + 1}")
    print(f"  Loss: {checkpoint['loss']:.4f}")
    
    return model


def evaluate_and_visualize(checkpoint_path="./checkpoints/colorization_epoch_20.pkl",
                           data_dir="./data/cifar-10-batches-py",
                           num_samples=10,
                           save_dir="./results"):
    """Evaluate model and save visualization."""
    
    # Setup
    os.makedirs(save_dir, exist_ok=True)
    
    # Try GPU, fallback to CPU
    try:
        device = ndl.cuda()
        _ = ndl.Tensor([0], device=device)
    except:
        device = ndl.cpu()
    print(f"Using device: {device}")
    
    # Load model
    model = load_model(checkpoint_path, device)
    model.eval()
    
    # Load test data
    test_dataset = CIFAR10Dataset(data_dir, train=False)
    color_dataset = ColorizationDataset(test_dataset, augmentation=None, return_rgb=True)
    
    # Collect samples
    print(f"\nEvaluating on {num_samples} samples...")
    
    psnr_scores = []
    ssim_scores = []
    l1_errors = []
    
    results = []
    
    H, W = 32, 32
    
    for i in range(min(num_samples, len(color_dataset))):
        L_np, ab_target_np, rgb_original = color_dataset[i]
        
        # Normalize ab for model input
        ab_target_normalized = ab_target_np / 128.0
        
        # Add batch dimension
        L_batch = L_np.reshape(1, 1, 32, 32)
        
        # Convert to tensor
        L_tensor = ndl.Tensor(L_batch, device=device, dtype="float32")
        
        # Predict
        ab_pred = model(L_tensor)
        ab_pred_np = ab_pred.numpy()[0]  # Remove batch dim
        
        # Get 2D L channel
        L_2d = L_np[0]  # (H, W)
        
        # Build flattened lists for Needle tensor metrics
        pred_list = []
        gt_list = []
        gray_list = []
        
        for hi in range(H):
            for wi in range(W):
                l = float(L_2d[hi, wi])
                pred_list.extend([l, float(ab_pred_np[0, hi, wi]), float(ab_pred_np[1, hi, wi])])
                gt_list.extend([l, float(ab_target_normalized[0, hi, wi]), float(ab_target_normalized[1, hi, wi])])
                gray_list.extend([l, 0.0, 0.0])
        
        pred_tensor = ndl.Tensor(pred_list).reshape((H, W, 3))
        gt_tensor = ndl.Tensor(gt_list).reshape((H, W, 3))
        
        # Compute metrics
        psnr = compute_psnr(pred_tensor, gt_tensor)
        ssim = compute_ssim_simple(pred_tensor, gt_tensor)
        l1 = compute_l1(pred_tensor, gt_tensor)
        
        psnr_scores.append(psnr)
        ssim_scores.append(ssim)
        l1_errors.append(l1)
        
        # Convert to RGB for visualization
        rgb_pred = lab_to_rgb_python(L_2d, ab_pred_np, H, W)
        rgb_gt = lab_to_rgb_python(L_2d, ab_target_normalized, H, W)
        rgb_gray = [[[float(L_2d[hi, wi])] * 3 for wi in range(W)] for hi in range(H)]
        
        results.append({
            'gray': rgb_gray,
            'predicted': rgb_pred,
            'ground_truth': rgb_gt,
            'psnr': psnr,
            'ssim': ssim,
            'l1': l1
        })
        
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{num_samples}")
    
    # Helper functions
    def mean(lst):
        return sum(lst) / len(lst) if lst else 0.0
    
    def std(lst):
        if len(lst) < 2:
            return 0.0
        m = mean(lst)
        variance = sum((x - m) ** 2 for x in lst) / len(lst)
        return math.sqrt(variance)
    
    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Samples evaluated: {len(psnr_scores)}")
    print(f"  PSNR:  {mean(psnr_scores):.2f} dB (std: {std(psnr_scores):.2f})")
    print(f"  SSIM:  {mean(ssim_scores):.4f} (std: {std(ssim_scores):.4f})")
    print(f"  L1:    {mean(l1_errors):.4f} (std: {std(l1_errors):.4f})")
    print("=" * 60)
    
    # Save visualization
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        
        # Create grid visualization
        n_show = min(8, len(results))
        fig, axes = plt.subplots(n_show, 3, figsize=(9, 3 * n_show))
        
        if n_show == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(n_show):
            r = results[i]
            
            axes[i, 0].imshow(r['gray'])
            axes[i, 0].set_title('Grayscale Input')
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(r['predicted'])
            axes[i, 1].set_title(f"Predicted (PSNR: {r['psnr']:.1f})")
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(r['ground_truth'])
            axes[i, 2].set_title('Ground Truth')
            axes[i, 2].axis('off')
        
        plt.tight_layout()
        save_path = os.path.join(save_dir, 'colorization_results.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\nVisualization saved to: {save_path}")
        
    except ImportError:
        print("\nMatplotlib not available, skipping visualization")
    
    # Save metrics to JSON
    metrics = {
        'num_samples': len(psnr_scores),
        'psnr_mean': mean(psnr_scores),
        'psnr_std': std(psnr_scores),
        'ssim_mean': mean(ssim_scores),
        'ssim_std': std(ssim_scores),
        'l1_mean': mean(l1_errors),
        'l1_std': std(l1_errors),
    }
    
    metrics_path = os.path.join(save_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to: {metrics_path}")
    
    return metrics


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate colorization model")
    parser.add_argument("--checkpoint", default="./checkpoints/colorization_epoch_20.pkl",
                        help="Path to model checkpoint")
    parser.add_argument("--data_dir", default="./data/cifar-10-batches-py",
                        help="Path to CIFAR-10 data")
    parser.add_argument("--num_samples", type=int, default=100,
                        help="Number of samples to evaluate")
    parser.add_argument("--save_dir", default="./results",
                        help="Directory to save results")
    
    args = parser.parse_args()
    
    evaluate_and_visualize(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        num_samples=args.num_samples,
        save_dir=args.save_dir
    )
