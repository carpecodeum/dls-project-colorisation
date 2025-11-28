#!/usr/bin/env python3
"""
Evaluate trained colorization model and visualize results.
"""

import sys
import os
import numpy as np
import pickle

sys.path.append('./python')
sys.path.append('./apps')

import needle as ndl
import needle.nn as nn
from needle.data import DataLoader
from needle.data.datasets import CIFAR10Dataset, ColorizationDataset

# Metrics
def compute_psnr(img1, img2, max_val=1.0):
    """Compute Peak Signal-to-Noise Ratio."""
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(max_val / np.sqrt(mse))


def compute_ssim_simple(img1, img2):
    """Compute simplified SSIM."""
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    mu1 = np.mean(img1)
    mu2 = np.mean(img2)
    
    sigma1_sq = np.var(img1)
    sigma2_sq = np.var(img2)
    sigma12 = np.mean((img1 - mu1) * (img2 - mu2))
    
    ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
           ((mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim


def lab_to_rgb_numpy(L, ab):
    """Convert L and ab channels to RGB image."""
    # L is in [0, 1], convert to [0, 100]
    L = L * 100.0
    
    # ab is normalized to [-1, 1], convert back to [-128, 127]
    ab = ab * 128.0
    
    # Combine into Lab image
    lab = np.zeros((L.shape[0], L.shape[1], 3))
    lab[:, :, 0] = L
    lab[:, :, 1] = ab[0] if len(ab.shape) == 3 else ab[:, :, 0]
    lab[:, :, 2] = ab[1] if len(ab.shape) == 3 else ab[:, :, 1]
    
    # Lab to XYZ
    y = (lab[:, :, 0] + 16) / 116
    x = lab[:, :, 1] / 500 + y
    z = y - lab[:, :, 2] / 200
    
    def f_inv(t):
        delta = 6/29
        return np.where(t > delta, t**3, 3 * delta**2 * (t - 4/29))
    
    X = 0.95047 * f_inv(x)
    Y = 1.00000 * f_inv(y)
    Z = 1.08883 * f_inv(z)
    
    # XYZ to RGB
    R = 3.2406 * X - 1.5372 * Y - 0.4986 * Z
    G = -0.9689 * X + 1.8758 * Y + 0.0415 * Z
    B = 0.0557 * X - 0.2040 * Y + 1.0570 * Z
    
    rgb = np.stack([R, G, B], axis=-1)
    rgb = np.clip(rgb, 0, 1)
    
    return rgb


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
    
    for i in range(min(num_samples, len(color_dataset))):
        L_np, ab_target_np, rgb_original = color_dataset[i]
        
        # Normalize ab for model input
        ab_target_normalized = ab_target_np / 128.0
        
        # Add batch dimension
        L_batch = L_np[np.newaxis, ...]
        
        # Convert to tensor
        L_tensor = ndl.Tensor(L_batch, device=device, dtype="float32")
        
        # Predict
        ab_pred = model(L_tensor)
        ab_pred_np = ab_pred.numpy()[0]  # Remove batch dim
        
        # Convert predictions to RGB
        L_2d = L_np[0]  # Remove channel dim (1, H, W) -> (H, W)
        
        # Predicted RGB
        rgb_pred = lab_to_rgb_numpy(L_2d, ab_pred_np)
        
        # Ground truth RGB (from ab_target)
        rgb_gt = lab_to_rgb_numpy(L_2d, ab_target_np / 128.0)
        
        # Grayscale RGB (for comparison)
        rgb_gray = np.stack([L_2d, L_2d, L_2d], axis=-1)
        
        # Compute metrics
        psnr = compute_psnr(rgb_pred, rgb_gt)
        ssim = compute_ssim_simple(rgb_pred, rgb_gt)
        l1 = np.mean(np.abs(rgb_pred - rgb_gt))
        
        psnr_scores.append(psnr)
        ssim_scores.append(ssim)
        l1_errors.append(l1)
        
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
    
    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Samples evaluated: {len(psnr_scores)}")
    print(f"  PSNR:  {np.mean(psnr_scores):.2f} dB (std: {np.std(psnr_scores):.2f})")
    print(f"  SSIM:  {np.mean(ssim_scores):.4f} (std: {np.std(ssim_scores):.4f})")
    print(f"  L1:    {np.mean(l1_errors):.4f} (std: {np.std(l1_errors):.4f})")
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
        'psnr_mean': float(np.mean(psnr_scores)),
        'psnr_std': float(np.std(psnr_scores)),
        'ssim_mean': float(np.mean(ssim_scores)),
        'ssim_std': float(np.std(ssim_scores)),
        'l1_mean': float(np.mean(l1_errors)),
        'l1_std': float(np.std(l1_errors)),
    }
    
    import json
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

