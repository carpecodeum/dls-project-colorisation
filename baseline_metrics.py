"""
Baseline Metrics Script for Image Colorization

Computes PSNR and SSIM metrics on grayscale baseline (no colorization).
This serves as a baseline to compare against trained models.

Usage:
    python baseline_metrics.py [--samples N] [--output results.json]
"""

import sys
sys.path.append('./python')

import numpy as np
import argparse
import json
from datetime import datetime


def compute_psnr(pred: np.ndarray, target: np.ndarray, max_val: float = 1.0) -> float:
    """
    Compute Peak Signal-to-Noise Ratio.
    
    Args:
        pred: Predicted image (H, W, C) or (C, H, W)
        target: Ground truth image
        max_val: Maximum pixel value (1.0 for normalized images)
    
    Returns:
        PSNR value in dB
    """
    mse = np.mean((pred - target) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(max_val / np.sqrt(mse))


def compute_ssim(pred: np.ndarray, target: np.ndarray, 
                 C1: float = 0.01**2, C2: float = 0.03**2) -> float:
    """
    Compute Structural Similarity Index (simplified global version).
    
    Args:
        pred: Predicted image
        target: Ground truth image
        C1, C2: Stability constants
    
    Returns:
        SSIM value in [0, 1]
    """
    # Flatten for global computation
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    
    # Compute means
    mu_pred = np.mean(pred_flat)
    mu_target = np.mean(target_flat)
    
    # Compute variances and covariance
    var_pred = np.var(pred_flat)
    var_target = np.var(target_flat)
    covar = np.mean((pred_flat - mu_pred) * (target_flat - mu_target))
    
    # SSIM formula
    numerator = (2 * mu_pred * mu_target + C1) * (2 * covar + C2)
    denominator = (mu_pred**2 + mu_target**2 + C1) * (var_pred + var_target + C2)
    
    return numerator / (denominator + 1e-8)


def compute_ssim_windowed(pred: np.ndarray, target: np.ndarray,
                          window_size: int = 7) -> float:
    """
    Compute SSIM using sliding window (more accurate).
    
    Args:
        pred: Predicted image (H, W, C)
        target: Ground truth image (H, W, C)
        window_size: Size of the sliding window
    
    Returns:
        Mean SSIM across all windows
    """
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    H, W, C = pred.shape
    ssim_values = []
    
    for i in range(0, H - window_size + 1, window_size // 2):
        for j in range(0, W - window_size + 1, window_size // 2):
            window_pred = pred[i:i+window_size, j:j+window_size, :]
            window_target = target[i:i+window_size, j:j+window_size, :]
            
            mu_p = np.mean(window_pred)
            mu_t = np.mean(window_target)
            var_p = np.var(window_pred)
            var_t = np.var(window_target)
            covar = np.mean((window_pred - mu_p) * (window_target - mu_t))
            
            num = (2 * mu_p * mu_t + C1) * (2 * covar + C2)
            den = (mu_p**2 + mu_t**2 + C1) * (var_p + var_t + C2)
            ssim_values.append(num / (den + 1e-8))
    
    return np.mean(ssim_values)


def lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    """
    Convert Lab to RGB.
    lab: (3, H, W) or (H, W, 3)
    """
    if lab.shape[0] == 3:
        lab = np.transpose(lab, (1, 2, 0))
    
    L, a, b = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]
    
    # Lab to XYZ
    fy = (L + 16.0) / 116.0
    fx = a / 500.0 + fy
    fz = fy - b / 200.0
    
    epsilon = 0.008856
    kappa = 903.3
    
    xr = np.where(fx**3 > epsilon, fx**3, (116.0 * fx - 16.0) / kappa)
    yr = np.where(L > kappa * epsilon, ((L + 16.0) / 116.0)**3, L / kappa)
    zr = np.where(fz**3 > epsilon, fz**3, (116.0 * fz - 16.0) / kappa)
    
    X = xr * 0.95047
    Y = yr * 1.0
    Z = zr * 1.08883
    
    xyz = np.stack([X, Y, Z], axis=-1)
    
    # XYZ to RGB
    M_inv = np.array([
        [ 3.2404542, -1.5371385, -0.4985314],
        [-0.9692660,  1.8760108,  0.0415560],
        [ 0.0556434, -0.2040259,  1.0572252]
    ])
    
    rgb = np.dot(xyz, M_inv.T)
    
    # Gamma correction
    mask = rgb > 0.0031308
    rgb = np.where(mask,
                   1.055 * np.power(np.clip(rgb, 0, None), 1.0/2.4) - 0.055,
                   12.92 * rgb)
    
    return np.clip(rgb, 0, 1)


def grayscale_baseline(L: np.ndarray) -> np.ndarray:
    """
    Create grayscale RGB image from L channel (baseline - no color).
    
    Args:
        L: Luminance channel (1, H, W) in [0, 1]
    
    Returns:
        RGB image (H, W, 3) where R=G=B
    """
    # Convert L from [0, 1] to grayscale RGB
    gray = L[0]  # (H, W)
    return np.stack([gray, gray, gray], axis=-1)


def run_baseline_metrics(num_samples: int = 100, seed: int = 42) -> dict:
    """
    Run PSNR/SSIM metrics on baseline (grayscale only).
    
    Args:
        num_samples: Number of samples to evaluate
        seed: Random seed for reproducibility
    
    Returns:
        Dictionary with metrics
    """
    import needle.data as data
    
    print("=" * 60)
    print("Baseline Metrics Evaluation")
    print("=" * 60)
    print(f"Samples: {num_samples}")
    print(f"Seed: {seed}")
    print("-" * 60)
    
    # Load test dataset
    print("\nLoading CIFAR-10 test dataset...")
    test_cifar = data.CIFAR10Dataset(
        base_folder='./data/cifar-10-batches-py',
        train=False
    )
    test_dataset = data.ColorizationDataset(test_cifar, augmentation=None)
    
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Sample indices
    np.random.seed(seed)
    indices = np.random.choice(len(test_dataset), num_samples, replace=False)
    
    # Compute metrics
    psnr_scores = []
    ssim_scores = []
    ssim_windowed_scores = []
    l1_errors = []
    
    print(f"\nComputing metrics on {num_samples} samples...")
    
    for i, idx in enumerate(indices):
        L, ab_target, rgb_original = test_dataset[idx]
        
        # Baseline: grayscale RGB (no colorization)
        rgb_baseline = grayscale_baseline(L)
        
        # Ground truth RGB
        rgb_gt = np.transpose(rgb_original, (1, 2, 0))  # CHW to HWC
        
        # Compute metrics
        psnr = compute_psnr(rgb_baseline, rgb_gt)
        ssim = compute_ssim(rgb_baseline, rgb_gt)
        ssim_w = compute_ssim_windowed(rgb_baseline, rgb_gt)
        l1 = np.mean(np.abs(rgb_baseline - rgb_gt))
        
        psnr_scores.append(psnr)
        ssim_scores.append(ssim)
        ssim_windowed_scores.append(ssim_w)
        l1_errors.append(l1)
        
        if (i + 1) % 20 == 0:
            print(f"  Processed {i + 1}/{num_samples} samples...")
    
    # Aggregate results
    results = {
        'experiment': 'grayscale_baseline',
        'timestamp': datetime.now().isoformat(),
        'num_samples': num_samples,
        'seed': seed,
        'metrics': {
            'psnr': {
                'mean': float(np.mean(psnr_scores)),
                'std': float(np.std(psnr_scores)),
                'min': float(np.min(psnr_scores)),
                'max': float(np.max(psnr_scores))
            },
            'ssim_global': {
                'mean': float(np.mean(ssim_scores)),
                'std': float(np.std(ssim_scores)),
                'min': float(np.min(ssim_scores)),
                'max': float(np.max(ssim_scores))
            },
            'ssim_windowed': {
                'mean': float(np.mean(ssim_windowed_scores)),
                'std': float(np.std(ssim_windowed_scores)),
                'min': float(np.min(ssim_windowed_scores)),
                'max': float(np.max(ssim_windowed_scores))
            },
            'l1_error': {
                'mean': float(np.mean(l1_errors)),
                'std': float(np.std(l1_errors)),
                'min': float(np.min(l1_errors)),
                'max': float(np.max(l1_errors))
            }
        },
        'raw_scores': {
            'psnr': [float(x) for x in psnr_scores],
            'ssim': [float(x) for x in ssim_scores],
            'ssim_windowed': [float(x) for x in ssim_windowed_scores],
            'l1': [float(x) for x in l1_errors]
        }
    }
    
    # Print summary
    print("\n" + "=" * 60)
    print("BASELINE METRICS RESULTS")
    print("=" * 60)
    print(f"\nPSNR (dB):")
    print(f"  Mean: {results['metrics']['psnr']['mean']:.2f}")
    print(f"  Std:  {results['metrics']['psnr']['std']:.2f}")
    print(f"  Range: [{results['metrics']['psnr']['min']:.2f}, {results['metrics']['psnr']['max']:.2f}]")
    
    print(f"\nSSIM (global):")
    print(f"  Mean: {results['metrics']['ssim_global']['mean']:.4f}")
    print(f"  Std:  {results['metrics']['ssim_global']['std']:.4f}")
    
    print(f"\nSSIM (windowed):")
    print(f"  Mean: {results['metrics']['ssim_windowed']['mean']:.4f}")
    print(f"  Std:  {results['metrics']['ssim_windowed']['std']:.4f}")
    
    print(f"\nL1 Error:")
    print(f"  Mean: {results['metrics']['l1_error']['mean']:.4f}")
    print(f"  Std:  {results['metrics']['l1_error']['std']:.4f}")
    
    print("=" * 60)
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Compute baseline metrics for colorization')
    parser.add_argument('--samples', type=int, default=100, 
                        help='Number of samples to evaluate (default: 100)')
    parser.add_argument('--output', type=str, default='baseline_results.json',
                        help='Output JSON file for results')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    results = run_baseline_metrics(
        num_samples=args.samples,
        seed=args.seed
    )
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {args.output}")


if __name__ == '__main__':
    main()

