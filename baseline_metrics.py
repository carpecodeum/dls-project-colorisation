"""
Baseline Metrics Script for Image Colorization

Computes PSNR and SSIM metrics on grayscale baseline (no colorization).
This serves as a baseline to compare against trained models.

Uses Needle operations where possible, with Python math for scalar operations.

Usage:
    python baseline_metrics.py [--samples N] [--output results.json]
"""

import sys
sys.path.append('./python')

import argparse
import json
import math
import random
from datetime import datetime

import needle as ndl
from needle import ops


def compute_psnr(pred: ndl.Tensor, target: ndl.Tensor, max_val: float = 1.0) -> float:
    """
    Compute Peak Signal-to-Noise Ratio using Needle operations.
    
    Args:
        pred: Predicted image tensor
        target: Ground truth image tensor
        max_val: Maximum pixel value (1.0 for normalized images)
    
    Returns:
        PSNR value in dB
    """
    diff = pred - target
    squared_diff = diff * diff
    
    # Compute mean
    numel = 1
    for dim in squared_diff.shape:
        numel *= dim
    mse_tensor = ops.summation(squared_diff) / numel
    mse = float(mse_tensor.numpy().flatten()[0])
    
    if mse == 0:
        return float('inf')
    return 20 * math.log10(max_val / math.sqrt(mse))


def compute_ssim(pred: ndl.Tensor, target: ndl.Tensor, 
                 C1: float = 0.01**2, C2: float = 0.03**2) -> float:
    """
    Compute Structural Similarity Index using Needle operations.
    
    Args:
        pred: Predicted image tensor
        target: Ground truth image tensor
        C1, C2: Stability constants
    
    Returns:
        SSIM value in [0, 1]
    """
    # Flatten tensors
    pred_flat = ops.reshape(pred, (pred.shape[0] * pred.shape[1] * pred.shape[2],))
    target_flat = ops.reshape(target, (target.shape[0] * target.shape[1] * target.shape[2],))
    
    numel = pred_flat.shape[0]
    
    # Compute means
    mu_pred = ops.summation(pred_flat) / numel
    mu_target = ops.summation(target_flat) / numel
    
    # Broadcast means for centered computation
    mu_pred_val = float(mu_pred.numpy().flatten()[0])
    mu_target_val = float(mu_target.numpy().flatten()[0])
    
    # Compute variances and covariance
    pred_centered = pred_flat - mu_pred_val
    target_centered = target_flat - mu_target_val
    
    var_pred = ops.summation(pred_centered * pred_centered) / numel
    var_target = ops.summation(target_centered * target_centered) / numel
    covar = ops.summation(pred_centered * target_centered) / numel
    
    var_pred_val = float(var_pred.numpy().flatten()[0])
    var_target_val = float(var_target.numpy().flatten()[0])
    covar_val = float(covar.numpy().flatten()[0])
    
    # SSIM formula
    numerator = (2 * mu_pred_val * mu_target_val + C1) * (2 * covar_val + C2)
    denominator = (mu_pred_val**2 + mu_target_val**2 + C1) * (var_pred_val + var_target_val + C2)
    
    return numerator / (denominator + 1e-8)


def compute_ssim_windowed(pred_np, target_np, window_size: int = 7) -> float:
    """
    Compute SSIM using sliding window (more accurate).
    Uses Python loops for windowing, Needle for computations.
    
    Args:
        pred_np: Predicted image numpy array (H, W, C)
        target_np: Ground truth image numpy array (H, W, C)
        window_size: Size of the sliding window
    
    Returns:
        Mean SSIM across all windows
    """
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    H, W, C = pred_np.shape
    ssim_values = []
    
    for i in range(0, H - window_size + 1, window_size // 2):
        for j in range(0, W - window_size + 1, window_size // 2):
            window_pred = pred_np[i:i+window_size, j:j+window_size, :]
            window_target = target_np[i:i+window_size, j:j+window_size, :]
            
            # Convert to Needle tensors
            wp = ndl.Tensor(window_pred.flatten())
            wt = ndl.Tensor(window_target.flatten())
            numel = wp.shape[0]
            
            mu_p = float((ops.summation(wp) / numel).numpy().flatten()[0])
            mu_t = float((ops.summation(wt) / numel).numpy().flatten()[0])
            
            wp_c = wp - mu_p
            wt_c = wt - mu_t
            
            var_p = float((ops.summation(wp_c * wp_c) / numel).numpy().flatten()[0])
            var_t = float((ops.summation(wt_c * wt_c) / numel).numpy().flatten()[0])
            covar = float((ops.summation(wp_c * wt_c) / numel).numpy().flatten()[0])
            
            num = (2 * mu_p * mu_t + C1) * (2 * covar + C2)
            den = (mu_p**2 + mu_t**2 + C1) * (var_p + var_t + C2)
            ssim_values.append(num / (den + 1e-8))
    
    return sum(ssim_values) / len(ssim_values) if ssim_values else 0.0


def lab_to_rgb_tensor(lab: ndl.Tensor) -> ndl.Tensor:
    """
    Convert Lab to RGB using Needle operations.
    lab: (3, H, W) tensor
    Returns: (H, W, 3) tensor
    """
    # Use the ops.lab_to_rgb function
    # First reshape to (1, 3, H, W) for batch processing
    C, H, W = lab.shape
    lab_batch = ops.reshape(lab, (1, C, H, W))
    rgb_batch = ops.lab_to_rgb(lab_batch)
    # Reshape to (H, W, 3)
    rgb = ops.reshape(rgb_batch, (3, H, W))
    # Permute to HWC
    rgb_np = rgb.numpy()
    return rgb_np.transpose(1, 2, 0)  # Return as numpy HWC for metrics


def grayscale_baseline_tensor(L: ndl.Tensor):
    """
    Create grayscale RGB image from L channel (baseline - no color).
    
    Args:
        L: Luminance channel tensor (1, H, W) in [0, 1]
    
    Returns:
        RGB numpy array (H, W, 3) where R=G=B
    """
    L_np = L.numpy()
    gray = L_np[0]  # (H, W)
    # Stack using list comprehension (avoiding numpy.stack)
    H, W = gray.shape
    rgb = [[[ gray[i, j], gray[i, j], gray[i, j] ] for j in range(W)] for i in range(H)]
    # Convert back to flat array structure
    import array
    result = []
    for i in range(H):
        for j in range(W):
            for c in range(3):
                result.append(gray[i, j])
    # Reshape manually
    result_arr = []
    idx = 0
    for i in range(H):
        row = []
        for j in range(W):
            pixel = [result[idx], result[idx+1], result[idx+2]]
            idx += 3
            row.append(pixel)
        result_arr.append(row)
    return result_arr


def grayscale_baseline(L_np):
    """
    Create grayscale RGB from L channel numpy array.
    
    Args:
        L_np: Luminance numpy array (1, H, W) in [0, 1]
    
    Returns:
        RGB numpy array (H, W, 3)
    """
    gray = L_np[0]  # (H, W)
    H, W = gray.shape
    # Build RGB array without numpy.stack
    rgb = [[[gray[i, j]] * 3 for j in range(W)] for i in range(H)]
    return rgb


def compute_l1_error(pred: ndl.Tensor, target: ndl.Tensor) -> float:
    """Compute L1 error using Needle operations."""
    diff = pred - target
    # Approximate abs using sqrt(x^2)
    abs_diff = ops.power_scalar(diff * diff, 0.5)
    numel = 1
    for dim in abs_diff.shape:
        numel *= dim
    return float((ops.summation(abs_diff) / numel).numpy().flatten()[0])


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
    print("Baseline Metrics Evaluation (Using Needle)")
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
    
    # Sample indices using Python random
    random.seed(seed)
    all_indices = list(range(len(test_dataset)))
    indices = random.sample(all_indices, min(num_samples, len(all_indices)))
    
    # Compute metrics
    psnr_scores = []
    ssim_scores = []
    ssim_windowed_scores = []
    l1_errors = []
    
    print(f"\nComputing metrics on {num_samples} samples...")
    
    for i, idx in enumerate(indices):
        L_np, ab_target_np, rgb_original_np = test_dataset[idx]
        
        # Create baseline grayscale RGB (no colorization)
        gray = L_np[0]  # (H, W)
        H, W = gray.shape
        
        # Build rgb_baseline as nested list then convert
        rgb_baseline_list = [[[float(gray[hi, wi])] * 3 for wi in range(W)] for hi in range(H)]
        
        # Ground truth RGB - transpose from CHW to HWC
        rgb_gt_list = [[[float(rgb_original_np[c, hi, wi]) for c in range(3)] for wi in range(W)] for hi in range(H)]
        
        # Convert to Needle tensors for metric computation
        # Flatten for PSNR/SSIM
        baseline_flat = [rgb_baseline_list[hi][wi][c] for hi in range(H) for wi in range(W) for c in range(3)]
        gt_flat = [rgb_gt_list[hi][wi][c] for hi in range(H) for wi in range(W) for c in range(3)]
        
        pred_tensor = ndl.Tensor(baseline_flat).reshape((H, W, 3))
        target_tensor = ndl.Tensor(gt_flat).reshape((H, W, 3))
        
        # Compute metrics
        psnr = compute_psnr(pred_tensor, target_tensor)
        ssim = compute_ssim(pred_tensor, target_tensor)
        
        # For windowed SSIM, convert to simple arrays
        pred_arr = pred_tensor.numpy()
        target_arr = target_tensor.numpy()
        ssim_w = compute_ssim_windowed(pred_arr, target_arr)
        
        # L1 error
        l1 = compute_l1_error(pred_tensor, target_tensor)
        
        psnr_scores.append(psnr)
        ssim_scores.append(ssim)
        ssim_windowed_scores.append(ssim_w)
        l1_errors.append(l1)
        
        if (i + 1) % 20 == 0:
            print(f"  Processed {i + 1}/{num_samples} samples...")
    
    # Aggregate results using Python
    def mean(lst):
        return sum(lst) / len(lst) if lst else 0.0
    
    def std(lst):
        if len(lst) < 2:
            return 0.0
        m = mean(lst)
        variance = sum((x - m) ** 2 for x in lst) / len(lst)
        return math.sqrt(variance)
    
    results = {
        'experiment': 'grayscale_baseline',
        'timestamp': datetime.now().isoformat(),
        'num_samples': num_samples,
        'seed': seed,
        'metrics': {
            'psnr': {
                'mean': mean(psnr_scores),
                'std': std(psnr_scores),
                'min': min(psnr_scores),
                'max': max(psnr_scores)
            },
            'ssim_global': {
                'mean': mean(ssim_scores),
                'std': std(ssim_scores),
                'min': min(ssim_scores),
                'max': max(ssim_scores)
            },
            'ssim_windowed': {
                'mean': mean(ssim_windowed_scores),
                'std': std(ssim_windowed_scores),
                'min': min(ssim_windowed_scores),
                'max': max(ssim_windowed_scores)
            },
            'l1_error': {
                'mean': mean(l1_errors),
                'std': std(l1_errors),
                'min': min(l1_errors),
                'max': max(l1_errors)
            }
        },
        'raw_scores': {
            'psnr': psnr_scores,
            'ssim': ssim_scores,
            'ssim_windowed': ssim_windowed_scores,
            'l1': l1_errors
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
