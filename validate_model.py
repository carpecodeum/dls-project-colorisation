#!/usr/bin/env python3
"""
Comprehensive validation of the colorization model.
Runs unit tests, validates model on test set, and compares to baseline.

Uses Needle operations and Python math instead of numpy.
"""

import sys
import os
import pickle
import json
import math
import random

sys.path.append('./python')

import needle as ndl
import needle.nn as nn
from needle import ops
from needle.data import DataLoader
from needle.data.datasets import CIFAR10Dataset, ColorizationDataset


def compute_psnr(pred: ndl.Tensor, target: ndl.Tensor, max_val: float = 1.0) -> float:
    """Peak Signal-to-Noise Ratio using Needle operations."""
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


def compute_ssim(pred: ndl.Tensor, target: ndl.Tensor) -> float:
    """Structural Similarity Index using Needle operations."""
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    
    # Flatten
    pred_flat = ops.reshape(pred, (pred.shape[0] * pred.shape[1] * pred.shape[2],))
    target_flat = ops.reshape(target, (target.shape[0] * target.shape[1] * target.shape[2],))
    
    numel = pred_flat.shape[0]
    
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
    """L1 error using Needle operations."""
    diff = pred - target
    abs_diff = ops.power_scalar(diff * diff, 0.5)
    
    numel = 1
    for dim in abs_diff.shape:
        numel *= dim
    
    return float((ops.summation(abs_diff) / numel).numpy().flatten()[0])


def lab_to_rgb_list(L_val, ab_pred, H, W):
    """Convert L and ab to RGB using pure Python.
    Returns list of lists for RGB values.
    """
    # L is in [0, 1], ab_pred is normalized [-1, 1]
    L_scaled = L_val * 100.0
    
    rgb_result = []
    for i in range(H):
        row = []
        for j in range(W):
            l = L_scaled[i][j] if hasattr(L_scaled, '__getitem__') else float(L_scaled)
            a = ab_pred[0][i][j] * 128.0 if len(ab_pred) > 0 else 0
            b = ab_pred[1][i][j] * 128.0 if len(ab_pred) > 1 else 0
            
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
            R = max(0, min(1, R))
            G = max(0, min(1, G))
            B = max(0, min(1, B))
            
            row.append([R, G, B])
        rgb_result.append(row)
    return rgb_result


class ValidationSuite:
    def __init__(self, checkpoint_path, data_dir):
        self.checkpoint_path = checkpoint_path
        self.data_dir = data_dir
        self.results = {}
        
        # Setup device
        try:
            self.device = ndl.cuda()
            _ = ndl.Tensor([0], device=self.device)
        except:
            self.device = ndl.cpu()
        print(f"Using device: {self.device}")
    
    def run_unit_tests(self):
        """Run unit tests for all components."""
        print("\n" + "=" * 60)
        print("UNIT TESTS")
        print("=" * 60)
        
        tests_passed = 0
        tests_failed = 0
        
        # Test 1: Color space operations
        print("\n[1/5] Testing color space operations...")
        try:
            from needle.ops.ops_colorspace import rgb_to_lab, lab_to_rgb
            # Create a test RGB tensor using random values
            random.seed(42)
            rgb_data = [[[[random.random() for _ in range(8)] for _ in range(8)] for _ in range(3)]]
            rgb_tensor = ndl.Tensor(rgb_data, device=self.device)
            
            # Convert RGB -> Lab -> RGB
            lab_tensor = rgb_to_lab(rgb_tensor)
            rgb_back_tensor = lab_to_rgb(lab_tensor)
            
            rgb_np = rgb_tensor.numpy()
            rgb_back = rgb_back_tensor.numpy()
            
            # Compute error using Python
            total_error = 0
            count = 0
            for n in range(1):
                for c in range(3):
                    for h in range(8):
                        for w in range(8):
                            total_error += abs(rgb_np[n][c][h][w] - rgb_back[n][c][h][w])
                            count += 1
            error = total_error / count
            
            assert error < 0.1, f"RGB->Lab->RGB error too high: {error}"
            print(f"      ✓ Color space conversion (error: {error:.4f})")
            tests_passed += 1
        except Exception as e:
            print(f"      ✗ Failed: {e}")
            tests_failed += 1
        
        # Test 2: Dataset loading
        print("\n[2/5] Testing dataset loading...")
        try:
            train_ds = CIFAR10Dataset(self.data_dir, train=True)
            test_ds = CIFAR10Dataset(self.data_dir, train=False)
            assert len(train_ds) > 0, "Empty train dataset"
            assert len(test_ds) > 0, "Empty test dataset"
            print(f"      ✓ CIFAR-10: {len(train_ds)} train, {len(test_ds)} test")
            tests_passed += 1
        except Exception as e:
            print(f"      ✗ Failed: {e}")
            tests_failed += 1
        
        # Test 3: Colorization dataset
        print("\n[3/5] Testing colorization dataset...")
        try:
            color_ds = ColorizationDataset(test_ds, augmentation=None, return_rgb=False)
            gray, ab = color_ds[0]
            assert gray.shape == (1, 32, 32), f"Wrong gray shape: {gray.shape}"
            assert ab.shape == (2, 32, 32), f"Wrong ab shape: {ab.shape}"
            print(f"      ✓ ColorizationDataset: gray {gray.shape}, ab {ab.shape}")
            tests_passed += 1
        except Exception as e:
            print(f"      ✗ Failed: {e}")
            tests_failed += 1
        
        # Test 4: Model forward pass
        print("\n[4/5] Testing model forward pass...")
        try:
            model = nn.ColorizationNet(device=self.device, dtype="float32")
            random.seed(42)
            x_data = [[[[random.gauss(0, 1) for _ in range(32)] for _ in range(32)]] for _ in range(2)]
            x = ndl.Tensor(x_data, device=self.device)
            y = model(x)
            assert y.shape == (2, 2, 32, 32), f"Wrong output shape: {y.shape}"
            print(f"      ✓ Forward pass: input (2,1,32,32) -> output {y.shape}")
            tests_passed += 1
        except Exception as e:
            print(f"      ✗ Failed: {e}")
            tests_failed += 1
        
        # Test 5: Loss functions
        print("\n[5/5] Testing loss functions...")
        try:
            random.seed(42)
            pred_data = [[[[random.gauss(0, 1) for _ in range(32)] for _ in range(32)] for _ in range(2)] for _ in range(2)]
            target_data = [[[[random.gauss(0, 1) for _ in range(32)] for _ in range(32)] for _ in range(2)] for _ in range(2)]
            
            pred = ndl.Tensor(pred_data, device=self.device)
            target = ndl.Tensor(target_data, device=self.device)
            
            l1 = nn.L1Loss()(pred, target)
            ssim = nn.SSIMLoss()(pred, target)
            
            l1_val = float(l1.numpy().flatten()[0])
            ssim_val = float(ssim.numpy().flatten()[0])
            
            assert not math.isnan(l1_val), "L1 loss is NaN"
            assert not math.isnan(ssim_val), "SSIM loss is NaN"
            print(f"      ✓ L1={l1_val:.4f}, SSIM={ssim_val:.4f}")
            tests_passed += 1
        except Exception as e:
            print(f"      ✗ Failed: {e}")
            tests_failed += 1
        
        self.results['unit_tests'] = {
            'passed': tests_passed,
            'failed': tests_failed,
            'total': tests_passed + tests_failed
        }
        
        print(f"\n  Summary: {tests_passed}/{tests_passed + tests_failed} tests passed")
        return tests_failed == 0
    
    def validate_trained_model(self, num_samples=500):
        """Validate trained model on test set."""
        print("\n" + "=" * 60)
        print("MODEL VALIDATION ON TEST SET")
        print("=" * 60)
        
        # Load model
        print(f"\nLoading model from: {self.checkpoint_path}")
        model = nn.ColorizationNet(device=self.device, dtype="float32")
        
        with open(self.checkpoint_path, 'rb') as f:
            checkpoint = pickle.load(f)
        
        params = model.parameters()
        for param, value in zip(params, checkpoint['param_values']):
            param.cached_data = ndl.Tensor(value, device=self.device, dtype="float32").cached_data
        
        print(f"  Checkpoint epoch: {checkpoint['epoch'] + 1}")
        print(f"  Checkpoint loss: {checkpoint['loss']:.4f}")
        
        model.eval()
        
        # Load test data
        test_ds = CIFAR10Dataset(self.data_dir, train=False)
        color_ds = ColorizationDataset(test_ds, augmentation=None, return_rgb=True)
        
        # Evaluate
        print(f"\nEvaluating on {num_samples} samples...")
        
        model_psnr, model_ssim, model_l1 = [], [], []
        baseline_psnr, baseline_ssim, baseline_l1 = [], [], []
        
        for i in range(min(num_samples, len(color_ds))):
            L_np, ab_target_np, rgb_original = color_ds[i]
            ab_target_normalized = ab_target_np / 128.0
            
            # Model prediction - add batch dimension
            L_batch = L_np.reshape(1, 1, 32, 32)
            L_tensor = ndl.Tensor(L_batch, device=self.device, dtype="float32")
            ab_pred = model(L_tensor)
            ab_pred_np = ab_pred.numpy()[0]
            
            # Get 2D L channel
            L_2d = L_np[0]  # (32, 32)
            H, W = 32, 32
            
            # Convert to RGB for metrics
            # Build tensors for PSNR/SSIM computation
            # Flatten everything for simpler metric computation
            pred_list = []
            gt_list = []
            gray_list = []
            
            for hi in range(H):
                for wi in range(W):
                    l = L_2d[hi, wi]
                    a_pred = ab_pred_np[0, hi, wi]
                    b_pred = ab_pred_np[1, hi, wi]
                    a_gt = ab_target_normalized[0, hi, wi]
                    b_gt = ab_target_normalized[1, hi, wi]
                    
                    # Simple approximation: use ab channels directly as proxy
                    pred_list.extend([l, a_pred, b_pred])
                    gt_list.extend([l, a_gt, b_gt])
                    gray_list.extend([l, 0, 0])
            
            pred_tensor = ndl.Tensor(pred_list).reshape((H, W, 3))
            gt_tensor = ndl.Tensor(gt_list).reshape((H, W, 3))
            gray_tensor = ndl.Tensor(gray_list).reshape((H, W, 3))
            
            # Model metrics
            model_psnr.append(compute_psnr(pred_tensor, gt_tensor))
            model_ssim.append(compute_ssim(pred_tensor, gt_tensor))
            model_l1.append(compute_l1(pred_tensor, gt_tensor))
            
            # Baseline metrics
            baseline_psnr.append(compute_psnr(gray_tensor, gt_tensor))
            baseline_ssim.append(compute_ssim(gray_tensor, gt_tensor))
            baseline_l1.append(compute_l1(gray_tensor, gt_tensor))
            
            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{num_samples}")
        
        # Helper functions for statistics
        def mean(lst):
            return sum(lst) / len(lst) if lst else 0.0
        
        def std(lst):
            if len(lst) < 2:
                return 0.0
            m = mean(lst)
            variance = sum((x - m) ** 2 for x in lst) / len(lst)
            return math.sqrt(variance)
        
        # Results
        self.results['model_metrics'] = {
            'psnr_mean': mean(model_psnr),
            'psnr_std': std(model_psnr),
            'ssim_mean': mean(model_ssim),
            'ssim_std': std(model_ssim),
            'l1_mean': mean(model_l1),
            'l1_std': std(model_l1),
        }
        
        self.results['baseline_metrics'] = {
            'psnr_mean': mean(baseline_psnr),
            'psnr_std': std(baseline_psnr),
            'ssim_mean': mean(baseline_ssim),
            'ssim_std': std(baseline_ssim),
            'l1_mean': mean(baseline_l1),
            'l1_std': std(baseline_l1),
        }
        
        # Improvement
        psnr_improve = self.results['model_metrics']['psnr_mean'] - self.results['baseline_metrics']['psnr_mean']
        ssim_improve = self.results['model_metrics']['ssim_mean'] - self.results['baseline_metrics']['ssim_mean']
        l1_improve = self.results['baseline_metrics']['l1_mean'] - self.results['model_metrics']['l1_mean']
        
        self.results['improvement'] = {
            'psnr_gain_db': psnr_improve,
            'ssim_gain': ssim_improve,
            'l1_reduction': l1_improve,
        }
        
        # Print comparison table
        print("\n" + "-" * 60)
        print(f"{'Metric':<15} {'Model':<20} {'Baseline (Gray)':<20}")
        print("-" * 60)
        mm = self.results['model_metrics']
        bm = self.results['baseline_metrics']
        print(f"{'PSNR (dB)':<15} {mm['psnr_mean']:.2f} ± {mm['psnr_std']:.2f}     {bm['psnr_mean']:.2f} ± {bm['psnr_std']:.2f}")
        print(f"{'SSIM':<15} {mm['ssim_mean']:.4f} ± {mm['ssim_std']:.4f}   {bm['ssim_mean']:.4f} ± {bm['ssim_std']:.4f}")
        print(f"{'L1 Error':<15} {mm['l1_mean']:.4f} ± {mm['l1_std']:.4f}   {bm['l1_mean']:.4f} ± {bm['l1_std']:.4f}")
        print("-" * 60)
        print(f"\n  Improvement over baseline:")
        print(f"    PSNR:  +{psnr_improve:.2f} dB")
        print(f"    SSIM:  +{ssim_improve:.4f}")
        print(f"    L1:    -{l1_improve:.4f} (lower is better)")
        
        return True
    
    def save_results(self, output_path="./validation_results.json"):
        """Save validation results to JSON."""
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\nResults saved to: {output_path}")
    
    def print_summary(self):
        """Print final summary."""
        print("\n" + "=" * 60)
        print("VALIDATION SUMMARY")
        print("=" * 60)
        
        # Unit tests
        ut = self.results.get('unit_tests', {})
        status = "✓ PASSED" if ut.get('failed', 1) == 0 else "✗ FAILED"
        print(f"\n  Unit Tests: {status} ({ut.get('passed', 0)}/{ut.get('total', 0)})")
        
        # Model performance
        mm = self.results.get('model_metrics', {})
        print(f"\n  Model Performance:")
        print(f"    PSNR: {mm.get('psnr_mean', 0):.2f} dB")
        print(f"    SSIM: {mm.get('ssim_mean', 0):.4f}")
        print(f"    L1:   {mm.get('l1_mean', 0):.4f}")
        
        # Improvement
        imp = self.results.get('improvement', {})
        print(f"\n  Improvement over Grayscale Baseline:")
        print(f"    PSNR: +{imp.get('psnr_gain_db', 0):.2f} dB")
        print(f"    SSIM: +{imp.get('ssim_gain', 0):.4f}")
        
        print("\n" + "=" * 60)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate colorization model")
    parser.add_argument("--checkpoint", default="./checkpoints/colorization_epoch_20.pkl")
    parser.add_argument("--data_dir", default="./data/cifar-10-batches-py")
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--output", default="./validation_results.json")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("COLORIZATION MODEL VALIDATION")
    print("=" * 60)
    
    validator = ValidationSuite(args.checkpoint, args.data_dir)
    
    # Run all validation steps
    validator.run_unit_tests()
    
    if os.path.exists(args.checkpoint):
        validator.validate_trained_model(args.num_samples)
    else:
        print(f"\nSkipping model validation: checkpoint not found at {args.checkpoint}")
    
    validator.save_results(args.output)
    validator.print_summary()


if __name__ == "__main__":
    main()
