#!/usr/bin/env python3
"""
Comprehensive validation of the colorization model.
Runs unit tests, validates model on test set, and compares to baseline.
"""

import sys
import os
import numpy as np
import pickle
import json

sys.path.append('./python')
sys.path.append('./apps')

import needle as ndl
import needle.nn as nn
from needle.data import DataLoader
from needle.data.datasets import CIFAR10Dataset, ColorizationDataset


def compute_psnr(img1, img2, max_val=1.0):
    """Peak Signal-to-Noise Ratio."""
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(max_val / np.sqrt(mse))


def compute_ssim(img1, img2):
    """Structural Similarity Index."""
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    mu1, mu2 = np.mean(img1), np.mean(img2)
    sigma1_sq, sigma2_sq = np.var(img1), np.var(img2)
    sigma12 = np.mean((img1 - mu1) * (img2 - mu2))
    
    ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
           ((mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim


def lab_to_rgb(L, ab):
    """Convert L and ab to RGB."""
    L = L * 100.0
    ab = ab * 128.0
    
    lab = np.zeros((L.shape[0], L.shape[1], 3))
    lab[:, :, 0] = L
    lab[:, :, 1] = ab[0] if len(ab.shape) == 3 else ab[:, :, 0]
    lab[:, :, 2] = ab[1] if len(ab.shape) == 3 else ab[:, :, 1]
    
    y = (lab[:, :, 0] + 16) / 116
    x = lab[:, :, 1] / 500 + y
    z = y - lab[:, :, 2] / 200
    
    def f_inv(t):
        delta = 6/29
        return np.where(t > delta, t**3, 3 * delta**2 * (t - 4/29))
    
    X = 0.95047 * f_inv(x)
    Y = 1.00000 * f_inv(y)
    Z = 1.08883 * f_inv(z)
    
    R = 3.2406 * X - 1.5372 * Y - 0.4986 * Z
    G = -0.9689 * X + 1.8758 * Y + 0.0415 * Z
    B = 0.0557 * X - 0.2040 * Y + 1.0570 * Z
    
    return np.clip(np.stack([R, G, B], axis=-1), 0, 1)


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
            # Create a test RGB tensor
            rgb_np = np.random.rand(1, 3, 8, 8).astype(np.float32)
            rgb_tensor = ndl.Tensor(rgb_np, device=self.device)
            
            # Convert RGB -> Lab -> RGB
            lab_tensor = rgb_to_lab(rgb_tensor)
            rgb_back_tensor = lab_to_rgb(lab_tensor)
            rgb_back = rgb_back_tensor.numpy()
            
            error = np.mean(np.abs(rgb_np - rgb_back))
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
            x = ndl.Tensor(np.random.randn(2, 1, 32, 32).astype(np.float32), 
                          device=self.device)
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
            pred = ndl.Tensor(np.random.randn(2, 2, 32, 32).astype(np.float32),
                             device=self.device)
            target = ndl.Tensor(np.random.randn(2, 2, 32, 32).astype(np.float32),
                               device=self.device)
            
            l1 = nn.L1Loss()(pred, target)
            ssim = nn.SSIMLoss()(pred, target)
            
            assert not np.isnan(l1.numpy()), "L1 loss is NaN"
            assert not np.isnan(ssim.numpy()), "SSIM loss is NaN"
            print(f"      ✓ L1={float(l1.numpy()):.4f}, SSIM={float(ssim.numpy()):.4f}")
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
            
            # Model prediction
            L_batch = L_np[np.newaxis, ...]
            L_tensor = ndl.Tensor(L_batch, device=self.device, dtype="float32")
            ab_pred = model(L_tensor)
            ab_pred_np = ab_pred.numpy()[0]
            
            L_2d = L_np[0]
            rgb_pred = lab_to_rgb(L_2d, ab_pred_np)
            rgb_gt = lab_to_rgb(L_2d, ab_target_normalized)
            rgb_gray = np.stack([L_2d, L_2d, L_2d], axis=-1)
            
            # Model metrics
            model_psnr.append(compute_psnr(rgb_pred, rgb_gt))
            model_ssim.append(compute_ssim(rgb_pred, rgb_gt))
            model_l1.append(np.mean(np.abs(rgb_pred - rgb_gt)))
            
            # Baseline metrics (grayscale vs ground truth)
            baseline_psnr.append(compute_psnr(rgb_gray, rgb_gt))
            baseline_ssim.append(compute_ssim(rgb_gray, rgb_gt))
            baseline_l1.append(np.mean(np.abs(rgb_gray - rgb_gt)))
            
            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{num_samples}")
        
        # Results
        self.results['model_metrics'] = {
            'psnr_mean': float(np.mean(model_psnr)),
            'psnr_std': float(np.std(model_psnr)),
            'ssim_mean': float(np.mean(model_ssim)),
            'ssim_std': float(np.std(model_ssim)),
            'l1_mean': float(np.mean(model_l1)),
            'l1_std': float(np.std(model_l1)),
        }
        
        self.results['baseline_metrics'] = {
            'psnr_mean': float(np.mean(baseline_psnr)),
            'psnr_std': float(np.std(baseline_psnr)),
            'ssim_mean': float(np.mean(baseline_ssim)),
            'ssim_std': float(np.std(baseline_ssim)),
            'l1_mean': float(np.mean(baseline_l1)),
            'l1_std': float(np.std(baseline_l1)),
        }
        
        # Improvement
        psnr_improve = self.results['model_metrics']['psnr_mean'] - self.results['baseline_metrics']['psnr_mean']
        ssim_improve = self.results['model_metrics']['ssim_mean'] - self.results['baseline_metrics']['ssim_mean']
        l1_improve = self.results['baseline_metrics']['l1_mean'] - self.results['model_metrics']['l1_mean']
        
        self.results['improvement'] = {
            'psnr_gain_db': float(psnr_improve),
            'ssim_gain': float(ssim_improve),
            'l1_reduction': float(l1_improve),
        }
        
        # Print comparison table
        print("\n" + "-" * 60)
        print(f"{'Metric':<15} {'Model':<20} {'Baseline (Gray)':<20}")
        print("-" * 60)
        print(f"{'PSNR (dB)':<15} {self.results['model_metrics']['psnr_mean']:.2f} ± {self.results['model_metrics']['psnr_std']:.2f}     {self.results['baseline_metrics']['psnr_mean']:.2f} ± {self.results['baseline_metrics']['psnr_std']:.2f}")
        print(f"{'SSIM':<15} {self.results['model_metrics']['ssim_mean']:.4f} ± {self.results['model_metrics']['ssim_std']:.4f}   {self.results['baseline_metrics']['ssim_mean']:.4f} ± {self.results['baseline_metrics']['ssim_std']:.4f}")
        print(f"{'L1 Error':<15} {self.results['model_metrics']['l1_mean']:.4f} ± {self.results['model_metrics']['l1_std']:.4f}   {self.results['baseline_metrics']['l1_mean']:.4f} ± {self.results['baseline_metrics']['l1_std']:.4f}")
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
    validator.validate_trained_model(args.num_samples)
    validator.save_results(args.output)
    validator.print_summary()


if __name__ == "__main__":
    main()

