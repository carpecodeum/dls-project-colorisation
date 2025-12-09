"""
Colorization Dataset with Augmentation Pipeline.

Provides train/val/test splits and data augmentation for image colorization.

Note: This module uses numpy for data preprocessing and augmentation.
This is intentional and follows standard deep learning practice:
- Data loading/preprocessing operates on raw numpy arrays
- DataLoader converts preprocessed numpy arrays to Needle Tensors
- For differentiable color space operations in the neural network,
  use needle.ops.rgb_to_lab, needle.ops.lab_to_rgb, needle.ops.gray_to_lab
"""

import numpy as np
from typing import Optional, List, Callable
from ..data_basic import Dataset


class ColorizationAugmentation:
    """
    Augmentation pipeline for colorization training.
    
    Includes:
    - Random horizontal flip
    - Random crop and resize
    - Color jitter (applied before grayscale conversion)
    - Random rotation
    """
    
    def __init__(self, 
                 flip_prob: float = 0.5,
                 crop_prob: float = 0.3,
                 jitter_prob: float = 0.2,
                 rotation_prob: float = 0.2):
        self.flip_prob = flip_prob
        self.crop_prob = crop_prob
        self.jitter_prob = jitter_prob
        self.rotation_prob = rotation_prob
    
    def random_horizontal_flip(self, img: np.ndarray) -> np.ndarray:
        """Flip image horizontally with probability flip_prob."""
        if np.random.random() < self.flip_prob:
            return img[:, :, ::-1].copy()
        return img
    
    def random_crop_resize(self, img: np.ndarray, 
                           min_crop: float = 0.8) -> np.ndarray:
        """Random crop and resize back to original size."""
        if np.random.random() < self.crop_prob:
            C, H, W = img.shape
            crop_h = int(H * np.random.uniform(min_crop, 1.0))
            crop_w = int(W * np.random.uniform(min_crop, 1.0))
            
            start_h = np.random.randint(0, H - crop_h + 1)
            start_w = np.random.randint(0, W - crop_w + 1)
            
            cropped = img[:, start_h:start_h+crop_h, start_w:start_w+crop_w]
            
            # Simple nearest-neighbor resize back to original size
            resized = np.zeros((C, H, W), dtype=img.dtype)
            for c in range(C):
                for i in range(H):
                    for j in range(W):
                        src_i = int(i * crop_h / H)
                        src_j = int(j * crop_w / W)
                        resized[c, i, j] = cropped[c, src_i, src_j]
            return resized
        return img
    
    def color_jitter(self, img: np.ndarray,
                    brightness: float = 0.1,
                    contrast: float = 0.1,
                    saturation: float = 0.1) -> np.ndarray:
        """Apply random color jittering."""
        if np.random.random() < self.jitter_prob:
            # Brightness
            img = img + np.random.uniform(-brightness, brightness)
            
            # Contrast
            mean = img.mean()
            img = (img - mean) * np.random.uniform(1-contrast, 1+contrast) + mean
            
            # Clip to valid range
            img = np.clip(img, 0, 1)
        return img
    
    def random_rotation_90(self, img: np.ndarray) -> np.ndarray:
        """Random 90-degree rotation."""
        if np.random.random() < self.rotation_prob:
            k = np.random.randint(1, 4)  # 90, 180, or 270 degrees
            img = np.rot90(img, k, axes=(1, 2)).copy()
        return img
    
    def __call__(self, img: np.ndarray) -> np.ndarray:
        """Apply all augmentations."""
        img = self.random_horizontal_flip(img)
        img = self.random_crop_resize(img)
        img = self.color_jitter(img)
        img = self.random_rotation_90(img)
        return img


class ColorizationDataset(Dataset):
    """
    Dataset wrapper for image colorization.
    
    Converts RGB images to grayscale + Lab color channel pairs.
    
    Returns:
        (grayscale, ab_target, rgb_original) where:
        - grayscale: L channel in [0, 1], shape (1, H, W)
        - ab_target: a* and b* channels, shape (2, H, W)
        - rgb_original: original RGB image for visualization
    """
    
    def __init__(self, 
                 base_dataset: Dataset, 
                 augmentation: Optional[ColorizationAugmentation] = None,
                 return_rgb: bool = True):
        """
        Parameters:
        base_dataset - RGB image dataset returning (image, label)
        augmentation - optional augmentation pipeline
        return_rgb - if True, also return original RGB for visualization
        """
        self.base_dataset = base_dataset
        self.augmentation = augmentation
        self.return_rgb = return_rgb
    
    def rgb_to_lab(self, rgb: np.ndarray) -> np.ndarray:
        """
        Convert RGB to Lab color space.
        rgb: (3, H, W) in [0, 1]
        returns: (3, H, W) with L in [0, 100], a in [-128, 127], b in [-128, 127]
        """
        # Transpose to HWC for easier computation
        rgb_hwc = np.transpose(rgb, (1, 2, 0))
        
        # Step 1: RGB to XYZ (with gamma correction)
        mask = rgb_hwc > 0.04045
        rgb_linear = np.where(mask, 
                             np.power((rgb_hwc + 0.055) / 1.055, 2.4),
                             rgb_hwc / 12.92)
        
        # RGB to XYZ matrix (D65 illuminant)
        M = np.array([
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041]
        ])
        
        H, W, _ = rgb_hwc.shape
        rgb_flat = rgb_linear.reshape(-1, 3)
        xyz_flat = np.dot(rgb_flat, M.T)
        xyz_hwc = xyz_flat.reshape(H, W, 3)
        
        # Normalize by D65 white point
        xyz_hwc = xyz_hwc / np.array([0.95047, 1.0, 1.08883])
        
        # Step 2: XYZ to Lab
        epsilon = 0.008856
        kappa = 903.3
        
        mask = xyz_hwc > epsilon
        f_xyz = np.where(mask,
                        np.power(xyz_hwc, 1.0/3.0),
                        (kappa * xyz_hwc + 16.0) / 116.0)
        
        L = 116.0 * f_xyz[:, :, 1] - 16.0
        a = 500.0 * (f_xyz[:, :, 0] - f_xyz[:, :, 1])
        b = 200.0 * (f_xyz[:, :, 1] - f_xyz[:, :, 2])
        
        # Stack to CHW format
        lab = np.stack([L, a, b], axis=0)
        
        return lab.astype(np.float32)
    
    def __getitem__(self, index):
        """
        Returns (grayscale, ab_target) or (grayscale, ab_target, rgb) for training.
        """
        rgb_img, label = self.base_dataset[index]
        
        # Apply augmentation if provided
        if self.augmentation is not None:
            rgb_img = self.augmentation(rgb_img)
        
        # Convert to Lab
        lab = self.rgb_to_lab(rgb_img)
        
        # Extract L channel (grayscale) and normalize to [0, 1]
        L = lab[0:1, :, :] / 100.0  # L is in [0, 100]
        
        # Extract ab channels
        ab = lab[1:3, :, :]  # a, b in [-128, 127]
        
        if self.return_rgb:
            return L, ab, rgb_img
        return L, ab
    
    def __len__(self):
        return len(self.base_dataset)


def create_colorization_splits(base_folder: str, 
                               val_size: int = 5000,
                               seed: int = 42,
                               augment_train: bool = True):
    """
    Create train/val/test dataset splits for colorization.
    
    Returns:
        train_dataset, val_dataset, test_dataset
    """
    from .cifar10_dataset import CIFAR10Dataset
    
    # Create augmentation for training
    augmentation = ColorizationAugmentation() if augment_train else None
    
    # Training set (with augmentation)
    train_cifar = CIFAR10Dataset(
        base_folder=base_folder,
        train=True,
        split='train',
        val_size=val_size,
        seed=seed
    )
    train_dataset = ColorizationDataset(train_cifar, augmentation=augmentation)
    
    # Validation set (no augmentation)
    val_cifar = CIFAR10Dataset(
        base_folder=base_folder,
        train=True,
        split='val',
        val_size=val_size,
        seed=seed
    )
    val_dataset = ColorizationDataset(val_cifar, augmentation=None)
    
    # Test set (no augmentation)
    test_cifar = CIFAR10Dataset(
        base_folder=base_folder,
        train=False
    )
    test_dataset = ColorizationDataset(test_cifar, augmentation=None)
    
    return train_dataset, val_dataset, test_dataset

