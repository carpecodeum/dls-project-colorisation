"""
Colorization Dataset with Augmentation Pipeline.

Provides train/val/test splits and data augmentation for image colorization.

This module uses pure Python for data preprocessing and augmentation,
avoiding numpy dependencies in the colorization-specific code.
"""

import random
import math
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
    
    def random_horizontal_flip(self, img):
        """Flip image horizontally with probability flip_prob."""
        if random.random() < self.flip_prob:
            C, H, W = img.shape
            # Flip along width axis - need to copy
            flipped = img[:, :, ::-1].copy()
            return flipped
        return img
    
    def random_crop_resize(self, img, min_crop: float = 0.8):
        """Random crop and resize back to original size."""
        if random.random() < self.crop_prob:
            C, H, W = img.shape
            crop_h = int(H * random.uniform(min_crop, 1.0))
            crop_w = int(W * random.uniform(min_crop, 1.0))
            
            start_h = random.randint(0, H - crop_h)
            start_w = random.randint(0, W - crop_w)
            
            cropped = img[:, start_h:start_h+crop_h, start_w:start_w+crop_w]
            
            # Simple nearest-neighbor resize back to original size
            # Create new array with same type
            resized = img.copy()
            for c in range(C):
                for i in range(H):
                    for j in range(W):
                        src_i = int(i * crop_h / H)
                        src_j = int(j * crop_w / W)
                        resized[c, i, j] = cropped[c, src_i, src_j]
            return resized
        return img
    
    def color_jitter(self, img, brightness: float = 0.1, contrast: float = 0.1):
        """Apply random color jittering."""
        if random.random() < self.jitter_prob:
            C, H, W = img.shape
            
            # Brightness adjustment
            bright_delta = random.uniform(-brightness, brightness)
            
            # Compute mean for contrast
            total = 0.0
            count = C * H * W
            for c in range(C):
                for h in range(H):
                    for w in range(W):
                        total += float(img[c, h, w])
            mean_val = total / count
            
            # Contrast factor
            contrast_factor = random.uniform(1 - contrast, 1 + contrast)
            
            # Apply jitter
            result = img.copy()
            for c in range(C):
                for h in range(H):
                    for w in range(W):
                        val = float(img[c, h, w])
                        val = val + bright_delta
                        val = (val - mean_val) * contrast_factor + mean_val
                        # Clip to [0, 1]
                        result[c, h, w] = max(0.0, min(1.0, val))
            return result
        return img
    
    def random_rotation_90(self, img):
        """Random 90-degree rotation."""
        if random.random() < self.rotation_prob:
            k = random.randint(1, 3)  # 90, 180, or 270 degrees
            C, H, W = img.shape
            
            result = img.copy()
            for _ in range(k):
                # Rotate 90 degrees clockwise
                rotated = result.copy()
                new_H, new_W = W, H
                for c in range(C):
                    for h in range(H):
                        for w in range(W):
                            # 90 degree rotation: (h, w) -> (w, H-1-h)
                            rotated[c, w, H - 1 - h] = result[c, h, w]
                result = rotated
                H, W = new_H, new_W
            return result
        return img
    
    def __call__(self, img):
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
    
    def rgb_to_lab(self, rgb):
        """
        Convert RGB to Lab color space using pure Python.
        rgb: numpy array (3, H, W) in [0, 1]
        returns: numpy array (3, H, W) with L in [0, 100], a in [-128, 127], b in [-128, 127]
        """
        C, H, W = rgb.shape
        
        # RGB to XYZ matrix (D65 illuminant)
        M = [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041]
        ]
        
        # D65 white point
        Xn, Yn, Zn = 0.95047, 1.0, 1.08883
        
        # Constants
        epsilon = 0.008856
        kappa = 903.3
        
        # Output Lab array (same type as input)
        lab = rgb.copy()
        
        for h in range(H):
            for w in range(W):
                # Get RGB values
                r = float(rgb[0, h, w])
                g = float(rgb[1, h, w])
                b = float(rgb[2, h, w])
                
                # Gamma correction (inverse sRGB companding)
                def gamma_expand(v):
                    if v > 0.04045:
                        return ((v + 0.055) / 1.055) ** 2.4
                    else:
                        return v / 12.92
                
                r_lin = gamma_expand(r)
                g_lin = gamma_expand(g)
                b_lin = gamma_expand(b)
                
                # RGB to XYZ
                X = M[0][0] * r_lin + M[0][1] * g_lin + M[0][2] * b_lin
                Y = M[1][0] * r_lin + M[1][1] * g_lin + M[1][2] * b_lin
                Z = M[2][0] * r_lin + M[2][1] * g_lin + M[2][2] * b_lin
                
                # Normalize by white point
                X /= Xn
                Y /= Yn
                Z /= Zn
                
                # XYZ to Lab (f function)
                def f_func(t):
                    if t > epsilon:
                        return t ** (1.0 / 3.0)
                    else:
                        return (kappa * t + 16.0) / 116.0
                
                fx = f_func(X)
                fy = f_func(Y)
                fz = f_func(Z)
                
                # Lab values
                L = 116.0 * fy - 16.0
                a = 500.0 * (fx - fy)
                b_val = 200.0 * (fy - fz)
                
                lab[0, h, w] = L
                lab[1, h, w] = a
                lab[2, h, w] = b_val
        
        return lab
    
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
