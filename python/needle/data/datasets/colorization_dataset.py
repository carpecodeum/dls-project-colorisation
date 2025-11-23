import numpy as np
from typing import Optional, List
from ..data_basic import Dataset


class ColorizationDataset(Dataset):
    """
    Wrapper dataset that converts RGB images to grayscale-color pairs for colorization training.
    
    Returns:
        (gray_L_channel, ab_channels_target) where:
        - gray_L_channel: grayscale image in [0, 1] shape (1, H, W)
        - ab_channels_target: ab color channels in Lab space, shape (2, H, W)
    """
    
    def __init__(self, base_dataset: Dataset, transforms: Optional[List] = None):
        """
        Parameters:
        base_dataset - Dataset that returns (rgb_image, label) where rgb_image is (3, H, W) in [0, 1]
        transforms - Optional list of transforms to apply
        """
        self.base_dataset = base_dataset
        self.transforms = transforms
    
    def __getitem__(self, index):
        """
        Returns (grayscale_L, ab_target) for colorization training.
        """
        rgb_img, label = self.base_dataset[index]
        
        # rgb_img is (3, H, W) in [0, 1]
        # Convert to grayscale using standard weights
        # Grayscale = 0.299*R + 0.587*G + 0.114*B
        gray = (0.299 * rgb_img[0:1, :, :] + 
                0.587 * rgb_img[1:2, :, :] + 
                0.114 * rgb_img[2:3, :, :])
        
        # For Lab conversion, we need to do a more complex conversion
        # For the MVP, we'll use a simplified approach:
        # 1. Convert grayscale to L channel (scale to [0, 100])
        L = gray * 100.0
        
        # 2. Convert RGB to Lab (full conversion)
        # We'll do this in numpy for simplicity
        rgb_np = rgb_img if isinstance(rgb_img, np.ndarray) else np.array(rgb_img)
        
        # Add batch dimension for conversion
        rgb_batch = rgb_np.reshape(1, 3, rgb_np.shape[1], rgb_np.shape[2])
        
        # Simplified Lab conversion
        lab = self._rgb_to_lab_simple(rgb_batch)
        
        # Extract ab channels (drop L channel)
        ab_target = lab[0, 1:3, :, :]  # shape (2, H, W)
        
        # Return grayscale in [0, 1] and ab channels
        return gray, ab_target
    
    def __len__(self):
        return len(self.base_dataset)
    
    def _rgb_to_lab_simple(self, rgb):
        """
        Simplified RGB to Lab conversion.
        rgb: (N, 3, H, W) in [0, 1]
        returns: (N, 3, H, W) in Lab space
        """
        # Convert to HWC for easier computation
        rgb_hwc = np.transpose(rgb, (0, 2, 3, 1))
        
        # Step 1: RGB to XYZ
        mask = rgb_hwc > 0.04045
        rgb_linear = np.where(mask, 
                             np.power((rgb_hwc + 0.055) / 1.055, 2.4),
                             rgb_hwc / 12.92)
        
        # Transformation matrix (D65 illuminant)
        M = np.array([
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041]
        ])
        
        shape = rgb_linear.shape
        rgb_flat = rgb_linear.reshape(-1, 3)
        xyz_flat = np.dot(rgb_flat, M.T)
        xyz_hwc = xyz_flat.reshape(shape)
        
        # Normalize by D65 white point
        xyz_hwc = xyz_hwc / np.array([0.95047, 1.0, 1.08883])
        
        # Step 2: XYZ to Lab
        epsilon = 0.008856
        kappa = 903.3
        
        mask = xyz_hwc > epsilon
        f_xyz = np.where(mask,
                        np.power(xyz_hwc, 1.0/3.0),
                        (kappa * xyz_hwc + 16.0) / 116.0)
        
        L = 116.0 * f_xyz[:, :, :, 1] - 16.0
        a = 500.0 * (f_xyz[:, :, :, 0] - f_xyz[:, :, :, 1])
        b = 200.0 * (f_xyz[:, :, :, 1] - f_xyz[:, :, :, 2])
        
        # Stack and convert back to NCHW
        lab_hwc = np.stack([L, a, b], axis=-1)
        lab = np.transpose(lab_hwc, (0, 3, 1, 2))
        
        return lab.astype(np.float32)

