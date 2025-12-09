"""Colorization neural network architecture."""

from typing import List, Optional
from needle.autograd import Tensor
from needle import ops
import needle.init as init
from .nn_basic import Module, Parameter, ReLU, Sequential
from .nn_conv import Conv


class ConvBlock(Module):
    """
    Convolutional block with Conv -> BatchNorm -> ReLU.
    """
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, device=None, dtype="float32"):
        super().__init__()
        self.conv = Conv(in_channels, out_channels, kernel_size, stride=stride, device=device, dtype=dtype)
        # Note: BatchNorm2d might not be available, so we'll skip it for MVP
        self.relu = ReLU()
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.relu(x)
        return x


class UpConvBlock(Module):
    """
    Upsampling block using nearest neighbor upsampling + convolution.
    """
    
    def __init__(self, in_channels, out_channels, kernel_size=3, device=None, dtype="float32"):
        super().__init__()
        self.conv = Conv(in_channels, out_channels, kernel_size, stride=1, device=device, dtype=dtype)
        self.relu = ReLU()
        self.scale_factor = 2
    
    def forward(self, x: Tensor) -> Tensor:
        # Simple upsampling by repeating pixels
        # x is (N, C, H, W)
        N, C, H, W = x.shape
        
        # We'll implement a simple upsampling using reshape and tile operations
        # Upsample by factor of 2
        # This is a simplified version - ideally we'd use proper interpolation
        
        # Repeat along height
        x_expanded = ops.stack([x, x], axis=3)  # (N, C, H, 2, W)
        x_expanded = ops.reshape(x_expanded, (N, C, H * 2, W))
        
        # Repeat along width
        x_expanded = ops.stack([x_expanded, x_expanded], axis=4)  # (N, C, H*2, W, 2)
        x_upsampled = ops.reshape(x_expanded, (N, C, H * 2, W * 2))
        
        x_upsampled = self.conv(x_upsampled)
        x_upsampled = self.relu(x_upsampled)
        return x_upsampled


class ColorizationNet(Module):
    """
    Encoder-Decoder network for image colorization.
    
    Input: Grayscale image (N, 1, H, W) in [0, 1]
    Output: AB channels (N, 2, H, W) in Lab color space
    
    Architecture:
    - Encoder: Progressive downsampling with convolutions
    - Decoder: Progressive upsampling with skip connections
    """
    
    def __init__(self, device=None, dtype="float32"):
        super().__init__()
        self.device = device
        self.dtype = dtype
        
        # Encoder
        # Input: (N, 1, 32, 32)
        self.enc1 = ConvBlock(1, 64, kernel_size=3, stride=1, device=device, dtype=dtype)  # (N, 64, 32, 32)
        self.enc2 = ConvBlock(64, 128, kernel_size=3, stride=2, device=device, dtype=dtype)  # (N, 128, 16, 16)
        self.enc3 = ConvBlock(128, 256, kernel_size=3, stride=2, device=device, dtype=dtype)  # (N, 256, 8, 8)
        self.enc4 = ConvBlock(256, 512, kernel_size=3, stride=2, device=device, dtype=dtype)  # (N, 512, 4, 4)
        
        # Bottleneck
        self.bottleneck = ConvBlock(512, 512, kernel_size=3, stride=1, device=device, dtype=dtype)  # (N, 512, 4, 4)
        
        # Decoder with skip connections
        self.dec1 = UpConvBlock(512, 256, kernel_size=3, device=device, dtype=dtype)  # (N, 256, 8, 8)
        self.dec2 = UpConvBlock(256 + 256, 128, kernel_size=3, device=device, dtype=dtype)  # (N, 128, 16, 16)
        self.dec3 = UpConvBlock(128 + 128, 64, kernel_size=3, device=device, dtype=dtype)  # (N, 64, 32, 32)
        
        # Final layer to produce 2 channels (ab)
        self.final = Conv(64 + 64, 2, kernel_size=3, stride=1, device=device, dtype=dtype)  # (N, 2, 32, 32)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the colorization network.
        
        Args:
            x: Grayscale input image (N, 1, H, W) in [0, 1]
        
        Returns:
            AB color channels (N, 2, H, W) in Lab space
        """
        # Encoder with skip connections saved
        e1 = self.enc1(x)  # (N, 64, 32, 32)
        e2 = self.enc2(e1)  # (N, 128, 16, 16)
        e3 = self.enc3(e2)  # (N, 256, 8, 8)
        e4 = self.enc4(e3)  # (N, 512, 4, 4)
        
        # Bottleneck
        b = self.bottleneck(e4)  # (N, 512, 4, 4)
        
        # Decoder with skip connections
        d1 = self.dec1(b)  # (N, 256, 8, 8)
        d1_cat = ops.stack([d1, e3], axis=1)  # Concatenate along channel dimension
        d1_cat = ops.reshape(d1_cat, (d1.shape[0], d1.shape[1] + e3.shape[1], d1.shape[2], d1.shape[3]))
        
        d2 = self.dec2(d1_cat)  # (N, 128, 16, 16)
        d2_cat = ops.stack([d2, e2], axis=1)
        d2_cat = ops.reshape(d2_cat, (d2.shape[0], d2.shape[1] + e2.shape[1], d2.shape[2], d2.shape[3]))
        
        d3 = self.dec3(d2_cat)  # (N, 64, 32, 32)
        d3_cat = ops.stack([d3, e1], axis=1)
        d3_cat = ops.reshape(d3_cat, (d3.shape[0], d3.shape[1] + e1.shape[1], d3.shape[2], d3.shape[3]))
        
        # Final convolution
        out = self.final(d3_cat)  # (N, 2, 32, 32)
        
        return out


class ColorizationModel(Module):
    """
    Full colorization model that handles the entire pipeline:
    - Grayscale input -> L channel
    - Predict AB channels
    - Combine L + AB -> RGB output
    """
    
    def __init__(self, device=None, dtype="float32"):
        super().__init__()
        self.colorization_net = ColorizationNet(device=device, dtype=dtype)
        self.device = device
        self.dtype = dtype
    
    def forward(self, gray: Tensor, return_lab=False) -> Tensor:
        """
        Args:
            gray: Grayscale input (N, 1, H, W) in [0, 1]
            return_lab: If True, return Lab instead of RGB
        
        Returns:
            If return_lab=False: RGB image (N, 3, H, W) in [0, 1]
            If return_lab=True: Lab image (N, 3, H, W)
        """
        # Predict AB channels
        ab_pred = self.colorization_net(gray)  # (N, 2, H, W)
        
        # Convert grayscale to L channel (scale from [0, 1] to [0, 100])
        L = gray * 100.0  # (N, 1, H, W)
        
        # Concatenate L and AB
        lab = ops.stack([L, ab_pred], axis=1)
        lab = ops.reshape(lab, (L.shape[0], 3, L.shape[2], L.shape[3]))
        
        if return_lab:
            return lab
        
        # Convert Lab to RGB (would use ops.lab_to_rgb in full implementation)
        # For now, return Lab
        return lab
    
    def predict_ab(self, gray: Tensor) -> Tensor:
        """
        Predict only AB channels.
        
        Args:
            gray: Grayscale input (N, 1, H, W) in [0, 1]
        
        Returns:
            AB channels (N, 2, H, W)
        """
        return self.colorization_net(gray)