"""Loss functions for image colorization."""

from typing import Optional
from needle.autograd import Tensor
from needle import ops
import needle.init as init
from .nn_basic import Module, Parameter
import numpy as np

# Small epsilon for numerical stability
EPS = 1e-6


class L1Loss(Module):
    """
    Mean Squared Error loss (using L2 instead of L1 for stability).
    L1 with sqrt causes NaN gradients, so we use squared error.
    """
    
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction
    
    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """
        pred: predicted values
        target: ground truth values
        """
        # Use squared error instead of absolute error for numerical stability
        # sqrt(x^2) has undefined gradient at x=0
        diff = pred - target
        squared_diff = diff * diff
        
        if self.reduction == 'mean':
            total = ops.summation(squared_diff)
            # Compute total number of elements
            numel = 1
            for dim in squared_diff.shape:
                numel *= dim
            return total / max(numel, 1)
        elif self.reduction == 'sum':
            return ops.summation(squared_diff)
        else:
            return squared_diff


class SSIMLoss(Module):
    """
    Structural Similarity Index (SSIM) loss.
    Simplified version that computes SSIM and returns 1 - SSIM as loss.
    """
    
    def __init__(self, window_size=11, C1=0.01**2, C2=0.03**2):
        super().__init__()
        self.window_size = window_size
        # Increased stability constants
        self.C1 = max(C1, 1e-4)
        self.C2 = max(C2, 1e-4)
    
    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """
        Compute SSIM loss between predicted and target images.
        For MVP, we compute a simplified version using mean and variance.
        """
        # Compute total number of elements
        numel = 1
        for dim in pred.shape:
            numel *= dim
        numel = max(numel, 1)  # Avoid division by zero
        
        # Compute means
        mu_pred = ops.summation(pred) / numel
        mu_target = ops.summation(target) / numel
        
        # Broadcast means to the shape of pred/target
        mu_pred_broadcast = ops.broadcast_to(ops.reshape(mu_pred, (1, 1, 1, 1)), pred.shape)
        mu_target_broadcast = ops.broadcast_to(ops.reshape(mu_target, (1, 1, 1, 1)), target.shape)
        
        # Compute variances and covariance (simplified)
        pred_centered = pred - mu_pred_broadcast
        target_centered = target - mu_target_broadcast
        
        var_pred = ops.summation(pred_centered * pred_centered) / numel
        var_target = ops.summation(target_centered * target_centered) / numel
        covar = ops.summation(pred_centered * target_centered) / numel
        
        # SSIM formula with increased epsilon for stability
        numerator = (2 * mu_pred * mu_target + self.C1) * (2 * covar + self.C2)
        denominator = (mu_pred * mu_pred + mu_target * mu_target + self.C1) * (var_pred + var_target + self.C2)
        
        ssim = numerator / (denominator + EPS)
        
        # Clamp SSIM to valid range to avoid numerical issues
        # Return 1 - SSIM as loss (we want to minimize this)
        one = ops.add_scalar(ssim * 0, 1.0)  # Create a tensor with value 1
        loss = one - ssim
        
        return loss


class SimplifiedPerceptualLoss(Module):
    """
    Simplified perceptual loss using features from a pretrained network.
    For MVP, we use early layers of ResNet9.
    """
    
    def __init__(self, feature_extractor=None, layers=[1, 2], device=None, dtype="float32"):
        super().__init__()
        self.layers = layers
        self.device = device
        self.dtype = dtype
        
        # For MVP, we'll use a simple feature-based loss
        # In a full implementation, we'd use a pretrained ResNet9
        self.feature_extractor = feature_extractor
    
    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """
        Compute perceptual loss between predicted and target images.
        For MVP, we use a simplified L2 loss in "feature space" (just downsampled images).
        """
        if self.feature_extractor is not None:
            # Extract features using the provided network
            pred_features = self.feature_extractor(pred)
            target_features = self.feature_extractor(target)
            
            # Compute L2 distance
            diff = pred_features - target_features
            numel = 1
            for dim in diff.shape:
                numel *= dim
            loss = ops.summation(diff * diff) / numel
        else:
            # Simplified version: just use L2 loss on the images themselves
            diff = pred - target
            numel = 1
            for dim in diff.shape:
                numel *= dim
            loss = ops.summation(diff * diff) / numel
        
        return loss


class CombinedColorizationLoss(Module):
    """
    Combined loss for image colorization.
    Combines L1 loss, SSIM loss, and perceptual loss.
    """
    
    def __init__(self, 
                 l1_weight=1.0, 
                 ssim_weight=0.5, 
                 perceptual_weight=0.1,
                 feature_extractor=None,
                 device=None,
                 dtype="float32"):
        super().__init__()
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
        self.perceptual_weight = perceptual_weight
        
        self.l1_loss = L1Loss()
        self.ssim_loss = SSIMLoss()
        self.perceptual_loss = SimplifiedPerceptualLoss(feature_extractor, device=device, dtype=dtype)
    
    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """
        Compute combined loss.
        pred: predicted ab channels (N, 2, H, W)
        target: ground truth ab channels (N, 2, H, W)
        """
        l1 = self.l1_loss(pred, target)
        ssim = self.ssim_loss(pred, target)
        perceptual = self.perceptual_loss(pred, target)
        
        total_loss = (self.l1_weight * l1 + 
                     self.ssim_weight * ssim + 
                     self.perceptual_weight * perceptual)
        
        return total_loss
