"""Color space conversion operations using Needle array_api."""

from numbers import Number
from typing import Optional, List, Tuple, Union

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from ..backend_selection import array_api, BACKEND


def _ndarray_where(condition: NDArray, x: NDArray, y: NDArray) -> NDArray:
    """Simulate np.where using NDArray operations: condition * x + (1 - condition) * y"""
    # condition should be 0 or 1 values
    return condition * x + (1.0 - condition) * y


def _ndarray_clip(arr: NDArray, low: float, high: float) -> NDArray:
    """Clip values to [low, high] range using maximum operations."""
    # First clip to minimum (low), then clip to maximum (high)
    clipped = arr.maximum(low)
    # For upper bound, we use: min(x, high) = -max(-x, -high)
    clipped = -((-clipped).maximum(-high))
    return clipped


def _ndarray_power(arr: NDArray, exp: float) -> NDArray:
    """Raise array elements to a power."""
    return arr ** exp


class RGBToLab(TensorOp):
    """
    Convert RGB image to Lab color space.
    Input: RGB tensor in NCHW format with values in [0, 1]
    Output: Lab tensor in NCHW format with L in [0, 100], a in [-128, 127], b in [-128, 127]
    """
    
    def compute(self, rgb: NDArray) -> NDArray:
        """
        RGB to Lab conversion following standard formulas.
        rgb: NCHW format with values in [0, 1]
        """
        # Get shape info
        N, C, H, W = rgb.shape
        device = rgb.device
        
        # Constants
        gamma = 2.4
        inv_gamma = 1.0 / gamma
        threshold = 0.04045
        epsilon = 0.008856
        kappa = 903.3
        
        # D65 white point
        Xn, Yn, Zn = 0.95047, 1.0, 1.08883
        
        # RGB to XYZ matrix (D65 illuminant)
        M = [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041]
        ]
        
        # Extract RGB channels: rgb is (N, 3, H, W)
        # Permute to (N, H, W, 3) for easier processing
        rgb_permuted = rgb.permute((0, 2, 3, 1)).compact()  # (N, H, W, 3)
        
        # Flatten spatial dimensions: (N*H*W, 3)
        flat_shape = (N * H * W, 3)
        rgb_flat = rgb_permuted.reshape(flat_shape).compact()
        
        # Extract R, G, B channels
        R = rgb_flat[:, 0:1].compact().reshape((N * H * W,))  # (N*H*W,)
        G = rgb_flat[:, 1:2].compact().reshape((N * H * W,))
        B = rgb_flat[:, 2:3].compact().reshape((N * H * W,))
        
        # Gamma correction (inverse sRGB companding)
        # mask = rgb > 0.04045
        # rgb_linear = where(mask, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)
        mask_R = (R >= threshold)  # Returns 0 or 1
        mask_G = (G >= threshold)
        mask_B = (B >= threshold)
        
        R_linear = _ndarray_where(mask_R, _ndarray_power((R + 0.055) / 1.055, gamma), R / 12.92)
        G_linear = _ndarray_where(mask_G, _ndarray_power((G + 0.055) / 1.055, gamma), G / 12.92)
        B_linear = _ndarray_where(mask_B, _ndarray_power((B + 0.055) / 1.055, gamma), B / 12.92)
        
        # RGB to XYZ transformation
        X = M[0][0] * R_linear + M[0][1] * G_linear + M[0][2] * B_linear
        Y = M[1][0] * R_linear + M[1][1] * G_linear + M[1][2] * B_linear
        Z = M[2][0] * R_linear + M[2][1] * G_linear + M[2][2] * B_linear
        
        # Normalize by D65 white point
        X = X / Xn
        Y = Y / Yn
        Z = Z / Zn
        
        # XYZ to Lab: apply f function
        # f(t) = t^(1/3) if t > epsilon else (kappa * t + 16) / 116
        def f_func(t):
            mask = (t >= epsilon)
            return _ndarray_where(mask, _ndarray_power(t + 1e-10, 1.0/3.0), (kappa * t + 16.0) / 116.0)
        
        fx = f_func(X)
        fy = f_func(Y)
        fz = f_func(Z)
        
        # Calculate Lab
        L = 116.0 * fy - 16.0
        a = 500.0 * (fx - fy)
        b = 200.0 * (fy - fz)
        
        # Reshape back to (N, H, W) for each channel
        L = L.reshape((N, H, W))
        a = a.reshape((N, H, W))
        b = b.reshape((N, H, W))
        
        # Stack channels and permute to NCHW
        # We need to create (N, 3, H, W)
        # Expand dims and concatenate
        L_exp = L.reshape((N, 1, H, W))
        a_exp = a.reshape((N, 1, H, W))
        b_exp = b.reshape((N, 1, H, W))
        
        # Create output array and fill it
        out = array_api.empty((N, 3, H, W), device=device)
        out[:, 0:1, :, :] = L_exp
        out[:, 1:2, :, :] = a_exp
        out[:, 2:3, :, :] = b_exp
        
        return out.compact()
    
    def gradient(self, out_grad: Tensor, node: Tensor):
        """
        Gradient computation - approximate with identity for simplicity in MVP.
        For full implementation, would need to compute Jacobian of the transformation.
        """
        return out_grad


class LabToRGB(TensorOp):
    """
    Convert Lab image to RGB color space.
    Input: Lab tensor in NCHW format with L in [0, 100], a in [-128, 127], b in [-128, 127]
    Output: RGB tensor in NCHW format with values in [0, 1]
    """
    
    def compute(self, lab: NDArray) -> NDArray:
        """
        Lab to RGB conversion following standard formulas.
        lab: NCHW format
        """
        N, C, H, W = lab.shape
        device = lab.device
        
        # Constants
        epsilon = 0.008856
        kappa = 903.3
        inv_gamma = 1.0 / 2.4
        threshold = 0.0031308
        
        # D65 white point
        Xn, Yn, Zn = 0.95047, 1.0, 1.08883
        
        # Extract L, a, b channels
        L = lab[:, 0:1, :, :].compact().reshape((N * H * W,))
        a = lab[:, 1:2, :, :].compact().reshape((N * H * W,))
        b = lab[:, 2:3, :, :].compact().reshape((N * H * W,))
        
        # Lab to XYZ
        fy = (L + 16.0) / 116.0
        fx = a / 500.0 + fy
        fz = fy - b / 200.0
        
        # Inverse f function
        def inv_f(f_val, is_y=False, L_val=None):
            f_cubed = _ndarray_power(f_val + 1e-10, 3.0)
            mask = (f_cubed >= epsilon)
            if is_y and L_val is not None:
                L_mask = (L_val >= kappa * epsilon)
                y_branch1 = _ndarray_power((L_val + 16.0) / 116.0 + 1e-10, 3.0)
                y_branch2 = L_val / kappa
                return _ndarray_where(L_mask, y_branch1, y_branch2)
            else:
                branch1 = f_cubed
                branch2 = (116.0 * f_val - 16.0) / kappa
                return _ndarray_where(mask, branch1, branch2)
        
        xr = inv_f(fx)
        yr = inv_f(fy, is_y=True, L_val=L)
        zr = inv_f(fz)
        
        # Denormalize by D65 white point
        X = xr * Xn
        Y = yr * Yn
        Z = zr * Zn
        
        # XYZ to RGB inverse transformation matrix
        M_inv = [
            [ 3.2404542, -1.5371385, -0.4985314],
            [-0.9692660,  1.8760108,  0.0415560],
            [ 0.0556434, -0.2040259,  1.0572252]
        ]
        
        R_linear = M_inv[0][0] * X + M_inv[0][1] * Y + M_inv[0][2] * Z
        G_linear = M_inv[1][0] * X + M_inv[1][1] * Y + M_inv[1][2] * Z
        B_linear = M_inv[2][0] * X + M_inv[2][1] * Y + M_inv[2][2] * Z
        
        # Apply gamma correction (sRGB companding)
        # mask = rgb > 0.0031308
        # rgb = where(mask, 1.055 * rgb^(1/2.4) - 0.055, 12.92 * rgb)
        mask_R = (R_linear >= threshold)
        mask_G = (G_linear >= threshold)
        mask_B = (B_linear >= threshold)
        
        R = _ndarray_where(mask_R, 1.055 * _ndarray_power(R_linear.maximum(1e-10), inv_gamma) - 0.055, 12.92 * R_linear)
        G = _ndarray_where(mask_G, 1.055 * _ndarray_power(G_linear.maximum(1e-10), inv_gamma) - 0.055, 12.92 * G_linear)
        B = _ndarray_where(mask_B, 1.055 * _ndarray_power(B_linear.maximum(1e-10), inv_gamma) - 0.055, 12.92 * B_linear)
        
        # Clip to [0, 1]
        R = _ndarray_clip(R, 0.0, 1.0)
        G = _ndarray_clip(G, 0.0, 1.0)
        B = _ndarray_clip(B, 0.0, 1.0)
        
        # Reshape to (N, H, W)
        R = R.reshape((N, H, W))
        G = G.reshape((N, H, W))
        B = B.reshape((N, H, W))
        
        # Stack to NCHW
        R_exp = R.reshape((N, 1, H, W))
        G_exp = G.reshape((N, 1, H, W))
        B_exp = B.reshape((N, 1, H, W))
        
        out = array_api.empty((N, 3, H, W), device=device)
        out[:, 0:1, :, :] = R_exp
        out[:, 1:2, :, :] = G_exp
        out[:, 2:3, :, :] = B_exp
        
        return out.compact()
    
    def gradient(self, out_grad: Tensor, node: Tensor):
        """
        Gradient computation - approximate with identity for simplicity in MVP.
        """
        return out_grad


class GrayToLab(TensorOp):
    """
    Convert grayscale image to Lab color space (only L channel, a and b are 0).
    Input: Grayscale tensor in NCHW format (C=1) with values in [0, 1]
    Output: Lab tensor in NCHW format (C=3) with L in [0, 100], a=0, b=0
    """
    
    def compute(self, gray: NDArray) -> NDArray:
        """
        Grayscale to Lab conversion.
        gray: NCHW format with C=1, values in [0, 1]
        """
        N, C, H, W = gray.shape
        device = gray.device
        
        # Convert grayscale [0, 1] to L [0, 100]
        L = gray * 100.0
        
        # Create output with zeros for a and b channels
        out = array_api.full((N, 3, H, W), 0.0, device=device)
        out[:, 0:1, :, :] = L
        
        return out.compact()
    
    def gradient(self, out_grad: Tensor, node: Tensor):
        """
        Gradient flows only through L channel.
        """
        # Extract gradient for L channel and scale back
        from . import ops_mathematic
        L_grad = ops_mathematic.slice(out_grad, (slice(None), slice(0, 1), slice(None), slice(None)))
        return L_grad * 100.0


def rgb_to_lab(rgb: Tensor) -> Tensor:
    """Convert RGB tensor to Lab color space."""
    return RGBToLab()(rgb)


def lab_to_rgb(lab: Tensor) -> Tensor:
    """Convert Lab tensor to RGB color space."""
    return LabToRGB()(lab)


def gray_to_lab(gray: Tensor) -> Tensor:
    """Convert grayscale tensor to Lab color space."""
    return GrayToLab()(gray)
