"""Color space conversion operations."""

from numbers import Number
from typing import Optional, List, Tuple, Union

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp
import numpy

from ..backend_selection import array_api, BACKEND


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
        # Convert to numpy for computation
        rgb_np = rgb.numpy() if hasattr(rgb, 'numpy') else numpy.array(rgb)
        
        # Step 1: RGB to XYZ
        # Apply gamma correction (inverse sRGB companding)
        mask = rgb_np > 0.04045
        rgb_linear = numpy.where(mask, 
                                 numpy.power((rgb_np + 0.055) / 1.055, 2.4),
                                 rgb_np / 12.92)
        
        # RGB to XYZ transformation matrix
        # Using D65 illuminant
        M = numpy.array([
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041]
        ])
        
        # Extract channels (N, C, H, W) -> (N, H, W, C)
        rgb_hwc = numpy.transpose(rgb_linear, (0, 2, 3, 1))
        shape = rgb_hwc.shape
        
        # Flatten spatial dimensions for matrix multiplication
        rgb_flat = rgb_hwc.reshape(-1, 3)
        xyz_flat = numpy.dot(rgb_flat, M.T)
        xyz_hwc = xyz_flat.reshape(shape)
        
        # Normalize by D65 white point
        xyz_hwc = xyz_hwc / numpy.array([0.95047, 1.0, 1.08883])
        
        # Step 2: XYZ to Lab
        # Apply f function
        epsilon = 0.008856  # (6/29)^3
        kappa = 903.3  # (29/3)^3
        
        mask = xyz_hwc > epsilon
        f_xyz = numpy.where(mask,
                           numpy.power(xyz_hwc, 1.0/3.0),
                           (kappa * xyz_hwc + 16.0) / 116.0)
        
        # Calculate Lab
        L = 116.0 * f_xyz[:, :, :, 1] - 16.0
        a = 500.0 * (f_xyz[:, :, :, 0] - f_xyz[:, :, :, 1])
        b = 200.0 * (f_xyz[:, :, :, 1] - f_xyz[:, :, :, 2])
        
        # Stack and convert back to NCHW
        lab_hwc = numpy.stack([L, a, b], axis=-1)
        lab = numpy.transpose(lab_hwc, (0, 3, 1, 2))
        
        return array_api.array(lab, dtype=rgb.dtype, device=rgb.device)
    
    def gradient(self, out_grad: Tensor, node: Tensor):
        """
        Gradient computation - approximate with identity for simplicity in MVP.
        For full implementation, would need to compute Jacobian of the transformation.
        """
        # For MVP, we return a scaled gradient
        # This is a simplification - full implementation would compute actual Jacobian
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
        # Convert to numpy for computation
        lab_np = lab.numpy() if hasattr(lab, 'numpy') else numpy.array(lab)
        
        # Convert NCHW to NHWC
        lab_hwc = numpy.transpose(lab_np, (0, 2, 3, 1))
        L = lab_hwc[:, :, :, 0]
        a = lab_hwc[:, :, :, 1]
        b = lab_hwc[:, :, :, 2]
        
        # Step 1: Lab to XYZ
        fy = (L + 16.0) / 116.0
        fx = a / 500.0 + fy
        fz = fy - b / 200.0
        
        epsilon = 0.008856
        kappa = 903.3
        
        # Inverse f function
        xr = numpy.where(fx**3 > epsilon, fx**3, (116.0 * fx - 16.0) / kappa)
        yr = numpy.where(L > kappa * epsilon, ((L + 16.0) / 116.0)**3, L / kappa)
        zr = numpy.where(fz**3 > epsilon, fz**3, (116.0 * fz - 16.0) / kappa)
        
        # Denormalize by D65 white point
        X = xr * 0.95047
        Y = yr * 1.0
        Z = zr * 1.08883
        
        # Stack XYZ
        xyz_hwc = numpy.stack([X, Y, Z], axis=-1)
        
        # Step 2: XYZ to RGB
        # Inverse transformation matrix
        M_inv = numpy.array([
            [ 3.2404542, -1.5371385, -0.4985314],
            [-0.9692660,  1.8760108,  0.0415560],
            [ 0.0556434, -0.2040259,  1.0572252]
        ])
        
        shape = xyz_hwc.shape
        xyz_flat = xyz_hwc.reshape(-1, 3)
        rgb_flat = numpy.dot(xyz_flat, M_inv.T)
        rgb_hwc = rgb_flat.reshape(shape)
        
        # Apply gamma correction (sRGB companding)
        mask = rgb_hwc > 0.0031308
        rgb_hwc = numpy.where(mask,
                             1.055 * numpy.power(rgb_hwc, 1.0/2.4) - 0.055,
                             12.92 * rgb_hwc)
        
        # Clip to [0, 1]
        rgb_hwc = numpy.clip(rgb_hwc, 0, 1)
        
        # Convert back to NCHW
        rgb = numpy.transpose(rgb_hwc, (0, 3, 1, 2))
        
        return array_api.array(rgb, dtype=lab.dtype, device=lab.device)
    
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
        gray_np = gray.numpy() if hasattr(gray, 'numpy') else numpy.array(gray)
        
        # Convert grayscale [0, 1] to L [0, 100]
        # Assuming linear mapping for simplicity
        L = gray_np * 100.0
        
        # Create a and b channels (zeros)
        N, C, H, W = gray_np.shape
        a = numpy.zeros((N, C, H, W), dtype=gray_np.dtype)
        b = numpy.zeros((N, C, H, W), dtype=gray_np.dtype)
        
        # Stack to create Lab image
        lab = numpy.concatenate([L, a, b], axis=1)
        
        return array_api.array(lab, dtype=gray.dtype, device=gray.device)
    
    def gradient(self, out_grad: Tensor, node: Tensor):
        """
        Gradient flows only through L channel.
        """
        # Extract gradient for L channel and scale back
        L_grad = out_grad[:, 0:1, :, :] * 100.0
        return L_grad


def rgb_to_lab(rgb: Tensor) -> Tensor:
    """Convert RGB tensor to Lab color space."""
    return RGBToLab()(rgb)


def lab_to_rgb(lab: Tensor) -> Tensor:
    """Convert Lab tensor to RGB color space."""
    return LabToRGB()(lab)


def gray_to_lab(gray: Tensor) -> Tensor:
    """Convert grayscale tensor to Lab color space."""
    return GrayToLab()(gray)
