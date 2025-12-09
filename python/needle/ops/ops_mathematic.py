"""Operator implementations."""

from numbers import Number
from typing import Optional, List, Tuple, Union

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks

from ..backend_selection import array_api, BACKEND
from .ops_tuple import *


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class EWisePow(TensorOp):
    """Op to element-wise raise a tensor to a power."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        
        raise NotImplementedError()
        

    def gradient(self, out_grad, node):
        
        raise NotImplementedError()
        


def power(a, b):
    return EWisePow()(a, b)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        
        return a ** self.scalar
        

    def gradient(self, out_grad, node):
        
        input_val = node.inputs[0]
        return out_grad * self.scalar * (input_val ** (self.scalar - 1))
        


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        
        return a / b
        

    def gradient(self, out_grad, node):
        
        lhs, rhs = node.inputs
        return out_grad / rhs, out_grad * (-lhs / (rhs ** 2))
        


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        
        return a / self.scalar
        

    def gradient(self, out_grad, node):
        
        return out_grad / self.scalar
        


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        
        if self.axes:
            return array_api.swapaxes(a, self.axes[0], self.axes[1])
        else:
            return array_api.swapaxes(a, a.ndim - 2, a.ndim - 1)
        

    def gradient(self, out_grad, node):
        
        return transpose(out_grad, self.axes)
        


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        
        # Ensure array is compact before reshaping
        return array_api.reshape(a.compact(), self.shape)
        

    def gradient(self, out_grad, node):
        
        input_shape = node.inputs[0].shape
        return reshape(out_grad, input_shape)
        


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        
        return array_api.broadcast_to(a, self.shape)
        

    def gradient(self, out_grad, node):
        
        input_shape = node.inputs[0].shape
        axes_to_sum = []
        ndim_diff = len(self.shape) - len(input_shape)
        for i in range(ndim_diff):
            axes_to_sum.append(i)
        for i in range(len(input_shape)):
            if input_shape[i] == 1 and self.shape[i + ndim_diff] != 1:
                axes_to_sum.append(i + ndim_diff)
        
        if axes_to_sum:
            grad = summation(out_grad, tuple(axes_to_sum))
        else:
            grad = out_grad
        return reshape(grad, input_shape)
        


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        
        if self.axes is None:
            return array_api.sum(a)
        if isinstance(self.axes, int):
            return array_api.sum(a, axis=self.axes)
        axes = sorted(self.axes, reverse=True) if isinstance(self.axes, tuple) else [self.axes]
        result = a
        for axis in axes:
            result = array_api.sum(result, axis=axis)
        return result
        

    def gradient(self, out_grad, node):
        
        input_shape = node.inputs[0].shape
        if self.axes is None:
            grad_shape = [1] * len(input_shape)
        else:
            grad_shape = list(input_shape)
            axes = [self.axes] if isinstance(self.axes, int) else self.axes
            for axis in axes:
                grad_shape[axis] = 1
        out_grad_reshaped = reshape(out_grad, tuple(grad_shape))
        return broadcast_to(out_grad_reshaped, input_shape)
        


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        
        return a @ b
        

    def gradient(self, out_grad, node):
        
        lhs, rhs = node.inputs
        lgrad = matmul(out_grad, transpose(rhs))
        rgrad = matmul(transpose(lhs), out_grad)
        
        if len(lhs.shape) < len(lgrad.shape):
            lgrad = summation(lgrad, tuple(range(len(lgrad.shape) - len(lhs.shape))))
        if len(rhs.shape) < len(rgrad.shape):
            rgrad = summation(rgrad, tuple(range(len(rgrad.shape) - len(rhs.shape))))
            
        return lgrad, rgrad
        


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        
        return -a
        

    def gradient(self, out_grad, node):
        
        return -out_grad
        


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        
        return array_api.log(a)
        

    def gradient(self, out_grad, node):
        
        return out_grad / node.inputs[0]
        


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        
        return array_api.exp(a)
        

    def gradient(self, out_grad, node):
        
        return out_grad * exp(node.inputs[0])
        


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        
        return array_api.maximum(a, 0)
        

    def gradient(self, out_grad, node):
        
        input_val = node.inputs[0].realize_cached_data()
        mask = Tensor(input_val > 0, device=out_grad.device, dtype=out_grad.dtype)
        return out_grad * mask
        


def relu(a):
    return ReLU()(a)


class Tanh(TensorOp):
    def compute(self, a):
        
        return array_api.tanh(a)
        

    def gradient(self, out_grad, node):
        
        tanh_val = tanh(node.inputs[0])
        return out_grad * (tanh_val * (-tanh_val) + 1)
        


def tanh(a):
    return Tanh()(a)


class Stack(TensorOp):
    def __init__(self, axis: int):
        """
        Concatenates a sequence of arrays along a new dimension.
        Parameters:
        axis - dimension to concatenate along
        All arrays need to be of the same size.
        """
        self.axis = axis

    def compute(self, args: TensorTuple) -> Tensor:
        
        n = len(args)
        shape = args[0].shape
        new_shape = list(shape)
        new_shape.insert(self.axis, n)
        
        out = array_api.empty(new_shape, device=args[0].device)
        
        slices = [slice(None)] * len(new_shape)
        for i, arr in enumerate(args):
            slices[self.axis] = i
            out[tuple(slices)] = arr
        
        return out
        

    def gradient(self, out_grad, node):
        
        return split(out_grad, self.axis)
        


def stack(args, axis):
    return Stack(axis)(make_tuple(*args))


class Split(TensorTupleOp):
    def __init__(self, axis: int):
        """
        Splits a tensor along an axis into a tuple of tensors.
        (The "inverse" of Stack)
        Parameters:
        axis - dimension to split
        """
        self.axis = axis

    def compute(self, A):
        
        n = A.shape[self.axis]
        result = []
        slices = [slice(None)] * len(A.shape)
        new_shape = list(A.shape)
        new_shape.pop(self.axis)
        
        for i in range(n):
            slices[self.axis] = i
            result.append(A[tuple(slices)].compact().reshape(tuple(new_shape)))
        
        return tuple(result)
        

    def gradient(self, out_grad, node):
        
        return stack(out_grad, self.axis)
        


def split(a, axis):
    return Split(axis)(a)


class Flip(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        
        return array_api.flip(a, self.axes)
        

    def gradient(self, out_grad, node):
        
        return flip(out_grad, self.axes)
        


def flip(a, axes):
    return Flip(axes)(a)


class Dilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        
        if self.dilation == 0:
            return a
        
        # Determine the maximum axis to ensure we have enough dimensions
        max_axis = max(self.axes) if self.axes else -1
        
        # Extend shape with singleton dimensions if necessary
        new_shape = list(a.shape)
        while len(new_shape) <= max_axis:
            new_shape.append(1)
        
        # Reshape input if we added dimensions
        if len(new_shape) > len(a.shape):
            a = a.reshape(tuple(new_shape))
        
        # Calculate dilated shape
        new_shape = list(a.shape)
        for axis in self.axes:
            new_shape[axis] = a.shape[axis] * (self.dilation + 1)
        
        out = array_api.full(tuple(new_shape), 0.0, device=a.device)
        
        slices = [slice(None)] * len(new_shape)
        for axis in self.axes:
            slices[axis] = slice(0, new_shape[axis], self.dilation + 1)
        
        out[tuple(slices)] = a
        return out
        

    def gradient(self, out_grad, node):
        
        grad = undilate(out_grad, self.axes, self.dilation)
        # If we extended dimensions in forward pass, we need to reshape back
        input_shape = node.inputs[0].shape
        if grad.shape != input_shape:
            grad = grad.reshape(input_shape)
        return grad
        


def dilate(a, axes, dilation):
    return Dilate(axes, dilation)(a)


class UnDilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        
        if self.dilation == 0:
            return a
        
        slices = [slice(None)] * len(a.shape)
        for axis in self.axes:
            slices[axis] = slice(0, a.shape[axis], self.dilation + 1)
        
        return a[tuple(slices)].compact()
        

    def gradient(self, out_grad, node):
        
        return dilate(out_grad, self.axes, self.dilation)
        


def undilate(a, axes, dilation):
    return UnDilate(axes, dilation)(a)


class Conv(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.stride = stride
        self.padding = padding

    def compute(self, A, B):
        
        A = A.pad(((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)))
        
        N, H, W, C_in = A.shape
        K, _, _, C_out = B.shape
        Ns, Hs, Ws, Cs = A.strides
        
        H_out = (H - K) // self.stride + 1
        W_out = (W - K) // self.stride + 1
        
        inner_dim = K * K * C_in
        im2col_shape = (N, H_out, W_out, K, K, C_in)
        im2col_strides = (Ns, Hs * self.stride, Ws * self.stride, Hs, Ws, Cs)
        im2col = A.as_strided(im2col_shape, im2col_strides).compact()
        
        im2col = im2col.reshape((N * H_out * W_out, inner_dim))
        B_reshaped = B.compact().reshape((inner_dim, C_out))
        out = im2col @ B_reshaped
        
        return out.reshape((N, H_out, W_out, C_out))
        

    def gradient(self, out_grad, node):
        
        X, W = node.inputs
        K, _, C_in, C_out = W.shape
        N, H, W_dim, _ = X.shape
        _, H_out, W_out, _ = out_grad.shape
        
        # Gradient w.r.t. X: convolve dilated out_grad with flipped W
        out_grad_for_X = out_grad
        if self.stride > 1:
            out_grad_for_X = dilate(out_grad, (1, 2), self.stride - 1)
        
        W_flipped = flip(W, (0, 1))
        W_flipped = transpose(W_flipped, (2, 3))
        
        pad = K - 1 - self.padding
        X_grad = conv(out_grad_for_X, W_flipped, stride=1, padding=pad)
        
        # Gradient w.r.t. W: Use im2col approach (transpose of forward matmul)
        # Forward: out = im2col(X) @ W_reshaped
        # Backward: W_grad = im2col(X).T @ out_grad
        
        # Pad X
        X_padded = X.realize_cached_data().pad(((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)))
        N_pad, H_pad, W_pad, C_in = X_padded.shape
        Ns, Hs, Ws, Cs = X_padded.strides
        
        # Create im2col view: same as in forward pass
        im2col_shape = (N, H_out, W_out, K, K, C_in)
        im2col_strides = (Ns, Hs * self.stride, Ws * self.stride, Hs, Ws, Cs)
        im2col = X_padded.as_strided(im2col_shape, im2col_strides).compact()
        
        # Reshape for matmul
        im2col = im2col.reshape((N * H_out * W_out, K * K * C_in))
        out_grad_reshaped = out_grad.realize_cached_data().compact().reshape((N * H_out * W_out, C_out))
        
        # W_grad = im2col.T @ out_grad_reshaped
        W_grad_flat = im2col.permute((1, 0)) @ out_grad_reshaped
        
        # Reshape to (K, K, C_in, C_out)
        from needle import Tensor
        W_grad = Tensor.make_const(W_grad_flat.reshape((K, K, C_in, C_out)))
        
        return X_grad, W_grad
        


def conv(a, b, stride=1, padding=1):
    return Conv(stride, padding)(a, b)


