from typing import Optional, Any, Union
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

from ..backend_selection import array_api, BACKEND 

class LogSoftmax(TensorOp):
    def compute(self, Z: NDArray) -> NDArray:
        
        raise NotImplementedError()
        

    def gradient(self, out_grad: Tensor, node: Tensor):
        
        raise NotImplementedError()
        


def logsoftmax(a: Tensor) -> Tensor:
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None) -> None:
        self.axes = axes

    def compute(self, Z: NDArray) -> NDArray:
        
        max_z = array_api.max(Z, axis=self.axes, keepdims=True)
        max_z_broadcast = max_z.broadcast_to(Z.shape)
        max_z_reduce = array_api.max(Z, axis=self.axes, keepdims=False)
        return array_api.log(array_api.sum(array_api.exp(Z - max_z_broadcast), axis=self.axes)) + max_z_reduce
        

    def gradient(self, out_grad: Tensor, node: Tensor):
        
        z = node.inputs[0]
        max_z = Tensor(array_api.max(z.realize_cached_data(), axis=self.axes, keepdims=True), 
                      device=z.device, dtype=z.dtype)
        # Broadcast max_z to z.shape for subtraction
        max_z_broadcast = broadcast_to(max_z, z.shape)
        exp_z = exp(z - max_z_broadcast)
        sum_exp_z = summation(exp_z, axes=self.axes)
        
        # Reshape for broadcasting
        if self.axes is not None:
            new_shape = list(z.shape)
            axes = [self.axes] if isinstance(self.axes, int) else self.axes
            for axis in axes:
                new_shape[axis] = 1
            # Realize and create new tensors to ensure they're compact
            sum_exp_z_data = sum_exp_z.realize_cached_data().compact().reshape(tuple(new_shape))
            out_grad_data = out_grad.realize_cached_data().compact().reshape(tuple(new_shape))
            sum_exp_z = Tensor(sum_exp_z_data, device=z.device, dtype=z.dtype)
            out_grad = Tensor(out_grad_data, device=z.device, dtype=z.dtype)
        else:
            new_shape = [1] * len(z.shape)
            sum_exp_z_data = sum_exp_z.realize_cached_data().compact().reshape(tuple(new_shape))
            out_grad_data = out_grad.realize_cached_data().compact().reshape(tuple(new_shape))
            sum_exp_z = Tensor(sum_exp_z_data, device=z.device, dtype=z.dtype)
            out_grad = Tensor(out_grad_data, device=z.device, dtype=z.dtype)
            
        return broadcast_to(out_grad / sum_exp_z, z.shape) * exp_z
        


def logsumexp(a: Tensor, axes: Optional[tuple] = None) -> Tensor:
    return LogSumExp(axes=axes)(a)