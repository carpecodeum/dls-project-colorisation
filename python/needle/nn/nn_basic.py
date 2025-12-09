"""The module.
"""
from typing import Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> list[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> list["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self) -> None:
        self.training = True

    def parameters(self) -> list[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> list["Module"]:
        return _child_modules(self.__dict__)

    def eval(self) -> None:
        self.training = False
        for m in self._children():
            m.training = False

    def train(self) -> None:
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device: Any | None = None, dtype: str = "float32") -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        
        self.weight = Parameter(
            init.kaiming_uniform(in_features, out_features, 
                               device=device, dtype=dtype, requires_grad=True)
        )
        if bias:
            self.bias = Parameter(
                init.kaiming_uniform(out_features, 1,
                                   shape=(out_features,),
                                   device=device, dtype=dtype, requires_grad=True)
            )
        else:
            self.bias = None
        

    def forward(self, X: Tensor) -> Tensor:
        
        out = X @ self.weight
        if self.bias:
            out = out + self.bias.broadcast_to(out.shape)
        return out
        


class Flatten(Module):
    def forward(self, X: Tensor) -> Tensor:
        
        batch_size = X.shape[0]
        # Calculate total size of all dimensions except batch
        total_size = 1
        for dim in X.shape[1:]:
            total_size *= dim
        return X.reshape((batch_size, total_size))
        


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        
        return ops.relu(x)
        

class Sequential(Module):
    def __init__(self, *modules: Module) -> None:
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        
        for module in self.modules:
            x = module(x)
        return x
        


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor) -> Tensor:
        
        batch_size, num_classes = logits.shape
        
        # Compute log(sum(exp(logits))) for each sample
        log_sum_exp = ops.logsumexp(logits, axes=(1,))  # Shape: (batch_size,)
        
        # Convert y to one-hot encoding
        y_one_hot = np.zeros((batch_size, num_classes), dtype=np.float32)
        y_one_hot[np.arange(batch_size), y.numpy().astype(np.int32)] = 1
        y_one_hot = Tensor(y_one_hot, device=logits.device, dtype=logits.dtype)
        
        # Extract logits for correct classes: logits[range(batch_size), y]
        correct_class_logits = (logits * y_one_hot).sum(axes=(1,))  # Shape: (batch_size,)
        
        # Loss = mean(log_sum_exp - correct_class_logits)
        loss = (log_sum_exp - correct_class_logits).sum() / batch_size
        
        return loss
        


class BatchNorm1d(Module):
    def __init__(self, dim: int, eps: float = 1e-5, momentum: float = 0.1, device: Any | None = None, dtype: str = "float32") -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        
        self.device = device
        self.dtype = dtype
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype, requires_grad=True))
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype, requires_grad=True))
        self.running_mean = init.zeros(dim, device=device, dtype=dtype)
        self.running_var = init.ones(dim, device=device, dtype=dtype)
        

    def forward(self, x: Tensor) -> Tensor:
        
        batch_size = x.shape[0]
        
        if self.training:
            # Compute batch statistics
            mean = x.sum(axes=(0,)) / batch_size  # (dim,)
            # Compute variance: E[(x - mean)^2]
            x_centered = x - mean.broadcast_to(x.shape)
            var = (x_centered ** 2).sum(axes=(0,)) / batch_size  # (dim,)
            
            # Update running statistics (detached from computational graph)
            mean_data = mean.realize_cached_data()
            var_data = var.realize_cached_data()
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean_data
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var_data
            
            # Normalize
            std = (var + self.eps) ** 0.5
            x_norm = (x - mean.broadcast_to(x.shape)) / std.broadcast_to(x.shape)
        else:
            # Use running statistics
            from needle import Tensor
            mean = Tensor(self.running_mean, device=self.device, dtype=self.dtype)
            var = Tensor(self.running_var, device=self.device, dtype=self.dtype)
            std = (var + self.eps) ** 0.5
            x_norm = (x - mean.broadcast_to(x.shape)) / std.broadcast_to(x.shape)
        
        # Scale and shift
        return self.weight.broadcast_to(x.shape) * x_norm + self.bias.broadcast_to(x.shape)
        

class BatchNorm2d(BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor):
        # nchw -> nhcw -> nhwc
        s = x.shape
        _x = x.transpose((1, 2)).transpose((2, 3)).reshape((s[0] * s[2] * s[3], s[1]))
        y = super().forward(_x).reshape((s[0], s[2], s[3], s[1]))
        return y.transpose((2,3)).transpose((1,2))


class LayerNorm1d(Module):
    def __init__(self, dim: int, eps: float = 1e-5, device: Any | None = None, dtype: str = "float32") -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        
        self.device = device
        self.dtype = dtype
        self.weight = Parameter(
            init.ones(dim, device=device, dtype=dtype, requires_grad=True)
        )
        self.bias = Parameter(
            init.zeros(dim, device=device, dtype=dtype, requires_grad=True)
        )
        

    def forward(self, x: Tensor) -> Tensor:
        
        original_shape = x.shape
        if len(original_shape) == 1:
            x = x.reshape((1, self.dim))
            reshaped = x
        else:
            leading = 1
            for dim in original_shape[:-1]:
                leading *= dim
            reshaped = x.reshape((leading, self.dim))

        mean = reshaped.sum(axes=(1,)) / self.dim
        mean = mean.reshape((reshaped.shape[0], 1))
        centered = reshaped - mean.broadcast_to(reshaped.shape)

        var = (centered ** 2).sum(axes=(1,)) / self.dim
        std = (var + self.eps) ** 0.5
        std = std.reshape((reshaped.shape[0], 1)).broadcast_to(reshaped.shape)

        normalized = centered / std

        weight = self.weight.reshape((1, self.dim)).broadcast_to(reshaped.shape)
        bias = self.bias.reshape((1, self.dim)).broadcast_to(reshaped.shape)
        out = normalized * weight + bias

        return out.reshape(original_shape)
        


class Dropout(Module):
    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        
        if (not self.training) or self.p == 0.0:
            return x

        keep_prob = 1.0 - self.p
        if keep_prob <= 0.0:
            return init.zeros_like(x)

        mask = init.randb(
            *x.shape, p=keep_prob, device=x.device, dtype=x.dtype, requires_grad=False
        )
        return x * mask / keep_prob
        


class Residual(Module):
    def __init__(self, fn: Module) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        
        return x + self.fn(x)
        
