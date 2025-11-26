from typing import List
from needle.autograd import Tensor
import needle.backend_ndarray.ndarray as ndarray
from needle import ops
import needle.init as init
import numpy as np
from .nn_sequence import Embedding
from .nn_basic import (
    Parameter, 
    Module, 
    ReLU,
    Dropout,
    LayerNorm1d,
    Linear,
    Sequential
)


class MultiHeadAttention(Module):
    """
    The multi-head self attention module.
    """
    def __init__(
        self,
        *,
        dropout = 0.,
        causal = False,
        device = None,
        dtype = "float32",
    ):

        super().__init__()

        self.device = device
        self.dtype = dtype

        self.causal = causal
        self.dropout = Dropout(dropout)

    def create_causal_mask(self, i, j, device):
        """
        return a triangular causal mask.
        Input: i, j: the shape of the mask to be created
        """
        mask = -np.finfo(np.float32).max * np.triu(
            np.ones((1, 1, i, j), dtype=np.float32), j - i + 1)

        return ndarray.array(
            mask, device=device)

    def matmul(self, a, b_transpose):
        """
        batched matrix multiplication;
        Input: a of shape (..., T_q, D), b_transpose of shape (..., D, T_k)
        Output: result of shape (..., T_q, T_k)
        """
        a_shape = (*a.shape, 1)
        a = a.reshape(a_shape)

        b_transpose_shape = (*b_transpose.shape[:-2], 1, *b_transpose.shape[-2:])
        b_transpose = b_transpose.reshape(b_transpose_shape)

        broadcast_shape = list(a_shape)
        broadcast_shape[-1] = b_transpose.shape[-1]
        a = a.broadcast_to(broadcast_shape)

        broadcast_shape = list(b_transpose_shape)
        broadcast_shape[-3] = a_shape[-3]
        b_transpose = b_transpose.broadcast_to(broadcast_shape)

        return (a * b_transpose).sum(len(a.shape) - 2)

    def softmax(self, logit):
        """
        The softmax function; 
        """
        max_val = Tensor(
            logit.realize_cached_data().max(axis=3),
            device=logit.device,
            dtype=logit.dtype,
            requires_grad=False
        )

        max_val = max_val.reshape((*logit.shape[:-1], 1))
        max_val = max_val.broadcast_to(logit.shape)

        probs = ops.exp(logit - max_val)

        denom = probs.sum(axes=3)
        denom = denom.reshape((*logit.shape[:-1], 1))
        denom = denom.broadcast_to(logit.shape)

        return probs / denom

    def forward(
        self,
        q, k, v,
    ):
        """
        The forward function of the MultiHeadAttention activation function.
        Input: three states q, k, v, with shape (batch_size, num_head, seq_len, dim_head)
        Output: the activation output `result` and attention softmax probability `probs` (with dropout applied)
        """
        batch_size, num_head, queries_len, q_dim = q.shape
        _, _, keys_values_len, k_dim = k.shape
        _, _, _, v_dim = v.shape

        assert q_dim == k_dim == v_dim

        result = None
        probs = None

        ### BEGIN YOUR SOLUTION
        scale = 1.0 / np.sqrt(q_dim)

        k_t = ops.transpose(k, axes=(2, 3))
        logits = self.matmul(q, k_t) * scale

        if self.causal:
            mask = self.create_causal_mask(
                queries_len, keys_values_len, device=q.device
            )
            mask_tensor = Tensor(
                mask, device=q.device, dtype=q.dtype, requires_grad=False
            )
            mask_tensor = mask_tensor.broadcast_to(logits.shape)
            logits = logits + mask_tensor

        probs = self.softmax(logits)
        probs = self.dropout(probs)

        result = self.matmul(probs, v)
        ### END YOUR SOLUTION

        return result, probs


class AttentionLayer(Module):

    def __init__(
        self,
        q_features: int,
        num_head: int,
        dim_head: int,
        *,
        k_features: int = None,
        v_features: int = None,
        out_features: int = None,
        dropout = 0.,
        causal = True,
        device = None,
        dtype = "float32",
    ):

        super().__init__()

        self.device = device
        self.dtype = dtype

        if k_features is None:
            k_features = q_features
        if v_features is None:
            v_features = q_features
        if out_features is None:
            out_features = q_features

        self.q_features = q_features
        self.k_features = k_features
        self.v_features = v_features
        self.out_features = out_features

        self.num_head = num_head
        self.dim_head = dim_head

        self.prenorm_q = LayerNorm1d(
            q_features, device=device, dtype=dtype)
        self.prenorm_k = LayerNorm1d(
            k_features, device=device, dtype=dtype)
        self.prenorm_v = LayerNorm1d(
            v_features, device=device, dtype=dtype)

        inner_dim = num_head * dim_head
        
        self.q_projection = Linear(
            q_features, inner_dim, bias=False,
            device=device, dtype=dtype)
        self.k_projection = Linear(
            k_features, inner_dim, bias=False,
            device=device, dtype=dtype)
        self.v_projection = Linear(
            v_features, inner_dim, bias=False,
            device=device, dtype=dtype)

        self.attn = MultiHeadAttention(
            dropout=dropout, causal=causal,
            device=device, dtype=dtype)

        self.out_projection = Linear(
            inner_dim, out_features, bias=False,
            device=device, dtype=dtype)

    def forward(
        self,
        q, k=None, v=None,
    ):
        """
        The forward function of the self-attention layer.
        Input: `q` with shape (batch_size, q_len, q_dim)
               `k` (if not None) with shape (batch_size, kv_len, k_dim)
               `v` (if not None) with shape (batch_size, kv_len, v_dim)
        Output: the output `result` with shape (batch_size, kv_len, out_features)
        """

        if k is None:
            k = q
        if v is None:
            v = q

        batch_size, queries_len, q_dim = q.shape
        _, keys_values_len, k_dim = k.shape
        _, _, v_dim = v.shape

        result = None

        ### BEGIN YOUR SOLUTION
        inner_dim = self.num_head * self.dim_head

        q_flat = q.reshape((batch_size * queries_len, q_dim))
        q_norm = self.prenorm_q(q_flat).reshape((batch_size, queries_len, q_dim))

        k_flat = k.reshape((batch_size * keys_values_len, k_dim))
        k_norm = self.prenorm_k(k_flat).reshape((batch_size, keys_values_len, k_dim))

        v_flat = v.reshape((batch_size * keys_values_len, v_dim))
        v_norm = self.prenorm_v(v_flat).reshape((batch_size, keys_values_len, v_dim))

        q_proj = self.q_projection(
            q_norm.reshape((batch_size * queries_len, q_dim))
        ).reshape((batch_size, queries_len, self.num_head, self.dim_head))
        k_proj = self.k_projection(
            k_norm.reshape((batch_size * keys_values_len, k_dim))
        ).reshape((batch_size, keys_values_len, self.num_head, self.dim_head))
        v_proj = self.v_projection(
            v_norm.reshape((batch_size * keys_values_len, v_dim))
        ).reshape((batch_size, keys_values_len, self.num_head, self.dim_head))

        q_proj = ops.transpose(q_proj, axes=(1, 2))
        k_proj = ops.transpose(k_proj, axes=(1, 2))
        v_proj = ops.transpose(v_proj, axes=(1, 2))

        attn_output, probs = self.attn(q_proj, k_proj, v_proj)
        self.probs = probs

        attn_output = ops.transpose(attn_output, axes=(1, 2))
        attn_output = attn_output.reshape((batch_size, queries_len, inner_dim))

        projected = self.out_projection(
            attn_output.reshape((batch_size * queries_len, inner_dim))
        )
        result = projected.reshape((batch_size, queries_len, self.out_features))
        ### END YOUR SOLUTION

        return result


class TransformerLayer(Module):

    def __init__(
        self,
        q_features: int,
        num_head: int,
        dim_head: int,
        hidden_size: int,
        *,
        dropout = 0.,
        causal = True,
        device = None,
        dtype = "float32",
    ):

        super().__init__()

        self.device = device
        self.dtype = dtype

        ### BEGIN YOUR SOLUTION
        self.attention = AttentionLayer(
            q_features, num_head, dim_head,
            dropout=dropout, causal=causal,
            device=device, dtype=dtype
        )

        self.dropout_attn = Dropout(dropout)
        self.dropout_ff = Dropout(dropout)
        self.dropout_ff_inner = Dropout(dropout)

        self.ff_norm = LayerNorm1d(q_features, device=device, dtype=dtype)
        self.linear1 = Linear(q_features, hidden_size, device=device, dtype=dtype)
        self.linear2 = Linear(hidden_size, q_features, device=device, dtype=dtype)
        self.activation = ReLU()

        self.hidden_size = hidden_size
        self.q_features = q_features
        ### END YOUR SOLUTION

    def forward(
        self,
        x
    ):
        """
        The forward function of a Transformer Layer.
        Input: the hidden states from previous layers `x` with shape (batch_size, seq_len, x_dim)
        Ouput: the hidden states after the Transformer Layer `x` with shape (batch_size, seq_len, x_dim)
        """

        batch_size, seq_len, x_dim = x.shape

        ### BEGIN YOUR SOLUTION
        attn_out = self.attention(x)
        attn_out = self.dropout_attn(attn_out)
        x = x + attn_out

        y = x.reshape((batch_size * seq_len, x_dim))
        y = self.ff_norm(y)
        y = y.reshape((batch_size, seq_len, x_dim))

        y = self.linear1(y.reshape((batch_size * seq_len, x_dim)))
        y = self.activation(y.reshape((batch_size, seq_len, self.hidden_size)))
        y = self.dropout_ff_inner(y)

        y = y.reshape((batch_size * seq_len, self.hidden_size))
        y = self.linear2(y).reshape((batch_size, seq_len, x_dim))
        y = self.dropout_ff(y)

        x = x + y
        ### END YOUR SOLUTION

        return x


class Transformer(Module):

    def __init__(
        self,
        embedding_size: int,
        hidden_size: int,
        num_layers: int, 
        *,
        num_head: int = 8,
        dim_head: int = 32,
        dropout = 0.,
        causal = True,
        device = None,
        dtype = "float32",
        batch_first = False,
        sequence_len = 2048
    ):

        super().__init__()

        self.device = device
        self.dtype = dtype
        self.batch_first = batch_first

        ### BEGIN YOUR SOLUTION
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.sequence_len = sequence_len

        self.positional_embedding = Embedding(
            sequence_len, embedding_size, device=device, dtype=dtype
        )

        self.layers = []
        for _ in range(num_layers):
            self.layers.append(
                TransformerLayer(
                    embedding_size, num_head, dim_head, hidden_size,
                    dropout=dropout, causal=causal,
                    device=device, dtype=dtype
                )
            )
        ### END YOUR SOLUTION

    def forward(
        self,
        x, h=None
    ):

        if not self.batch_first:
            x = ops.transpose(x, axes=(0, 1))

        ### BEGIN YOUR SOLUTION
        batch_size, seq_len, embed_dim = x.shape
        if seq_len > self.sequence_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds maximum {self.sequence_len}"
            )

        positions = np.arange(seq_len, dtype=np.int32).reshape((seq_len, 1))
        positions = np.broadcast_to(positions, (seq_len, batch_size))
        pos_tensor = Tensor(
            positions,
            device=x.device,
            dtype="float32",
            requires_grad=False,
        )

        pos_emb = self.positional_embedding(pos_tensor)
        pos_emb = ops.transpose(pos_emb, axes=(0, 1))

        x = x + pos_emb

        for layer in self.layers:
            x = layer(x)
        ### END YOUR SOLUTION

        if not self.batch_first:
            x = ops.transpose(x, axes=(0, 1))

        return x, init.zeros_like(x)