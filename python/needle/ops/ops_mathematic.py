"""Operator implementations."""

from numbers import Number
from typing import Optional, List, Tuple, Union

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks
import numpy as array_api

import sys
sys.path.append('./python')
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
        ### BEGIN YOUR SOLUTION
        return a ** b
        ### END YOUR SOLUTION
        
    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a, b = node.inputs
        grad_a = out_grad * b * a ** (b - 1)
        grad_b = out_grad * node * log(a)
        return grad_a, grad_b
        ### END YOUR SOLUTION


def power(a, b):
    return EWisePow()(a, b)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return a ** self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        n = node.inputs[0]
        return out_grad * self.scalar * (n ** (self.scalar-1))
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a / b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION        
        a, b = node.inputs[0], node.inputs[1]
        grad_a = out_grad / b
        grad_b = out_grad * (-a) / (b * b)
        return grad_a, grad_b
        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a / self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad / self.scalar
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple]):
        self.axes = axes if axes else (-1, -2)

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        # return array_api.swapaxes(a, self.axes[0], self.axes[1])
        order = list(range(len(a.shape)))
        if self.axes:
            order[self.axes[0]], order[self.axes[1]] = order[self.axes[1]], order[self.axes[0]]
        else:
            order = order[ :-2] + [order[-1], order[-2]]
        return a.permute(tuple(order))
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return transpose(out_grad, self.axes)
        ### END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.reshape(a, self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        shape0 = node.inputs[0].shape
        return reshape(out_grad, shape0)
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


class Permute(TensorOp):
    def __init__(self, axes):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.permute(self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return permute(out_grad, self.axes)
        ### END YOUR SOLUTION

def permute(a, axes):
    return Permute(axes)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.broadcast_to(a, self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        shape0 = node.inputs[0].shape
        shape = [1] * (len(self.shape) - len(shape0)) + list(shape0)
        reduced_axes = []
        if not shape or len(shape) == 1:
            reduced_axes = list(range(len(self.shape)))
        else:
            for i, (input_axis, broadcasted_axis) in enumerate(zip(shape, self.shape)):
                if input_axis != broadcasted_axis:
                    reduced_axes.append(i)
        return reshape(summation(out_grad, tuple(reduced_axes)), shape0)
        ### END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        if axes is None:
            self.axes = None
        elif isinstance(axes, int):
            self.axes = (axes,)
        else:
            self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        if self.axes is None:
            return a.sum(axis = None)
        elif isinstance(self.axes, int) or (isinstance(self.axes, (list, tuple)) and len(self.axes) == 1):
            return a.sum(self.axes)
        else:
            for axis in reversed(sorted(self.axes)):
                a = a.sum(axis = axis)
            return a
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        shape = node.inputs[0].shape
        axes = self.axes if self.axes else list(range(len(shape)))
        aug_axes = [1 if i in axes else s for i, s in enumerate(shape)]
        return broadcast_to(reshape(out_grad, aug_axes), shape)
        ### END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return array_api.matmul(a, b)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a, b = node.inputs
        if len(a.shape) < len(b.shape):
            axes = tuple(range(len(b.shape) - len(a.shape)))
            return summation(matmul(out_grad, transpose(b)), axes), matmul(transpose(a), out_grad)
        elif len(a.shape) > len(b.shape):
            axes = tuple(range(len(a.shape) - len(b.shape)))
            return matmul(out_grad, transpose(b)), summation(matmul(transpose(a), out_grad), axes)
        return matmul(out_grad, transpose(b)), matmul(transpose(a), out_grad)
        ### END YOUR SOLUTION


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return -a
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return negate(out_grad)
        ### END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.log(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return divide(out_grad, node.inputs[0])
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.exp(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return multiply(out_grad, node)
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.maximum(a, 0)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raw = node.realize_cached_data().numpy()
        raw[raw > 0] = 1
        return out_grad * Tensor(raw, device=out_grad.device, dtype=out_grad.dtype)
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)


class Tanh(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.tanh(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        x = init.ones(*out_grad.shape, device=out_grad.device, requires_grad=False)
        y = power_scalar(tanh(node.inputs[0]), 2)
        return out_grad * (x - y)
        ### END YOUR SOLUTION


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
        ### BEGIN YOUR SOLUTION
        shape = list(args[0].shape)
        for x in args:
            assert x.shape != shape, 'stack dimensions mismatch'
        new_shape = shape[:self.axis] + [len(args)] + shape[self.axis:]
        out = array_api.empty(new_shape, args[0].dtype, args[0].device)
        for i, tensor in enumerate(args):
            idx = (slice(None),) * self.axis + (i,) + (slice(None),) * (len(shape) - self.axis)
            out[idx] = tensor.compact()
        return out
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return split(out_grad, axis=self.axis)
        ### END YOUR SOLUTION


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
        ### BEGIN YOUR SOLUTION
        shape = list(A.shape)
        new_shape = shape[:self.axis] + shape[self.axis+1:]
        out = []
        for i in range(A.shape[self.axis]):
            idx = (slice(None),) * self.axis + (i,) + (slice(None),) * (len(new_shape) - self.axis)
            out.append(array_api.reshape(A[idx].compact(),new_shape))
        return tuple(out)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return stack(out_grad, self.axis)
        ### END YOUR SOLUTION


def split(a, axis):
    return Split(axis)(a)


class Flip(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.flip(self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return flip(out_grad, self.axes)
        ### END YOUR SOLUTION


def flip(a, axes):
    return Flip(axes)(a)


class Dilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        if self.dilation:
            stride = self.dilation + 1
            dilated_shape = tuple(
                size * stride if axis in self.axes else size
                for axis, size in enumerate(a.shape)
            )
            out = array_api.full(dilated_shape, fill_value=0.0, device=a.device)
            idx = tuple(
                slice(None, None, stride) if axis in self.axes else slice(None)
                for axis in range(a.ndim)
            )
            out[idx] = a
            return out
        return a
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return undilate(out_grad, self.axes, self.dilation)
        ### END YOUR SOLUTION


def dilate(a, axes, dilation):
    return Dilate(axes, dilation)(a)


class UnDilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        if self.dilation:
            stride = self.dilation + 1
            undilated_shape = tuple(
                size // stride if axis in self.axes else size
                for axis, size in enumerate(a.shape)
            )
            idx = tuple(
                slice(None, None, stride) if axis in self.axes else slice(None)
                for axis in range(a.ndim)
            )
            out = a[idx]
            return out
        return a
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return dilate(out_grad, self.axes, self.dilation)
        ### END YOUR SOLUTION


def undilate(a, axes, dilation):
    return UnDilate(axes, dilation)(a)


class Conv(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.stride = stride
        self.padding = padding

    def compute(self, A, B):
        ### BEGIN YOUR SOLUTION
        A = A.compact()
        B = B.compact()
        A = A.pad(((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)))

        N, H, W, C_in = A.shape
        K, _, _, C_out = B.shape
        Ns, Hs, Ws, Cs = A.strides
        inner_dim = K * K * C_in
        _H = (H - K) // self.stride + 1
        _W = (W - K) // self.stride + 1
        out_shape = (N, _H, _H, C_out)
        outer_dim = N * _H * _W
        A = A.as_strided(shape=(N, _H, _W, K, K, C_in), 
                    strides=(Ns, (Hs * self.stride), (Ws * self.stride), Hs, Ws, Cs)). \
                    compact(). \
                    reshape((outer_dim, inner_dim))
        out = A @ (B.reshape((K * K * C_in, C_out)))
        return out.reshape(out_shape)
        ### END YOUR SOLUTION


    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # out_grad: (N, H', W', C_out) == (N, H+2P-K+1, W+2P-K+1, C_out)
        A = node.inputs[0] # (N, H, W, C_in)
        B = node.inputs[1] # (K, K, C_in, C_out)
        N, H, W, C_in  = A.shape
        K, _, _, C_out = B.shape 

        # if stride > 1, dilate out_grad with stride-1
        out_grad = dilate(out_grad, (1, 2), self.stride - 1)
        # W_T: (C_out, C_in, K, K) 
        B_T = transpose(flip(B, axes=(0, 1)), (2, 3))
        # do conv(dilate(out_grad), W_T)
        # X_grad has the same height & width as X
        # H' + 2*newP - K + 1 = H
        # (H + 2P + 1) + 2*newP - K + 1 = H
        # newP = K - 1 - P 
        A_grad = conv(out_grad, B_T, padding=K-1-self.padding)

        # X_permute: (C_in, H, W, N)
        A_permute = A.permute((3, 1, 2, 0))
        # A_permute = transpose(A, (0, 3))
        # out_grad_permute: (H', W', N, C_out)
        out_grad_permute = out_grad.permute((1, 2, 0, 3))
        # out_grad_permute = transpose(transpose(out_grad, (0, 1)), (1, 2))
        # do conv(X_permute, out_grad_permute), we can take H' as new kernel size
        # W_grad has the same height & width as X
        # H + 2*newP - H' + 1 = K
        # H + 2*newP - (H + 2P - K + 1) + 1 = K
        # newP = P
        B_grad = conv(A_permute, out_grad_permute, padding=self.padding)
        B_grad = B_grad.permute((1, 2, 0, 3))
        # B_grad = transpose(transpose(B_grad, (0, 1)), (1, 2))
        
        return A_grad, B_grad 


def conv(a, b, stride=1, padding=1):
    return Conv(stride, padding)(a, b)


