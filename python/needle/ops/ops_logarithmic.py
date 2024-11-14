from typing import Optional
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

from ..backend_selection import array_api, BACKEND 

class LogSoftmax(TensorOp):
    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        z = Tensor(Z, dtype='float32')
        x = logsumexp(z, axes=1)
        return (z - reshape(x, (x.shape[0], 1))).numpy()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        Z = node.inputs[0]
        return [
            out_grad
            - summation(out_grad, axes=(1,))
            .reshape((out_grad.shape[0], 1))
            .broadcast_to(out_grad.shape)
            * exp(logsoftmax(Z))
        ]
        # raise NotImplementedError()
        ### END YOUR SOLUTION


def logsoftmax(a):
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        if axes is None:
            self.axes = None
        elif isinstance(axes, int):
            self.axes = (axes,)
        else:
            self.axes = axes
                     
    def compute(self, Z):
        ### BEGIN YOUR SOLUTION

        # max_Z = array_api.max(Z, axis=self.axes, keepdims=True)
        brocasted_Z = Z.max(axis=self.axes, keepdims=True)
        brocasted_Z = array_api.broadcast_to(brocasted_Z, Z.shape)
        max_Z = Z.max(axis=self.axes)
        x = array_api.exp(Z - brocasted_Z)
        x = array_api.sum(x, axis=self.axes)
        x = array_api.log(x) + max_Z
        return x
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        in_shape = node.inputs[0].shape

        if self.axes is not None:
            out_axes = node.inputs[0].shape
            shape = [
                out_axes[i] if i not in self.axes else 1 for i in range(len(out_axes))
            ]
            softmax = Exp()(node.inputs[0] - node.reshape(shape).broadcast_to(in_shape))
            return out_grad.reshape(shape).broadcast_to(in_shape) * softmax
        else:
            softmax = Exp()(node.inputs[0] - node.broadcast_to(in_shape))
            return out_grad.broadcast_to(in_shape) * softmax
        ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)