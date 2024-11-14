"""Optimization module"""
import needle as ndl
import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        ### BEGIN YOUR SOLUTION
        for param in self.params:
            if param.grad is None:
                continue
            grad = param.grad.data + self.weight_decay * param.data
            u_t = self.momentum * self.u.get(param, 0) + (1 - self.momentum) * grad
            self.u[param] = u_t
            param.data -= self.lr * self.u[param]
        ### END YOUR SOLUTION

    def clip_grad_norm(self, max_norm=0.25):
        """
        Clips gradient norm of parameters.
        """
        ### BEGIN YOUR SOLUTION
        ### END YOUR SOLUTION


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):
        ### BEGIN YOUR SOLUTION
        self.t += 1
        for param in self.params:
            if param.grad is None:
                continue
            grad = param.grad.data + self.weight_decay * param.data
            u_t = self.beta1 * self.m.get(param, 0) + (1 - self.beta1) * grad
            self.m[param] = u_t
            v_t = self.beta2 * self.v.get(param, 0) + (1 - self.beta2) * (grad ** 2)
            self.v[param] = v_t
            unbiased_u = self.m[param] / (1 - self.beta1 ** self.t)
            unbiased_v = self.v[param] / (1 - self.beta2 ** self.t)
            x = self.lr * unbiased_u.data / (unbiased_v.data ** 0.5 + self.eps)
            param.data -= x.data

        ### END YOUR SOLUTION