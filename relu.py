"""Lucid's optional ReLU gradient warm-up, ported to PyTorch.

Copyright 2018 The Lucid Authors. Licensed under Apache-2.0;
see LICENSES/Lucid.txt and the original implementation:
https://github.com/tensorflow/lucid/blob/master/lucid/misc/redirected_relu_grad.py
"""
import torch
from torch.nn import functional as F


class ReLUWarmup(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return F.relu(x)

    @staticmethod
    def backward(ctx, gradient):
        (x,) = ctx.saved_tensors
        ordinary = gradient.masked_fill(x < 0, 0)
        # Only redirect when the entire ordinary gradient is blocked, per image.
        active = ordinary.flatten(1).ne(0).any(dim=1)[:, None, None, None]
        # Under gradient descent, negative gradients push negative inputs up.
        redirected = gradient.masked_fill((x < 0) & (gradient > 0), 0)
        return torch.where(active, ordinary, redirected)
