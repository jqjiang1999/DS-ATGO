import math
import torch
import numpy as np
from torch import nn


def heaviside(x: torch.Tensor):
    return (x >= 0.).float()


class ActFun(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, width):
        ctx.save_for_backward(x)
        ctx.width = width
        return heaviside(x)

    @staticmethod
    def backward(ctx, grad_output):
        width, input = ctx.width, ctx.saved_tensors[0]
        s_x = 1.0 / width * (torch.abs(input) < width / 2.0)
        grad_input = s_x * grad_output.clone()
        return grad_input, None


class SG(nn.Module):
    def __init__(self, init_Vth=1.0, init_width=1.0):
        super().__init__()
        self.init_Vth= init_Vth
        self.init_width = init_width
        self.dynamic_width = None

    def forward(self, x: torch.Tensor, v_param: dict):
        """ Threshold-driven Gradient Optimization """
        if self.training:
            v_stdev = v_param['v_stdev']
            if v_stdev <= self.init_Vth:
                self.dynamic_width = self.init_width * (1. - torch.tanh((self.init_Vth - v_stdev)))
            elif v_stdev > self.init_Vth:
                self.dynamic_width = self.init_width * (1. + torch.tanh((v_stdev - self.init_Vth)))
            return ActFun.apply(x, self.dynamic_width)
        else:
            return heaviside(x)

    def extra_repr(self):
        return f'type=rectangular, width={self.init_width}'
