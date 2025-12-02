import torch
from torch import nn

from neuron import LIFNode, PLIFNode


class BN1d_SNN(nn.Module):
    def __init__(self, num_features, alpha=1.0, Vth=1.0):
        super().__init__()
        self.alpha, self.Vth = alpha, Vth
        self.tdBN = nn.BatchNorm1d(num_features)
        torch.nn.init.constant_(self.tdBN.weight, alpha * Vth)

    def forward(self, input):
        y = input.transpose(1, 2).contiguous()  # B T C -> B C T
        y = self.tdBN(y)
        y = y.contiguous().transpose(1, 2)  # B C T -> B T C
        return y

    def extra_repr(self):
        return f'alpha={self.alpha}, threshold={self.Vth}'


"tdBN: From 'Going Deeper With Directly-Trained Larger Spiking Neural Networks, AAAI 2021'"
class BN2d_SNN(nn.Module):
    def __init__(self, num_features, alpha=1.0, Vth=1.0):
        super().__init__()
        self.alpha, self.Vth = alpha, Vth
        self.tdBN = nn.BatchNorm3d(num_features)
        torch.nn.init.constant_(self.tdBN.weight, alpha * Vth)

    def forward(self, input):
        # This code is form spikingjelly https://github.com/fangwei123456/spikingjelly
        y = input.transpose(1, 2).contiguous()  # B T C H W -> B C T H W
        y = self.tdBN(y)
        y = y.contiguous().transpose(1, 2)  # B C T H W -> B T C H W
        return y

    def extra_repr(self):
        return f'alpha={self.alpha}, threshold={self.Vth}'


class tdLayer(nn.Module):
    def __init__(self, layer, bn=None):
        super().__init__()
        self.layer = layer
        self.bn = bn

    def forward(self, x: torch.Tensor):
        x_shape = [x.shape[0], x.shape[1]]
        x_ = self.layer(x.flatten(0, 1).contiguous())
        x_shape.extend(x_.shape[1:])
        x_ = x_.view(x_shape)

        if self.bn is not None:
            x_ = self.bn(x_)
        return x_


class SpikingNeuron(nn.Module):
    def __init__(self, neuron_type='PLIF'):
        super().__init__()

        self.neuron = PLIFNode(init_a=5.0, threshold=1.0) if neuron_type == 'PLIF' \
            else LIFNode(decay=0.2, threshold=1.0)

    def forward(self, x: torch.Tensor):
        time_steps = x.size(1)
        spikes = torch.zeros(x.shape, device=x.device)
        for step in range(time_steps):
            spikes[:, step, ...] = self.neuron(x[:, step, ...], step)
        return spikes