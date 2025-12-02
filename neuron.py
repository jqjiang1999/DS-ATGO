import math
import torch
from surrogate import SG
from base_config import args


T = args.T
class BaseNode(torch.nn.Module):
    def __init__(self, threshold=1.0, surrogate_function=SG()):
        super().__init__()
        self.threshold = threshold
        self.surrogate_function = surrogate_function

        self.v = 0.0
        self.running_threshold = torch.full((T,), float(threshold))
    def forward(self, dv: torch.Tensor, step: int):
        raise NotImplementedError

    def spiking(self, step):
        """ Adaptive Threshold Mechanism """
        v_param = {'v_mean': torch.mean(self.v.clone().detach()),
                   'v_stdev': torch.std(self.v.clone().detach()),
                   'step': step}

        if self.training:
            self.threshold = 1.0 * (v_param['v_mean'] + v_param['v_stdev']) # 1.0 denotes the firing rate control factor $f_c$
            " Moving Average of Training Threshold "
            self.running_threshold[step] = 0.9 * self.running_threshold[step] + 0.1 * self.threshold
        else:
            " Inference Threshold "
            self.threshold = self.running_threshold[step]

        spikes = self.surrogate_function(self.v - self.threshold, v_param)
        self.v = self.v * (1. - spikes)
        return spikes

    def neuron_reset(self):
        self.v = 0.0


class LIFNode(BaseNode):
    def __init__(self, decay=0.2, threshold=1.0):
        super().__init__(threshold, surrogate_function=SG())
        assert isinstance(decay, float) and 0. < decay < 1.
        self.decay = torch.tensor(decay,)

    def forward(self, dv: torch.Tensor, step: int):
        self.v = self.decay * self.v + dv
        return self.spiking(step)

    def extra_repr(self):
        return f'threshold={self.threshold}, tau={self.decay}'


"PLIF: From 'Incorporating Learnable Membrane Time Constant to Enhance Learning of Spiking Neural Networks, ICCV 2021'"
class PLIFNode(BaseNode):
    def __init__(self, init_a=5.0, threshold=1.0):
        super().__init__(threshold, surrogate_function=SG())
        assert isinstance(init_a, float) and init_a > 1.
        init_w = - math.log(init_a - 1)
        self.w = torch.nn.Parameter(torch.tensor(init_w, dtype=torch.float))

    def forward(self, dv: torch.Tensor, step: int):
        self.v = self.w.sigmoid() * self.v + dv
        return self.spiking(step)

    def extra_repr(self):
        return f'threshold={self.threshold}, init_tau={self.w.sigmoid():.2f}'