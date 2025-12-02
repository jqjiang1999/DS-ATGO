import math
import torch
import torch.nn as nn

from network_model.layers import BN1d_SNN, BN2d_SNN, tdLayer, SpikingNeuron


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = tdLayer(conv3x3(inplanes, planes, stride),
                             BN2d_SNN(planes))
        self.spike1 = SpikingNeuron()
        self.conv2 = tdLayer(conv3x3(planes, planes),
                             BN2d_SNN(planes))
        self.spike2 = SpikingNeuron()
        self.downsample = downsample

    def forward(self, x):
        """ basic ResNet """
        # identity = x
        # out = self.conv1(x)
        # out = self.spike1(out)
        # out = self.conv2(out)
        # if self.downsample is not None:
        #     identity = self.downsample(x)
        # out += identity
        # out = self.spike2(out)

        "SEW-ResNet: From 'Deep Residual Learning in Spiking Neural Networks, NeurIPS 2021'"
        identity = x
        out = self.conv1(x)
        out = self.spike1(out)
        out = self.conv2(out)
        out = self.spike2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity # ADD
        return out


class ResNet19(nn.Module):
    def __init__(self, T, output_size):
        super(ResNet19, self).__init__()
        self.T, self.output_size = T, output_size
        self.planes = [128, 256, 512]
        self.stride = [1, 2, 2]

        self.inplanes = 128
        self.encode = nn.Sequential(
            tdLayer(
                nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False),
                BN2d_SNN(self.inplanes)),
            SpikingNeuron(),
        )

        self.ResNetBlocks = nn.Sequential(
            self._make_layer(BasicBlock, self.planes[0], self.stride[0]),
            self._make_layer(BasicBlock, self.planes[0]),
            self._make_layer(BasicBlock, self.planes[0]),
            self._make_layer(BasicBlock, self.planes[1], self.stride[1]),
            self._make_layer(BasicBlock, self.planes[1]),
            self._make_layer(BasicBlock, self.planes[1]),
            self._make_layer(BasicBlock, self.planes[2], self.stride[2]),
            self._make_layer(BasicBlock, self.planes[2])
        )

        self.global_ap = tdLayer(nn.AdaptiveAvgPool2d((1, 1)))
        self.fc = nn.Sequential(
            tdLayer(nn.Linear(self.planes[-1] * BasicBlock.expansion, 256, bias=False),
                    BN1d_SNN(256),
            ),
            SpikingNeuron(),
            # tdLayer(nn.Dropout(0.5)),
            tdLayer(nn.Linear(256, self.output_size, bias=False))
        )
        self._initialize_weights()


    def _make_layer(self, block, planes, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                tdLayer(conv1x1(self.inplanes, planes * block.expansion, stride),
                        BN2d_SNN(planes * block.expansion)),
                SpikingNeuron() #SEW-ResNet
            )
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        return nn.Sequential(*layers)


    def forward(self, inputs):
        inputs_seqT = inputs.unsqueeze(1).repeat(1, self.T, 1, 1, 1)
        x = self.encode(inputs_seqT.float())
        x = self.ResNetBlocks(x)

        x = self.global_ap(x)
        x = x.view(x.shape[0], x.shape[1], -1)
        x = self.fc(x)
        return x


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)