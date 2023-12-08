import torch
from torch import nn
import torch.nn.functional as F


def conv3x3x3(in_planes, out_planes, stride=1, kernel=3):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=kernel,
                     stride=stride,
                     padding=(kernel - 1) // 2 if isinstance(kernel,
                                                             int) else (i // 2 for i in kernel),
                     bias=False)


def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None, kernel=3, no_norm=False, gelu_act=False,
                 conv_2plus1d=False,
                 zero_init_last_layer=False,
                 ):
        super().__init__()

        self.no_norm = no_norm

        if conv_2plus1d:
            self.conv1 = nn.Sequential(
                conv3x3x3(in_planes, planes, stride,
                          kernel=(3, 3, 1)),  # spatial
                nn.GroupNorm(8, planes),
                nn.ReLU(inplace=True),
                conv3x3x3(planes, planes, stride, kernel=(1, 1, 3),  # depth
                          ))
        else:
            self.conv1 = conv3x3x3(in_planes, planes, stride, kernel=kernel)
        if not no_norm:
            self.bn1 = nn.GroupNorm(8, planes)
        if gelu_act:
            self.act = nn.GELU()
        else:
            self.act = nn.ReLU(inplace=True)

        if conv_2plus1d:
            self.conv2 = nn.Sequential(
                conv3x3x3(planes, planes, stride, kernel=(3, 3, 1)),  # spatial
                nn.GroupNorm(8, planes),
                nn.ReLU(inplace=True),
                conv3x3x3(planes, planes, stride, kernel=(1, 1, 3),  # depth
                          ))
        else:
            self.conv2 = conv3x3x3(planes, planes, kernel=kernel)
        if not no_norm:
            self.bn2 = nn.GroupNorm(8, planes)
        self.downsample = downsample
        self.stride = stride

        if zero_init_last_layer:
            # resume from a pretrained small model, maintain the initial behavior
            self.bn2.weight.data.zero_()
            self.bn2.bias.data.zero_()

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        if not self.no_norm:
            out = self.bn1(out)
        out = self.act(out)

        out = self.conv2(out)
        if not self.no_norm:
            out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.act(out)

        return out
