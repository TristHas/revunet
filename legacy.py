import torch
from torch import nn
from torch.nn import functional as F

from .Imodules import *
from .utils import pad_size

class ConvT(nn.Module):
    """
    """
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=0, bias=True):
        super(ConvT, self).__init__()
        self.conv = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=bias)
        nn.init.kaiming_normal_(self.conv.weight)
        if bias:
            nn.init.constant_(self.conv.bias, 0)
    def forward(self, x):
        return self.conv(x)

class Conv(nn.Module):
    """
    3D convolution w/ MSRA init.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, bias=True):
        super(Conv, self).__init__()
        self.conv = nn.Conv3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=bias)
        nn.init.kaiming_normal_(self.conv.weight)
        if bias:
            nn.init.constant_(self.conv.bias, 0)

    def forward(self, x):
        return self.conv(x)

class UpsampleMod(nn.Module):
    """
    """
    def __init__(self, in_channels, out_channels, up=(1,2,2), mode='bilinear',
                 activation=F.elu, bn_stats=False, invert=False, bn_ieps=0.1):
        super(UpsampleMod, self).__init__()
        # Convolution params.
        ks = (1,1,1)
        st = (1,1,1)
        pad = (0,0,0)
        bias = True
        # Upsampling.
        if mode == 'bilinear':
            self.up = lambda x:F.interpolate(x, scale_factor=up, mode='trilinear', align_corners=False) 
            self.conv = Conv(in_channels, out_channels, ks, st, pad, bias)
        elif mode == 'nearest':
            self.up = lambda x:F.interpolate(x, scale_factor=up, mode='nearest') 
            self.conv = Conv(in_channels, out_channels, ks, st, pad, bias)
        elif mode == 'transpose':
            self.up = ConvT(in_channels, out_channels,
                            kernel_size=up, stride=up, bias=bias)
            self.conv = lambda x: x
        else:
            assert False, "unknown upsampling mode {}".format(mode)
        # BatchNorm and activation.
        self.bn = IBatchNorm3d(out_channels, invert=invert, track_running_stats=bn_stats, ieps=bn_ieps)
        self.activation = activation

    def forward(self, x, skip):
        self.bn.set_invert(False)
        x = self.up(x)
        x = self.conv(x)
        x = self.bn(x + skip)
        return self.activation(x)

class EmbeddingMod(nn.Module):
    """
    """
    def __init__(self, in_channels, out_channels, kernel_size,
                 activation=F.elu):
        super(EmbeddingMod, self).__init__()
        pad = pad_size(kernel_size, 'same')
        self.conv = Conv(in_channels, out_channels, kernel_size,
                         stride=1, padding=pad, bias=True)
        self.activation = activation

    def forward(self, x):
        return self.activation(self.conv(x))
# Number of feature maps.