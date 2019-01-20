import torch
import torch.nn as nn

from .rev_block import RevBlock

class Permutation(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.permut = torch.randperm(channels).cuda(0)
        
        self.permut_inv = torch.arange(channels).cuda(0)
        self.permut_inv[self.permut] = torch.arange(channels).cuda(0)
        
    def forward(self, x):
        return x.index_select(1, self.permut)
    
    def inverse(self, y):
        return y.index_select(1, self.permut_inv)


class IConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, *args, invert=True, **kwargs):
        super().__init__()
        self.set_invert(invert)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.args = args
        self.kwargs = kwargs
        self.perm = Permutation(out_channels).cuda(0)
        if self.invert:
            if self.in_channels == self.out_channels:
                assert (self.in_channels % 2) == 0
                f_conv =  nn.Conv3d(self.in_channels//2, self.out_channels//2, *self.args, **self.kwargs)
                g_conv =  nn.Conv3d(self.in_channels//2, self.out_channels//2, *self.args, **self.kwargs)
                self.module = RevBlock(f_conv, g_conv, invert=True)
            else:
                raise Exception(f'Cannot inverse convolution with in_channels {self.in_channels} and out_channels {self.out_channels}')
        else:
            self.module = nn.Conv3d(self.in_channels, self.out_channels, *self.args, **self.kwargs)

    def set_invert(self, invert):
    	self.invert = invert

    def forward(self, x):
            return self.perm(self.module(x))
