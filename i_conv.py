import torch
import torch.nn as nn

from .rev_block import RevBlock

class IConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, *args, invert=False, add_skip=True, **kwargs):
        super().__init__()
        self.set_invert(invert)
        self.add_skip = add_skip
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.args = args
        self.kwargs = kwargs

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
        if not self.invert and self.add_skip:
            return x + self.module(x)
        else:
            return self.module(x)
