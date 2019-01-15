import torch
import torch.nn as nn
import torch.nn.functional as F

from rev_block import RevBlock
from utils import pad_size

class IBatchNorm3d(nn.BatchNorm3d):
    """
    """
    def __init__(self, *args, ieps=0, invert=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.ieps = ieps
        self.invert = invert
        
    def set_invert(self, invert):
        self.invert = invert
        
    def forward(self, x):
        if self.invert:
            return self.i_forward(x)
        else:
            return super().forward(x)

    def i_forward(self, x):
        with torch.no_grad():
            x_ = x.view(x.size(0), x.size(1), -1)
            mean, std = x_.mean(2).squeeze(), x_.std(2).squeeze()
            
        out = super().forward(x)
        
        if self.training:
            handle_ref = [0]
            handle_ref_ = out.register_hook(self.get_variable_backward_hook(x, out, std, mean, handle_ref))
            handle_ref[0] = handle_ref_
        x.data.set_()
        return out
        
    def inverse(self, y, x, std, mean):
        with torch.no_grad():
            x_ =  F.batch_norm(
                        y, None, None, std, mean, 
                        True, 0.0, self.ieps
                    )
        x.data.set_(x_)
        y.data.set_()

    def get_variable_backward_hook(self, x, output, std, mean, handle_ref):
        def backward_hook(grad):
            self.inverse(output, x, std, mean)
            handle_ref[0].remove()
        return backward_hook

class ISequential(nn.Sequential):
    def set_invert(self, invert):
        self.invert = invert
        for m in self.children():
            if hasattr(m, "set_invert"):
                m.set_invert(invert)

class IModuleList(nn.ModuleList):
    def set_invert(self, invert):
        self.invert = invert
        for m in self.children():
            if hasattr(m, "set_invert"):
                m.set_invert(invert)
                
class IUpsample(nn.Upsample):
    def __init__(self, *args, invert=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.invert = invert
        
    def set_invert(self, invert):
        self.invert = invert
        
    def forward(self, x):
        if self.invert:
            return self.i_forward(x)
        else:
            return super().forward(x)
        
    def i_forward(self, x):
        y=super().forward(x)
        if self.mode == 'nearest':

            handle_ref = []
            handle_ref.append(
                y.register_hook(self.get_variable_backward_hook(x, y, handle_ref))
            )
            x.data.set_()

        return y
    
    def inverse(self, x, y):
        with torch.no_grad():
            slices = [slice(None), slice(None)] + [slice(None, None, scale_factor) for _ in y.size()[2:]]
            x_ = y[slices]
        x.data.set_(x_)
        y.data.set_()
            
    def get_variable_backward_hook(self, x, y, handle_ref):
        def backward_hook(grad):
            self.inverse(x, y)
            handle_ref[0].remove()
        return backward_hook
    
class ILeakyReLU(nn.LeakyReLU):
    def __init__(self, *args, invert=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.invert = invert
        
    def set_invert(self, invert):
        self.invert = invert
                
    def forward(self, x):
        if self.invert:
            return self.i_forward(x)
        else:
            return super().forward(x)

    def i_forward(self, x):
        y=super().forward(x)
        if self.training:
            handle_ref = [0]
            handle_ref_ = y.register_hook(self.get_variable_backward_hook(x, y, handle_ref))
            handle_ref[0] = handle_ref_
        x.data.set_()
        return y
        
    def inverse(self, x, y):
        with torch.no_grad():
            x_ = F.leaky_relu(y, 1/self.negative_slope, self.inplace)
        x.data.set_(x_)
        y.data.set_()

    def get_variable_backward_hook(self, x, y, handle_ref):
        def backward_hook(grad):
            self.inverse(x, y)
            handle_ref[0].remove()
        return backward_hook
    

class IConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, *args, invert=True, **kwargs):
        super().__init__()
        self.invert = invert
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.args = args
        self.kwargs = kwargs
        
        f_conv =  nn.Conv3d(self.in_channels//2, self.out_channels//2, *self.args, **self.kwargs)
        g_conv =  nn.Conv3d(self.in_channels//2, self.out_channels//2, *self.args, **self.kwargs)
        self.inv_module = RevBlock(f_conv, g_conv)
        
        self.noinv_module = nn.Conv3d(self.in_channels, self.out_channels, *self.args, **self.kwargs)
        self.set_module()

    def set_module(self):
        if self.invert:
            if self.in_channels == self.out_channels:
                assert (self.in_channels % 2) == 0
                self.module =self.inv_module

            else:
                raise Exception(f'Cannot inverse convolution with in_channels {self.in_channels} and out_channels {self.out_channels}')
        else:
            self.module = self.noinv_module

    def set_invert(self, invert):
        if invert != self.invert:
            print(f"WARNING: Reinitialization of IConv3d from {self.invert} to {invert}")
            self.invert = invert
            self.set_module()    

    def forward(self, x):
        return self.module(x)
    
class IConvMod(nn.Module):
    """
        Convolution module.
    """
    def __init__(self, in_channels, out_channels, activation, invert=True,
                       nlayer=3, ks=(3,3,3), bias=False, bn_stats=False):
        super().__init__()
        st   = (1,1,1)
        pad  = pad_size(ks, 'same')
        conv = [nn.Conv3d(in_channels,  out_channels, kernel_size=ks, stride=st, padding=pad, bias=bias)]
        bn   = [IBatchNorm3d(out_channels, track_running_stats=bn_stats)]
        for i in range(nlayer-1):
            conv.append(IConv3d(out_channels, out_channels, kernel_size=ks, stride=st, padding=pad, bias=bias))
            bn.append(IBatchNorm3d(out_channels, track_running_stats=bn_stats))
        # Activation function.
        self.activation = activation
        self.conv = IModuleList(conv)
        self.bn = IModuleList(bn)
        #self.set_invert(invert)
        
    def forward(self, x):
        return self._forward_(x)
        
    def _forward(self, x):
        for i,(conv,bn) in enumerate(zip(self.conv, self.bn)):
            x = conv(x)
            if i == 0:
                skip = x + 0
            if i == len(self.conv)-1:
                x = x+skip
            x = self.activation(bn(x))
        return x

    def _forward_(self, x):
        conv, bn = self.conv[0], self.bn[0]
        x = conv(x)
        x = bn(x)
        x = self.activation(x)
        #x = self.activation(bn(conv(x)))
        skip = x + 0
        for i,(conv,bn) in enumerate(zip(self.conv[1:], self.bn[1:])):
            x = conv(x)
            if i == len(self.conv[1:])-1:
                x = x+skip
            x = self.activation(bn(x))
        return x

    def set_invert(self, invert, layers=None):
        self.invert=invert
        if layers is None: 
            layers = set(map(lambda x:type(x), self.modules()))-{type(self)}
        else:
            layers = set(layers)
        for m in self.modules():
            if type(m) in layers and hasattr(m, "set_invert"):
                m.set_invert(invert)
