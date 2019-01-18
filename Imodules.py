import torch
import torch.nn as nn
import torch.nn.functional as F

from .rev_block import RevBlock
from .utils import pad_size

class IConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, *args, invert=True, **kwargs):
        super().__init__()
        self.set_invert(invert)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.args = args
        self.kwargs = kwargs

        if self.invert:
            if self.in_channels == self.out_channels:
                assert (self.in_channels % 2) == 0
                f_conv =  nn.Conv3d(self.in_channels//2, self.out_channels//2, *self.args, **self.kwargs)
                g_conv =  nn.Conv3d(self.in_channels//2, self.out_channels//2, *self.args, **self.kwargs)
                self.module = RevBlock(f_conv, g_conv)
            else:
                raise Exception(f'Cannot inverse convolution with in_channels {self.in_channels} and out_channels {self.out_channels}')
        else:
            self.module = nn.Conv3d(self.in_channels, self.out_channels, *self.args, **self.kwargs)

    def set_invert(self, invert):
    	self.invert = invert

    def forward(self, x):
        return self.module(x)

class ISkip(nn.Module):
    def __init__(self, channels, skip_module_fn, invert=True):
        super().__init__()
        self.set_invert(invert)
        if self.invert:
            self.module = RevBlock(
                skip_module_fn(channels//2),
                skip_module_fn(channels//2),
            )
        else:
            self.module = skip_module_fn(channels)
    
    def forward(self, x):
        if self.invert:
            return self.module(x) 
        else:
            return x + self.module(x)
    
    def set_invert(self, invert):
        self.invert = invert

            
class IBatchNorm3d(nn.BatchNorm3d):
    """
    """
    def __init__(self, *args, ieps=0, invert=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.ieps = ieps
        self.set_invert(invert)
        
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
        if self.training and out.requires_grad:
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
    

class ILeakyReLU(nn.LeakyReLU):
    def __init__(self, *args, invert=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_invert(invert)
        
    def set_invert(self, invert):
        self.invert = invert
                
    def forward(self, x):
        if self.invert:
            return self.i_forward(x)
        else:
            return super().forward(x)

    def i_forward(self, x):
        y=super().forward(x)
        if self.training and y.requires_grad:
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

class ISequential(nn.Sequential):
    def set_invert(self, invert):
        self.invert = invert
        for m in self.children():
            if hasattr(m, "set_invert"):
                m.set_invert(invert)


class IModuleList(nn.ModuleList):
    def set_invert(self, invert):
        return None
        self.invert = invert
        for m in self.children():
            if hasattr(m, "set_invert"):
                m.set_invert(invert)


class IUpsample(nn.Upsample):
    def __init__(self, *args, invert=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_invert( invert)
        
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
            if self.training and y.requires_grad:
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
    

class IConvMod(nn.Module):
    """
        Convolution module.
    """
    def __init__(self, in_channels, out_channels, activation=None, 
                       invert=True, skip_invert=True,
                       nlayer=3, ks=(3,3,3), bias=False, bn_stats=False):
        super().__init__()
        st   = (1,1,1)
        pad  = pad_size(ks, 'same')
        activation = activation or ILeakyReLU()
        self.activation = activation
        self.skip_invert = skip_invert
        self.skip = RevAdd(invert)
        self.set_invert(invert)

        if not skip_invert:
            conv = [nn.Conv3d(in_channels,  out_channels, kernel_size=ks, stride=st, padding=pad, bias=bias)]
            bn   = [IBatchNorm3d(out_channels, track_running_stats=bn_stats)]

                    
            for i in range(nlayer-1):
                conv.append(IConv3d(out_channels, out_channels, kernel_size=ks, 
                                    stride=st, padding=pad, bias=bias, invert=invert))
                bn.append(IBatchNorm3d(out_channels, track_running_stats=bn_stats))
            # Activation function.
            self.conv = IModuleList(conv)
            self.bn = IModuleList(bn)
        
        if skip_invert:
            self.conv = IModuleList([nn.Conv3d(in_channels,  out_channels, kernel_size=ks, stride=st, padding=pad, bias=bias)])
            self.bn   = IModuleList([
                IBatchNorm3d(out_channels, track_running_stats=bn_stats),
                IBatchNorm3d(out_channels, track_running_stats=bn_stats),
            ])

            class SkipModule(nn.Module):
                def __init__(self, channels):
                    super().__init__()
                    self.conv = IModuleList([nn.Conv3d(channels,  channels, kernel_size=ks, stride=st, padding=pad, bias=bias)])
                    self.bn = IModuleList()
                    self.activation = activation
                    for i in range(nlayer-2):
                        self.conv.append(IConv3d(channels, channels, kernel_size=ks, 
                                                 stride=st, padding=pad, bias=bias, invert=invert))
                        self.bn.append(IBatchNorm3d(channels, track_running_stats=bn_stats))

                def forward(self, x):
                    for conv, bn in zip(self.conv[:-1], self.bn):
                        x = self.activation(bn(conv(x)))

                    return self.conv[-1](x)
                
            self.skip = ISkip(out_channels, SkipModule, invert=True)


    def forward(self, x):
        if self.skip_invert:
            return self._forward_iskip(x)
        return self._forward(x)
        
    def _forward(self, x):
        for i,(conv,bn) in enumerate(zip(self.conv, self.bn)):
            x = conv(x)
            if i==0:
                
                x = self.skip.register_skip(x)
            if i == len(self.conv)-1:
                x = self.skip(x)
            x = self.activation(bn(x))
        return x

    def _forward_iskip(self,x):
        conv, bn = self.conv[0], self.bn[0]
        x = conv(x)
        x = bn(x)
        x = self.activation(x)
        x = self.skip(x)
        return self.activation(self.bn[-1](x))   

    def set_invert(self, invert, layers=None):
        self.invert=invert


class RevAdd(nn.Module):
    def __init__(self, invert=True):
        super().__init__()
        self.set_invert(invert)
        
    def set_invert(self, invert):
        self.invert=invert
        
    def register_skip(self, skip):
        self.skip = skip
        if True:#self.invert:
            return skip+0
        #else:
        #    return skip
        
    def forward(self, x):
        out = x + self.skip
        if self.invert:
            if self.training and out.requires_grad:
                handle_ref = [0]
                handle_ref_ = out.register_hook(self.get_variable_backward_hook(x, out, handle_ref))
                handle_ref[0] = handle_ref_
            x.data.set_()
        return out
            
    def inverse(self, x, out):
        with torch.no_grad():
            x_ = out - self.skip
        x.data.set_(x_)
        out.data.set_()

    def get_variable_backward_hook(self, x, out, handle_ref):
        def backward_hook(grad):
            self.inverse(x, out)
            handle_ref[0].remove()
        return backward_hook
    

class IBroadcast(nn.Module):
    def __init__(self, in_channels, out_channels, invert=True):
        if (out_channels  % in_channels) != 0:
            raise Exception(f'Cannot broadcast {in_channels} in_channels to {out_channels} out_channels')
        super().__init__()
        self.set_invert( invert)
        
        self.in_channels = in_channels
        self.repeat_count = out_channels // in_channels
    
    def set_invert(self, invert):
        self.invert = invert

    def forward(self, x):
        out = x.repeat(1, self.repeat_count, 1, 1, 1)
        if self.invert:
            if self.training and out.requires_grad:
                handle_ref = []
                handle_ref.append(
                    out.register_hook(self.get_variable_backward_hook(x, out, handle_ref))
                )
            x.data.set_()
        return out
        
    def inverse(self, y, x):
        x_ = y.narrow(1, 0, self.in_channels)
        x.data.set_(x_)
        y.data.set_()

    def get_variable_backward_hook(self, x, output, handle_ref):
        def backward_hook(grad):
            self.inverse(output, x)
            handle_ref[0].remove()
        return backward_hook
    

class ISkip(nn.Module):
    def __init__(self, channels, skip_module_fn, invert=True):
        super().__init__()
        self.set_invert( invert)
        if self.invert:
            self.module = RevBlock(
                skip_module_fn(channels//2),
                skip_module_fn(channels//2),
            )
            self.module.set_invert(True)
        else:
            self.module = skip_module_fn(channels)
    
    def forward(self, x):
        if self.invert:
            return self.module(x) 
        else:
            return x + self.module(x)
    
    def set_invert(self, invert):
        self.invert=invert