import torch
import torch.nn as nn
import torch.nn.functional as F

from .rev_block import RevBlock
from .utils import pad_size, flatten

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
                self.module = RevBlock(f_conv, g_conv, invert=True)
            else:
                raise Exception(f'Cannot inverse convolution with in_channels {self.in_channels} and out_channels {self.out_channels}')
        else:
            self.module = nn.Conv3d(self.in_channels, self.out_channels, *self.args, **self.kwargs)

    def set_invert(self, invert):
    	self.invert = invert

    def forward(self, x):
        return self.module(x)

class ISkip(nn.Module):
    def __init__(self, channels, skip_module_fn, skip_invert=True, invert=True):
        super().__init__()
        self.set_invert(skip_invert)
        self.skip_invert = skip_invert

        if skip_invert:
            self.module = RevBlock(
                skip_module_fn(channels//2),
                skip_module_fn(channels//2),
                invert = True
            )
        else:
            self.skip = RevAdd(invert)
            self.module = skip_module_fn(channels)
    
    def forward(self, x):
        if self.skip_invert:
            return self.module(x) 
        else:
            x = self.skip.register_skip(x)
            return self.skip(self.module(x))
    
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
            x_ = x.permute(1, 0, 2, 3, 4).contiguous().view(x.size(1), -1)
            mean, std = x_.mean(1).squeeze(), x_.std(1).squeeze()
        out = F.batch_norm( 
            x, None, None, self.weight.abs() + self.ieps, self.bias, 
            True, 0.0, self.eps
        )

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
                        True, 0.0, 0
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
    

class ResidualConvMod(nn.Module):
    def __init__(self, channels, activation=None, bn_ieps=0.1,
                 invert=True, nlayer=3, pad=(1,1,1),
                 ks=(3,3,3), bias=False, bn_stats=False, st=(1,1,1)):
        super().__init__()
        layers = [(
            IBatchNorm3d(channels, track_running_stats=bn_stats, invert=invert, ieps=bn_ieps),
            activation,
            IConv3d(channels, channels, kernel_size=ks, stride=st, padding=pad, bias=bias, invert=invert),
        )
                   for layer in range(nlayer-2)]
        layers = [IConv3d(channels, channels, kernel_size=ks, stride=st, padding=pad, bias=bias, invert=invert)] + flatten(layers)
        self.layers = nn.ModuleList(layers)
        
        self.invert = invert
        self.delete_intermediaries = False
        self.fill_intermediaries = False
        self.intermediaries = []

    def forward(self, x):
        if self.delete_intermediaries:
            self.intermediaries = []

        for i, layer in enumerate(self.layers):
            y = layer(x)
            if self.fill_intermediaries:
                self.intermediaries[i].data.set_(y)
            
            if self.delete_intermediaries:
                self.intermediaries.append(y)
                if i>0:
                    x.data.set_()
                
            x = y
        
        if self.fill_intermediaries:
            del self.intermediaries

        return x


class IConvMod(nn.Module):
    """
        Convolution module.
    """
    def __init__(self, in_channels, out_channels, activation=None, 
                       invert=True, skip_invert=True, bn_ieps=0.1,
                       nlayer=3, ks=(3,3,3), bias=False, bn_stats=False):
        super().__init__()
        st   = (1,1,1)
        pad  = pad_size(ks, 'same')
        self.activation = activation or ILeakyReLU(invert=invert)
        self.skip_invert = skip_invert
        self.set_invert(invert)

        self.conv = IModuleList([nn.Conv3d(in_channels,  out_channels, kernel_size=ks, stride=st, padding=pad, bias=bias)])
        self.bn   = IModuleList([
            IBatchNorm3d(out_channels, track_running_stats=bn_stats, invert=invert, ieps=bn_ieps),
            IBatchNorm3d(out_channels, track_running_stats=bn_stats, invert=invert, ieps=bn_ieps),
        ])

        residual_module_fn = lambda channels: ResidualConvMod(
                 channels=channels, activation=self.activation, 
                 invert=invert, nlayer=nlayer, pad=pad, bn_ieps=bn_ieps,
                 ks=ks, bias=bias, bn_stats=bn_stats, st=st)

        self.skip = ISkip(out_channels, residual_module_fn, skip_invert=skip_invert, invert=invert)

    def forward(self,x):
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
        if self.invert:
            return skip+0
        else:
            return skip
        
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

class ShapePool2D(nn.Module):
    def __init__(self, block_size):
        super().__init__()
        self.block_size = block_size
        self.block_size_sq = block_size*block_size
        
    def inverse(self, input):
        output = input.permute(0, 2, 3, 1)
        input.data.set_()
        (batch_size, d_height, d_width, d_depth) = output.size()
        s_depth = int(d_depth / self.block_size_sq)
        s_width = int(d_width * self.block_size)
        s_height = int(d_height * self.block_size)
        t_1 = output.contiguous().view(batch_size, d_height, d_width, self.block_size_sq, s_depth)
        spl = t_1.split(self.block_size, 3)
        output.data.set_()
        output = torch.cuda.FloatTensor(size=(len(spl), batch_size, d_height, s_width, s_depth), device=input.device)
        for i,t_t in enumerate(spl):
            output[i,:,:,:,:]=t_t.contiguous().view(batch_size, d_height, s_width, s_depth)
        output = output.transpose(0, 1)
        output = output.permute(0, 2, 1, 3, 4).contiguous()
        output = output.view(batch_size, s_height, s_width, s_depth)
        output = output.permute(0, 3, 1, 2)
        return output.contiguous()

        
    def forward(self, input):
        ref = input
        output = input.permute(0, 2, 3, 1)
        (batch_size, s_height, s_width, s_depth) = output.size()
        d_depth = s_depth * self.block_size_sq
        d_height = int(s_height / self.block_size)
        d_width  = int(s_width / self.block_size)
        
        t_1 = output.split(self.block_size, 2)
        output.data.set_()
        output = torch.cuda.FloatTensor(size=(batch_size, d_height, d_width, d_depth), device=input.device)
        
        for i,t_t in enumerate(t_1):
            output[:,i,:,:] = t_t.contiguous().view(batch_size, d_height, d_depth)
        
        output = output.permute(0, 2, 1, 3)
        output = output.permute(0, 3, 1, 2)
        output = output.contiguous()

        if self.training and output.requires_grad:
            handle_ref = [0]
            handle_ref_ = output.register_hook(self.get_variable_backward_hook(ref, output, handle_ref))
            handle_ref[0] = handle_ref_
        ref.data.set_()
        return output
    
    def get_variable_backward_hook(self, x, out, handle_ref):
        def backward_hook(grad):
            x.data.set_(self.inverse(out))
            handle_ref[0].remove()
        return backward_hook

class IConv2d(nn.Module):
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
                f_conv =  nn.Conv2d(self.in_channels//2, self.out_channels//2, *self.args, **self.kwargs)
                g_conv =  nn.Conv2d(self.in_channels//2, self.out_channels//2, *self.args, **self.kwargs)
                self.module = RevBlock(f_conv, g_conv, invert=True)
            else:
                raise Exception(f'Cannot inverse convolution with in_channels {self.in_channels} and out_channels {self.out_channels}')
        else:
            self.module = nn.Conv2d(self.in_channels, self.out_channels, *self.args, **self.kwargs)

    def set_invert(self, invert):
    	self.invert = invert

    def forward(self, x):
        return self.module(x)
            
class IBatchNorm2d(nn.BatchNorm2d):
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
            x_ = x.permute(1,0,2,3).contiguous().view(x.size(1), -1)
            mean, std = x_.mean(1).squeeze(), x_.std(1).squeeze()

        out = F.batch_norm( 
            x, None, None, self.weight.abs() + self.ieps, self.bias, 
            True, 0.0, self.eps
        )

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
                        True, 0.0, 0
                    )
        x.data.set_(x_)
        y.data.set_()

    def get_variable_backward_hook(self, x, output, std, mean, handle_ref):
        def backward_hook(grad):
            self.inverse(output, x, std, mean)
            handle_ref[0].remove()
        return backward_hook
    
