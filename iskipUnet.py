import torch
from torch import nn
from torch.nn import functional as F

# from . import *
from .legacy import *
from .utils import *
from .Imodules import *


class Bridge(nn.Module):
    def __init__(self, pooling, upsample, activation, nfeatures, invert, skip_invert, bn_ieps=0.1, nlayer=3):
        super().__init__()
        self.layers = nn.ModuleList([
            pooling,
            IConvMod( 
                nfeatures[0],
                nfeatures[1],
                activation=activation,
                nlayer=nlayer,
                invert=invert,
                skip_invert=skip_invert,
                bn_ieps=bn_ieps
            ),
            upsample,
            nn.Conv3d(
                nfeatures[1],
                nfeatures[0],
                kernel_size=(1, 1, 1),
            )
        ])
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

            
class RecursiveUModule(nn.Module):
    def __init__(self, pooling, upsample, activation, nfeatures, invert, skip_invert, skip_U_invert, bn_ieps=0.1, nlayer=3):
        super().__init__()
        
        if len(nfeatures) == 2:
            self.module = Bridge(pooling, upsample, activation, nfeatures, invert, skip_invert, bn_ieps=bn_ieps, nlayer=nlayer)
        
        else:
            next_nfeatures = [x//2 if skip_U_invert else x for x in nfeatures[1:]]
            u_skip_module = ISkip(nfeatures[0], lambda _: RecursiveUModule(pooling, upsample, activation,
                                                              next_nfeatures, invert, skip_invert, skip_U_invert,
                                                              bn_ieps=bn_ieps, nlayer=nlayer),
                                  skip_invert=skip_U_invert,
                                  invert=False
                                 )
            self.module = nn.Sequential(
                pooling,
                IConvMod(
                    nfeatures[0],
                    nfeatures[1],
                    activation=activation,
                    invert=invert,
                    skip_invert=skip_invert,
                    bn_ieps=bn_ieps,
                    nlayer=nlayer,
                ),
                u_skip_module,
                IBatchNorm3d(nfeatures[1], track_running_stats=False, ieps=bn_ieps, invert=invert),
                activation,
                IConvMod(
                    nfeatures[1],
                    nfeatures[1],
                    activation=activation,
                    invert=invert,
                    skip_invert=skip_invert,
                    bn_ieps=bn_ieps,
                    nlayer=nlayer
                ),
                upsample,
                nn.Conv3d(
                    nfeatures[1],
                    nfeatures[0],
                    kernel_size=(1, 1, 1),
                )
            )
            
    def forward(self, x):
            return self.module(x)

class UpsampleMod(nn.Module):
    """
    """
    def __init__(self, up=(1,2,2), mode='bilinear'):
        super().__init__()
        # Upsampling.
        if mode == 'bilinear':
            self.up = lambda x:F.interpolate(x, scale_factor=up, mode='trilinear', align_corners=False) 
        elif mode == 'nearest':
            self.up = lambda x:F.interpolate(x, scale_factor=up, mode='nearest') 
            assert False, "unknown upsampling mode {}".format(mode)

    def forward(self, x):
        x = self.up(x)
        return x

class ISkipUNet(nn.Module):
    """
        
    """
    def __init__(self, out_ch, depth=4, 
                 ks=(3,3,3), embed_ks=(1,5,5), 
                 activation=None, invert=True, skip_invert=True, skip_U_invert=True,
                 bn_ieps=0.1, neg_slope=.5,
                 nfeatures = [16,32,64,128,256,512,1024],
                 upsample='bilinear', pool=(1,2,2), nlayer=3):
        super().__init__()
        self.upsample   = UpsampleMod(mode=upsample)
        self.depth      = depth
        self.activation = ILeakyReLU(negative_slope=neg_slope) if activation is None else activation
        self.pooling    = nn.MaxPool3d(pool)



        self.embed_in = EmbeddingMod(1, nfeatures[0], embed_ks, activation=self.activation)
        
        next_nfeatures = [x//2 if skip_U_invert else x for x in nfeatures[:depth+1]]
        u_skip_module = ISkip(nfeatures[0], lambda _: RecursiveUModule(self.pooling, self.upsample, 
                                                                       self.activation, next_nfeatures,
                                                                       invert, skip_invert, skip_U_invert,
                                                                       nlayer=nlayer, bn_ieps=bn_ieps),
                              skip_invert=skip_U_invert,
                              invert=False)

        self.unet = nn.Sequential(
            IConvMod(
                nfeatures[0],
                nfeatures[0],
                activation=self.activation,
                invert=invert,
                nlayer=nlayer,
                skip_invert=skip_invert,
                bn_ieps=bn_ieps
            ),
            u_skip_module,
            IBatchNorm3d(nfeatures[0] , track_running_stats=False, ieps=bn_ieps, invert=invert),
            self.activation,
        )
        self.embed_out = EmbeddingMod(nfeatures[0], nfeatures[0], embed_ks, activation=self.activation)
        self.output    = nn.Conv3d(nfeatures[0], out_ch, (1,1,1), bias=True)


    def forward(self, x):
        """
        """
        x = self.embed_in(x)
        x = self.unet(x)
        x = self.embed_out(x)
        return self.output(x)

    def set_invert(self, invert):
        """
        """
        self.invert = invert