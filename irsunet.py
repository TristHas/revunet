import torch
from torch import nn
from torch.nn import functional as F

#from . import *
from .legacy import *
from .utils import *
from .Imodules import *

class IRSUNet(nn.Module):
    """
        
    """
    def __init__(self, out_ch, depth=4, 
                 ks=(3,3,3), embed_ks=(1,5,5), 
                 activation=None, invert=True, skip_invert=True,
                 nfeatures = [24,32,48,72,104,144],
                 upsample='bilinear', pool=(1,2,2)):
        super(IRSUNet, self).__init__()
        activation      = ILeakyReLU(invert=invert) if activation is None else activation
        self.upsample   = upsample
        self.depth      = depth
        self.activation = activation
        self.pooling    = nn.MaxPool3d(pool)

        # Contracting Path
        self.embed_in = EmbeddingMod(1, nfeatures[0], embed_ks, activation=activation)
        self.contract = IModuleList([IConvMod(nfeatures[0], nfeatures[0], 
                                              invert=invert, skip_invert=skip_invert,
                                              activation=activation, ks=ks)])
        for d in range(depth):
            self.contract.append(IConvMod(nfeatures[d], nfeatures[d+1], 
                                          invert=invert, skip_invert=skip_invert,
                                          activation=activation, ks=ks))
        
        # Expanding Path
        self.expand, self.upsamp =IModuleList(), IModuleList() 
        for d in reversed(range(depth)):
            self.upsamp.append(UpsampleMod(nfeatures[d+1], nfeatures[d], 
                                           up=pool, mode=self.upsample, activation=activation))
            self.expand.append(IConvMod(nfeatures[d], nfeatures[d], 
                                        invert=invert, skip_invert=skip_invert,
                                        activation=activation, ks=ks))

        # Output feature embedding without batchnorm.
        self.embed_out = EmbeddingMod(nfeatures[0], nfeatures[0], embed_ks, activation=activation)
        self.output    = nn.Conv3d(nfeatures[0], out_ch, (1,1,1), bias=True)
        self.set_invert(invert)

    def forward(self, x):
        """
        """
        skip = []
        x = self.embed_in(x)
        for convmod in self.contract[:-1]:
            x = convmod(x)
            skip.append(x)
            x = self.pooling(x)
            
        # Bridge.
        x = self.contract[-1](x)
        
        # Expanding/upsampling pathway.
        for convmod, upsample, sk in zip(self.expand, self.upsamp, reversed(skip)):
            x = upsample(x, sk)
            x = convmod(x)

        # Output feature embedding without batchnorm.
        x = self.embed_out(x)
        return self.output(x)

    def set_invert(self, invert, layers=None):
        """
        """
        if layers is None: 
            layers = set(map(lambda x:type(x), self.modules()))-{type(self)}
        else:
            layers = set(layers)
        for m in self.modules():
            if type(m) in layers and hasattr(m, "set_invert"):
                m.set_invert(invert)