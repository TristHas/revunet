import torch
from torch import nn
from torch.nn import functional as F

# from . import *
from .legacy import *
from .utils import *
from .Imodules import *

class ISkipUNet(nn.Module):
    """
        
    """
    def __init__(self, out_ch, depth=4, 
                 ks=(3,3,3), embed_ks=(1,5,5), 
                 activation=None, invert=True,
                 nfeatures = [16,32,64,128,256,512, 1024],
                 upsample='trilinear', pool=(1,2,2), skip_invert=True):
        super().__init__()
        self.upsample   = IUpsample(scale_factor=(1, 2, 2), mode=upsample)
        self.depth      = depth
        self.activation = ILeakyReLU() if activation is None else activation
        self.pooling    = nn.MaxPool3d(pool)


        def get_Unet_fn_with_channels(nfeatures):
            if len(nfeatures) == 2:
                return lambda channels: ISequential(
                    self.pooling,
                    IConvMod(
                        nfeatures[0],
                        nfeatures[1],
                        self.activation,
                        invert=invert,
                        skip_invert=skip_invert,
                    ),
                    self.upsample,
                    nn.Conv3d(
                        nfeatures[1],
                        nfeatures[0],
                        kernel_size=(1, 1, 1),
                    )
                )

            else:
                return lambda channels: ISequential(
                    self.pooling,
                    IConvMod(
                        nfeatures[0],
                        nfeatures[1],
                        self.activation,
                        invert=invert,
                        skip_invert=skip_invert,
                    ),
                    ISkip(nfeatures[0], get_Unet_fn_with_channels([x//2 if skip_invert else x for x in nfeatures[1:]]), invert=skip_invert),
                    IBatchNorm3d(nfeatures[1], track_running_stats=False),
                    self.activation,
                    IConvMod(
                        nfeatures[1],
                        nfeatures[1],
                        self.activation,
                        invert=invert,
                        skip_invert=skip_invert,
                    ),
                    self.upsample,
                    nn.Conv3d(
                        nfeatures[1],
                        nfeatures[0],
                        kernel_size=(1, 1, 1),
                    )
                )



        self.embed_in = EmbeddingMod(1, nfeatures[0], embed_ks)
        self.unet = ISequential(
            IConvMod(
                nfeatures[0],
                nfeatures[0],
                self.activation,
                invert=invert,
                skip_invert=skip_invert,
            ),
            ISkip(nfeatures[0], get_Unet_fn_with_channels([x//2 if skip_invert else x for x in nfeatures[:depth+1]]), invert=skip_invert),
            IBatchNorm3d(nfeatures[0] , track_running_stats=False),
            self.activation,
        )
        self.embed_out = EmbeddingMod(nfeatures[0], nfeatures[0], embed_ks)
        self.output    = nn.Conv3d(nfeatures[0], out_ch, (1,1,1), bias=True)


    def forward(self, x):
        """
        """
        x = self.embed_in(x)
        x = self.unet(x)

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
