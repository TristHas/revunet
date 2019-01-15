import torch.nn as nn
import torch

class RevBlock(nn.Module):

    def __init__(self, F_module, G_module, invert=True):
        super().__init__()
        self.F_module = F_module        
        self.G_module = G_module
        self.invert = self.set_invert(invert)

    def forward(self, x):
        x1, x2 = torch.chunk(x, 2, dim=1)
        y1 = self.F_module(x2) + x1
        y2 = self.G_module(y1) + x2
        y = torch.cat([y1, y2], dim=1)
        
        if self.invert:
            if self.training:
                handle_ref = []
                handle_ref.append(y.register_hook(self.get_variable_backward_hook((x, x1, x2, y1, y2), y, handle_ref)))
            x.data.set_()
            x1.data.set_()
            x2.data.set_()
            y1.data.set_()
            y2.data.set_()
        return y

    def inverse(self, output, inp):
        y1_, y2_ = torch.chunk(output, 2, dim=1)
        with torch.no_grad():
            x2_ = y2_ - self.G_module(y1_)
            x1_ = y1_ - self.F_module(x2_)
            x_ = torch.cat((x1_, x2_), 1)

        x, x1, x2, y1, y2 = inp
        x.data.set_(x_)
        x1.data.set_(x1_)
        x2.data.set_(x2_)
        y1.data.set_(y1_)
        y2.data.set_(y2_)
        output.data.set_()

    def get_variable_backward_hook(self, inp_to_fill, output, handle_ref):
        def backward_hook(grad):
            self.inverse(output, inp_to_fill)
            handle_ref[0].remove()
        return backward_hook
    
    def set_invert(self, invert):
        self.invert=invert