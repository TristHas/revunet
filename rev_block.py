import torch.nn as nn
import torch

class Checkpoint(object):
    def __init__(self, invert):
        self.invert = invert
        
    def set(self, x):
        self.cp = x
        if self.invert:
            return x+0
        else:
            return x

    def get(self):
        return self.cp

    def drop(self):
        self.cp.data.set_()

class RevBlock(nn.Module):

    def __init__(self, F_module, G_module, invert=True):
        super().__init__()
        self.F_module = F_module        
        self.G_module = G_module
        self.cp1 = Checkpoint(getattr(self.F_module, 'invert', False))
        self.cp2 = Checkpoint(getattr(self.G_module, 'invert', False))
        self.set_invert(invert)

    def forward(self, x):
        setattr(self.F_module, "delete_intermediaries",  True)
        setattr(self.G_module, "delete_intermediaries",  True)
        
        
        x1, x2 = torch.chunk(x, 2, dim=1)
        if self.invert:
            x.data.set_()
        
        x2 = self.cp1.set(x2)
        F_x2 = self.F_module(x2) 
        y1 = F_x2+ x1
        if self.invert:
            F_x2.data.set_()
            x1.data.set_()

        y1 = self.cp2.set(y1)
        G_y1= self.G_module(y1)
        y2 = self.cp1.get() + G_y1
        if self.invert:
            G_y1.data.set_()
            x2.data.set_() 

        y = torch.cat([self.cp2.get(), y2], dim=1)
        
        self.cp1.drop()
        self.cp2.drop()
        if self.invert:
            y1.data.set_()
            y2.data.set_()
            
            if self.training and y.requires_grad:
                handle_ref = []
                handle_ref.append(y.register_hook(self.get_variable_backward_hook((x, x1, x2, F_x2, y1, G_y1, y2), y, handle_ref)))

                
        setattr(self.F_module, "delete_intermediaries", False)
        setattr(self.G_module, "delete_intermediaries",False)
        return y

    def inverse(self, output, inp):
        setattr(self.F_module, "fill_intermediaries", not getattr(self.F_module, "invert", False))
        setattr(self.G_module, "fill_intermediaries", not getattr(self.G_module, "invert", False))
        x, x1, x2, F_x2, y1, G_y1, y2 = inp
        
        y1_data, y2_data = torch.chunk(output, 2, dim=1)

        output.data.set_()
        
        with torch.no_grad():
            y1_data = self.cp1.set(y1_data)
            G_y1_data =self.G_module(y1_data)
            x2_data = y2_data - G_y1_data
            
            x2_data = self.cp2.set(x2_data)
            F_x2_data = self.F_module(x2_data)
            
            x1_data = self.cp1.get() - F_x2_data
            
            x_data = torch.cat((x1_data, self.cp2.get()), 1)
            
            x.data.set_(x_data)
            y1.data.set_(y1_data)
            G_y1.data.set_(G_y1_data)
            F_x2.data.set_(F_x2_data)
            x2.data.set_(x2_data)
            self.cp1.drop()
            self.cp2.drop()
        
        
        setattr(self.F_module, "fill_intermediaries",  False)
        setattr(self.G_module, "fill_intermediaries", False)

    def get_variable_backward_hook(self, inp_to_fill, output, handle_ref):
        def backward_hook(grad):
            self.inverse(output, inp_to_fill)
            handle_ref[0].remove()
        return backward_hook

    def set_invert(self, invert):
        self.invert = invert