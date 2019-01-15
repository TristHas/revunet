import numpy as np
import torch
import torch.nn.functional as F

topy = lambda x: torch.from_numpy(x).cuda(device)

def pad_size(ks, mode):
    assert mode in ['valid', 'same', 'full']
    if mode == 'valid':
        pad = (0,0,0)
    elif mode == 'same':
        assert all([x %  2 for x in ks])
        pad = tuple(x // 2 for x in ks)
    elif mode == 'full':
        pad = tuple(x - 1 for x in ks)
    return pad

def layers_mod(model):
    return list(set(map(lambda x:type(x).__name__, model.modules())))#__name__

def params_mod(model):
    return list(model.parameters())

def params_size_mod(model):
    return np.sum(list(map(lambda x:np.prod(x.size()), model.parameters())))
    
def select_mod(model, modules):
    modules = list(map(lambda x: x if isinstance(x, str) else x.__name__, modules))
    return list(filter(lambda x:type(x).__name__ in modules, model.modules()))#
    
def distrib_params_per_mod(model, rel=False):
    res = {}
    for mod in layers_mod(model):
        layers = select_mod(model, [mod])
        count = np.sum(list(map(lambda x:params_size_mod(x), layers)))
        res[mod]=count
    if rel:
        nparams=params_size_mod(model)
        res = {k:v/nparams for k,v in res.items()}
    return res

def distrib_mod(model):
    res = {}
    for mod in layers_mod(model):
        #print(mod)
        layers = select_mod(model, [mod])
        res[mod]=len(layers)
    return res

def train(ds):
    model.train()
    x,y = ds.sample()
    x = topy(x)
    y = topy(y)

    w_pos = y.mean(2, keepdim=True).mean(3, keepdim=True).mean(4, keepdim=True)
    w_neg = 1 - w_pos
    msk_pos = (y == 1).float()
    msk_neg = (y == 0).float()
    msk = n_msk = msk_pos * w_neg + msk_neg * w_pos

    opt.zero_grad()
    out = model(x)
    loss = F.binary_cross_entropy_with_logits(out, y, msk)
    loss.backward()
    opt.step()
    return loss.item()
    
def valid(ds, nrun=20):
    model.eval()
    with torch.no_grad():
        accs = 0
        for i in range(nrun):
            x,y = ds.sample()
            x = topy(x)
            y = topy(y)

            w_pos = y.mean(2, keepdim=True).mean(3, keepdim=True).mean(4, keepdim=True)
            w_neg = 1 - w_pos
            msk_pos = (y == 1).float()
            msk_neg = (y == 0).float()
            msk = n_msk = msk_pos * w_neg + msk_neg * w_pos
            
            out = model(x)
            accs += F.binary_cross_entropy_with_logits(out, y, msk).item()
    return accs / nrun
    
def monitor(name, y, x):
    writer.add_scalar(name, y, x)