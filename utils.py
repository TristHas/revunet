import numpy as np

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

