import io
import os
import subprocess
import torch
import pandas as pd
import numpy as np

def get_gpu_mem(synchronize=True, empty_cache=True):
    return torch.cuda.memory_allocated(), torch.cuda.memory_cached()


def generate_mem_hook(handle_ref, mem, idx, hook_type, exp):
    def hook(self, *args):
        if len(mem)==0 or mem[-1]["exp"]!=exp:
            call_idx = 0
        else:
            call_idx = mem[-1]["call_idx"]+1
        
        mem_all, mem_cached = get_gpu_mem()
        torch.cuda.synchronize()
        mem.append({
            'layer_idx': idx,
            'call_idx': call_idx,
            'layer_type': type(self).__name__,
            'exp': exp,
            'hook_type': hook_type,
            'mem_all': mem_all,
            'mem_cached': mem_cached,
        })
    return hook

def add_memory_hooks(idx, mod, mem_log, exp, hr):
    
    h = mod.register_forward_pre_hook(generate_mem_hook(hr, mem_log, idx, 'pre', exp))
    hr.append(h)
    
    
    h = mod.register_forward_hook(generate_mem_hook(hr, mem_log, idx, 'fwd', exp))
    hr.append(h)
    
    h = mod.register_backward_hook(generate_mem_hook(hr, mem_log, idx, 'bwd', exp))
    hr.append(h)
    
def log_mem(model, inp, mem_log, exp):
    hr = []
    
    for idx, module in enumerate(model.modules()):
        add_memory_hooks(idx, module, mem_log, exp, hr)
        
    out = model(inp)
    loss=out.sum()
    loss.backward()
    
    [h.remove for h in hr]
    
def pp(df, exp):
	df_exp = df[df.exp==exp]
	df_pprint =(
		df_exp.assign(open_layer = lambda ddf: ddf.hook_type.map(lambda x: {"pre":0, "fwd":1, "bwd":2}[x]).rolling(2).apply(lambda x: x[0]==0 and x[1] == 0))
	   	.assign(close_layer = lambda ddf: ddf.hook_type.map(lambda x: {"pre":0, "fwd":1, "bwd":2}[x]).rolling(2).apply(lambda x: x[0]==1 and x[1] == 1))
	   	.assign(indent_level =  lambda ddf: (ddf.open_layer.cumsum() - ddf.close_layer.cumsum()).fillna(0).map(int))
	   	.sort_values(by = "call_idx")
	   	.assign(mem_diff = lambda ddf: ddf.mem_all.diff()//2**20)
	)
	pprint_lines = [f"{'    '*row[1].indent_level}{row[1].layer_type} {row[1].hook_type}  {row[1].mem_diff or ''}" for row in  df_pprint.iterrows()]
	for x in pprint_lines:
		print(x)