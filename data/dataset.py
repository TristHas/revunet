from torch.utils.data import DataLoader
import torch
import numpy as np
from .preprocess import Warp, Flip, Greyscale, Affinity

default_dst = list()
default_dst.append((0,0,1))
default_dst.append((0,1,0))
default_dst.append((1,0,0))
default_dst.append((0,0,3))
default_dst.append((0,3,0))
default_dst.append((2,0,0))
default_dst.append((0,0,9))
default_dst.append((0,9,0))
default_dst.append((3,0,0))
default_dst.append((0,0,27))
default_dst.append((0,27,0))
default_dst.append((4,0,0))

class DataSet(object):
    def __init__(self, fov, img, lbl, 
                 mode="train", w=True, g=True, f=True, 
                 dst=default_dst, nsamp=None):
        self.img = img
        self.lbl = lbl
        
        self.fov   = fov
        self.dst   = dst
        self.size  = self.img.shape
        self.range = np.max(dst, 0)        
        self._w = w
        self._f = f
        self._g = g
        self.mode = mode
        self.nsamp = nsamp if nsamp else int(10**10)
        
        self.w = Warp()
        self.f = Flip()
        self.g = Greyscale()
        self.a = Affinity(dst=self.dst, recompute=False)
        
        
    def __getitem__(self, _):
        return self.sample()
    
    def __len__(self):
        return self.nsamp
        
    def sample(self):
        img, lbl = self._sample()
        img, lbl = self.aug(img, lbl)
        aff, msk = self.a(lbl)
        return img, aff, msk
        
    def aug(self, img, lbl):
        if self.mode=="train":
            img, lbl = self.w(img,lbl) if self._w else (img[None,:], lbl[None,:])
            img, lbl = self.f(img,lbl) if self._f else (img, lbl)
            img = self.g(img) if self._g else img
        else:
            img, lbl = img[None,:], lbl[None,:]
        return img, lbl
        
    def _sample(self):
        fov_  = self._prepare_aug()
        delta = (fov_ // 2) 
        coord = self.gen_coord(delta)
        return self._slice(coord, delta)
    
    def gen_coord(self, delta):
        mins  = delta
        maxs  = np.array(self.size) - np.array(delta) - 1
        coord = *map(lambda x: np.random.randint(*x), zip(mins, maxs)),
        return np.array(coord)
        
    def _slice(self, coord, delta):
        bbox = self._bbox(coord, delta)
        a = self.img[bbox]
        b = self.lbl[bbox]
        return a, b
    
    def _bbox(self, coord, delta):
        mins = coord - delta
        maxs = coord + delta
        return tuple(map(lambda x: slice(*x), zip(mins, maxs)))
    
    def _prepare_aug(self):
        if self.mode=="train":
            fov_ = self.w.prepare(self.fov)
            self.f.prepare(None)
            self.g.prepare(None)
            return np.array(fov_)
        else:
            return np.array(self.fov)
        
class DataSetLoader():
    def __init__(self, fov, img, lbl, 
                 mode="train", w=True, g=True, f=True, 
                 dst=default_dst, nsamp=None, num_workers=0):
        self.mode = mode
        self.ds   = DataSet(fov=fov, img=img, lbl=lbl, mode=mode, 
                            w=w, g=g, f=f, dst=dst, nsamp=nsamp)
        self.dl = iter(DataLoader(self.ds, num_workers=num_workers, pin_memory=True))
        
    def sample(self, mode):
        if mode == self.mode:
            return next(self.dl)
        else:
            mode_ = self.mode
            self.ds.mode=mode
            out = self.ds.sample()
            self.ds.mode = mode_
            return map(lambda x:torch.from_numpy(x[None,:]), out)