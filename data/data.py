import tifffile
import numpy as np
from .preprocess import check_volume, affinitize, affinitize_mask
from .dataset import default_dst, DataSetLoader

class OldDataSet(object):
    def __init__(self, fov, img, aff):
        self.fov = fov
        self.delta = np.array(self.fov) // 2
        self.size = img.shape
        self.img = img
        self.aff = aff
        
    def sample(self):
        coord = self.gen_coord()
        return self._slice(coord)
    
    def gen_coord(self):
        mins = self.delta
        maxs = np.array(self.size) - np.array(self.delta) - 1
        coord = *map(lambda x: np.random.randint(*x), zip(mins, maxs)),
        return np.array(coord)
        
    def _slice(self, coord):
        bbox = self._bbox(coord)
        aff_bbox = (slice(None),) + bbox
        a = self.img[bbox]
        b = self.aff[aff_bbox]
        return a[None, None, :], b[None, :]
    
    def _bbox(self, coord):
        mins = coord - self.delta
        maxs = coord + self.delta
        return tuple(map(lambda x: slice(*x), zip(mins, maxs)))
    
def load_all_data(root="/home/tristan/workspace/data/snemi3d", dst=default_dst):
    max_or = np.array(dst_or).max(0)
    boundary = f"{root}/train-membranes-idsia.tif"
    labels   = f"{root}/train-labels.tif"
    img      = f"{root}/train-input.tif"

    seg = tifffile.imread(labels)
    msk = tifffile.imread(boundary)
    img = tifffile.imread(img).astype("float32")[max_or[0]:,max_or[1]:,max_or[2]:]

    img  = (img - img.mean()) / img.std()
    aff  = affinitize_all(seg, dst_or)[:,max_or[0]:,max_or[1]:,max_or[2]:]
    return img, aff

def affinitize_all(seg, dst):
    shape = (len(dst),) + seg.shape[-3:]
    affs = np.zeros(shape, dtype='float32')
    for i, dst in enumerate(dst):
        affinitize(seg, ret=affs[i,...], dst=dst)
    return affs

def get_dataloaders(root, fov, num_workers=2):
    img   = (tifffile.imread(root+"train-input.tif") / 256).astype("float32")
    lbl   = tifffile.imread(root+"train-labels.tif")
    train_ds = DataSetLoader(fov, img[:,:,:-200], lbl[:,:,:-200], num_workers=num_workers)
    test_ds  = DataSetLoader(fov, img[:,:,-200:], lbl[:,:,-200:], mode="test")
    return train_ds, test_ds