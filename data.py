import tifffile
import numpy as np

def check_volume(data):
    """Ensure that data is a numpy 3D array."""
    assert isinstance(data, np.ndarray)
    if data.ndim == 2:
        data = data[np.newaxis,...]
    elif data.ndim == 3:
        pass
    elif data.ndim == 4:
        assert data.shape[0]==1
        data = np.reshape(data, data.shape[-3:])
    else:
        raise RuntimeError('data must be a numpy 3D array')

    assert data.ndim==3
    return data

def affinitize(img, ret=None, dst=(1,1,1), dtype='float32'):
    """
    Transform segmentation to an affinity map.
    Args:
        img: 3D indexed image, with each index corresponding to each segment.
    Returns:
        ret: an affinity map (4D tensor).
    """
    img = check_volume(img)
    if ret is None:
        ret = np.zeros(img.shape, dtype=dtype)
    # Sanity check.
    (dz,dy,dx) = dst
    assert abs(dx) < img.shape[-1]
    assert abs(dy) < img.shape[-2]
    assert abs(dz) < img.shape[-3]

    # Slices.
    s0 = list()
    s1 = list()
    s2 = list()
    for i in range(3):
        if dst[i] == 0:
            s0.append(slice(None))
            s1.append(slice(None))
            s2.append(slice(None))
        elif dst[i] > 0:
            s0.append(slice(dst[i],  None))
            s1.append(slice(dst[i],  None))
            s2.append(slice(None, -dst[i]))
        else:
            s0.append(slice(None,  dst[i]))
            s1.append(slice(-dst[i], None))
            s2.append(slice(None,  dst[i]))
    ret[s0] = (img[s1]==img[s2]) & (img[s1]>0)
    return ret[np.newaxis,...]

def affinitize_mask(msk, ret=None, dst=(1,1,1), dtype='float32'):
    """
    Transform binary mask to affinity mask.
    Args:
        msk: 3D binary mask.
    Returns:
        ret: 3D affinity mask (4D tensor).
    """
    msk = check_volume(msk)
    if ret is None:
        ret = np.zeros(msk.shape, dtype=dtype)
    # Sanity check.
    (dz,dy,dx) = dst
    assert abs(dx) < msk.shape[-1]
    assert abs(dy) < msk.shape[-2]
    assert abs(dz) < msk.shape[-3]
    # Slices.
    s0 = list()
    s1 = list()
    s2 = list()
    for i in range(3):
        if dst[i] == 0:
            s0.append(slice(None))
            s1.append(slice(None))
            s2.append(slice(None))
        elif dst[i] > 0:
            s0.append(slice(dst[i],  None))
            s1.append(slice(dst[i],  None))
            s2.append(slice(None, -dst[i]))
        else:
            s0.append(slice(None,  dst[i]))
            s1.append(slice(-dst[i], None))
            s2.append(slice(None,  dst[i]))
    ret[s0] = (msk[s1]>0) & (msk[s2]>0)
    return ret[np.newaxis,...]

class DataSet(object):
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
    
def load_data(root="/home/tristan/workspace/data/snemi3d"):
    dst = list()
    dst.append((0,0,1))
    dst.append((0,1,0))
    dst.append((1,0,0))
    dst.append((0,0,2))
    dst.append((0,2,0))
    dst.append((2,0,0))
    dst.append((0,0,4))
    dst.append((0,4,0))
    dst.append((3,0,0))
    dst.append((0,0,12))
    dst.append((0,12,0))
    dst.append((4,0,0))

    dst_sub = dst

    dst = list()
    dst.append((0,0,1))
    dst.append((0,1,0))
    dst.append((1,0,0))
    dst.append((0,0,4))
    dst.append((0,4,0))
    dst.append((2,0,0))
    dst.append((0,0,8))
    dst.append((0,8,0))
    dst.append((3,0,0))
    dst.append((0,0,24))
    dst.append((0,24,0))
    dst.append((4,0,0))

    dst_or = dst

    max_or = np.array(dst_or).max(0)
    max_sub = np.array(dst_sub).max(0)
    boundary = f"{root}/train-membranes-idsia.tif"
    labels   = f"{root}/train-labels.tif"
    img      = f"{root}/train-input.tif"

    seg = tifffile.imread(labels)
    msk = tifffile.imread(boundary)
    img = tifffile.imread(img).astype("float32")[max_or[0]:,max_or[1]:,max_or[2]:]

    img  = (img - img.mean()) / img.std()
    aff  = affinite(seg, dst_or)[:,max_or[0]:,max_or[1]:,max_or[2]:]
    return img, aff

def affinite(seg, dst):
    shape = (len(dst),) + seg.shape[-3:]
    affs = np.zeros(shape, dtype='float32')
    for i, dst in enumerate(dst):
        affinitize(seg, ret=affs[i,...], dst=dst)
    return affs

