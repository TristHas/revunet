import numpy as np
import time
from .warping import warping
import numpy as np

def flip(data, rule):
    """
        Flip data according to a specified rule.
        Args:
            data:   4D numpy array to be transformed.
            rule:   Transform rule, specified as a Boolean array.
                    [z reflection, y reflection, x reflection, xy transpose]
        Returns:
            Transformed data.
    """
    data = check_tensor(data)
    assert np.size(rule)==4
    # z reflection.
    if rule[0]:
        data = data[:,::-1,:,:]
    # y reflection.
    if rule[1]:
        data = data[:,:,::-1,:]
    # x reflection.
    if rule[2]:
        data = data[:,:,:,::-1]
    # Transpose in xy.
    if rule[3]:
        data = data.transpose(0,1,3,2)
    # Prevent potential negative stride issues by copying.
    return np.copy(data)

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

    ret[tuple(s0)] = (img[tuple(s1)]==img[tuple(s2)]) & (img[tuple(s1)]>0)
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

    ret[tuple(s0)] = (msk[tuple(s1)]>0) & (msk[tuple(s2)]>0)
    return ret[np.newaxis,...]

def rebalance_binary_class(img, msk=None, base_w=0.0, dtype='float32'):
    """
        Binary-class rebalancing.
    """
    img = check_volume(img)
    ret = np.zeros(img.shape, dtype=dtype)
    if msk is None:
        msk   = np.ones(img.shape, dtype=bool)
        idx   = img > 0
        total = img.size
    else:
        msk   = check_volume(msk)
        msk   = msk > 0
        idx   = (img > 0) & msk
        total = np.count_nonzero(msk)

    count = np.count_nonzero(idx)
    if count > 0 and (total - count) > 0:
        weight = [1.0/count, 1.0/(total - count)]
        weight = weight/np.sum(weight)
    else:
        weight = [base_w]*2
    ret[idx] = weight[0]
    ret[~idx & msk] = weight[1]
    return ret

def check_tensor(data):
    """
        Ensure that data is a numpy 4D array.
    """
    assert isinstance(data, np.ndarray)
    if data.ndim == 2:
        data = data[np.newaxis,np.newaxis,...]
    elif data.ndim == 3:
        data = data[np.newaxis,...]
    elif data.ndim == 4:
        pass
    else:
        raise RuntimeError('data must be a numpy 4D array')
    assert data.ndim==4
    return data

def check_volume(data):
    """
        Ensure that data is a numpy 3D array.
    """
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

class Affinity():
    """
        Expand segmentation into affinity represntation.
    """
    def __init__(self, dst, crop=None, base_w=1.0, recompute=True):
        """
            Initialize parameters.
            Args:
                dst    : List of 3-tuples, each indicating affinity distance in (z,y,x).
                source : Key to source data from which to construct affinity.
                target : Key to target data.
                crop   : 3-tuple indicating crop offset.
                base_w : base weight for class-rebalanced gradient weight mask.
        """
        self.dst    = dst
        self.crop   = crop
        self.base_w = base_w
        self.recompute = recompute

    def __call__(self, seg, **kwargs):
        """
            Affinity label processing.
        """
        msk = np.ones(seg.shape, 'float32')
        # Recompute connected components.
        if self.recompute:
            shape = (3,) + seg.shape[-3:]
            aff = np.zeros(shape, dtype='float32')
            affinitize(seg, ret=aff[0,...], dst=(0,0,1))
            affinitize(seg, ret=aff[1,...], dst=(0,1,0))
            affinitize(seg, ret=aff[2,...], dst=(1,0,0))
            seg = datatools.get_segmentation(aff)
        # Affinitize.
        shape = (len(self.dst),) + seg.shape[-3:]
        affs = np.zeros(shape, dtype='float32')
        msks = np.zeros(shape, dtype='float32')
        for i, dst in enumerate(self.dst):
            affinitize(seg, ret=affs[i,...], dst=dst)
            affinitize_mask(msk, ret=msks[i,...], dst=dst)
        lbl = affs
        msk = msks
        # Rebalancing.
        if self.base_w is not None:
            for c in range(lbl.shape[0]):
                msk[c,...] = rebalance_binary_class(lbl[c,...], msk=msk[c,...], base_w=self.base_w)
        # Crop.
        if self.crop is not None:
            #lbl = tf.crop(lbl, offset=self.crop)
            #msk = tf.crop(msk, offset=self.crop)
            raise NotImplementedError("Crooping not supported yet")
        return lbl, msk

class Flip():
    """
        Random flip.
    """

    def __init__(self):
        pass

    def prepare(self, x):
        return x

    def __call__(self, img, lbl, **kwargs):
        rule = np.random.rand(4) > 0.5
        return flip(img, rule), flip(lbl, rule)
    
class Greyscale():
    """
        Greyscale value augmentation.
        Randomly adjust contrast/brightness, and apply random gamma correction.
    """

    def __init__(self, mode='mix', skip_ratio=0.3):
        """
            Initialize parameters.

            Args:
                mode: '2D', '3D', 'mix'
                skip_ratio: Probability of skipping augmentation.
        """
        self.set_mode(mode)
        self.set_skip_ratio(skip_ratio)
        self.CONTRAST_FACTOR   = 0.3
        self.BRIGHTNESS_FACTOR = 0.3

    def prepare(self, x):
        """
        """
        self.skip = np.random.rand() < self.skip_ratio
        return x

    def __call__(self, img, lbl=None):
        """
        """
        if not self.skip:
            if self.mode == 'mix':
                mode = '3D' if np.random.rand() > 0.5 else '2D'
            else:
                mode = self.mode
            if mode is '2D': img = self.augment2D(img)
            if mode is '3D': img = self.augment3D(img)
        return img

    def augment2D(self, img):
        """
            Adapted from ELEKTRONN (http://elektronn.org/).
        """
        for z in range(img.shape[-3]):
            img_ = img[...,z,:,:]
            img_ *= 1 + (np.random.rand() - 0.5)*self.CONTRAST_FACTOR
            img_ += (np.random.rand() - 0.5)*self.BRIGHTNESS_FACTOR
            img_ = np.clip(img_, 0, 1)
            img_ **= 2.0**(np.random.rand()*2 - 1)
            img[...,z,:,:] = img_
        return img

    def augment3D(self, img):
        """
            Adapted from ELEKTRONN (http://elektronn.org/).
        """
        img *= 1 + (np.random.rand() - 0.5)*self.CONTRAST_FACTOR
        img += (np.random.rand() - 0.5)*self.BRIGHTNESS_FACTOR
        img = np.clip(img, 0, 1)
        img **= 2.0**(np.random.rand()*2 - 1)
        return img

    def set_mode(self, mode):
        """
            Set 2D/3D/mix greyscale value augmentation mode.
        """
        assert mode=='2D' or mode=='3D' or mode=='mix'
        self.mode = mode

    def set_skip_ratio(self, ratio):
        """
            Set the probability of skipping augmentation.
        """
        assert ratio >= 0.0 and ratio <= 1.0
        self.skip_ratio = ratio
        
class Warp():
    """
        Warping data augmentation.
        1. Continuous rotation.
        2. Shear.
        3. Twist.
        4. Scale.
        5. Perspective stretch.
    """
    def __init__(self, skip_ratio=0.3):
        self.set_skip_ratio(skip_ratio)

    def prepare(self, fov):
        """
            Randomly draw warp parameters and compute required (mostly larger
            than original) image sizes.
        """
        self.skip = False
        if self.skip_ratio > np.random.rand():
            self.skip = True
            return fov                
        params    = warping.getWarpParams(fov)
        self.size = tuple(x for x in params[0])
        size_diff = tuple(x - y for x, y in zip(self.size, fov))
        self.rot     = params[1]
        self.shear   = params[2]
        self.scale   = params[3]
        self.stretch = params[4]
        self.twist   = params[5]
        self.fov     = fov
        return self.size

    def __call__(self, img, lbl):
        """
            Apply warp data augmentation.
        """
        lbl = check_tensor(lbl)
        img = check_tensor(img)
        
        if self.skip:
            return img, lbl
        
        img = np.transpose(img, (1,0,2,3))
        img = warping.warp3d(img, self.fov, self.rot, self.shear, 
                             self.scale, self.stretch, self.twist)
        img = np.copy(np.transpose(img, (1,0,2,3)))
        lbl = np.transpose(lbl, (1,0,2,3))
        lbl = warping.warp3dLab(lbl, self.fov, self.size,
             self.rot, self.shear, self.scale, self.stretch, self.twist)
        lbl = np.copy(np.transpose(lbl, (1,0,2,3)))
        return img, lbl

    def set_skip_ratio(self, ratio):
        """
            Set the probability of skipping augmentation.
        """
        assert ratio >= 0.0 and ratio <= 1.0
        self.skip_ratio = ratio