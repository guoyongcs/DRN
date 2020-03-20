import random
import numpy as np
import skimage.color as sc
import torch


def get_patch(*args, patch_size=96, scale=[2], multi_scale=False):
    th, tw = args[-1].shape[:2] # target images size

    tp = patch_size  # patch size of target hr image
    ip = [patch_size // s for s in scale] # patch size of lr images

    # tx and ty are the top  and left coordinate of the patch 
    tx = random.randrange(0, tw - tp + 1)
    ty = random.randrange(0, th - tp + 1)
    tx, ty = tx- tx % scale[0], ty - ty % scale[0]
    ix, iy = [ tx // s for s in scale], [ty // s for s in scale]
   
    lr = [args[0][i][iy[i]:iy[i] + ip[i], ix[i]:ix[i] + ip[i], :] for i in range(len(scale))]
    hr = args[-1][ty:ty + tp, tx:tx + tp, :]

    return [lr, hr]

def set_channel(*args, n_channels=3):
    def _set_channel(img):
        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)

        c = img.shape[2]
        if n_channels == 1 and c == 3:
            img = np.expand_dims(sc.rgb2ycbcr(img)[:, :, 0], 2)
        elif n_channels == 3 and c == 1:
            img = np.concatenate([img] * n_channels, 2)

        return img

    return [_set_channel(a) for a in args[0]], _set_channel(args[-1])


def np2Tensor(*args, rgb_range=255):
    def _np2Tensor(img):
        np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
        tensor = torch.from_numpy(np_transpose).float()
        tensor.mul_(rgb_range / 255)

        return tensor

    return [_np2Tensor(a) for a in args[0]], _np2Tensor(args[1])


def augment(*args, hflip=True, rot=True):
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5

    def _augment(img):
        if hflip: img = img[:, ::-1, :]
        if vflip: img = img[::-1, :, :]
        if rot90: img = img.transpose(1, 0, 2)
        
        return img

    return [_augment(a) for a in args[0]], _augment(args[-1])

