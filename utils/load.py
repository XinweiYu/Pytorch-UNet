#
# load.py : utils on generators / lists of ids to transform from strings to
#           cropped images and masks

import os

import numpy as np
from PIL import Image

from .utils import resize_and_crop, get_square, normalize, hwc_to_chw


def get_ids(dir):
    """Returns a list of the ids in the directory"""
    return (f[:-4] for f in os.listdir(dir))


def split_ids(ids, n=2):
    """Split each id in n, creating n tuples (id, k) for each id"""
    return ((id, i) for i in range(n) for id in ids)


def to_cropped_imgs(ids, dir, suffix, scale):
    """From a list of tuples, returns the correct cropped img"""
    for id, pos in ids:
            # currently only scale.
        im = resize_and_crop(Image.open(dir + id + suffix), scale=scale)
        yield get_square(im, pos)

def get_imgs_and_masks(ids, dir_img, dir_mask, scale):
    """Return all the couples (img, mask)"""

    imgs = to_cropped_imgs(ids, dir_img, '.jpg', scale)

    # need to transform from HWC to CHW
    imgs_switched = map(hwc_to_chw, imgs)
    # normalize the image. The input image is [0-255] here.
    imgs_normalized = map(normalize, imgs_switched)

    masks = to_cropped_imgs(ids, dir_mask, '_mask.gif', scale)

    return zip(imgs_normalized, masks)


def get_full_img_and_mask(id, dir_img, dir_mask):
    im = Image.open(dir_img + id + '.jpg')
    mask = Image.open(dir_mask + id + '_mask.gif')
    return np.array(im), np.array(mask)


def BinarizeMask(true_masks,n_classes=1):
    # binarize the mask for each class.
    #num_image=true_masks.shape[0]
    #masks=np.zeros(true_masks.shape)
#    for i in range(num_image):
#        image=true_masks[i]
    masks=list()
    for j in range(n_classes):
        mask=true_masks==j+1
        tmp=np.zeros(true_masks.shape)
        tmp[mask]=1
        masks.append(tmp)
        
#        mask=(image==j+1 for j in range(n_classes))
#        masks.append(np.array(mask))
    masks=np.array(masks)
    n_dim=len(masks.shape)
    #print(n_dim)
    #print(masks.shape)
    if n_dim==4:  
        masks = np.moveaxis(masks,[0,1,2,3],[1,0,2,3])
    return masks.astype('float32')
