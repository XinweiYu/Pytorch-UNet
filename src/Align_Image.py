# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 11:38:41 2019
This is for image registration in low-res images
@author: xinweiy
"""

from scipy import interpolate
import cv2
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

def interpolate_img(image, pt, k_sz=5):
    width, height = image.shape[:2]
    x_min = int(max(0, np.floor(pt[0]) - k_sz))
    x_max = int(min(width, np.ceil(pt[0]) + k_sz + 1))
    y_min = int(max(0, np.floor(pt[1]) - k_sz))
    y_max = int(min(height, np.ceil(pt[1]) + k_sz + 1))
    x = np.arange(x_min, x_max)
    y = np.arange(y_min, y_max)
    xx, yy = np.meshgrid(x, y)
    z = image[x_min:x_max, y_min:y_max]

    f = interpolate.interp2d(x, y, z, kind='cubic')
    return f(pt[0], pt[1])

def crop_coordinate(cline_lim, crop_size=512):
    cline_width_mid = 0.5 * (cline_lim[1] + cline_lim[0])
    cline_height_mid = 0.5 * (cline_lim[3] + cline_lim[2])
    cline_width_mid = min(max((cline_width_mid, crop_size//2)), 1088-crop_size//2)
    cline_height_mid = min(max((cline_height_mid, crop_size // 2)), 1088 - crop_size // 2)
    return [int(cline_width_mid-crop_size//2), int(cline_width_mid+crop_size//2),\
            int(cline_height_mid-crop_size//2), int(cline_height_mid+crop_size//2)]

if __name__ == "__main__":
    path = '/tigress/LEIFER/PanNeuronal/2018/20180503/BrainScanner20180503_164957/LowMagBrain20180503_164957/cam1.avi'
    cline_path = '/tigress/LEIFER/PanNeuronal/2018/20180503/BrainScanner20180503_164957/BehaviorAnalysis/centerline.mat'
    cline_mat = sio.loadmat(cline_path)
    cline = cline_mat['centerline'][:,:,0]
    vidcap = cv2.VideoCapture(path)
    success = True
    count = 0
    # optical flow estimator
    of_estim = cv2.optflow.createOptFlow_DeepFlow()
    img0, img1 = None, None

    while success:
        success, img1 = vidcap.read()
        img1 = img1[:, :, 0]
        # crop the image.
        width, height = img1.shape
        sz= 100
        crop = [int(max(min(cline[:, 0])-sz, 0)), int(min(max(cline[:, 0])+sz, width)),
                int(max(min(cline[:, 1])-sz, 0)), int(min(max(cline[:, 1])+sz, height))]
        cline_lim = [int(min(cline[:, 0])), int(max(cline[:, 0])),
                     int(min(cline[:, 1])), int(max(cline[:, 0]))]
        crop = crop_coordinate(cline_lim, crop_size=512)
        origin = [crop[0], crop[2]]
        if img0 is not None and img1 is not None:
            flow = of_estim.calc(img0[crop[0]:crop[1], crop[2]:crop[3]],
                                 img1[crop[0]:crop[1], crop[2]:crop[3]], None)
            for i in range(len(cline)):
                dx = interpolate_img(flow[:, :, 1], cline[i]-origin)[0]
                dy = interpolate_img(flow[:, :, 0], cline[i]-origin)[0]
                cline[i] += [dx, dy]
        img0 = img1
        if count % 10 == 0:
            plt.imshow(img1)
            plt.scatter(cline[:, 1], cline[:, 0], c='black', s=0.1)
            plt.show()
            # plt.savefig('./output/'+str(count)+'.png')
            # plt.clf()
            print(crop[1]-crop[0], crop[3]-crop[2])
        count += 1