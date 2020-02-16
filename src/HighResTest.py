"""
Created on Mon Nov 14 11:38:41 2019
This is for looking at High resolution neural recording image.
@author: xinweiy
"""

import wormdatamodel
import matplotlib.pyplot as plt
import savitzkygolay as sg
import cv2
import numpy as np


def segment_cv_singleplane(Araw, C, resize=True, threshold=0.04):
    '''
    Single plane version
    '''

    if resize:
        A = cv2.resize(src=Araw, dsize=(256, 256), fx=0.5, fy=0.5, \
                       interpolation=cv2.INTER_AREA)
    else:
        A = Araw

    blur = 0.65
    A = cv2.GaussianBlur(A, (3, 3), blur, blur)

    BX = cv2.sepFilter2D(src=A, ddepth=-1, kernelX=C, kernelY=np.ones(1))
    BY = cv2.sepFilter2D(src=A, ddepth=-1, kernelX=np.ones(1), kernelY=C)
    B = BX + BY

    threshX = threshold * np.min(BX)
    threshY = threshold * np.min(BY)
    Bth = (BX < threshX) * (BY < threshY) * B

    sz_xy = 3
    sz_z = 3
    K = np.ones((sz_xy, sz_xy), np.float64)
    Bdil = -cv2.dilate(-Bth, K)

    Neuron = np.array(np.where(((B - Bdil) == 0.0) * B != 0.0))

    return Neuron


class Multi_Slice_Viewer(object):
    def __init__(self, volume):
        super(Multi_Slice_Viewer, self).__init__()
        self.remove_keymap_conflicts({'j', 'k'})
        fig, ax = plt.subplots()
        ax.volume = volume
        ax.index = volume.shape[0] // 2
        ax.imshow(volume[ax.index])
        fig.canvas.mpl_connect('key_press_event', self.process_key)
        plt.show()

    def process_key(self, event):
        fig = event.canvas.figure
        ax = fig.axes[0]
        if event.key == 'j':
            self.previous_slice(ax)
        elif event.key == 'k':
            self.next_slice(ax)
        fig.canvas.draw()

    def previous_slice(self, ax):
        volume = ax.volume
        ax.index = (ax.index - 1) % volume.shape[0]  # wrap around using %
        ax.images[0].set_array(volume[ax.index])
        print(ax.index)

    def next_slice(self, ax):
        volume = ax.volume
        ax.index = (ax.index + 1) % volume.shape[0]
        ax.images[0].set_array(volume[ax.index])
        print(ax.index)

    def remove_keymap_conflicts(self, new_keys_set):
        for prop in plt.rcParams:
            if prop.startswith('keymap.'):
                keys = plt.rcParams[prop]
                remove_list = set(keys) & new_keys_set
                for key in remove_list:
                    keys.remove(key)

if __name__ == "__main__":
    folder = "/tigress/LEIFER/PanNeuronal/20191106/BrainScanner20191106_143222"
    N = 1
    i = 1
    C = sg.get_1D_derivative(mu=7, poly=5, order=2)
    with wormdatamodel.data.recording(folder, legacy=True) as rec:
        rec.load(startVolume=i, nVolume=N)
        s_idx = 21
        t_idx = 22
        Multi_Slice_Viewer(rec.frame[:, 0, :, :])
        pts_s = segment_cv_singleplane(rec.frame[s_idx, 0, :, :].astype(np.float64), C, resize=True, threshold=0.04) * 2
        pts_t = segment_cv_singleplane(rec.frame[t_idx, 0, :, :].astype(np.float64), C, resize=True, threshold=0.04) * 2
        plt.subplot(1,2,1)
        plt.imshow(rec.frame[s_idx, 0, :, :])
        plt.scatter(pts_s[1], pts_s[0], s=10, c='red')
        plt.subplot(1,2,2)
        plt.imshow(rec.frame[t_idx, 0, :, :])
        plt.scatter(pts_t[1], pts_t[0], s=10, c='red')
        plt.show()
        print(rec.frame.shape)
        print(pts_t)

