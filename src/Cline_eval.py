# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 11:38:41 2019
This is for evaluating centerline from image with Neural Network training
@author: xinweiy
"""
import matplotlib.pyplot as plt
import os
import scipy.io as sio


if __name__ == "__main__":
    Folders = ['/tigress/LEIFER/PanNeuronal/2018/20180329/BrainScanner20180329_152141']
    for folder in Folders:
        path = os.path.join(folder, 'LowMagBrain' + folder[-15:] + '/cam1.avi')
        cline_path = os.path.join(folder, 'BehaviorAnalysis/centerline.mat')
        cline_mat = sio.loadmat(cline_path)
        cline = cline_mat['centerline']