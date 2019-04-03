#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 19:17:35 2019

@author: xinweiy
"""


import glob
import os


path = '/tigress/LEIFER/PanNeuronal/2018/20181210/BrainScanner20181210_153955'
cam_path = glob.glob(path+'/LowMagBrain*')[0]+'/cam1.avi'
print(cam_path)

directory = path+'/BehaviorAnalysis/'
if not os.path.exists(directory):
    os.makedirs(directory)