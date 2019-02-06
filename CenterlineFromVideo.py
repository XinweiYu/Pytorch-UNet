# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 11:38:41 2019

@author: xinweiy
"""
import argparse
import os
import pickle
import numpy as np
import torch
import torch.nn.functional as F
import cv2
import skimage.morphology
from PIL import Image
from predict import predict_img
from FindCenterline import FindCenterline
from unet import UNet
from utils import resize_and_crop, normalize, split_img_into_squares, hwc_to_chw, merge_masks#, dense_crf
#from utils import plot_img_and_mask



# This script is to get the centerline for the whole video.
class CenterlineFromVideo(object):
  def __init__(self):
    self.net_cline =  UNet(n_channels=3, n_classes=1)
    self.net_direction = UNet(n_channels=3, n_classes=8)
    self.net_tip = UNet(n_channels=3, n_classes=2)
    self.model_cline = "/home/xinweiy/github/checkpoints/cline.pth"
    self.model_tip = "/home/xinweiy/github/checkpoints/tip.pth"
    self.model_direction = "/home/xinweiy/github/checkpoints/direction.pth"
    if not args.cpu:
        print("Using CUDA version of the net, prepare your GPU !")
        self.net_cline.cuda()
        self.net_cline.load_state_dict(torch.load(self.model_cline))
        self.net_tip.cuda()
        self.net_tip.load_state_dict(torch.load(self.model_tip))    
        self.net_direction.cuda()
        self.net_direction.load_state_dict(torch.load(self.model_direction))    
    else:
        self.net_cline.cpu()
        self.net_cline.load_state_dict(torch.load(self.model_cline, map_location='cpu'))
        self.net_tip.cpu()
        self.net_tip.load_state_dict(torch.load(self.model_tip, map_location='cpu'))    
        self.net_direction.cpu()
        self.net_direction.load_state_dict(torch.load(self.model_direction, map_location='cpu'))    

    print("Model loaded !")
    
    self.fCline = FindCenterline(tip_r=3)
    self.cline = list()
    self.tips = np.array([[256,256],[256,256]])
    
    
    
  def GetCenterlines(self,path):
    # get centerlines for the video.
    vidcap = cv2.VideoCapture(path)
    selem = skimage.morphology.disk(5)
    success = True
    count = 0
    
    centerlines = list()
    while success and count<100:
      success,image = vidcap.read()
      image = Image.fromarray(image)
      image = image.crop((150,150,900,900))
      image = image.resize((512,512),Image.NEAREST)
      count += 1
      img_c = predict_img(net=self.net_cline,
                         full_img=image,
                         scale_factor=0.5,
                         out_threshold=0.5,
                         use_dense_crf= 0,
                         use_gpu=1)  
      img_tip = predict_img(net=self.net_tip,
                         full_img=image,
                         scale_factor=0.5,
                         out_threshold=0.5,
                         use_dense_crf= 0,
                         use_gpu=1)  
      img_dir = predict_img(net=self.net_direction,
                         full_img=image,
                         scale_factor=0.5,
                         out_threshold=0.5,
                         use_dense_crf= 0,
                         use_gpu=1)  
      

      img_c_erosion = skimage.morphology.erosion(img_c[0,:,:], selem) + img_tip[0,:,:]
      for i in range(len(img_dir)):
        img_dir[i,:,:] = skimage.morphology.binary_dilation(img_dir[i,:,:],selem)
      
      clines = self.fCline.GetCenterline(img_c_erosion,img_dir,img_tip)
      cline = self.SelectCline(clines)
      
      centerlines.append(cline)
      self.tips[0,:] = cline[0,:]
      self.tips[-1,:] = cline[-1,:]
    
    return centerlines
    
  def SelectCline(self,clines):
    # choose the cline that fit the former frame.   
    num_heads = len(clines)
    if num_heads > 1:
      for i in range(num_heads):
        dist = list()
        head_pt = clines[i][0][0,:]
        dist.append(np.sum(np.abs(head_pt-self.tips[0,:])))
      head_idx = np.argmin(dist)
    else:
      head_idx = 0
    
    cline = clines[head_idx].copy()  
    num_tails = len(cline)
    if num_tails > 1:
      for i in range(num_tails):
        dist = list()
        tail_pt = cline[i][-1,:]
        dist.append(np.sum(np.abs(tail_pt-self.tips[1,:])))
      tail_idx = np.argmin(dist)
    else:
      tail_idx = 0
      
    return clines[head_idx][tail_idx]
    
    
    
    
    
    
    
    
if __name__ == "__main__":
  clineFromVideo = CenterlineFromVideo()
  path = "/scratch/network/xinweiy/cam1.avi"
  cline = clineFromVideo.GetCenterlines(path)  
  centerline = dict()
  centerline['centerline'] = cline
  with open("/scratch/network/xinweiy/cline.txt", "wb") as fp:   #Pickling
    pickle.dump(centerline, fp)