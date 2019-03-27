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

#import matplotlib.pyplot as plt
import time
import pickle
from scipy.interpolate import interp1d
import scipy.io as sio

# This script is to get the centerline for the whole video.
class CenterlineFromVideo(object):
  
  def __init__(self,cpu=False,erode=5,dilation=5):
    self.net_cline =  UNet(n_channels=3, n_classes=1)
    self.net_direction = UNet(n_channels=3, n_classes=8)
    self.net_tip = UNet(n_channels=3, n_classes=5)
    self.model_cline = "/home/xinweiy/github/checkpoints/cline_087.pth"
    self.model_tip = "/home/xinweiy/github/checkpoints/tip_076.pth"
    self.model_direction = "/home/xinweiy/github/checkpoints/direction_081.pth"
    self.cpu = cpu
    self.erode = erode
    self.dilation = dilation
    if not cpu:

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
    self.tips = np.array([[256,256], [256,256]])
    self.length = 240

    
  def Video2Centerlines(self,path):
    # get centerlines for the video.
    vidcap = cv2.VideoCapture(path)
    success = True
    count = 0    
    centerlines = list()
    head_last = []
    tail_last = []
    self.cline_last = []
    head_list = list()
    tail_list = list()
    cline_list = list()
    while success:
      success,image = vidcap.read()     
      count += 1
      print( 'frame:{}'.format(count) )
      if success:
        cline = self.Image2Centerline(image, head_last=head_last, tail_last=tail_last)                     

      if len(cline)>10:
        centerlines.append(cline)
        head_last = (cline[0,:]-150)*512/750
        tail_last = (cline[-1,:]-150)*512/750
        self.cline_last = np.copy((cline-150)*512/750)
        if len(head_list)>=10:
          head_list.pop(0)
        if len(tail_list)>=10:
          tail_list.pop(0)
        if len(cline_list)>=50:
          cline_list.pop(0)
        head_list.append(head_last)
        tail_list.append(tail_last)
        cline_list.append(self.cline_last)
        if len(head_list) > 0:
          head_last = np.mean(np.array(head_list), axis=0)
        if len(tail_list) > 0:
          tail_last = np.mean(np.array(tail_list), axis=0)
        if len(cline_list) > 0:
          self.cline_last = np.mean(np.array(cline_list), axis=0)
      else:
        centerlines.append(centerlines[-1])
    
    return centerlines
  
  def Image2Centerline(self, img, plot=0, head_last=[], tail_last=[]):
    
    image = Image.fromarray(img)
    image = image.crop((150,150,900,900))
    image = image.resize((512,512), Image.NEAREST)
    

    img_c = predict_img(net = self.net_cline,
                         full_img = image,
                         scale_factor = 0.5,
                         out_threshold = 0.5,
                         use_dense_crf = 0,
                         use_gpu = not self.cpu)  
    
    img_c[0,:,:] = skimage.morphology.remove_small_objects(img_c[0,:,:], 1000)
#    selem = skimage.morphology.disk(5)
#    img_c[0,:,:] = skimage.morphology.binary_opening(img_c[0,:,:], selem)
    # try morphology method. to find skeleton and tips.
    # skel_coords, img_skel = self.skeleton_tip(img_c)
    #img_skel = skimage.morphology.skeletonize(img_c[0,:,:])
            
    img_tip = predict_img(net=self.net_tip,
                         full_img=image,
                         scale_factor = 0.5,
                         out_threshold = 0.5,
                         use_dense_crf= 0,
                         use_gpu = not self.cpu) 

    img_dir = predict_img(net=self.net_direction,
                          full_img=image,
                          scale_factor=0.5,
                          out_threshold=0.5,
                          use_dense_crf= 0,
                          use_gpu = not self.cpu)  

    img_tip = img_tip * img_c
    img_dir = img_dir * img_c
#    selem = skimage.morphology.disk(self.dilation)
#    for i in range(len(img_dir)):
#      img_dir[i,:,:] = skimage.morphology.binary_dilation(img_dir[i,:,:],selem)
    
    head_ref = self.fCline.GetTip(img_tip[0,:,:].astype(np.float32), head_last)
    tail_ref = self.fCline.GetTip(img_tip[1,:,:].astype(np.float32), tail_last)
        
    cline_dict = dict()
    cline_dict["clines"] = list()
    cline_dict["penalty"] = list()
    cline_dict["length"] = list()
    
    if len(head_last)==0 and len(head_ref)==1:
      head_last = head_ref[0]
      
    if len(tail_last)==0 and len(tail_ref)==1:
      tail_last = tail_ref[0]
   
    if len(head_ref):
      for i in range(len(head_ref)):
        cline_dict = self.fCline.cline_from_skel(img_c, img_dir, cline_dict,  
                                         head_ref[i, :], tail_ref, head_last, tail_last)
    else:
      head_ref = head_last
      cline_dict = self.fCline.cline_from_skel(img_c, img_dir, cline_dict,  
                                         head_ref, tail_ref, head_last, tail_last)
          
    cline = self.SelectCline(cline_dict, head_last, tail_last)

  
#    selem = skimage.morphology.disk(self.erode)
#    # use the centerline mask to clear tip and direction
#    img_tip = img_tip*img_c
#    img_dir = img_dir*img_c
#    
#    img_c_erosion = skimage.morphology.erosion(img_c[0,:,:], selem) + img_tip[0,:,:] +img_tip[1,:,:]
#    selem = skimage.morphology.disk(self.dilation)
#    for i in range(len(img_dir)):
#      img_dir[i,:,:] = skimage.morphology.binary_dilation(img_dir[i,:,:],selem)
    
#    clines = self.fCline.GetCenterline(img_c_erosion,img_dir,img_tip)
    if len(cline)>0:
      cline = 750/512*cline + 150
    if plot:
      plt.imshow(img)
      plt.plot(cline[:, 1], cline[:, 0])
      plt.scatter(cline[0, 1],cline[0, 0],
                  s=50,c='red',alpha=0.3,marker='x')
      plt.show()
    return cline


  def distanceInterp(self, cline, n=100):
    # interpolates curve to create n evenly spaced points along the curve
    if len(cline)>10:    
        cline_diff = np.diff(cline, axis=0)
        ds = np.sqrt(np.sum(cline_diff**2, axis=1))
        s = np.cumsum(ds)
        s = np.insert(s,0,0)
        Lsearch = np.linspace(0, s[-1], num=100, endpoint=True)
        f = interp1d(s, cline, axis=0, kind='quadratic')
        return f(Lsearch)
    else:
        return []
    
    
    
  def SelectCline(self, cline_dict, head_last=[], tail_last=[]):
    # choose the cline that fit the former frame.
    num_cline = len(cline_dict["clines"])
    
    if len(head_last)==0:
      head_last = [256, 256]
    if num_cline > 1:
      score_cline = list()
      
      for i in range(num_cline):
        cline = cline_dict["clines"][i]
        if len(cline) > 10:
          cline_diff = np.diff(cline[0::5, :], axis=0)
          length = np.sum( np.sqrt(np.sum(cline_diff**2, axis=1)))
          
          dist_head = np.sqrt(np.sum((head_last - cline[0,:])**2))
          
          if len(self.cline_last):
            cline_num = len(cline)-1
            cline_idx = np.floor(np.arange(0,100,1)*cline_num/99)
            cline_idx = cline_idx.astype(int)
            dist_old = np.sqrt(np.sum((self.cline_last - cline[cline_idx])**2, axis=1))
            dist_old = np.mean(dist_old)
          else:
            dist_old = 0
          
          
          score = [(cline_dict["penalty"][i]-1)*100, 0.05*(length-self.length)**2, dist_head*5, dist_old*3]
#          print('penalty{}'.format(cline_dict["penalty"][i]))
#          print('length{}'.format(length))
#          print('distance head{}'.format(dist_head))
#          print('distance old{}'.format(dist_old))
          score_cline.append(score)
        else:
          score_cline.append([float("Inf"), float("Inf"), float("Inf"), float("Inf")])
      
      cline_index = np.argmin(np.sum(np.array(score_cline), axis=1))
      score_min = np.min(np.sum(np.array(score_cline), axis=1))
      print(cline_index)
      print(score_cline)
      if score_min < 500:
          return self.distanceInterp(cline_dict["clines"][cline_index])
      else:
          return []
    else:
      if num_cline:
        cline = cline_dict["clines"][0]
        cline_diff = np.diff(cline[0::5, :], axis=0)
        length = np.sum( np.sqrt(np.sum(cline_diff**2, axis=1)))
        self.length = 0.99*self.length + 0.01*length
        return self.distanceInterp(cline_dict["clines"][0])
      else:
        return []
      
    
#    num_heads = len(clines)
#    if num_heads > 1:
#      dist = list()
#      for i in range(num_heads):
#        head_pt = clines[i][0, :]
#        dist.append(np.sum(np.abs(head_pt - self.tips[0, :])))
#      head_idx = np.argmin(dist)
#    else:
#      head_idx = 0
#      
#    cline = clines[head_idx].copy()  
#    num_tails = len(cline)
#    if num_tails > 1:
#      dist = list()      
#      for i in range(num_tails):
#        tail_pt = cline[i][-1,:]
#        dist.append(np.sum(np.abs(tail_pt-self.tips[1,:])))
#      tail_idx = np.argmin(dist)
#    else:
#      tail_idx = 0
#      
#    return clines[head_idx][tail_idx]
    
    
if __name__ == "__main__":
  clineFromVideo = CenterlineFromVideo()
  path = "/scratch/network/xinweiy/cam1.avi"

  tic = time.time()
  cline = clineFromVideo.Video2Centerlines(path) 
  toc = time.time()
  print("run time is: {}".format(toc-tic))
  with open("/scratch/network/xinweiy/test.txt", "wb") as fp:   #Pickling
    pickle.dump(cline, fp)

  cline_dict = dict()
  cline_dict["centerline"] = cline
  sio.savemat('/scratch/network/xinweiy/test.mat', cline_dict)

#  centerline = dict()
#  centerline['centerline'] = cline
#  with open("/scratch/network/xinweiy/cline.txt", "wb") as fp:   #Pickling
#    pickle.dump(centerline, fp)
#  plt.figure(figsize=(20,10))
#  clineFromVideo = CenterlineFromVideo(cpu=True)
#  path = "C:\\Users\\xinweiy\\Desktop\\cam1.avi"
#  vidcap = cv2.VideoCapture(path)
#  vidcap.set(1,11350)
#  success,image = vidcap.read() 
#  clines = clineFromVideo.Image2Centerline(image,plot=0)


  

