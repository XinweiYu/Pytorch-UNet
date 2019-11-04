from scipy import interpolate
import cv2
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import os
from Align_Image import crop_coordinate
from PIL import Image
import pickle
import glob

if __name__=="__main__":
    Folders = ['/tigress/LEIFER/PanNeuronal/2018/20180329/BrainScanner20180329_152141', \
                   '/tigress/LEIFER/PanNeuronal/2018/20180511/BrainScanner20180511_134913', \
                   '/tigress/LEIFER/PanNeuronal/2018/20180223/BrainScanner20180223_141721', \
                   '/tigress/LEIFER/PanNeuronal/2018/20180223/BrainScanner20180223_142554', \
                   '/tigress/LEIFER/PanNeuronal/2018/20180330/BrainScanner20180330_160650', \
                   '/tigress/LEIFER/PanNeuronal/2018/20180330/BrainScanner20180330_162137', \
                   '/tigress/LEIFER/PanNeuronal/2018/20180327/BrainScanner20180327_152059', \
                   '/tigress/LEIFER/PanNeuronal/2018/20180430/BrainScanner20180430_141614', \
                   '/tigress/LEIFER/PanNeuronal/2018/20180709/BrainScanner20180709_100433',\
                   '/tigress/LEIFER/PanNeuronal/2017/20170424/BrainScanner20170424_105620', \
                   '/tigress/LEIFER/PanNeuronal/2017/20170610/BrainScanner20170610_105634', \
                   '/tigress/LEIFER/PanNeuronal/2017/20170613/BrainScanner20170613_134800', \
                    '/tigress/LEIFER/PanNeuronal/2018/20181210/BrainScanner20181210_163927', \
                    '/tigress/LEIFER/PanNeuronal/2018/20181210/BrainScanner20181210_172207', \
                    '/tigress/LEIFER/PanNeuronal/2018/20181114/BrainScanner20181114_214842', \
                    '/tigress/LEIFER/PanNeuronal/2018/20180328/BrainScanner20180328_154502', \
                    '/tigress/LEIFER/PanNeuronal/2018/20180327/BrainScanner20180327_152059', \
                    '/tigress/LEIFER/PanNeuronal/2018/20180410/BrainScanner20180410_144953', \
                    '/tigress/LEIFER/PanNeuronal/2018/20180926/BrainScanner20180926_205648', \
                    '/tigress/LEIFER/PanNeuronal/2018/20180503/BrainScanner20180503_182227', \
                    '/tigress/LEIFER/PanNeuronal/2018/20180509/BrainScanner20180509_120413', \
                    '/tigress/LEIFER/PanNeuronal/2018/20180503/BrainScanner20180503_190433', \
                    '/tigress/LEIFER/PanNeuronal/2018/20180503/BrainScanner20180503_173901', \
                    '/tigress/LEIFER/PanNeuronal/2018/20180503/BrainScanner20180503_144513', \
                    '/tigress/LEIFER/PanNeuronal/2018/20180503/BrainScanner20180503_164957']
    img_idx = 1
    for folder in Folders:
        path = glob.glob(os.path.join(folder, 'LowMagBrain*/cam1.avi'))[0]
        cline_path = os.path.join(folder, 'BehaviorAnalysis/centerline.mat')
        try:
            tip_path = glob.glob(os.path.join(folder, 'LowMagBrain*/tip_coodinates.mat'))[0]
        except:
            print(folder, 'no tip')
            continue
        cline_mat = sio.loadmat(cline_path)
        tip_mat = sio.loadmat(tip_path)
        frame_head = np.where(tip_mat['head_pts'][:,0])[0]
        cline = cline_mat['centerline']
        vidcap = cv2.VideoCapture(path)
        success, img1 = vidcap.read()
        count = 0
        cline_dict = dict()
        cline_prev = cline[:, :, 0]
        while success:
            success, img1 = vidcap.read()
            if not success:
                break
            count += 1
            img1 = img1[:, :, 0]
            if not count in frame_head:
                continue
            # dist = np.sum((cline[:50, :, count] - cline_prev[:50, :])**2)
            # if dist < 1000:
            #     continue
            # cline_prev = cline[:, :, count]
            cline_dict['folder'] = folder
            cline_dict['img_path'] = 'output_img/' + str(img_idx) +'.png'

            cline_lim = [int(min(cline[:, 0, count-1])), int(max(cline[:, 0, count-1])),
                         int(min(cline[:, 1, count-1])), int(max(cline[:, 1, count-1]))]
            crop = crop_coordinate(cline_lim, crop_size=512)
            image_crop = Image.fromarray(img1[crop[0]:crop[1], crop[2]:crop[3]])
            image_crop.save('../' + cline_dict['img_path'])
            cline_dict['last_cline'] = cline[:, :, count-1] - [crop[0], crop[2]]
            cline_dict['current_cline'] = cline[:, :, count] - [crop[0], crop[2]]
            cline_dict['head_pt'] = tip_mat['head_pts'][count, ::-1] - [crop[0], crop[2]]

            # save match
            filename = os.path.join('../output', str(img_idx)) + '.txt'
            with open(filename, "wb") as fp:  # Pickling
                pickle.dump(cline_dict, fp)
                fp.close()
            img_idx += 1
