import skimage.morphology as skmorp
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
import time
from skimage.segmentation import active_contour


def get_curve_representation(cline, degree=10, num_point=300):
    rep = np.zeros(2 * (degree+1))
    pt = np.zeros((num_point, 2))
    x_new = np.arange((num_point)) / num_point
    x = np.arange(cline.shape[0]) / cline.shape[0]
    y = cline[:, 0]
    z = np.polyfit(x, y, degree)
    fn = np.poly1d(z)
    pt[:, 0] = fn(x_new)
    rep[:degree+1] = z
    y = cline[:, 1]
    z = np.polyfit(x, y, degree)
    rep[degree+1:] = z
    fn = np.poly1d(z)
    pt[:, 1] = fn(x_new)
    print(pt.shape)
    return rep, pt


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
            tic = time.time()
            cline_dict['folder'] = folder
            cline_dict['img_path'] = 'output_img/' + str(img_idx) +'.png'

            cline_lim = [int(min(cline[:, 0, count-1])), int(max(cline[:, 0, count-1])),
                         int(min(cline[:, 1, count-1])), int(max(cline[:, 1, count-1]))]
            crop = crop_coordinate(cline_lim, crop_size=512)
            image_crop = Image.fromarray(img1[crop[0]:crop[1], crop[2]:crop[3]])
            cline_dict['last_cline'] = cline[:, :, count-1] - [crop[0], crop[2]]
            cline_dict['current_cline'] = cline[:, :, count] - [crop[0], crop[2]]
            cline_dict['head_pt'] = tip_mat['head_pts'][count, ::-1] - [crop[0], crop[2]]

            # a test of active contour centerline algorithm
            init = cline_dict['last_cline']
            snake = active_contour(image_crop, init,  boundary_condition='fixed-free',
                                    alpha=0.015, beta=0.01, gamma=0.01, w_line=1, w_edge=0,
                                   coordinates='rc')

            plt.imshow(image_crop)
            plt.scatter(init[:, 1], init[:, 0], s=1, c='red')
            plt.scatter(snake[:, 1], snake[:, 0], s=1, c='yellow')
            plt.show()



            # produce a flow image
            worm_img = img1[crop[0]:crop[1], crop[2]:crop[3]]
            cline_crop = cline_dict['current_cline']
            cline_img = np.zeros(worm_img.shape) > 1
            #rep, cline_pt = get_curve_representation(cline_crop, degree=10, num_point=300)

            for pt in cline_crop[1:98, :]:
                x, y = int(pt[0]), int(pt[1])
                if x >= 0 and x < 512 and y >= 0 and y < 512:
                    cline_img[x, y] = 1
            selem = skmorp.disk(3)
            cline_img_dilate = skmorp.dilation(cline_img, selem)
            cline_skel = skmorp.skeletonize(cline_img_dilate)
            cline_pt = np.array(np.where(cline_skel)).T
            selem = skmorp.disk(15)
            worm_img_b = skmorp.dilation(cline_skel, selem)
            worm_img_b *= np.logical_not(cline_img_dilate > 0)
            #flow_img = np.zeros((worm_img_b.shape, 3))
            num_f = np.sum(worm_img_b) + np.sum(cline_img_dilate)
            num_total = worm_img_b.shape[0] * worm_img_b.shape[1]
            # set weight for pixels for unbalanced class.
            weight_img = np.ones(worm_img_b.shape) * num_f / num_total
            weight_img[worm_img_b] = 1 - num_f / num_total
            weight_img[cline_img_dilate] = 1 - num_f / num_total
            # set flow image
            pixel_flow_all = np.array(np.where(worm_img_b)).T
            flow_x = np.zeros(worm_img_b.shape)
            flow_y = np.zeros(worm_img_b.shape)
            for pixel_flow in pixel_flow_all:
                dir_vecs = cline_pt - np.array([pixel_flow])
                dis2 = np.sum(dir_vecs ** 2, axis=1)
                idx = np.argmin(dis2)
                dir_vec = dir_vecs[idx, :] / np.sqrt(dis2[idx])
                flow_x[pixel_flow[0], pixel_flow[1]] = dir_vec[0]
                flow_y[pixel_flow[0], pixel_flow[1]] = dir_vec[1]

            cline_dict['output_path'] = 'output_output/' + str(img_idx) + '.npy'
            flow_img = np.dstack((0.5 * (flow_x + 1), 0.5 * (flow_y + 1), weight_img))



            # plt.subplot(1,3,1)
            # plt.imshow(flow_img)
            # plt.subplot(1,3,2)
            # plt.imshow(flow_y)
            # plt.subplot(1,3,3)
            # plt.scatter(cline_pt[1, :], cline_pt[0, :], s=0.1, c='black')
            # plt.imshow(worm_img)
            # plt.show()


            # save match
            # filename = os.path.join('../output', str(img_idx)) + '.txt'
            # np.save(cline_dict['output_path'], flow_img.astype(np.float16))
            # image_crop.save('../' + cline_dict['img_path'])
            # with open(filename, "wb") as fp:  # Pickling
            #     pickle.dump(cline_dict, fp)
            #     fp.close()
            img_idx += 1
            #print('run time:', time.time()-tic)
