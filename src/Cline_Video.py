"""
Created on Mon Nov 11 11:38:41 2019
This is for getting centerline from video of worm.
@author: xinweiy
"""

import skimage.morphology as skmorp
from skimage.segmentation import active_contour
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
import numpy as np
from unet import UNet
import os
import glob
import scipy.io as sio
import cv2
from Align_Image import crop_coordinate
import time
from scipy.interpolate import interp1d
from Trainset_Preprocess import get_curve_representation

def distanceInterp(cline, n=100):
    # interpolates curve to create n evenly spaced points along the curve
    cline_diff = np.diff(cline, axis=0)
    ds = np.sqrt(np.sum(cline_diff ** 2, axis=1)) + 1e-4
    s = np.cumsum(ds)
    s = np.insert(s, 0, 0)
    Lsearch = np.linspace(0, s[-1], num=100, endpoint=True)
    f = interp1d(s, cline, axis=0, kind='quadratic')
    return f(Lsearch)



if __name__ == "__main__":
    data_folder = "/tigress/LEIFER/PanNeuronal/20191106/BrainScanner20191106_143222"
    video_path = glob.glob(os.path.join(data_folder, 'LowMagBrain*/cam1.avi'))[0]
    cline_path = os.path.join(data_folder, 'BehaviorAnalysis/centerline.mat')
    cline_mat = sio.loadmat(cline_path)
    cline = cline_mat['centerline']
    init = cline[:, :, 579]


    model = UNet(n_channels=1, n_classes=2)
    model_name = 'flow15'
    model.load_state_dict(torch.load(os.path.join('../trained_model', model_name + '.pth')))
    use_gpu = True
    if use_gpu:
        model.cuda()
    model.eval()

    vidcap = cv2.VideoCapture(video_path)
    tsfm = transforms.ToTensor()
    success = True
    count = 0
    # basic direction
    sqrt2 = np.sqrt(2) / 2
    basic_dir = np.array([[1, 0], [sqrt2, sqrt2], [0, 1],
                          [-sqrt2, sqrt2], [-1, 0], [-sqrt2, -sqrt2],
                          [0, -1], [sqrt2, -sqrt2]])
    basic_disp = np.array([[1, 0], [1, 1], [0, 1],
                           [-1, 1], [-1, 0], [-1, -1],
                           [0, -1], [1, -1]])

    def check_pt_valid(pt, image_size=[512, 512]):
        valid = True
        if pt[0] < 0 or pt[0] >= image_size[0]:
            valid = False
        if pt[1] < 0 or pt[1] >= image_size[1]:
            valid = False
        return valid





    def find_cline_pt(worm_pt):
        # go to the centerline pt with flow.
        next_pt = [int(worm_pt[0]), int(worm_pt[1])]
        if not check_pt_valid(next_pt):
            # if the point is outside the image return itself.
            return next_pt
        if boundary_img[next_pt[0], next_pt[1]]:
            # if the point is on the boundary return itself.
            return next_pt

        while flow_amp[next_pt[0], next_pt[1]] > flow_threshold:
            pt_dir = np.array([flow_out_i[0, next_pt[0], next_pt[1]],
                               flow_out_i[1, next_pt[0], next_pt[1]]])
            dir_score = pt_dir.dot(basic_dir.T)
            next_disp = basic_disp[np.argmax(dir_score)]
            next_pt = next_pt + next_disp
            if not check_pt_valid(next_pt):
                # if the point is outside the image return itself.
                return next_pt
            #init_flow.append([next_pt[0], next_pt[1]])
        return next_pt


    def min_dis_l1(tip, body_pts):
        dis = np.sum(np.abs(body_pts - np.array([tip])), axis=1)
        return np.min(dis)



    # treat video frame by frame
    while success:
        tic_begin = time.time()
        tic = time.time()
        success, img1 = vidcap.read()
        print('read data time:', time.time()-tic)
        if not success:
            break
        count += 1
        if count < 579:
            continue
        img1 = img1[:, :, 0]
        # crop the image
        cline_lim = [int(min(init[:, 0])), int(max(init[:, 0])),
                     int(min(init[:, 1])), int(max(init[:, 1]))]
        crop = crop_coordinate(cline_lim, crop_size=512)
        worm_img = img1[crop[0]:crop[1], crop[2]:crop[3]]
        init = init - [crop[0], crop[2]]

        worm_img = tsfm(worm_img)
        worm_img = worm_img[None, :, :].float()
        tic = time.time()
        if use_gpu:
            input_img = worm_img.cuda()
        else:
            input_img = worm_img

        with torch.no_grad():
            flow_out = model(input_img) * 2 - 1
        # threshold the flow to find pixel on worm

        print('network time:', time.time() - tic)

        flow_out_i = flow_out[0].detach().cpu().numpy()
        flow_amp = np.sum(flow_out_i ** 2, axis=0)
        flow_threshold = 0.1
        # find c_line points
        # flow_amp_b is all potential boundary points.
        flow_amp_b = flow_amp > flow_threshold
        selem = skmorp.square(3)
        flow_amp_b = np.invert(skmorp.binary_erosion(flow_amp_b, selem)) * flow_amp_b
        worm_pts = np.array(np.where(flow_amp_b)).T

        # generate the cline image and boundary image
        cline_img = np.zeros(flow_amp.shape) > 1
        boundary_img = np.zeros(flow_amp.shape) > 1
        tic = time.time()
        for worm_pt in worm_pts:
            pt_dir = np.array([flow_out_i[0, worm_pt[0], worm_pt[1]],
                               flow_out_i[1, worm_pt[0], worm_pt[1]]])
            dir_score = pt_dir.dot(basic_dir.T)
            next_disp = basic_disp[np.argmax(dir_score)]
            next_pt = worm_pt + next_disp
            if check_pt_valid(next_pt) and flow_amp[next_pt[0], next_pt[1]] <= flow_threshold:
                cline_img[next_pt[0], next_pt[1]] = 1
            inv_next_pt = worm_pt - next_disp
            if check_pt_valid(inv_next_pt):
                # boundary pt is direction opposite point and background boundary.
                if flow_amp[inv_next_pt[0], inv_next_pt[1]] <= flow_threshold:
                    boundary_img[inv_next_pt[0], inv_next_pt[1]] = 1
                pt_dir_inv = np.array([flow_out_i[0, inv_next_pt[0], inv_next_pt[1]],
                                       flow_out_i[1, inv_next_pt[0], inv_next_pt[1]]])
                if pt_dir.dot(pt_dir_inv) < 0:
                    boundary_img[inv_next_pt[0], inv_next_pt[1]] = 1

        print('find cline pt time:', time.time()-tic)

        tic = time.time()
        selem = skmorp.disk(3)
        cline_img = skmorp.binary_closing(cline_img, selem) * (flow_amp <= flow_threshold)
        boundary_img = skmorp.binary_dilation(boundary_img, selem)
        cline_img *= np.invert(boundary_img)


        # use active contour to find potential cline.
        # initialize snake from cline last frame.
        init_flow = list()
        for worm_pt in init:
            worm_pt_cline = find_cline_pt(worm_pt)
            init_flow.append(worm_pt_cline)

        init_flow = distanceInterp(np.array(init_flow))

        image_crop = worm_img.numpy()[0][0]
        worm_img = np.copy(image_crop)
        image_crop[cline_img] *= 2
        image_crop[boundary_img] *= 0.25

        print('morphlogical operation:', time.time()-tic)
        tic = time.time()
        snake = active_contour(image_crop, init_flow, boundary_condition='fixed',
                               alpha=0.015, beta=0.5, gamma=0.01, w_line=1, w_edge=0.2,
                               coordinates='rc', max_iterations=5)
        print('snake time:', time.time()-tic)

        tic = time.time()
        # extend the tip point if it is not at boundary
        snake = np.clip(np.rint(snake), 0, 511).astype(np.int16)
        head_dir = np.array(snake[0, :] - snake[5, :])
        dir_score = head_dir.dot(basic_dir.T)
        next_disp = basic_disp[np.argmax(dir_score)]
        head_tip = np.copy(snake[0, :])
        for _ in range(20):#min([10, int(min_head_dis-15)])):
            tmp = head_tip + next_disp
            if (not check_pt_valid(tmp)) or cline_img[tmp[0], tmp[1]] == 0:
                tmp_cline = find_cline_pt(tmp)
                if not np.any(tmp_cline - tmp):
                    break
                tmp = tmp_cline
            head_tip = tmp
        snake[0, :] = head_tip
        #snake[0, :] = find_cline_pt(snake[0, :])

        tail_dir = np.array(snake[-1, :] - snake[-6, :])
        dir_score = tail_dir.dot(basic_dir.T)
        next_disp = basic_disp[np.argmax(dir_score)]
        tail_tip = np.copy(snake[-1, :])
        for _ in range(20):#min([10, int(min_tail_dis-15)])):
            tmp = snake[-1, :] + next_disp
            if (not check_pt_valid(tmp)) or cline_img[tmp[0], tmp[1]] == 0:
                tmp_cline = find_cline_pt(tmp)
                if not np.any(tmp_cline - tmp):
                    break
                tmp = tmp_cline
            snake[-1, :] = tmp
        snake = np.clip(np.rint(distanceInterp(snake)), 0, 511).astype(np.int16)

        # check if there are bad segments of head and tip

        head_idx = 0
        bad_idxs = np.where(boundary_img[snake[:20, 0], snake[:20, 1]] == 1)[0]
        if len(bad_idxs) > 0:
            head_idx = max(bad_idxs)

        tail_idx = len(snake) - 1
        bad_idxs = np.where(boundary_img[snake[-20:, 0], snake[-20:, 1]] == 1)[0]
        if len(bad_idxs) > 0:
            tail_idx = min(bad_idxs) + 79
        snake = snake[head_idx:tail_idx+1, :]

        # check wheter tip point is too close to avoid body(itself for exampel)
        head_idx = 0
        while cline_img[snake[head_idx, 0], snake[head_idx, 1]] == 0 and head_idx < 5:
            head_idx += 1
        min_head_dis = min_dis_l1(snake[head_idx], snake[15:, :])
        while min_head_dis < 10 and head_idx < 5:
            head_idx += 1
            min_head_dis = min_dis_l1(snake[head_idx], snake[15:, :])

        tail_idx = len(snake) - 1
        while cline_img[snake[tail_idx, 0], snake[tail_idx, 1]] == 0 and tail_idx > len(snake)-6:
            tail_idx -= 1
        min_tail_dis = min_dis_l1(snake[tail_idx], snake[:-15, :])
        while min_tail_dis < 10 and tail_idx > len(snake)-6:
            tail_idx -= 1
            min_tail_dis = min_dis_l1(snake[tail_idx], snake[:-15, :])

        snake = snake[head_idx:tail_idx+1, :]
        _, snake = get_curve_representation(distanceInterp(snake), degree=10, num_point=100)


        print('post analysis time:', time.time()-tic)
        #snake[-1, :] = find_cline_pt(snake[-1, :])



        # # post analysis
        # snake = np.rint(snake).astype(np.int16)
        # # get rid of the head or tail that go too far into worm body.
        # head_avoid = np.zeros(image_crop.shape) > 1
        # tail_avoid = np.zeros(image_crop.shape) > 1
        # head_avoid[snake[10:, 0], snake[10:, 1]] = 1
        # tail_avoid[snake[:-10, 0], snake[:-10, 1]] = 1
        # selem = skmorp.disk(7)
        # head_avoid = skmorp.binary_dilation(head_avoid, selem)
        # tail_avoid = skmorp.binary_dilation(tail_avoid, selem)
        # head_avoid = np.logical_or(head_avoid, boundary_img)
        # tail_avoid = np.logical_or(tail_avoid, boundary_img)
        # head_idx = 0
        # while head_avoid[snake[head_idx, 0], snake[head_idx, 1]] and head_idx <= 5:
        #     head_idx += 1
        # tail_idx = -1
        # while tail_avoid[snake[tail_idx, 0], snake[tail_idx, 1]] and tail_idx >= -5:
        #     tail_idx -= 1
        # print(head_idx, tail_idx)
        # snake = snake[head_idx:tail_idx, :]





        init_old = init + [crop[0], crop[2]]

        init = distanceInterp(np.clip(snake, 0, 511) + [crop[0], crop[2]])

        cline_dis = np.mean(np.abs(init_old - init))
        if cline_dis > 30:
            init = init_old
        print(count, cline_dis)
        print('run time:', time.time() - tic_begin)
        plt.subplot(121)
        plt.imshow(worm_img)
        #plt.scatter(init[:, 1]-crop[2], init[:, 0]-crop[0], s=0.1, c='red')
        plt.scatter(snake[:, 1], snake[:, 0], s=0.1, c='yellow')
        plt.subplot(122)
        plt.imshow(image_crop)
        plt.savefig('./video_output/'+str(count)+'.png')
        plt.clf()




