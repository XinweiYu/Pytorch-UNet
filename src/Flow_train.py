# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 11:38:41 2019
This is for getting centerline from image with Neural Network training
@author: xinweiy
"""
import skimage.morphology as skmorp
from skimage.segmentation import active_contour
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import glob
import os
from PIL import Image
import pickle
from torchvision import transforms
from ClineNet import vgg_cline16_bn
from torch.nn import MSELoss, L1Loss
from torch.optim import Adam, lr_scheduler
from tqdm import tqdm
import time
import numpy as np
from Cline_train import Worm_Cline_Dataset
from unet import UNet



class Worm_Cline_Flow_Dataset(Worm_Cline_Dataset):
    # this class is used to read worm data for center-line.
    def __init__(self, data_folder, mode='train', tsfm=None):
        # tsfm stands for transform.
        super(Worm_Cline_Flow_Dataset, self).__init__(data_folder, mode, tsfm)

    def __getitem__(self, idx):
        # get idx the worm.
        #tic = time.time()
        sample = dict()
        file_name = self.worm_list[idx]
        with open(file_name, "rb") as fp:  # Pickling
            try:
                cline_dict = pickle.load(fp)
                fp.close()
            except:
                os.system('rm ' + file_name)
                print(file_name)

        img_path = os.path.join(self.data_folder, cline_dict['img_path'])
        worm_img = Image.open(img_path)
        worm_img = self.transform(worm_img)
        output_path = os.path.join(self.data_folder, cline_dict['output_path'][3:])
        output_img = np.load(output_path)
        output_img = self.transform(output_img)
        cline_prev = cline_dict['last_cline']
        worm_cline_prev = torch.zeros(worm_img.size())
        for i in range(len(cline_prev)):
            x = int(cline_prev[i, 0])
            y = int(cline_prev[i, 1])
            if x >= 0 and x < 512 and y >= 0 and y < 512:
                worm_cline_prev[0, x, y] = 1 - i / 255

        sample['image'] = worm_img.float()
            #torch.cat((worm_img, worm_cline_prev), dim=0).float()
        sample['target'] = output_img.float()
        sample['cline'] = cline_dict['current_cline']#[0:100:5, :]
        sample['last_cline'] = cline_dict['last_cline']
        # curve representation
        # get the cline direction
        cline_dir = np.diff(cline_dict['current_cline'], axis=0)[0:100:5, :]
        norm = np.sqrt(np.sum(cline_dir ** 2, axis=1, keepdims=True))
        norm[norm < 1e-6] = 1e-6
        cline_dir = cline_dir / norm
        sample['cline_dir'] = np.copy(cline_dir)
        sample['cline_dir'][:, 0] = -cline_dir[:, 1]
        sample['cline_dir'][:, 1] = -cline_dir[:, 0]
        sample['head_pt'] = cline_dict['head_pt']
        #print('data time:', time.time()-tic)
        return sample


def train(data_dir, use_gpu=True):
    tsfm = transforms.ToTensor()
    num_epoch = 40
    worm_data = Worm_Cline_Flow_Dataset(data_dir, mode='train', tsfm=tsfm)
    batch_size = 10
    data_loader = DataLoader(worm_data, batch_size=batch_size, shuffle=True, num_workers=1)
    model = UNet(n_channels=1, n_classes=2)

    criterion_coord = MSELoss()

    if use_gpu:
        model.cuda()
        criterion_coord.cuda()

    optimizer = Adam(model.parameters(), lr=1.0e-4)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.3)
    for epoch_idx in tqdm(range(num_epoch)):
        tic1 = time.time()
        for i, batch in enumerate(data_loader):
            tic = time.time()
            worm_img = batch['image']
            cline = batch['cline']
            head_pt = batch['head_pt']
            cline_dir = batch['cline_dir']
            target = batch['target']
            #print(worm_img.size())
            if use_gpu:
                worm_img = worm_img.cuda()
                cline = cline.cuda().float()
                head_pt = head_pt.cuda().float()
                cline_dir = cline_dir.cuda().float()
                target = target.cuda().float()
            model.train()
            optimizer.zero_grad()
            flow_out = model(worm_img)
            loss = torch.mean((target[:, :2, :, :] - flow_out) ** 2 * target[:, 2:3, :, :]) * 1000
            #loss = loss_cline #+ loss_dir * 1
            loss.backward()
            optimizer.step()
            #print('time:', time.time()-tic)
            if i % 100 == 0:
                print('loss_flow', loss.item())
                #print('loss_dir', loss_dir.item())

            # worm_img = worm_img.numpy()
            # ax1 = plt.subplot(1, 2, 1)
            # ax1.imshow(worm_img[0,0,:,:])
            # ax2 = plt.subplot(1, 2, 2)
            # ax2.imshow(worm_img[0,1,:,:])
            # plt.show()
        #print('epoch time', time.time()-tic1)
        if scheduler.get_lr()[0] >= 2e-6:
            scheduler.step()
        model_name = 'flow'
        tic = time.time()
        torch.save(model.state_dict(), '../trained_model/'+ model_name + str(epoch_idx) + '.pth')
        print('save time:', time.time()-tic)
    return model

def eval(data_dir, use_gpu=True):
    tsfm = transforms.ToTensor()
    num_epoch = 1
    worm_data = Worm_Cline_Flow_Dataset(data_dir, mode='train', tsfm=tsfm)
    batch_size = 1
    data_loader = DataLoader(worm_data, batch_size=batch_size, shuffle=False, num_workers=2)
    model = UNet(n_channels=1, n_classes=2)
    model_name = 'flow15'
    model.load_state_dict(torch.load(os.path.join('../trained_model', model_name + '.pth')))

    if use_gpu:
        model.cuda()
    model.eval()
    for i, batch in enumerate(data_loader):
        worm_img = batch['image']
        cline = batch['cline']
        head_pt = batch['head_pt']
        target = batch['target']



        if use_gpu:
            worm_img = worm_img.float().cuda()
            cline = cline.float().cuda()
        with torch.no_grad():
            flow_out = model(worm_img) * 2 -1
        worm_img = worm_img.detach().cpu().numpy()
        flow_out = flow_out.detach().cpu().numpy()

        # threshold the flow to find pixel on worm
        flow_out_i = flow_out[0]
        flow_amp = np.sum(flow_out_i ** 2, axis=0)
        flow_threshold = 0.5

        # find c_line points
        worm_pts = np.array(np.where(flow_amp > flow_threshold)).T
        sqrt2 = np.sqrt(2) / 2
        basic_dir = np.array([[1, 0], [sqrt2, sqrt2], [0, 1],
                              [-sqrt2, sqrt2], [-1, 0], [-sqrt2, -sqrt2],
                              [0, -1], [sqrt2, -sqrt2]])
        basic_disp = np.array([[1, 0], [1, 1], [0, 1],
                              [-1, 1], [-1, 0], [-1, -1],
                              [0, -1], [1, -1]])
        cline_img = np.zeros(flow_amp.shape) > 1
        for worm_pt in worm_pts:
            pt_dir = np.array([flow_out_i[0, worm_pt[0], worm_pt[1]],
                               flow_out_i[1, worm_pt[0], worm_pt[1]]])
            dir_score = pt_dir.dot(basic_dir.T)
            next_disp = basic_disp[np.argmax(dir_score)]
            next_pt = worm_pt + next_disp
            if flow_amp[next_pt[0], next_pt[1]] <= flow_threshold:
                cline_img[next_pt[0], next_pt[1]] = 1
        selem = skmorp.disk(5)
        cline_img = skmorp.dilation(cline_img, selem) * (flow_amp <= flow_threshold)
        selem = skmorp.disk(3)
        cline_img = skmorp.closing(cline_img, selem)
        # cline_img = skmorp.remove_small_objects(cline_img, min_size=300)
        # cline_img1 = skmorp.thin(cline_img)
        # cline_img2 = skmorp.skeletonize(cline_img)

        # a test of active contour centerline algorithm
        init = batch['last_cline'].numpy() #+ np.random.randn(100,2) * 2

        image_crop = worm_img[0][0]
        image_crop[cline_img] += 0.2

        snake = active_contour(image_crop, init, boundary_condition='fixed-free',
                               alpha=0.015, beta=0.5, gamma=0.01, w_line=1, w_edge=0.2,
                               coordinates='rc', max_iterations=200)

        plt.imshow(image_crop)
        plt.scatter(init[:, 1], init[:, 0], s=1, c='red')
        plt.scatter(snake[:, 1], snake[:, 0], s=1, c='yellow')
        plt.show()





        # plt.subplot(1,2,1)
        # plt.imshow(cline_img1)
        # plt.subplot(1,2,2)
        # plt.imshow(cline_img2)
        # plt.show()

        # plt.subplot(2,2,1)
        # plt.imshow(target[0, 0, :, :])
        # plt.title('target')
        # plt.subplot(2,2,2)
        # plt.imshow(target[0, 1, :, :])
        # plt.title('target')
        # plt.subplot(2,2,3)
        # plt.imshow(flow_out[0, 0, :, :])
        # plt.title('output')
        # plt.subplot(2,2,4)
        # plt.imshow(flow_out[0, 1, :, :])
        # plt.title('output')
        # plt.show()




if __name__ == "__main__":
    data_folder = "/tigress/LEIFER/Xinwei/github/Pytorch-Unet"
    #data_dir = os.join.path(data_folder, "output")
    #model = train(data_folder)
    eval(data_folder)