# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 11:38:41 2019
This is for getting centerline from image with Neural Network training
@author: xinweiy
"""
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


class Worm_Cline_Dataset(Dataset):
    # this class is used to read worm data for center-line.
    def __init__(self, data_folder, mode='train', tsfm=None):
        # tsfm stands for transform.
        self.data_folder = data_folder
        if mode == 'train':
            cline_dir = os.path.join(data_folder, "output")
        elif mode == 'test':
            cline_dir = os.path.join(data_folder, "test_output")
        self.worm_list = sorted(glob.glob(os.path.join(cline_dir, '*.txt')))
        self.num_worm = len(self.worm_list)
        self.mode = mode
        # transform is process of the loaded image
        self.transform = tsfm

    def __len__(self):
        # for the number of image data.
        return self.num_worm

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
        cline_prev = cline_dict['last_cline']
        worm_cline_prev = torch.zeros(worm_img.size())
        for i in range(len(cline_prev)):
            x = max(0, min(511, int(cline_prev[i, 0])))
            y = max(0, min(511, int(cline_prev[i, 1])))
            worm_cline_prev[0, x, y] = 1 - i / 255

        xv, yv = torch.meshgrid([torch.arange(0, worm_img.size(1)), torch.arange(0, worm_img.size(2))])

        xv = xv[None, :, :].float() / 512.
        yv = yv[None, :, :].float() / 512.
        sample['image'] = torch.cat((worm_img, worm_cline_prev, xv.float(), yv.float()), dim=0)
        sample['cline'] = cline_dict['current_cline'][0:100:5, :]
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

def line_dis(cline1, cline2):
    loss = 0
    cline_diff = cline1 - cline2
    for i in range(cline_diff.size(1)-1):
        loss += torch.sum(cline_diff[:, i:i+2, :] ** 2)
        loss += torch.sum(cline_diff[:, i, :] * cline_diff[:, i+1, :])
    return torch.sqrt(loss / cline_diff.size(1))


def train(data_dir, use_gpu=True):
    tsfm = transforms.ToTensor()
    num_epoch = 40
    worm_data = Worm_Cline_Dataset(data_dir, mode='train', tsfm=tsfm)
    batch_size = 16
    data_loader = DataLoader(worm_data, batch_size=batch_size, shuffle=True, num_workers=2)
    all_point = True
    if all_point:
        model = vgg_cline16_bn(channel_in=4, channel_out=40)
    else:
        model = vgg_cline16_bn(channel_in=4, channel_out=2)
    criterion_coord = L1Loss()

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
            #print(worm_img.size())
            if use_gpu:
                worm_img = worm_img.cuda()
                cline = cline.cuda().float()
                head_pt = head_pt.cuda().float()
                cline_dir = cline_dir.cuda().float()
            model.train()
            optimizer.zero_grad()
            cline_out = model(worm_img) * 512
            if all_point:
                cline_out = cline_out.view(-1, 20, 2)
                #loss_cline = criterion_coord(cline_out, cline)
                loss_cline = line_dis(cline_out, cline)
                cline_diff = cline_out - cline
                loss_dir = torch.mean(torch.abs(torch.sum(cline_diff * cline_dir, dim=2)))
            else:
                loss_cline = criterion_coord(cline_out, head_pt)
                cline_diff = cline_out - head_pt
                loss_dir = torch.mean(torch.abs(torch.sum(cline_diff * cline_dir[:, 0, :], dim=1)))

            loss = loss_cline #+ loss_dir * 1
            loss.backward()
            optimizer.step()
            #print('time:', time.time()-tic)
            if i % 100 == 0:
                print('loss_cline', loss_cline.item())
                print('loss_dir', loss_dir.item())

            # worm_img = worm_img.numpy()
            # ax1 = plt.subplot(1, 2, 1)
            # ax1.imshow(worm_img[0,0,:,:])
            # ax2 = plt.subplot(1, 2, 2)
            # ax2.imshow(worm_img[0,1,:,:])
            # plt.show()
        #print('epoch time', time.time()-tic1)
        if scheduler.get_lr()[0] >= 2e-6:
            scheduler.step()
        model_name = 'all_dir1_testtime'
        tic = time.time()
        torch.save(model.state_dict(), '../trained_model/'+ model_name + str(epoch_idx) + '.pth')
        print('save time:', time.time()-tic)
    return model

def eval(data_dir, use_gpu=True):
    tsfm = transforms.ToTensor()
    num_epoch = 1
    worm_data = Worm_Cline_Dataset(data_dir, mode='train', tsfm=tsfm)
    batch_size = 1
    data_loader = DataLoader(worm_data, batch_size=batch_size, shuffle=False, num_workers=2)
    model = vgg_cline16_bn(channel_in=2, channel_out=40)
    model_name = 'all_dir1_run3'
    model.load_state_dict(torch.load(os.path.join('../trained_model', model_name + '.pth')))

    if use_gpu:
        model.cuda()
    model.train()
    for i, batch in enumerate(data_loader):
        worm_img = batch['image']
        cline = batch['cline']
        head_pt = batch['head_pt']
        if use_gpu:
            worm_img = worm_img.float().cuda()
            cline = cline.float().cuda()

        cline_out = model(worm_img) * 512
        cline_out = cline_out.view(-1, 20, 2)

        worm_img = worm_img.detach().cpu().numpy()
        cline_out = cline_out.detach().cpu().numpy()
        cline = cline.detach().cpu().numpy()
        plt.imshow(worm_img[0, 0, :, :])
        plt.scatter(cline_out[0, :, 1], cline_out[0, :, 0], c='black', s=0.1)
        #plt.scatter(cline_out[0, 1], cline_out[0, 0], c='black', marker='o')
        plt.scatter(cline[0, :, 1], cline[0, :, 0], c='red', s=0.1)

        plt.scatter(head_pt[0,1], head_pt[0,0], marker='x', c='black')
        plt.show()


if __name__ == "__main__":
    data_folder = "/tigress/LEIFER/Xinwei/github/Pytorch-Unet"
    #data_dir = os.join.path(data_folder, "output")
    model = train(data_folder)
    #eval(data_folder)