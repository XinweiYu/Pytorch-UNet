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


class Worm_Cline_Dataset(Dataset):
    # this class is used to read worm data for center-line.
    def __init__(self, data_folder, mode='train', tsfm=None):
        # tsfm stands for transform.
        self.data_folder = data_folder
        cline_dir = os.path.join(data_folder, "output")
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

        sample['image'] = torch.cat((worm_img, worm_cline_prev), dim=0)
        sample['cline'] = cline_dict['current_cline'][0:100:5, :]
        return sample

def train(data_dir, use_gpu=True):
    tsfm = transforms.ToTensor()
    num_epoch = 1
    worm_data = Worm_Cline_Dataset(data_dir, mode='train', tsfm=tsfm)
    batch_size = 16
    data_loader = DataLoader(worm_data, batch_size=batch_size, shuffle=True, num_workers=2)
    model = vgg_cline16_bn(channel_in=2, channel_out=40)
    criterion_coord = L1Loss()

    if use_gpu:
        model.cuda()
        criterion_coord.cuda()

    optimizer = Adam(model.parameters(), lr=1.0e-4)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.3)
    tic = time.time()
    for epoch_idx in tqdm(range(num_epoch)):
        for i, batch in enumerate(data_loader):
            worm_img = batch['image']
            cline = batch['cline']
            if use_gpu:
                worm_img = worm_img.cuda()
                cline = cline.cuda().float()
            model.train()
            optimizer.zero_grad()
            cline_out = model(worm_img)
            cline_out = cline_out.view(-1, 20, 2)
            loss_cline = criterion_coord(cline_out, cline)


            loss_cline.backward()
            optimizer.step()
            if i % 100 == 0:
                print('loss_cline', loss_cline.item())
            # worm_img = worm_img.numpy()
            # ax1 = plt.subplot(1, 2, 1)
            # ax1.imshow(worm_img[0,0,:,:])
            # ax2 = plt.subplot(1, 2, 2)
            # ax2.imshow(worm_img[0,1,:,:])
            # plt.show()
    torch.save(model.state_dict(), '../trained_model/cline_model_' + str(epoch_idx) + '.pth')
    return model

if __name__ == "__main__":
    data_folder = "/tigress/LEIFER/Xinwei/github/Pytorch-Unet"
    #data_dir = os.join.path(data_folder, "output")
    model = train(data_folder)