import sys
import os
from optparse import OptionParser
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler
from eval import eval_net
from unet import UNet
from utils import get_ids, split_ids, split_train_val, get_imgs_and_masks, batch, BinarizeMask
import time
import copy

    
def train_net(net,
              epochs=100,
              batch_size=1,
              lr=0.1,
              val_percent=0.05,
              save_cp=True,
              gpu=True,
              img_scale=0.5,
              dir_mask = 'data/train_masks/',
              n_classes=1):

    dir_img = '/scratch/network/xinweiy/data/train/'
    #dir_mask = 'data/train_masks/'
    dir_checkpoint = '/home/xinweiy/github/checkpoints/'
    # the best result for validation.
    best_dice = 0.0
    #Returns a list of the ids in the directory
    ids = get_ids(dir_img)
    # what is this?
    ids = split_ids(ids)

    iddataset = split_train_val(ids, val_percent)

    print('''
    Starting training:
        Epochs: {}
        Batch size: {}
        Class number: {}
        Learning rate: {}
        Training size: {}
        Validation size: {}
        Checkpoints: {}
        CUDA: {}
    '''.format(epochs, batch_size, n_classes, lr, len(iddataset['train']),
               len(iddataset['val']), str(save_cp), str(gpu)))

    N_train = len(iddataset['train'])

    optimizer = optim.SGD(net.parameters(),
                          lr=lr,
                          momentum=0.9,
                          weight_decay=0.0005)

    # Decay LR by a factor of 0.1 every 7 epochs
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    criterion = nn.BCELoss()
    
    for epoch in range(epochs):
        print('Starting epoch {}/{}.'.format(epoch + 1, epochs))
        net.train()
        # step variant learning rate.
        scheduler.step()
        # reset the generators
        train = get_imgs_and_masks(iddataset['train'], dir_img, dir_mask, img_scale)
        val = get_imgs_and_masks(iddataset['val'], dir_img, dir_mask, img_scale)

        epoch_loss = 0
            
        for i, b in enumerate(batch(train, batch_size)):
            imgs = np.array([i[0] for i in b]).astype(np.float32)
            true_masks = np.array([i[1] for i in b])
            #print(true_masks.shape)
            true_masks = BinarizeMask(true_masks,n_classes)
            
            imgs = torch.from_numpy(imgs)
            true_masks = torch.from_numpy(true_masks)

            if gpu:
                imgs = imgs.cuda()
                true_masks = true_masks.cuda()

            masks_pred = net(imgs)
            masks_probs_flat = masks_pred.view(-1)

            true_masks_flat = true_masks.view(-1)

            loss = criterion(masks_probs_flat, true_masks_flat)
            epoch_loss += loss.item()
	    
            if not np.mod(i,int(N_train/5)):
                print('{0:.4f} --- loss: {1:.6f}'.format(i * batch_size / N_train, loss.item()))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('Epoch finished ! Loss: {}'.format(epoch_loss / i))

        if 1:
            val_dice = eval_net(net, val, gpu, n_classes)
            print('Validation Dice Coeff: {}'.format(val_dice))
            if val_dice > best_dice:
                best_dice = val_dice
                best_model_wts = copy.deepcopy(net.state_dict())

#        if save_cp and not np.mod(epoch+1,10):
#            torch.save(net.state_dict(),dir_checkpoint + 'Chl{}CP{}.pth'.format(n_classes, epoch + 1))                           
#            print('Checkpoint {} saved !'.format(epoch + 1))
        
    if save_cp:
        torch.save(net.state_dict(),dir_checkpoint + 'Channel{}dice{}.pth'.format(n_classes,best_dice))                           
        print('Model saved, best val_dice:{} !'.format(best_dice))
                
def get_args():
    parser = OptionParser()
    parser.add_option('-e', '--epochs', dest='epochs', default=400, type='int',
                      help='number of epochs')
    parser.add_option('-b', '--batch-size', dest='batchsize', default=10,
                      type='int', help='batch size')
    parser.add_option('-l', '--learning-rate', dest='lr', default=1e-1,
                      type='float', help='learning rate')
    parser.add_option('-g', '--gpu', action='store_true', dest='gpu',
                      default=True, help='use cuda')
    parser.add_option('-c', '--load', dest='load',
                      default=False, help='load file model')
    parser.add_option('-s', '--scale', dest='scale', type='float',
                      default=0.5, help='downscaling factor of the images')

    parser.add_option('-m', '--mask', dest='mask', type='int',
                      default=1, help='which mask to use')

    (options, args) = parser.parse_args()
    return options

if __name__ == '__main__':
    args = get_args()

    if int(args.mask)==1:
        n_classes = 1
        dir_mask = '/scratch/network/xinweiy/data/train_mask_centerline/'
    if int(args.mask)==2:
        n_classes = 2
        dir_mask = '/scratch/network/xinweiy/data/train_mask_tip/'
    if int(args.mask)==3:
        n_classes = 8
        dir_mask = '/scratch/network/xinweiy/data/train_mask_direction/'


    net = UNet(n_channels=3, n_classes=n_classes)

    if args.load:
        net.load_state_dict(torch.load(args.load))
        print('Model loaded from {}'.format(args.load))

    if args.gpu:
        net.cuda()
        # cudnn.benchmark = True # faster convolutions, but more memory
    tic=time.time()
    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  gpu=True,#args.gpu,
                  img_scale=args.scale,
                  dir_mask = dir_mask,
                  n_classes=n_classes)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        print('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
    toc=time.time()
    print('time is {}'.format(toc-tic))