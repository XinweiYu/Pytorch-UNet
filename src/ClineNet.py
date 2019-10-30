# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 11:38:41 2019
This is for getting centerline from image with Neural Network
@author: xinweiy
"""
import torch
import torch.nn as nn
import math

cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M',
          512, 512, 512, 512, 'M'],
    'Cline': [32, 32, 'M', 64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
}

def make_layers(cfg, in_channels, batch_norm=False):
    layers = []
    #in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

class VGG_Cline(nn.Module):
    '''
    VGG model for predicting centerline from image.
    '''
    def __init__(self, features, channel_out):
        super(VGG_Cline, self).__init__()
        self.channel_out = channel_out
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, self.channel_out),
        )
         # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()


    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def vgg_cline16(channel_in, channel_out):
    """VGG 16-layer model (configuration "D")"""
    return VGG_Cline(make_layers(cfg['Cline'], channel_in), channel_in, channel_out)

def vgg_cline16_bn(channel_in, channel_out):
    """VGG 16-layer model (configuration "D") with batch normalization"""
    return VGG_Cline(make_layers(cfg['Cline'], channel_in, batch_norm=True), channel_out)