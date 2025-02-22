#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 16:57:51 2024

@author: rajs
"""

import torch
from torchvision import transforms
from torch.nn import BCELoss, BCEWithLogitsLoss, MSELoss, L1Loss
from torchvision.models import vgg11, VGG11_Weights

class BCE:
    def __init__(self):
        # self.criterion = BCELoss()
        self.criterion = BCEWithLogitsLoss()

    def compute(self, prediction, ground_truth):
        return self.criterion(prediction, ground_truth)
          
            
class MSE:
    def __init__(self):
        self.criterion = MSELoss()
    
    def compute(self, prediction, ground_truth):
        return self.criterion(prediction, ground_truth)
    

class MAE:
    def __init__(self):
        self.criterion = L1Loss()

    def compute(self, prediction, ground_truth):
        return self.criterion(prediction, ground_truth)
        
        

'''
Info on downloading pretraiend models
pip install pretrainedmodels

import pretrainedmodels

>>> print(pretrainedmodels.model_names)
['fbresnet152', 'bninception', 'resnext101_32x4d', 'resnext101_64x4d', 'inceptionv4', 'inceptionresnetv2', 'alexnet', 'densenet121', 'densenet169', 'densenet201', 'densenet161', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'inceptionv3', 'squeezenet1_0', 'squeezenet1_1', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19_bn', 'vgg19', 'nasnetamobile', 'nasnetalarge', 'dpn68', 'dpn68b', 'dpn92', 'dpn98', 'dpn131', 'dpn107', 'xception', 'senet154', 'se_resnet50', 'se_resnet101', 'se_resnet152', 'se_resnext50_32x4d', 'se_resnext101_32x4d', 'cafferesnet101', 'pnasnet5large', 'polynet']

>>> print(pretrainedmodels.pretrained_settings['vgg11'])
{'imagenet': {'url': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth', 
              'input_space': 'RGB', 
              'input_size': [3, 224, 224], 
              'input_range': [0, 1], 
              'mean': [0.485, 0.456, 0.406], 
              'std': [0.229, 0.224, 0.225], 
              'num_classes': 1000}}

'''

class Perceptual:
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    perceptual_layer = ['4', '9', '14', '19']

    def __init__(self, device='cpu'):
        # self.model = torch.hub.load('pytorch/vision:v0.13.0', 'vgg11', pretrained = True)
        self.model = vgg11(weights=VGG11_Weights.IMAGENET1K_V1)
        self.model.to(device)
        self.criterion = MAE()
    
    def compute(self, prediction, ground_truth):
        prediction = self.preprocess(prediction)
        ground_truth = self.preprocess(ground_truth)
        
        loss = 0
        for layer, module in self.model.features._modules.items():
            prediction = module(prediction)
            ground_truth = module(ground_truth)
            if layer in self.perceptual_layer:
                loss += self.criterion.compute(prediction, ground_truth)
        return loss


if __name__ == '__main__':
    print('help: to check torch hub directory: ', torch.hub.get_dir())
    device = torch.device('cuda:1')
    p = Perceptual(device = device)
    
    x = torch.rand((3, 3, 640, 480)).to(device)
    y = torch.rand((3, 3, 640, 480)).to(device)
    loss = p.compute(x, y)
    print(loss)
    
