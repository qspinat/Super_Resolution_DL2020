#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 19:56:57 2021

@author: quentin
"""

import torchvision.models as models
import torch
import torch.nn as nn


def perceptual_loss(Img1,Img2,device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
    model = models.vgg16(pretrained=True).to(device)
    model.eval()
    idx = torch.tensor([3,8,15,22,29]) #[1,3,6,8,11,13,15,18,20,22,25,27,29]
    features1 = model.features(Img1)[:,idx]
    features2 = model.features(Img2)[:,idx]
    loss = nn.MSELoss()(features1,features2)
    return loss
    