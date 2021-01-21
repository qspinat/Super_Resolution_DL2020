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
    features1 = model.features[:4](Img1)
    features2 = model.features[:4](Img2)
    loss = nn.MSELoss()(features1,features2)
    for i in range(1,idx.shape[0]):
        features1 = model.features[idx[i-1]:idx[i]+1](features1)
        features2 = model.features[idx[i-1]:idx[i]+1](features2)
        loss = loss + nn.MSELoss()(features1,features2)
    return loss
    
def PSNR(Img_pred,Img_true):
    return 10*torch.log10(torch.max(Img_pred)**2/nn.MSELoss()(Img_pred,Img_true))

def SSIM(Img_pred,Img_true):
    L = torch.max(Img_true)-torch.min(Img_true)
    k1 = 0.01
    k2 = 0.03
    c1 = (k1*L)**2
    c2 = (k2*L)**2
    mu1 = torch.mean(Img_pred)
    mu2 = torch.mean(Img_true)
    sig1 = torch.var(Img_pred)
    sig2 = torch.var(Img_true)
    sig12 = torch.mean((Img_pred-mu1)*(Img_true-mu2))
    return (2*mu1*mu2+c1)*(2*sig12+c2)/(mu1**2+mu2**2+c1)/(sig1+sig2+c2)
    
    