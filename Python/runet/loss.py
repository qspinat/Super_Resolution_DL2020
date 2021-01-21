#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 19:56:57 2021

@author: quentin
"""

import torchvision.models as models
import torch
import torch.nn as nn


class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True,device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(models.vgg16(pretrained=True).features[:4].to(device).eval())
        blocks.append(models.vgg16(pretrained=True).features[4:9].to(device).eval())
        blocks.append(models.vgg16(pretrained=True).features[9:16].to(device).eval())
        blocks.append(models.vgg16(pretrained=True).features[16:23].to(device).eval())
        for bl in blocks:
            for p in bl:
                p.requires_grad = False
        self.device=device
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.mean = torch.nn.Parameter(torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)).to(device)
        self.std = torch.nn.Parameter(torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)).to(device)
        self.resize = resize

    def forward(self, input, target):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1).to(self.device)
            target = target.repeat(1, 3, 1, 1).to(self.device)
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for block in self.blocks:
            x = block(x)
            y = block(y)
            loss = loss + nn.MSELoss()(x, y)
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
    
    