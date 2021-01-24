#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 12:42:37 2021

@author: quentin
"""

from skimage import transform
import numpy as np
import torch
import matplotlib.pyplot as plt
from .AE_preprocess import patch_decomp, patch_recomp

def visu(model_CDA,test_dataset,num_images=5,device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
    
    fig, axes = plt.subplots(nrows=num_images, ncols=3, figsize=(3*3,3*num_images))
    axes[0,0].set_title('Original image low res')
    axes[0,1].set_title('Super resolution image')
    axes[0,2].set_title('Original image high res')
    
    indices = np.random.choice(np.arange(len(test_dataset)),num_images,replace=False)

    for i,ind in enumerate(indices):
        sample = test_dataset[ind]
        img = sample["image"].numpy()
        label = sample["label"].numpy()

        img = transform.resize(img, label.shape)

        # Super resolution

        patch_img = torch.FloatTensor(patch_decomp(img)).to(device)
        patch_img = model_CDA(patch_img).cpu().detach().numpy()
        img_super = patch_recomp(patch_img,img.shape)


        #plot
        axes[i,0].imshow(img.transpose((1,2,0)))
        axes[i,0].axis('off')
        axes[i,1].imshow(img_super.transpose((1,2,0)))
        axes[i,1].axis('off')
        axes[i,2].imshow(label.transpose((1,2,0)))
        axes[i,2].axis('off')
    
    fig.tight_layout()
    fig.show()
    