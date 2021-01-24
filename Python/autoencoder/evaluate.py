import torch
import torch.nn as nn
import numpy as np
from skimage import transform
from .AE_preprocess import patch_decomp, patch_recomp
from .loss import SSIM,PSNR,VGGPerceptualLoss

def test_model(model,test_dataset,scale=2,device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
    N = len(test_dataset)
    list_PSNR = np.zeros(N)
    list_SSIM = np.zeros(N)
    list_MSE = np.zeros(N)
    list_VGG = np.zeros(N)
    for i in range(N):
        sample= test_dataset[i]
        img = sample["image"].numpy()
        label = sample["label"]
        img = transform.resize(img, label.shape)
        patch_img = torch.FloatTensor(patch_decomp(img,patch_size=scale*3)).to(device)
        patch_img = model(patch_img).cpu().detach().numpy()
        img_super = torch.FloatTensor(patch_recomp(patch_img,img.shape))
        list_PSNR[i] = PSNR(img_super,label)
        list_SSIM[i] = SSIM(img_super,label)
        list_MSE[i] = nn.MSELoss()(img_super,label)
        img_super = img_super[None,:,:,:].to(device)
        label = label.float()[None,:,:,:].to(device)
        list_VGG[i] = VGGPerceptualLoss()(img_super,label).cpu().detach().numpy()
        
    res = np.zeros((4,2))
    res[0,0] = np.mean(list_PSNR)
    res[1,0] = np.mean(list_SSIM)
    res[2,0] = np.mean(list_MSE)
    res[3,0] = np.mean(list_VGG)
    res[0,1] = np.std(list_PSNR)
    res[1,1] = np.std(list_SSIM)
    res[2,1] = np.std(list_MSE)
    res[3,1] = np.std(list_VGG)

    print("Mean PSNR of {:.03f} on test set with std of {:.03f}".format(res[0,0],res[0,1]))
    print("Mean SSIM of {:.05f} on test set with std of {:.05f}".format(res[1,0],res[1,1]))
    print("Mean MSE of {:.07f} on test set with std of {:.07f}".format(res[2,0],res[2,1]))
    print("Mean VGG-Perceptual of {:.05f} on test set with std of {:.05f}".format(res[3,0],res[3,1]))
    
    return res
