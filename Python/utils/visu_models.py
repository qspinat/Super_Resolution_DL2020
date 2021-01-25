import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from torchvision import utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from ISR.models import RRDN
from skimage import transform
import matplotlib.pyplot as plt
from autoencoder.AE_preprocess import patch_decomp, patch_recomp
from common.constants import DRIVE_ROOT, DATA_ROOT, DEFAULT_INPUT_SIZE
from common.dataset import SatelliteDataset
from common.transforms import create_transforms
from runet.runet import RUNet

def visu_all(model_CDA,model_RUNET,model_RRDN,test_dataset,num_images=5,device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),image_id=0):

    if num_images>1:    
      fig, axes = plt.subplots(nrows=num_images, ncols=6, figsize=(2.5*6,2.5*num_images))
      axes[0,0].set_title('Bilinear interpolation')
      axes[0,1].set_title('Spline-5 interpolation')
      axes[0,2].set_title('CDA')
      axes[0,3].set_title('RUNET')
      axes[0,4].set_title('ESRGAN')
      axes[0,5].set_title('High res')
      indices = np.random.choice(np.arange(len(test_dataset)),num_images,replace=False)

    elif num_images==1:
      fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(2.5*3,2.5*2))
      axes[0,0].set_title('Bilinear interpolation')
      axes[0,1].set_title('Spline-5 interpolation')
      axes[0,2].set_title('CDA')
      axes[1,0].set_title('RUNET')
      axes[1,1].set_title('ESRGAN')
      axes[1,2].set_title('High res')      
      indices = [int(image_id)]


    for i,ind in enumerate(indices):
        sample = test_dataset[ind]
        img = sample["image"]
        label = sample["label"].numpy()

        linear = transform.resize(img.numpy(), label.shape)
        spline = transform.resize(img.numpy(), label.shape, order=5)

        ############# Super resolution ##############

        #### CDA

        patch_img = torch.FloatTensor(patch_decomp(linear)).to(device)
        patch_img = model_CDA(patch_img).cpu().detach().numpy()
        img_CDA = patch_recomp(patch_img,linear.shape)

        ### RUNET

        linear = torch.FloatTensor(linear)[None,:,:,:].to(device)
        img_RUNET = model_RUNET(linear).cpu().detach().numpy().squeeze()
        linear = linear.cpu().detach().numpy().squeeze()

        ### ERSGAN

        img = img.permute(1,2,0) * 255
        img_GAN = model_RRDN.predict(img)
        img_GAN = transform.resize(img_GAN, (img_GAN.shape[0] / 2, img_GAN.shape[1] / 2))
        img_GAN = np.array(img_GAN)

        #plot
        if num_images>1:
            axes[i,0].imshow(linear.transpose((1,2,0)))
            axes[i,0].axis('off')
            axes[i,1].imshow(spline.transpose((1,2,0)))
            axes[i,1].axis('off')
            axes[i,2].imshow(img_CDA.transpose((1,2,0)))
            axes[i,2].axis('off')
            axes[i,3].imshow(img_RUNET.transpose((1,2,0)))
            axes[i,3].axis('off')
            axes[i,4].imshow(img_GAN)
            axes[i,4].axis('off')
            axes[i,5].imshow(label.transpose((1,2,0)))
            axes[i,5].axis('off')
        elif num_images==1:
            axes[0,0].imshow(linear.transpose((1,2,0)))
            axes[0,0].axis('off')
            axes[0,1].imshow(spline.transpose((1,2,0)))
            axes[0,1].axis('off')
            axes[0,2].imshow(img_CDA.transpose((1,2,0)))
            axes[0,2].axis('off')
            axes[1,0].imshow(img_RUNET.transpose((1,2,0)))
            axes[1,0].axis('off')
            axes[1,1].imshow(img_GAN)
            axes[1,1].axis('off')
            axes[1,2].imshow(label.transpose((1,2,0)))
            axes[1,2].axis('off')
    
    fig.tight_layout(pad=0.25)
    fig.show()

