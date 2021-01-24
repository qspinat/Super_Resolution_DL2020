import time
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from skimage import transform
import cv2
from .loss import SSIM,PSNR,VGGPerceptualLoss


class Cv2ResizerEvaluation():
    def __init__(self, display_time=False):
        self.display_time = display_time

    def interpolate(self, img, output_size):
        start = time.time()
        interpolate_result = cv2.resize(img, output_size, interpolation=cv2.INTER_LINEAR)
        
        if self.display_time:
            print(f'Took {time.time() - start}s to compute a linear interpolation on an image of shape {img.shape}')
        
        return interpolate_result

    def evaluate_dataset(self, dataset):
        total_loss = 0
        for data in tqdm(dataset):
            image = data['image'].permute(1,2,0).numpy()
            label = data['label'].permute(1,2,0).numpy()

            interpolated = self.interpolate(image, output_size=label.shape[:2])
            loss = np.linalg.norm(label - interpolated)
            total_loss += loss

        print(f'Mean L2 loss on the dataset : {total_loss / len(dataset)}')

    def evaluate_dataset_bis(self, dataset, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        N = len(dataset)
        list_PSNR = np.zeros(N)
        list_SSIM = np.zeros(N)
        list_MSE = np.zeros(N)
        list_VGG = np.zeros(N)
        for i in tqdm(range(N)):
            sample= dataset[i]
            img = sample["image"]
            label = sample["label"]
            img_super = torch.FloatTensor(transform.resize(img, label.shape,order=1))
            list_PSNR[i] = PSNR(img_super,label)
            list_SSIM[i] = SSIM(img_super,label)
            list_MSE[i] = nn.MSELoss()(img_super,label)
            img_super = img_super[None,:,:,:].float().to(device)
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
