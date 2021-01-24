import os
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from ISR.models import RRDN

from common.constants import DATA_ROOT, DEFAULT_INPUT_SIZE
from common.dataset import SatelliteDataset
from common.transforms import create_transforms
from common.loss import SSIM, PSNR, VGGPerceptualLoss


class RRDNEvaluation:
    def __init__(self, weights="gans", patch_size=DEFAULT_INPUT_SIZE, scale_factor=2):
        train_transforms, test_transforms = create_transforms(patch_size, patch_size, scale_factor=scale_factor, same_size_input_label=True)
        self.train_dataloader = DataLoader(SatelliteDataset(DATA_ROOT, train_transforms, is_training_set=True), 1)
        self.test_dataloader = DataLoader(SatelliteDataset(DATA_ROOT, test_transforms, is_training_set=False), 1)
        
        self.rrdn  = RRDN(weights=weights)

    def evaluate(self, evaluate_testset=True):
        dataloader = self.test_dataloader if evaluate_testset else self.train_dataloader
        MSE = nn.MSELoss()
        perceptual_loss = VGGPerceptualLoss()

        self.list_PSNR, self.list_SSIM, self.list_MSE, self.list_VGG = [], [], [], []

        with torch.no_grad():
            total_loss = 0
            for sample in tqdm(dataloader):
                image = sample['image']
                label = sample["label"]
                label = label.squeeze().permute(1,2,0)
                image = image.squeeze().permute(1,2,0) * 255

                output = self.rrdn.predict(image)

                output = transform.resize(output, (label.shape[0], label.shape[1]))
                output = torch.Tensor(output)

                psnr = PSNR(output, label)
                ssim = SSIM(output, label)
                mse = MSE(output, label)

                output = Variable(output).float().cuda().permute(2,0,1).unsqueeze(0)
                label = Variable(label).float().cuda().permute(2,0,1).unsqueeze(0)
                vgg_loss = perceptual_loss(output, label).cpu()

                self.list_PSNR.append(psnr)
                self.list_SSIM.append(ssim)
                self.list_MSE.append(mse)
                self.list_VGG.append(vgg_loss)

        set_used = "test" if evaluate_testset else "train"
        print("Mean PSNR of {:.03f} on {} set with std of {:.03f}".format(np.mean(self.list_PSNR), set_used, np.std(self.list_PSNR)))
        print("Mean SSIM of {:.05f} on {} set with std of {:.05f}".format(np.mean(self.list_SSIM), set_used, np.std(self.list_SSIM)))
        print("Mean MSE of {:.07f} on {} set with std of {:.07f}".format(np.mean(self.list_MSE), set_used, np.std(self.list_MSE)))
        print("Mean VGG-Perceptual of {:.05f} on {} set with std of {:.05f}".format(np.mean(self.list_VGG), set_used, np.std(self.list_VGG)))
