import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from common.loss import SSIM, PSNR, VGGPerceptualLoss
from common.transforms import create_transforms
from common.dataset import SatelliteDataset
from common.constants import DATA_ROOT
from runet.runet import RUNet


class RUNetEvaluation:
    def __init__(self, img_size=128, scale_factor=2):
        train_transforms, test_transforms = create_transforms(img_size, img_size, scale_factor=scale_factor, same_size_input_label=True)
        self.train_dataloader = DataLoader(SatelliteDataset(DATA_ROOT, train_transforms, is_training_set=True), 1)
        self.test_dataloader = DataLoader(SatelliteDataset(DATA_ROOT, test_transforms, is_training_set=False), 1)

        self.model = RUNet().cuda()
        self.model.eval()

    def evaluate(self, checkpoint, evaluate_testset=True):
        dataloader = self.test_dataloader if evaluate_testset else self.train_dataloader
        MSE = nn.MSELoss()
        perceptual_loss = VGGPerceptualLoss()

        print(f"Loading RUNet weights from {checkpoint}")
        self.model.load_state_dict(torch.load(checkpoint))

        self.list_PSNR, self.list_SSIM, self.list_MSE, self.list_VGG = [], [], [], []

        with torch.no_grad():
            for data in tqdm(dataloader):
                image = Variable(data['image']).float().cuda()
                label = Variable(data['label']).float().cuda()
                
                output = self.model(image)
                
                psnr = PSNR(output, label).cpu()
                ssim = SSIM(output, label).cpu()
                mse = MSE(output, label).cpu()
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
