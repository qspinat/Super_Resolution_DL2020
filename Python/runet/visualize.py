import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import utils

from common.transforms import create_transforms
from common.dataset import SatelliteDataset
from common.constants import DATA_ROOT
from runet.runet import RUNet


class RUNetVisualizer:
    def __init__(self, img_size=128, scale_factor=2,device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        train_transforms, test_transforms = create_transforms(img_size, img_size, scale_factor=scale_factor, same_size_input_label=True)
        self.train_dataloader = SatelliteDataset(DATA_ROOT, train_transforms, is_training_set=True)
        self.test_dataloader = SatelliteDataset(DATA_ROOT, test_transforms, is_training_set=False)
        self.device=device
        self.model = RUNet().to(device)
        self.model.eval()

    def format_image(self, img):
        return utils.make_grid(img).cpu().numpy().transpose((1,2,0))

    def visualize_batch(self, image, output, label):
        plt.figure(figsize=(20,20))
        plt.subplot(1,3,1)
        plt.imshow(self.format_image(image))
        
        plt.subplot(1,3,2)
        plt.imshow(self.format_image(output))
        
        plt.subplot(1,3,3)
        plt.imshow(self.format_image(label))
        plt.show()

    def visualize_runet(self, checkpoint, eval_test=True, batch_size=5,seed=1):
        dataloader = self.test_dataloader if eval_test else self.train_dataloader

        print(f"Loading RUNet weights from {checkpoint}")
        self.model.load_state_dict(torch.load(checkpoint))

        fig, axes = plt.subplots(nrows=batch_size, ncols=3, figsize=(3*3,3*batch_size))
        axes[0,0].set_title('Low res (linear interpolation)')
        axes[0,1].set_title('Super resolution')
        axes[0,2].set_title('High res')
        np.random.seed(seed)
        indices = np.random.choice(np.arange(len(dataloader)),batch_size,replace=False)

        with torch.no_grad():
            for i,ind in enumerate(indices):
                sample = dataloader[ind]
                img = sample["image"][None,:,:,:].float().to(self.device)
                label = sample["label"][None,:,:,:].float().to(self.device)
                img_super = self.model(img)

                axes[i,0].imshow(self.format_image(img))
                axes[i,0].axis('off')
                axes[i,1].imshow(self.format_image(img_super))
                axes[i,1].axis('off')
                axes[i,2].imshow(self.format_image(label))
                axes[i,2].axis('off')
            fig.tight_layout(pad=0.25)

            #for idx, data in enumerate(dataloader):
            #    if idx > limit:
            #        break
            #    image = Variable(data['image']).float().cuda()
            #    label = Variable(data['label']).float().cuda()
            #    output = self.model(image)

            #    self.visualize_batch(image, output, label)
