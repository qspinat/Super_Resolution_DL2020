import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import utils

from common.transforms import create_transforms
from common.dataset import SatelliteDataset
from common.constants import DATA_ROOT
from runet.runet import RUNet


class RUNetVisualizer:
    def __init__(self, img_size=128, scale_factor=2):
        train_transforms, test_transforms = create_transforms(img_size, img_size, scale_factor=scale_factor, same_size_input_label=True)
        self.train_dataloader = DataLoader(SatelliteDataset(DATA_ROOT, train_transforms, is_training_set=True), 1)
        self.test_dataloader = DataLoader(SatelliteDataset(DATA_ROOT, test_transforms, is_training_set=False), 1)

        self.model = RUNet().cuda()
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

    def visualize_runet(self, checkpoint, eval_test=True, limit=10):
        dataloader = self.test_dataloader if eval_test else self.train_dataloader

        print(f"Loading RUNet weights from {checkpoint}")
        self.model.load_state_dict(torch.load(checkpoint))

        with torch.no_grad():
            for idx, data in enumerate(dataloader):
                if idx > limit:
                    break
                image = Variable(data['image']).float().cuda()
                label = Variable(data['label']).float().cuda()
                
                output = self.model(image)
                
                self.visualize_batch(image, output, label)
