import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from ISR.models import RRDN
from skimage import transform

from common.constants import DATA_ROOT
from common.dataset import SatelliteDataset
from common.transforms import create_transforms


class GANsVisualizer:
    def __init__(self, weights="gans", img_size=128, scale_factor=2, patch_size=64):
        train_transforms, test_transforms = create_transforms(img_size, img_size, scale_factor=scale_factor, same_size_input_label=False)
        self.train_dataloader = SatelliteDataset(DATA_ROOT, train_transforms, is_training_set=True)
        self.test_dataloader = SatelliteDataset(DATA_ROOT, test_transforms, is_training_set=False)
        self.weights = weights
        self.model = RRDN(weights=weights, patch_size=patch_size)

    def visualize_gans(self, eval_test=True, batch_size=5,seed=1):
        dataloader = self.test_dataloader if eval_test else self.train_dataloader

        fig, axes = plt.subplots(nrows=batch_size, ncols=3, figsize=(3*3,3*batch_size))
        axes[0,0].set_title('Low res (linear interpolation)')
        axes[0,1].set_title('Super resolution')
        axes[0,2].set_title('High res')
        np.random.seed(seed)
        indices = np.random.choice(np.arange(len(dataloader)), batch_size, replace=False)

        with torch.no_grad():
            for i, ind in enumerate(indices):
                sample = dataloader[ind]
                img = sample["image"]
                label = sample["label"]

                img = img.permute(1,2,0) * 255
                output = self.model.predict(img)

                output = transform.resize(output, (output.shape[0] / 2, output.shape[1] / 2))
                label = label.permute(1,2,0)
                img = np.array(img) / 255
                img = transform.resize(img,label.shape)
                output = np.array(output)

                axes[i,0].imshow(img)
                axes[i,0].axis('off')
                axes[i,1].imshow(output)
                axes[i,1].axis('off')
                axes[i,2].imshow(label)
                axes[i,2].axis('off')
            fig.tight_layout(pad=0.25)
