from PIL import Image
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from ISR.models import RRDN
from skimage import transform

from common.constants import DRIVE_ROOT, DATA_ROOT, DEFAULT_INPUT_SIZE
from common.dataset import SatelliteDataset
from common.transforms import create_transforms


class RRDNEvaluator:
    def __init__(self, criterion, weights="gans", patch_size=DEFAULT_INPUT_SIZE):
        self.weights = weights
        self.rrdn  = RRDN(weights=weights, patch_size=patch_size)
        self.criterion = criterion

    def evaluate_set_with_rrdn(self, dataloader, save_folder='train/'):
        SAVE_PATH = DRIVE_ROOT + "gans_results/"
        if not os.path.exists(SAVE_PATH + save_folder):
            os.makedirs(SAVE_PATH + save_folder)

        with torch.no_grad():
            total_loss = 0
            for idx, sample in tqdm(enumerate(dataloader), total=len(dataloader)):
                input = sample['image']
                label = sample["label"]
                input = input.squeeze(0).permute(1,2,0) * 255

                output = self.rrdn.predict(input)
                output = transform.resize(output, (output.shape[0] / 2, output.shape[1] / 2))
                label = label.squeeze().permute(1,2,0)

                loss = self.criterion(torch.Tensor(output), label)
                total_loss += loss.data.item()

            print(f'Total loss for {save_folder[:-1]} dataset with {self.criterion.__repr__()} loss: {total_loss / len(dataloader.dataset)}')


    def evaluate_all(self):
        train_transforms, test_transforms = create_transforms()
        train_dataset = SatelliteDataset(DATA_ROOT, train_transforms, is_training_set=True)
        test_dataset = SatelliteDataset(DATA_ROOT, test_transforms, is_training_set=False)

        train_dataloader = DataLoader(train_dataset, 1)
        test_dataloader = DataLoader(test_dataset, 1)

        self.evaluate_set_with_rrdn(train_dataloader, 'train/')
        self.evaluate_set_with_rrdn(test_dataloader, 'test/')
