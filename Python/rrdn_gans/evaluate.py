from PIL import Image
import torch
from tqdm import tqdm
import os
from ISR.models import RRDN

from common.constants import DRIVE_ROOT, DATA_ROOT, DEFAULT_INPUT_SIZE
from common.dataset import SatelliteDataset
from common.transforms import create_transforms


class RRDNEvaluator:
    def __init__(self, weights="gans", patch_size=DEFAULT_INPUT_SIZE):
        self.weights = weights
        self.rrdn  = RRDN(weights=WEIGHTS, patch_size=patch_size)

    def evaluate_set_with_rrdn(self, dataset, save_folder='train/'):
        SAVE_PATH = DRIVE_ROOT + "gans_results/"
        if not os.path.exists(SAVE_PATH + save_folder):
            os.makedirs(SAVE_PATH + save_folder)

        with torch.no_grad():
            for idx, sample in tqdm(enumerate(train_dataset), total=len(train_dataset)):
                input = sample['image']
                input = input.squeeze(0).permute(1,2,0) * 255

                output = self.rrdn.predict(input)
                
                Image.fromarray(output).save(SAVE_PATH + save_folder + f'img-{idx}.png')


    def evaluate_all(self):
        train_transforms, test_transforms = create_transforms()
        train_dataset = SatelliteDataset(DATA_ROOT, train_transforms, is_training_set=True)
        test_dataset = SatelliteDataset(DATA_ROOT, test_transforms, is_training_set=False)

        self.evaluate_set_with_rrdn(train_dataset, 'train/')
        self.evaluate_set_with_rrdn(test_dataset, 'test/')
