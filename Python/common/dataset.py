from glob import glob
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


class SatelliteDataset(Dataset):
  def __init__(self, data_root, transformations, is_training_set=True):
    directory = 'training/' if is_training_set else 'testing/'
    self.path = data_root + directory
    self.input_paths = glob(self.path + 'input/**')
    self.transformations = transformations

  def __getitem__(self, index):
    img_path = self.input_paths[index]
    img = Image.open(img_path)

    sample = { 'image': np.array(img), 'label': np.array(img) }

    if self.transformations:
      sample = self.transformations(sample)
    
    return sample

  def __len__(self):
    return len(self.input_paths)
