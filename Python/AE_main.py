#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 13:39:45 2021

@author: quentin
"""

import os
import torch
import numpy as np

os.chdir('/media/OS/Users/Quentin/Documents/ENPC/3A/DL/Projet/Super_Resolution_DL2020/Python')

from utils.visualizer import Visualizer
from common.dataset import SatelliteDataset
from common.transforms import create_transforms

DATA_ROOT = "/content/drive/MyDrive/road_segmentation_ideal/"

train_transforms, test_transforms = create_transforms()
train_dataset = SatelliteDataset(DATA_ROOT, train_transforms, is_training_set=True)
test_dataset = SatelliteDataset(DATA_ROOT, test_transforms, is_training_set=False)

visualizer = Visualizer()

print('Training example')
visualizer.visualize_sample(train_dataset[0])

print('Test example')
visualizer.visualize_sample(test_dataset[0])

