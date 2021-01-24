import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from common.constants import DATA_ROOT
from common.dataset import SatelliteDataset
from common.transforms import create_transforms_runet, create_transforms
from runet.runet import RUNet
from runet.train import train
from runet.loss import VGGPerceptualLoss


def train_runet(img_size = 64, train_bs = 16, test_bs = 1, lr=0.01, n_epochs=300):
    model = RUNet()
    model = model.cuda()

    #train_transforms, test_transforms = create_transforms_runet(img_size, img_size, same_size_input_label=True)
    train_transforms, test_transforms = create_transforms(img_size, img_size, same_size_input_label=True)
    train_dataset = SatelliteDataset(DATA_ROOT, train_transforms, is_training_set=True)
    test_dataset = SatelliteDataset(DATA_ROOT, test_transforms, is_training_set=False)

    train_dataloader = DataLoader(train_dataset, batch_size=train_bs, shuffle=True, num_workers=20)
    test_dataloader = DataLoader(test_dataset, batch_size=test_bs, shuffle=False, num_workers=0)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, 'min')

    criterion = VGGPerceptualLoss()
    #criterion = nn.MSELoss()

    train(model, train_dataloader, test_dataloader, criterion, optimizer, scheduler, n_epochs=n_epochs)
