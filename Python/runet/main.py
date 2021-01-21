import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from common.constants import DATA_ROOT
from common.dataset import SatelliteDataset
from common.transforms import create_transforms_runet
from runet.runet import RUNet
from runet.train import train


def train_runet(train_bs = 32, test_bs = 1):
    model = RUNet()
    model = model.cuda()

    train_transforms, test_transforms = create_transforms_runet(64, 64, same_size_input_label=True)
    train_dataset = SatelliteDataset(DATA_ROOT, train_transforms, is_training_set=True)
    test_dataset = SatelliteDataset(DATA_ROOT, test_transforms, is_training_set=False)

    train_dataloader = DataLoader(train_dataset, batch_size=train_bs, shuffle=True, num_workers=20)
    test_dataloader = DataLoader(test_dataset, batch_size=test_bs, shuffle=False, num_workers=0)

    optimizer = optim.Adam(model.parameters(), lr=0.01)
    scheduler = ReduceLROnPlateau(optimizer, 'min')

    # criterion = PERCEPTUAL_LOSS
    criterion = nn.MSELoss()

    train(model, train_dataloader, test_dataloader, criterion, optimizer, scheduler, n_epochs=300)
