import os
import numpy as np
import torch
from torch.autograd import Variable
from tqdm import tqdm
from utils.formatter import format_checkpoint_name


def train(model, train_dataloader, test_dataloader, criterion, optimizer, scheduler, n_epochs, log_interval=100):
    best_loss = np.inf
    checkpoint_file = format_checkpoint_name()
    parent_folder = '/'.join(checkpoint_file.split('/')[:-1])
    if not os.path.exists(parent_folder):
        os.makedirs(parent_folder)

    for epoch in range(n_epochs):
        model.train()
        for idx, data in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            input, label = data["image"], data["label"]
            input = Variable(input).float().cuda()
            label = Variable(label).float().cuda()
            
            optimizer.zero_grad()

            output = model(input)
            loss = criterion(output, label)

            loss.backward()
            optimizer.step()
            if idx % log_interval == 0:
              print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                  epoch, idx * len(data), len(train_dataloader.dataset),
                  100. * idx / len(train_dataloader), loss.data.item()))

        model.eval()
        with torch.no_grad():
            total_loss = 0
            for idx, data in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
                input, label = data["image"], data["label"]
                input = Variable(input).float().cuda()
                label = Variable(label).float().cuda()
                
                optimizer.zero_grad()

                output = model(input)
                loss = criterion(output, label)

                optimizer.step()

                total_loss += loss
            
            print(f'Test loss: {total_loss}')
            if total_loss < best_loss:
                best_loss = total_loss
                print('Saving best model at', checkpoint_file)
                torch.save(model.state_dict(), checkpoint_file)
        scheduler.step(total_loss)
    return model
