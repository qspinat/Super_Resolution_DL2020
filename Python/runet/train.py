import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
from torchvision import utils
from tqdm import tqdm
from utils.formatter import format_best_checkpoint_name, format_current_checkpoint_name


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def show_imgs(images, titles=["Input", "Output", "Label"]):
    plt.figure(figsize=(20,20))
    for ind, img in enumerate(images):
        plt.subplot(1, len(images), ind+1)
        plt.imshow(utils.make_grid(images[ind]).cpu().numpy().transpose((1,2,0)))
        plt.title(titles[ind])
    plt.show()


def train(model, train_dataloader, test_dataloader, criterion, optimizer, scheduler, n_epochs, log_interval=10):
    best_loss = np.inf
    best_checkpoint_file = format_best_checkpoint_name()
    current_checkpoint_file = format_current_checkpoint_name()
    parent_folder = '/'.join(best_checkpoint_file.split('/')[:-1])
    if not os.path.exists(parent_folder):
        os.makedirs(parent_folder)

    for epoch in range(n_epochs):
        print('Learning rate :', get_lr(optimizer))
        model.train()
        total_train_loss = 0
        for idx, data in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            input, label = data["image"], data["label"]
            input = Variable(input).float().cuda()
            label = Variable(label).float().cuda()
            
            optimizer.zero_grad()

            output = model(input)
            loss = criterion(output, label)

            loss.backward()
            total_train_loss += loss.data.item()
            optimizer.step()

            if idx < 1:
                show_imgs([input, output.detach(), label])

            if idx % log_interval == 0:
              print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                  epoch, idx * len(data), len(train_dataloader.dataset),
                  100. * idx / len(train_dataloader), loss.data.item()))

        print(f'Mean loss on trainset: {total_train_loss}')


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

                if idx < 3:
                    show_imgs([input, output, label])

                if idx % log_interval == 0:
                    print('Test Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, idx * len(data), len(test_dataloader.dataset),
                        100. * idx / len(test_dataloader), loss.data.item()))
            
            print(f'Test loss: {total_loss}')
            if total_loss < best_loss:
                best_loss = total_loss
                print('Saving best model at', best_checkpoint_file)
                torch.save(model.state_dict(), best_checkpoint_file)

        print('Saving current model at', current_checkpoint_file)
        torch.save(model.state_dict(), current_checkpoint_file)
        scheduler.step(total_loss)
    return model
