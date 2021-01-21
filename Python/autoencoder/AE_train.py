import numpy as np
import torch 
import torch.nn as nn
import torch.optim as optim
import collections
from .AE_model import *

#%% preprocess

# TO DO : 
#   - create funtion that from a dataset of images, gives a dataset of patches
#   - upscale LR image with bicubib interpolation

#%% training

def train_AE(model_AE,train_loader,epoch,lr=0.01,log_interval=1000,criterion="MSE"):
    model_AE.train()
    optimizer = optim.Adam(model_AE.parameters(),lr=lr)
    if criterion=="MSE":
        criterion = nn.MSELoss()
    elif criterion=="perceptual":
        criterion = VGGPerceptualLoss()
    for batch_idx, (data, target) in enumerate(train_loader): #to modify
        optimizer.zero_grad()
        output = model_AE(data)
        loss = criterion(output, data)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data.item()))
    return

def train_mapping(LR_AE,model_mapping,HR_AE,train_loader,epoch,lr=0.001,log_interval=1000,criterion="MSE"):
    model_mapping.train()
    optimizer = optim.Adam(model_mapping.parameters(),lr=lr)
    if criterion=="MSE":
        criterion = nn.MSELoss()
    elif criterion=="perceptual":
        criterion = VGGPerceptualLoss()
    for batch_idx, (data, target) in enumerate(train_loader):
        hl = LR_AE(data,path="encoding")
        hh = HR_AE(target,path="encoding")
        optimizer.zero_grad()
        output = model_mapping(hl)
        loss = criterion(output, hh)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data.item()))
            
    return
            
def fine_tuning(model_CDA,train_loader,epoch,lr=0.001,log_interval=1000,criterion="MSE"):
    model_CDA.train()
    optimizer = optim.Adam(model_CDA.parameters(),lr=lr)
    if criterion=="MSE":
        criterion = nn.MSELoss()
    elif criterion=="perceptual":
        criterion = VGGPerceptualLoss()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model_CDA(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data.item()))
            
    return
            
def train_global(model_CDA,LR_AE,model_mapping,HR_AE,patch_LR,patch_HR,num_epochs,bs,log_interval=1000):
    
    #step1
    print("LR autoencoder training")
    train_dataset = torch.utils.data.TensorDataset(patch_LR,patch_LR)
    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=bs)
    for epoch in range(1,num_epochs+1):
        train_AE(LR_AE,train_loader,epoch,log_interval=1000)

    #step2
    print("HR autoencoder training")
    train_dataset = torch.utils.data.TensorDataset(patch_HR,patch_HR)
    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=bs)
    for epoch in range(1,num_epochs+1):
        train_AE(HR_AE,train_loader,epoch,log_interval=1000)
    
    #step3
    print("mapping training")
    train_dataset = torch.utils.data.TensorDataset(patch_LR,patch_HR)
    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=bs)
    for epoch in range(1,num_epochs+1):
        train_mapping(LR_AE,model_mapping,HR_AE,train_loader,epoch,log_interval=1000)
    
    #step4
    #copy weights from step 1 2 and 3
    model_CDA.load_state_dict(collections.OrderedDict([('enc.weight',LR_AE.enc.weight),
                                           ('enc.bias',LR_AE.enc.bias),
                                           ('map.weight',model_mapping.map.weight),
                                           ('map.bias',model_mapping.map.bias),
                                           ('dec.weight',HR_AE.dec.weight),
                                           ('dec.bias',HR_AE.dec.bias)
                                                ]))
    #fine_tuning
    for epoch in range(1,num_epochs+1):
      fine_tuning(model_CDA,train_loader,epoch,log_interval=1000)
    
    return