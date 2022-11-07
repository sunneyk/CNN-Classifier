# File system and OS
import sys
import os
from unittest import TestLoader
import CNN_function as functions
import pandas as pd
import numpy as np
# PyTorch
import torch
from torch import save, load
import torch.nn as nn
from torch import optim
from tqdm import tqdm
import models as md

file = 'data.pkl'
save_model_flag = True
load_model_flag = False
load_model = 'model_state.pt'
# model
train_model_flag = True
num_conv1 = 32    
kernel_size = 3   
bn_mom = 0.1
batch_size = 64
num_epochs = 20
# Loss
loss_function = 'MSE'
# Optimizer
opt = 'Adam'
lr = 0.001
momentum = 0.5
model_name = "CNN8_Filters{}_{}_{}_LR{}".format(num_conv1, loss_function, opt, lr)
load_model = "models" + os.path.sep + model_name + ".pt"
save_model = "models" + os.path.sep + model_name + ".pt"


if __name__ == "__main__":
    device, gpu_flag = functions.gpuCheck()
    model = md.Net(num_conv1, kernel_size, bn_mom)
    model = model.to(device)

    # Build Pytorch data loader
    trainloader, testloader = functions.loadData(file, batch_size, device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor = 0.5, patience = 30, verbose = True)

    if load_model_flag:
        print("\nLoading Model: {}\n".format(load_model))
        checkpoint = torch.load(load_model)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        prev_epochs = checkpoint['epoch']
        train_loss = checkpoint['train_loss']
        min_train_loss = train_loss[-1]
		
        val_loss = checkpoint['val_loss']
        
        accuracy = checkpoint['accuracy']
        min_val_loss = val_loss[-1]

        
    else:
        print("Building model from scratch\n")
        prev_epochs = 0
        train_loss = []
        val_loss = []
        accuracy = []
        min_val_loss = 1000000000.0
        min_train_loss = 1000000000.0

    # Train
    if train_model_flag:
        for epoch in range(prev_epochs, num_epochs + prev_epochs):

            train_epoch_loss = functions.train(model, trainloader, trainloader.dataset, criterion, optimizer, device)
            train_loss.append(train_epoch_loss)

            val_epoch_loss, val_accuracy = functions.validate(model, testloader, testloader.dataset, criterion, device)
            val_loss.append(val_epoch_loss)
            accuracy.append(val_accuracy)
            scheduler.step(val_epoch_loss)

            # Save the Best Model
            if val_epoch_loss < min_val_loss:
                min_val_loss = val_epoch_loss
                min_epoch = epoch
                min_train_loss = train_epoch_loss

                if save_model_flag:
                    checkpoint = {
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'lr_scheduler_state_dict': scheduler.state_dict(),
                        #'amp': amp.state_dict(),
                        'train_loss': train_loss,
                        'train_loss_last': train_epoch_loss,
                        'val_loss': val_loss,
                        'val_loss_last': val_epoch_loss,
                        'accuracy': accuracy}
                    torch.save(checkpoint, save_model)
                    print("Saved Checkpoint at epoch: ", epoch)

            print('epoch [{}/{}], train loss: {:.7f}, val loss: {:.7f}, accuracy: {:.7f}'.format(
                epoch + 1, num_epochs + prev_epochs, train_epoch_loss, val_epoch_loss, val_accuracy))

        epoch = epoch + 1
        
        try:
            print('--------------------\nMinimum validation loss at epoch {} with a loss of {:.7f}'.format(min_epoch, min_val_loss))
        except:
            print('--------------------\nMinimum validation loss at epoch {} with a loss of {:.7f}'.format(epoch, min_val_loss))

	# Plot

    # Take test samples, predict what it is (top 3-5)