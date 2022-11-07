from dataclasses import dataclass
import sys
# PyTorch
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
import models as md
import pickle
from tqdm import tqdm

def loadData(file, batch_size, device):
    
    with open(file, 'rb') as f:
        dataset = pickle.load(f)

    num_spectra = dataset.shape[0]

    dataset = np.random.permutation(dataset)

    # Split 75/25 training/testing
    int_split = int(0.75*num_spectra)

    train_data = dataset[0:int_split]
    test_data = dataset[int_split:]

    # Reshape Numpy arrays to be (# samples, 1, 2048)
    train_data = train_data.reshape((train_data.shape[0], 1, train_data.shape[1]))
    test_data = test_data.reshape((test_data.shape[0], 1, test_data.shape[1]))

    train = torch.from_numpy(train_data)
    test = torch.from_numpy(test_data)

    train = train.to(device)
    test = test.to(device)

    trainloader = DataLoader(train, batch_size = batch_size, shuffle = True, num_workers = 0)
    testloader = DataLoader(test, batch_size = batch_size, shuffle = True, num_workers = 0)
    
    return trainloader, testloader

def gpuCheck():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("\nGPU Available")
        gpu_flag = True
        
    else:
        device = torch.device('cpu')
        print("\nGPU Not Available")
        gpu_flag = False
    
    return device, gpu_flag

def train(model, trainloader, dataset, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    counter = 0
    for i, data in tqdm(enumerate(trainloader), total = int(len(dataset)/trainloader.batch_size)):
            
        counter += 1

        X = data[:, :, 0:2048]
            
        y_indexes = data[:, :, 2048]
        y_indexes = y_indexes.long()

        y = torch.zeros(len(y_indexes[:, 0]), 2020).to(device) 
        y[np.arange(len(y_indexes[:, 0])), y_indexes[:, 0]] = 1

        predict_y = model(X.float())
        
        loss = criterion(y, predict_y)

        optimizer.zero_grad()   
        loss.backward()
        running_loss += loss.item()
        optimizer.step()
            
    return running_loss / counter

def validate(model, testloader, dataset, criterion, device):
    model.eval()
    running_loss = 0.0
    counter = 0
    running_accuracy = 0.0
    with torch.no_grad():
        for i, data in tqdm(enumerate(testloader), total=int(len(dataset)/testloader.batch_size)):
			
            counter += 1

            X = data[:, :, 0:2048]
            y_indexes = data[:, :, 2048]
            y_indexes = y_indexes.long()
            y = torch.zeros(len(y_indexes[:, 0]), 2020).to(device)
            y[np.arange(len(y_indexes[:, 0])), y_indexes[:, 0]] = 1

            predict_y = model(X.float())

            loss = criterion(y, predict_y)
            running_loss += loss.item()
            running_accuracy += accuracy(predict_y, y_indexes)
        
    return running_loss / counter, running_accuracy

def accuracy(predictions, labels):
    classes = torch.argmax(predictions, dim = 1)
    return torch.mean((classes == labels).float())
    