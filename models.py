# File system and OS
import sys
# PyTorch
import torch
import torchvision as tv
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F



class Net(nn.Module):
    def __init__(self, output, ks, bn_mom):
        super().__init__()
        pad = int((ks - 1) / 2)
        stride = 1

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels = 1, out_channels = output, kernel_size = ks, stride = stride, padding = pad),
            nn.BatchNorm1d(output, momentum = bn_mom),
            nn.MaxPool1d(2),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels = output, out_channels = output, kernel_size = ks, stride = stride, padding = pad),
            nn.BatchNorm1d(output, momentum = bn_mom),
            nn.MaxPool1d(2),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels = output, out_channels = output, kernel_size = ks, stride = stride, padding = pad),
            nn.BatchNorm1d(output, momentum = bn_mom),
            nn.MaxPool1d(2),
            nn.ReLU()
        )        



        self.lin1 = nn.Linear(8192, 2020)
        self.softmax = nn.Softmax(1)


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)       
        x = self.lin1(x)
        x = self.softmax(x)
        return x
