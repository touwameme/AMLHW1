import torch
from torch import nn


class YawNN(nn.Module):
    def __init__(self,inputsize):
        super(YawNN(),self).__init__()
        self.inputsize=inputsize
        self.fc=nn.Sequential(
            nn.Linear(inputsize, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 1)
        )
    def forward(self,x):
        x = fc(x)
        return x