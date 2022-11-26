import torch
from torch import nn
from torch.utils.data import Dataset
import numpy as np
import math

class RNN(nn.Module):
    def __init__(self,inputsize,statesize):
        super(RNN, self).__init__()
        self.inputsize = inputsize
        self.statesize = statesize
        self.layer1 = [224, 128, 64]
        self.fc1=nn.Sequential(
            nn.Linear(inputsize, self.layer1[0]),
            nn.Sigmoid(),
            nn.Linear(self.layer1[0], self.layer1[1]),
            nn.Sigmoid(),
            nn.Linear(self.layer1[1], self.layer1[2]),
            nn.Sigmoid()
        )
        self.layer2 = [32]
        self.fc2=nn.Sequential(
            nn.Linear(self.layer1[-1]+self.statesize, self.layer2[0]),
            nn.Sigmoid(),
            nn.Linear(self.layer2[0], 1)
        )
    def forward(self,x,state):
        batch_size = x.size(0)
        x =x.reshape(batch_size, -1).type(torch.float32)
        x =self.fc1(x)
        x = torch.cat([x, state.reshape(batch_size,1)], 1).type(torch.float32)
        x = self.fc2(x)
        return x

def train_epoch(model, dataloader, optimizer, loss_function):
    for ii, (input, gt) in enumerate(dataloader):
        state = input[:,-1]
        x = input[:,0:-1]
        pre = model(x, state)
        loss = loss_function(x, pre)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()