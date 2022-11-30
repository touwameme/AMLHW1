import torch
from torch import nn
from torch.utils.data import Dataset
import numpy as np
import math

class RNN2(nn.Module):
    def __init__(self,inputsize,statesize):
        super(RNN2, self).__init__()
        self.inputsize = inputsize
        self.statesize = statesize
        self.layer1 = [224, 256, 128, 64]
        self.dropout = nn.Dropout(0.2)
        self.fc1=nn.Sequential(
            nn.Linear(inputsize, self.layer1[0]),
            nn.Sigmoid(),
            nn.Linear(self.layer1[0], self.layer1[1]),
            nn.Sigmoid(),
            nn.Linear(self.layer1[1], self.layer1[2]),
            nn.Sigmoid(),
            nn.Linear(self.layer1[2], self.layer1[3]),
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
        x = self.dropout(x)
        x = torch.cat([x, state.reshape(batch_size,1)], 1).type(torch.float32)
        x = self.fc2(x)
        return x

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
        if self.statesize > 0:
            x = torch.cat([x, state.reshape(batch_size,1)], 1).type(torch.float32)
        x = self.fc2(x)
        return x
        
        
class LocationModel(nn.Module):  
    #state=[dir,log,lat]
    def __init__(self,inputsize,statesize): 
        super(LocationModel,self).__init__()
        self.inputsize = inputsize
        self.statesize = statesize
        self.dropout = nn.Dropout(0.2)
        self.directionModel = RNN2(600,1)
        param = torch.load('./experiment/w_decay_rnn2.pkl')
        self.directionModel.load_state_dict(param)
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
        self.fc2=nn.Sequential(   #64+3
            nn.Linear(self.layer1[-1]+self.statesize, self.layer2[0]),
            nn.Sigmoid(),
            nn.Linear(self.layer2[0], 2)
        )
        
        
    def forward(self,x,state):
        batch_size = x.size(0)
        x = x.reshape(batch_size,-1).type(torch.float32)
        direc = self.directionModel(x,state[:,0])
        x = self.fc1(x)  #veclocity  1
        x = self.dropout(x)
        x = torch.cat([x,direc.reshape(batch_size,1),state[:,1:].reshape(batch_size,2)], 1).type(torch.float32)
        x = self.fc2(x)
        return torch.cat([direc, x], axis=1)