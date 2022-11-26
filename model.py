import torch
from torch import nn
from torch.utils.data import Dataset
import numpy as np
from torch.nn import functional as F
import math
from torch.autograd import Variable
from matplotlib  import pyplot as plt   
class AccNN(nn.Module):
    def __init__(self,inputsize):
        super(AccNN,self).__init__()
        self.inputsize=inputsize
        self.fc=nn.Sequential(
            nn.Linear(inputsize, 64),
            #nn.LeakyReLU(),
            nn.Linear(64, 128),
            #nn.LeakyReLU(),
            nn.Linear(128, 256),
            #nn.LeakyReLU(),
            nn.Linear(256, 3)
        )
    def forward(self,x):
        batch_size = x.size(0)
        x =x.reshape(batch_size,-1).type(torch.float32)
        x =self.fc(x)
        return x
    
class Mydata(Dataset):
    def __init__(self,acc,linear_acc,gyro,mag,direction):
        self.acc = acc
        self.linear_acc =linear_acc
        self.gyro = gyro
        self.mag = mag
        self.direction = direction
        self.data = np.concatenate((acc,linear_acc,gyro,mag,direction.reshape(1,len(self.direction))),axis=0)
        self.label = direction
    def __getitem__(self, idx):
        assert idx<max(self.acc.shape)
        if idx==len(self.acc):
            return (torch.tensor(self.data[:,idx]),torch.tensor(self.direction[-1]))
        else:
            return (torch.tensor(self.data[:,idx]),torch.tensor(self.label[idx+1]))
    def __len__(self):
        return self.data.shape[1]
    
class MydataP(Dataset):
    def __init__(self,acc,linear_acc,gyro,mag,direction):
        self.acc = acc
        self.linear_acc =linear_acc
        self.gyro = gyro
        self.mag = mag
        self.direction = direction
        self.data = np.concatenate((acc,linear_acc,gyro,mag),axis=0)
        self.label = direction
    def __getitem__(self, idx):
        assert idx<max(self.acc.shape)
        
        if idx==len(self.acc):
            didx=-1
        else:
            didx=idx
        da = self.data[:,idx].reshape(-1)
        return (torch.tensor(da),torch.tensor(self.label[idx]),torch.tensor(self.label[idx+1]))
        
    def __len__(self):
        return self.data.shape[1]-1
    
def inference(model,dataset):
    print('Inferencing')
    loss_sum = 0
    #state = dataset[0][1]
    #dir = [state]
    #gt = [state]
    dir=[]
    labl=[]
    loss_func = torch.nn.MSELoss()
    cnt=0
    for ii,(input,state0,label) in enumerate(dataset):
        # gt.append(label)
        cnt+=1
        if (ii==0):
            state=state0
        else:
            input = Variable(input.reshape(1, -1)).type(torch.float32)
            state = Variable(state.type(torch.float32))
            label = label.type(torch.float32)
            state = model(input, state).squeeze()
        dir.append(state.detach().numpy())
        labl.append(label.detach().numpy())
        loss = loss_func(state, label)
        loss_sum += loss.item()
    print(loss_sum/cnt)
    plt.plot(dir)
    plt.plot(labl)
    plt.show()
'''  
period = min(findperiod(Ax),findperiod(Ay),findperiod(Az))

fx = np.array(wavelettransform(Ax, period))
fy=np.array(wavelettransform(Ay, period))
fz=np.array(wavelettransform(Az, period))

Batchsize=16
dataset =Mydata(fx, fy, fz)
dataloader = DataLoader(dataset,batch_size=Batchsize)
Amodel = AccNN(fx.shape[1]*fx.shape[2]*3)

optimizer = torch.optim.Adam(Amodel.parameters(),lr=0.01)
loss_func = torch.nn.MSELoss()

for epoch in range(1000):
    print('epoch {}'.format(epoch + 1))
    train_loss = 0
    for ii,data in enumerate(dataloader):
        a,b=[(ii)*period*Batchsize,(ii+1)*period*Batchsize]
        meanMagx=[ np.mean([Mx[np.where(np.logical_and(IndxM>=(ii*Batchsize+i)*period ,IndxM<(ii*Batchsize+i+1)*period))]]) for i in range(Batchsize)]
        meanMagy=[ np.mean([My[np.where(np.logical_and(IndxM>=(ii*Batchsize+i)*period ,IndxM<(ii*Batchsize+i+1)*period))]]) for i in range(Batchsize)]
        meanMagz=[ np.mean([Mz[np.where(np.logical_and(IndxM>=(ii*Batchsize+i)*period ,IndxM<(ii*Batchsize+i+1)*period))]]) for i in range(Batchsize)]
        meanM = list(zip(meanMagx,meanMagy,meanMagz))
        temDir = [getEstiDir(direction,Indx,(ii*Batchsize+i)*period,period) for i in range(Batchsize)]
        #print(sum(np.where(np.logical_and(Indx>=(ii*Batchsize)*period ,Indx<(ii*Batchsize+1)*period))))
        temDir = Variable(torch.tensor(temDir),requires_grad=True)
        data =Variable(data,requires_grad=True)
        predA = Amodel(data)
        meanM = Variable(torch.tensor(meanM),requires_grad=True)
        predir = [getDirection(getYaw(predA[i],meanM[i])) for i in range(Batchsize)]
        loss = 0
        for i in range(Batchsize):
            loss += loss_func(predir[i],temDir[i])
            train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('trainloss {:.6f}'.format(train_loss/Batchsize/68))

'''