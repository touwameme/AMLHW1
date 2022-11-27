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
    def __init__(self,acc,linear_acc,gyro,mag,direction,table):#[L1,L1+L2]
        self.acc = acc
        self.linear_acc =linear_acc
        self.gyro = gyro
        self.mag = mag
        self.direction = direction
        self.data = np.concatenate((acc,linear_acc,gyro,mag),axis=0)
        self.label = direction
        self.table=table
    def __getitem__(self, idx):
        assert idx<max(self.acc.shape)
        if idx==len(self.acc):
            dir1idx=idx
        else:
            dir1idx=idx+1
        da = self.data[:, idx].reshape(-1)
        return (torch.tensor(da),torch.tensor(self.label[idx]),torch.tensor(self.label[dir1idx]))
        
    def __len__(self):
        return self.data.shape[1]-1
    
def inference(model,dataset):
    model.eval()
    print('Inferencing')
    loss_sum = 0
    #state = dataset[0][1]
    #dir = [state]
    #gt = [state]
    ts_loss = []
    loss_func = torch.nn.MSELoss()
    with torch.no_grad():
        for ii,(ts,state0,dir) in enumerate(dataset):
            assert ts.shape[0] == dir.shape[0]
            pre = []
            gt = dir.numpy()
            # print(gt)
            time = ts.shape[0]
            state = state0.type(torch.float32).cuda()
            for t in range(time):
                input = ts[t].reshape(1, -1).type(torch.float32).cuda()
                label = dir[t].type(torch.float32).cuda()
                state = model(input, state).squeeze()
                # print(label, state)
            # if (ii==0):
            #     state=state0.cuda()
            # else:
            #     input = Variable(input.reshape(1, -1)).type(torch.float32).cuda()
            #     state = Variable(state.type(torch.float32)).cuda()
            #     label = label.type(torch.float32).cuda()
            #     state = model(input, state).squeeze()
                loss = loss_func(state, label)
                loss_sum += loss.item()
                pre.append(state.cpu().numpy())
                # gt.append(label.cpu().numpy())

            # print(pre)
            # print(len(pre),len(gt))
            ts_loss.append(loss_sum/time)
            # print(pre, gt)
            plt.plot(pre)
            plt.plot(gt)
            plt.title(dataset.path_list[ii])
            plt.savefig('./experiment/fig/'+str(ii))
            plt.show()

    print(ts_loss, np.mean(ts_loss))
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