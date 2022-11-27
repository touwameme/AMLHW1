import math
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from model import *
from filter import *
from torch.autograd import Variable
from torch.utils.data import DataLoader
import lowpass as lp
from RNN import *
from dataset import ts_data

#dataset = Mydata(phone_acc_filtered, phone_linear_acc_filtered, phone_gyro_filtered, phone_gyro_filtered, phone_direction_filtered)
#dataset = MydataP(phone_acc, phone_linear_acc, phone_gyro, phone_mag, phone_direction)

#print(phone_acc.shape)
#print(phone_direction.shape)
#plt.plot(phone_direction)
#plt.show()


path_list = ['data/Hand/00/', 'data/Hand/01/', 'data/Hand/04/', 'data/Hand/08/', 'data/Hand/12/','data/Hand/21/','data/Hand/25/','data/Hand/31/','data/Pocket/01/','data/Pocket/02/', 'data/Pocket/04/','data/Pocket/08/','data/Pocket/12/','data/Bag/00/','data/Bag/01/']
train_dataset = ts_data(path_list, 'train')
val_dataset = ts_data(path_list, 'val')

print('the size of train_dataset is:', len(train_dataset))
Batchsize=128
dataloader = DataLoader(train_dataset,num_workers=10,pin_memory=True,persistent_workers=True,batch_size=Batchsize,shuffle=True)
model = RNN(inputsize=600,statesize=1).cuda()
optimizer = torch.optim.Adam(model.parameters(),lr=0.002)
loss_func = torch.nn.MSELoss()

#torch.save(dataset,'./testcase0.dt')
param = torch.load('./experiment/dir.pkl')
model.load_state_dict(param)


for epoch in range(30000):
    model.train()
    train_loss = 0
    cnt=0
    for ii,(data,dir0,label) in enumerate(dataloader):
        data =Variable(data.type(torch.float32),requires_grad=True).cuda()
        state = Variable(dir0.type(torch.float32),requires_grad=True).cuda()
        label = label.type(torch.float32).cuda()
        pred = model(data,state).reshape(label.shape)
        loss = loss_func(pred,label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss+=loss.item()
        cnt+=1
    if epoch%2000==0:
        print('trainloss {:.6f}'.format(train_loss/cnt))
torch.save(model.state_dict(),'./experiment/dir2.pkl')

# test_dataloader=DataLoader(val_dataset,pin_memory=True,batch_size=1,shuffle=False)
print('the size of val_dataset is:', len(val_dataset))
inference(model, val_dataset)
#print(dataset.data.shape)
#print(dataset.direction.shape)
#dir = []
#for ii,(_,_,label) in enumerate(test_dataloader):
#    dir.append(label.detach().numpy())
#plt.plot(dir)
#plt.show()