import math
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from filter import *
from torch.autograd import Variable
from torch.utils.data import DataLoader
import lowpass as lp
from RNN import *
from dataset import ts_data
from test import get_dir_error
import os 

#dataset = Mydata(phone_acc_filtered, phone_linear_acc_filtered, phone_gyro_filtered, phone_gyro_filtered, phone_direction_filtered)
#dataset = MydataP(phone_acc, phone_linear_acc, phone_gyro, phone_mag, phone_direction)

def train_step(dataloader, model, optimizer, loss_func, step=5):
    for epoch in range(30000):
        model.train()
        train_loss = 0
        cnt = 0
        for ii, (data, dir0, label) in enumerate(dataloader):
            data = Variable(data.type(torch.float32), requires_grad=True).cuda()
            state = Variable(dir0.type(torch.float32), requires_grad=True).cuda()
            # print(data.shape, dir0.shape)
            label = label.type(torch.float32).cuda()
            for i in range(step):
                input = data[:, i]
                state = model(input, state).squeeze()
                # print(state.shape,label.shape)
            loss = loss_func(state, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            cnt += 1
        if epoch % 2000 == 0:
            print('trainloss {:.6f}'.format(train_loss / cnt))
    torch.save(model.state_dict(), './experiment/dir_step.pkl')

def train(dataloader, model, optimizer, loss_func):
    weight = 1000
    for epoch in range(4001):
        model.train()
        train_loss = 0
        local_loss = 0
        direction_loss = 0
        cnt=0
        for ii,(data,dir0,label) in enumerate(dataloader):
            data =Variable(data.type(torch.float32),requires_grad=True).cuda()
            state = Variable(dir0.type(torch.float32),requires_grad=True).cuda()
            label = label.type(torch.float32).cuda()
            pred = model(data, state).reshape(label.shape)
            assert pred.shape==label.shape
            diff = torch.abs(pred[:,0]-label[:,0])
            cir_err = torch.min(diff,360-diff)
            dir_loss = loss_func(torch.zeros(cir_err.shape).cuda(), cir_err)
            #loc_loss = loss_func(pred[:,1:],label[:,1:])
            vel_loss = loss_func(pred[:,1],label[:,1])
            # dir_loss = loss_func(pred[:,0],label[:,0])
            loss = weight * vel_loss + dir_loss 
            # loss = loss_func(pred,label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss+=loss.item()
            local_loss+=weight*vel_loss.item()
            direction_loss += dir_loss.item()
            cnt+=1
        if epoch%100 == 0:
            print('trainloss: {:.6f}, direction_loss: {:.6f}, velloss: {:.6f}'.format(train_loss/cnt, direction_loss/cnt, local_loss/weight/cnt))
            #weight *=(direction_loss/local_loss) 
        #    if local_loss < 10*direction_loss:
        #        weight = 10*weight
            print('=========',weight,'==========')

        if epoch%1000==0:
            param_dir = './experiment/velocity'
            if not os.path.exists(param_dir):
                os.makedirs(param_dir)
            torch.save(model.state_dict(),os.path.join(param_dir, str(int(epoch/1000))+'.pkl'))

# train_list = ['data/Hand/00/', 'data/Hand/01/', 'data/Hand/04/', 'data/Hand/08/', 'data/Hand/12/','data/Hand/21/','data/Hand/25/','data/Hand/31/','data/Pocket/01/','data/Pocket/02/', 'data/Pocket/04/','data/Pocket/08/','data/Pocket/12/','data/Bag/00/','data/Bag/01/']
# test_list = ['data/test_case0/test_case0-00-000/']
train_list = ['./data/Bag/00/', './data/Bag/01/', './data/Hand/03/', './data/Hand/04/', './data/Hand/05/', './data/Hand/02/', './data/Hand/20/', './data/Hand/18/', './data/Hand/27/', './data/Hand/11/', './data/Hand/29/', './data/Hand/19/', './data/Hand/26/', './data/Hand/07/', './data/Hand/00/',  './data/Hand/31/', './data/Hand/30/', './data/Hand/08/', './data/Hand/01/', './data/Hand/06/', './data/Hand/24/', './data/Hand/23/',  './data/Hand/12/', './data/Hand/25/', './data/Pocket/03/', './data/Pocket/02/', './data/Pocket/11/', './data/Pocket/10/', './data/Pocket/07/', './data/Pocket/00/', './data/Pocket/09/', './data/Pocket/08/']

test_list = ['./data/Bag/02/', './data/Hand/10/', './data/Hand/15/', './data/Hand/21/', './data/Hand/22/', './data/Hand/30/', './data/Pocket/04/', './data/Pocket/01/', './data/Pocket/12/', './data/test_case0/']

# train_dataset = ts_data(train_list, 'train', 1)
# model = RNN2(inputsize=600,statesize=1).cuda()
train_dataset = ts_data(train_list, 'train', 2)
model = VelocityModel(600,2).cuda()
#model = LocationModel(inputsize=600, statesize=3).cuda()
#param = torch.load('./experiment/loc_adaptive/4.pkl')
#model.load_state_dict(param)
print('the size of train_dataset is:', len(train_dataset))
Batchsize=128
dataloader = DataLoader(train_dataset,num_workers=10,pin_memory=True,persistent_workers=True,batch_size=Batchsize,shuffle=True)
optimizer = torch.optim.Adam(model.parameters(),lr=0.002,weight_decay=1e-4)
loss_func = torch.nn.MSELoss()

train(dataloader, model, optimizer, loss_func)

# val_dataset = ts_data(path_list, 'val')
# model = RNN(inputsize=600,statesize=1).cuda()
# param = torch.load('./experiment/dir_step.pkl')
# model.load_state_dict(param)
# print('the size of val_dataset is:', len(val_dataset))
# inference(model, val_dataset)

# print('the size of val_dataset is:', len(val_dataset))
# inference(model, val_dataset)

# for epoch in range(30000):
#     model.train()
#     train_loss = 0
#     cnt=0
#     for ii,(data,dir0,label) in enumerate(dataloader):
#         data =Variable(data.type(torch.float32),requires_grad=True).cuda()
#         state = Variable(dir0.type(torch.float32),requires_grad=True).cuda()
#         label = label.type(torch.float32).cuda()
#         pred = model(data,state).reshape(label.shape)
#         loss = loss_func(pred,label)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         train_loss+=loss.item()
#         cnt+=1
#     if epoch%2000==0:
#         print('trainloss {:.6f}'.format(train_loss/cnt))
# torch.save(model.state_dict(),'./experiment/dir2.pkl')

# test_dataloader=DataLoader(val_dataset,pin_memory=True,batch_size=1,shuffle=False)
