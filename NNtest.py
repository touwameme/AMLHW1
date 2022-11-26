import math
import numpy as np
from matplotlib  import pyplot as plt   
import pandas as pd
from period import *
from test import get_dir_error
from model import *
from filter import *
from torch.autograd import Variable
from torch.utils.data import DataLoader
import lowpass as lp
from RNN import *
DATA_PATH='./'
order = 4
fs = 5000
cutoff = 2
mag_data  = pd.read_csv(DATA_PATH+'Magnetometer.csv')
acc_data  = pd.read_csv(DATA_PATH+'Accelerometer.csv')
gyro_data = pd.read_csv(DATA_PATH+'Gyroscope.csv')
try:
    linear_acc_data = pd.read_csv(DATA_PATH+'Linear Accelerometer.csv')
except:
    linear_acc_data = pd.read_csv(DATA_PATH+'Linear Acceleration.csv')
location_data = pd.read_csv(DATA_PATH+'Location.csv')
PERIOD=50

phone_mag = np.array([wavelettransform(mag_data[mag_data.keys()[1]],PERIOD),
                      wavelettransform(mag_data[mag_data.keys()[2]],PERIOD),
                      wavelettransform(mag_data[mag_data.keys()[3]],PERIOD)])

phone_acc = np.array([wavelettransform(acc_data[acc_data.keys()[1]],PERIOD),
                      wavelettransform(acc_data[acc_data.keys()[2]],PERIOD),
                      wavelettransform(acc_data[acc_data.keys()[3]],PERIOD)])
phone_linear_acc = np.array([wavelettransform(linear_acc_data[linear_acc_data.keys()[1]],PERIOD),
                             wavelettransform(linear_acc_data[linear_acc_data.keys()[2]],PERIOD),
                             wavelettransform(linear_acc_data[linear_acc_data.keys()[3]],PERIOD)])
phone_gyro = np.array([wavelettransform(gyro_data[gyro_data.keys()[1]],PERIOD),
                       wavelettransform(gyro_data[gyro_data.keys()[2]],PERIOD),
                       wavelettransform(gyro_data[gyro_data.keys()[3]],PERIOD)])
phone_direction = np.array(dirtransform(location_data[location_data.keys()[5]],PERIOD))

phone_mag_filtered = np.array([lp.butter_lowpass_filter(mag_data[mag_data.keys()[1]],cutoff,fs,order),
                               lp.butter_lowpass_filter(mag_data[mag_data.keys()[2]],cutoff,fs,order),
                               lp.butter_lowpass_filter(mag_data[mag_data.keys()[3]],cutoff,fs,order)])

phone_acc_filtered = np.array([lp.butter_lowpass_filter(acc_data[acc_data.keys()[1]],cutoff,fs,order),
                               lp.butter_lowpass_filter(acc_data[acc_data.keys()[2]],cutoff,fs,order),
                               lp.butter_lowpass_filter(acc_data[acc_data.keys()[3]],cutoff,fs,order)])
phone_linear_acc_filtered = np.array([lp.butter_lowpass_filter(linear_acc_data[linear_acc_data.keys()[1]],cutoff,fs,order),
                                      lp.butter_lowpass_filter(linear_acc_data[linear_acc_data.keys()[2]],cutoff,fs,order),
                                      lp.butter_lowpass_filter(linear_acc_data[linear_acc_data.keys()[3]],cutoff,fs,order)])
phone_gyro_filtered = np.array([lp.butter_lowpass_filter(gyro_data[gyro_data.keys()[1]],cutoff,fs,order),
                                lp.butter_lowpass_filter(gyro_data[gyro_data.keys()[2]],cutoff,fs,order),
                                lp.butter_lowpass_filter(gyro_data[gyro_data.keys()[3]],cutoff,fs,order)])

phone_direction_filtered=np.array(lp.butter_lowpass_filter(location_data[location_data.keys()[5]], cutoff, fs))

#dataset = Mydata(phone_acc_filtered, phone_linear_acc_filtered, phone_gyro_filtered, phone_gyro_filtered, phone_direction_filtered)
dataset = MydataP(phone_acc, phone_linear_acc, phone_gyro, phone_mag, phone_direction)


#print(phone_acc.shape)
#print(phone_direction.shape)
#plt.plot(phone_direction)
#plt.show()




Batchsize=16
dataloader = DataLoader(dataset,batch_size=Batchsize,shuffle=False)
model = RNN(inputsize=600,statesize=1)
optimizer = torch.optim.Adam(model.parameters(),lr=0.002)
loss_func = torch.nn.MSELoss()

torch.save(dataset,'./testcase0.dt')
#param = torch.load('./param.pkl')
#model.load_state_dict(param)



for epoch in range(30000):
    model.train()
    train_loss = 0
    cnt=0
    for ii,(data,dir0,label) in enumerate(dataloader):
        data =Variable(data.type(torch.float32),requires_grad=True)
        state = Variable(dir0.type(torch.float32),requires_grad=True)
        label = label.type(torch.float32)
        pred = model(data,state).reshape(label.shape)
        loss = loss_func(pred,label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss+=loss.item()
        cnt+=1
    if epoch%2000==0:
        print('trainloss {:.6f}'.format(train_loss/cnt))
torch.save(model.state_dict(),'./param.pkl')

#model.eval()
#test_dataloader=DataLoader(dataset,batch_size=1,shuffle=False)
#inference(model, tesd_dataloader)
#print(dataset.data.shape)
#print(dataset.direction.shape)
#dir = []
#for ii,(_,_,label) in enumerate(test_dataloader):
#    dir.append(label.detach().numpy())
#plt.plot(dir)
#plt.show()