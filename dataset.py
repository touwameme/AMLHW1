import math
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from model import *
from filter import *
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
import lowpass as lp
from RNN import *
from period import *


def process_ts(DATA_PATH='./', order=4, fs=5000, cutoff=3):
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
    x = np.array(dirtransform(location_data[location_data.keys()[1]],PERIOD)).reshape(-1,1)
    y = np.array(dirtransform(location_data[location_data.keys()[2]],PERIOD)).reshape(-1,1)
    phone_loc = np.concatenate((x,y),axis=1)
    phone_vel = np.array(dirtransform(location_data[location_data.keys()[4]],PERIOD))

    phone_mag_filtered = np.array([wavelettransform(lp.butter_lowpass_filter(mag_data[mag_data.keys()[1]],cutoff,fs,order),PERIOD),
                                   wavelettransform(lp.butter_lowpass_filter(mag_data[mag_data.keys()[2]],cutoff,fs,order),PERIOD),
                                   wavelettransform(lp.butter_lowpass_filter(mag_data[mag_data.keys()[3]],cutoff,fs,order),PERIOD)])

    phone_acc_filtered = np.array([wavelettransform(lp.butter_lowpass_filter(acc_data[acc_data.keys()[1]],cutoff,fs,order),PERIOD),
                                   wavelettransform(lp.butter_lowpass_filter(acc_data[acc_data.keys()[2]],cutoff,fs,order),PERIOD),
                                   wavelettransform(lp.butter_lowpass_filter(acc_data[acc_data.keys()[3]],cutoff,fs,order),PERIOD)])
    phone_linear_acc_filtered = np.array([wavelettransform(lp.butter_lowpass_filter(linear_acc_data[linear_acc_data.keys()[1]],cutoff,fs,order),PERIOD),
                                          wavelettransform(lp.butter_lowpass_filter(linear_acc_data[linear_acc_data.keys()[2]],cutoff,fs,order),PERIOD),
                                          wavelettransform(lp.butter_lowpass_filter(linear_acc_data[linear_acc_data.keys()[3]],cutoff,fs,order),PERIOD)])
    phone_gyro_filtered = np.array([wavelettransform(lp.butter_lowpass_filter(gyro_data[gyro_data.keys()[1]],cutoff,fs,order),PERIOD),
                                    wavelettransform(lp.butter_lowpass_filter(gyro_data[gyro_data.keys()[2]],cutoff,fs,order),PERIOD),
                                    wavelettransform(lp.butter_lowpass_filter(gyro_data[gyro_data.keys()[3]],cutoff,fs,order),PERIOD)])

    phone_direction_filtered=np.array(dirtransform(lp.butter_lowpass_filter(location_data[location_data.keys()[5]], cutoff, fs),PERIOD))
    return phone_acc, phone_linear_acc, phone_gyro, phone_mag, phone_direction, phone_loc,phone_vel
    # return phone_acc_filtered, phone_linear_acc_filtered, phone_gyro_filtered, phone_mag_filtered, phone_direction_filtered

class ts_data(Dataset):
    def __init__(self, path_list, mode, statesize=1, step=5):
        self.mode = mode
        self.statesize = statesize
        self.step = step
        self.path_list = path_list
        self.input = []
        self.state = []
        self.label = []
        self.labellen = np.inf
        for path in self.path_list:
            acc, lac, gyro, mag, dir, loc,vel = process_ts(path)  # shape (12, t, 50) dir(t,)
            input = np.concatenate((acc, lac, gyro, mag), axis=0).reshape(-1, 600) # shape (t, 600)
            if path == './Bag/01/':
                input = input[:600]
                dir = dir[:600]
                loc = loc[:600]
            if  not self.mode=='test':
                input = input[3:-1-3]
                dir = dir[3:-3].reshape(-1,1)
                loc = loc[3:-3]
                vel = vel[3:-3]
                assert loc.shape[0]==dir.shape[0]
            else:
                dir = dir.reshape(-1,1)
                vel = vel.reshape(-1,1)
            if self.statesize==1:
                state = dir
            elif self.statesize==2:
                dir = dir.reshape(-1,1)
                vel = vel.reshape(-1,1)
                state = np.concatenate((dir,vel),axis=1)
            elif self.statesize==3:
                state = np.concatenate((dir, loc),axis=1)
            elif self.statesize==4:
                vel = vel.reshape(-1,1)
                state = np.concatenate((dir,loc,vel),axis=1)
            if self.mode == 'step_train':
                num = input.shape[0]//self.step
                input = input[:num*self.step].reshape(num,self.step,600)
                state = state[::self.step]
            elif self.mode == 'test':
                tmpstate = state[:state.shape[0]//2,:]# delete the None NAN tail
                state = np.array (tmpstate[np.isfinite(tmpstate)]).reshape(-1,statesize)
            self.input.append(input)
            # print(dir.shape)
            self.label.append(state[1:])
            self.state.append(state[:-1])

        if self.mode == 'train' or self.mode == 'step_train':
            self.input = np.concatenate(self.input, axis=0)
            self.label = np.concatenate(self.label)
            self.state = np.concatenate(self.state)
            self.len = self.input.shape[0]
        elif self.mode == 'val':
            self.len = len(self.input)
        elif self.mode =='finetune':
            self.len = len(self.label)
        elif self.mode=='finetune':
            self.labellen = len(self.label)

    def __getitem__(self, idx):
        if self.mode =='test':
            labelidx = min(idx,self.labellen-1)
            return torch.tensor(self.input[idx]), torch.tensor(self.state[labelidx]), torch.tensor(self.label[labelidx])
        elif self.mode == 'train' or self.mode == 'step_train' or self.mode=='val' or self.mode=='finetune':
            return (torch.tensor(self.input[idx]), torch.tensor(self.state[idx]), torch.tensor(self.label[idx]))
       

    def __len__(self):
        return self.len

# path_list = ['data/Hand/00/', 'data/Hand/01/', 'data/Hand/04/', 'data/Hand/08/', 'data/Hand/12/','data/Hand/21/','data/Hand/25/','data/Hand/31/','data/Pocket/01/','data/Pocket/02/', 'data/Pocket/04/','data/Pocket/08/','data/Pocket/12/','data/Bag/00/','data/Bag/01/']
# dataset = ts_data(path_list)
#
# torch.save(dataset, './experiment/dataset.dt')
# dataset = torch.load('./experiment/dataset.dt')
# print(len(dataset))
# print(dataset[0][1])

