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


def process_ts(DATA_PATH='./', order=4, fs=5000, cutoff=2):
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
    return phone_acc, phone_linear_acc, phone_gyro, phone_mag, phone_direction

class ts_data(Dataset):
    def __init__(self, path_list, mode):
        self.mode = mode
        self.path_list = path_list
        self.input = []
        self.state = []
        self.label = []
        for path in self.path_list:
            acc, lac, gyro, mag, dir = process_ts(path)  # shape (3, t, 50) dir(t,)
            input = np.concatenate((acc, lac, gyro, mag), axis=0).reshape(-1, 600)[:-1]  # shape (t, 600)
            # print(input.shape)
            self.input.append(input)
            # print(dir.shape)
            self.label.append(dir[1:])
            self.state.append(dir[:-1])
        if self.mode == 'train':
            self.input = np.concatenate(self.input, axis=0)
            self.label = np.concatenate(self.label)
            self.state = np.concatenate(self.state)
            self.len = self.input.shape[0]
        elif self.mode == 'val':
            self.len = len(self.input)

    def __getitem__(self, idx):
        if self.mode == 'train':
            return (torch.tensor(self.input[idx]), torch.tensor(self.state[idx]), torch.tensor(self.label[idx]))
        elif self.mode == 'val':
            return torch.tensor(self.input[idx]), torch.tensor(self.state[idx][0]), torch.tensor(self.label[idx])

    def __len__(self):
        return self.len

# path_list = ['data/Hand/00/', 'data/Hand/01/', 'data/Hand/04/', 'data/Hand/08/', 'data/Hand/12/','data/Hand/21/','data/Hand/25/','data/Hand/31/','data/Pocket/01/','data/Pocket/02/', 'data/Pocket/04/','data/Pocket/08/','data/Pocket/12/','data/Bag/00/','data/Bag/01/']
# dataset = ts_data(path_list)
#
# torch.save(dataset, './experiment/dataset.dt')
# dataset = torch.load('./experiment/dataset.dt')
# print(len(dataset))
# print(dataset[0][1])

