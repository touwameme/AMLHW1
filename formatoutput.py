import torch
import os
from matplotlib  import pyplot as plt   
from preprocess import *
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
from model import *
import time as Time
import geopy
test_final_list = ['./data/test/test1/', './data/test/test2/', './data/test/test3/', './data/test/test5/', './data/test/test6/', './data/test/test7/', './data/test/test8/', './data/test/test9/', './data/test/test10/', './data/test/test11/']


def calculateLoc(loc0,dir,vel):
    distance = vel*1/1000  #km
    distance = geopy.distance.geodesic(distance)
    next_loc = distance.destination(loc0,dir)
    return [next_loc.latitude,next_loc.longitude]


    

def pre_test_time_align(path):
    local_input = pd.read_csv(os.path.join(path,'Location_input.csv'))
    local_input.to_csv(os.path.join(path,'Location.csv'),index=None)
    prepro(path)
        

def dummy_model(x,state):
    return torch.tensor(torch.ones(x.shape[0],3))       


#model = LocationModel(inputsize=600, statesize=3).cuda()
#param = torch.load('./experiment/loc_adaptive/4.pkl')


def outputformat(path):
    target = pd.read_csv(os.path.join(path,'Location_input.csv'))
    pd.DataFrame(pre).to_csv('./output')
    
def inferenceOutput(model,dataset):
    model.eval()
    print('InferencingOutput')
    
    with torch.no_grad():
        for ii,(ts,states,dir) in enumerate(dataset):
            # print(ts.shape, dir.shape)
            #assert ts.shape[0] == dir.shape[0]
            time_start = Time.time()
            pre = []
            gt = dir.numpy()
            Loc=[]
            # print(gt)
            time = ts.shape[0]
            state = states[0].type(torch.float32).cuda()
            
            dir_c = state.reshape(-1)[0]
            lat_c = state.reshape(-1)[1].cpu().numpy().tolist()
            long_c = state.reshape(-1)[2].cpu().numpy().tolist()
            vel_c = state.reshape(-1)[3]
            state = torch.tensor([dir_c,vel_c]).type(torch.float32).cuda()
            for t in range(time):
                # state = states[t].type(torch.float32).cuda()
                input = ts[t].reshape(1, -1).type(torch.float32).cuda()
                dir_c = state.reshape(-1)[0].cpu().numpy().tolist()
                vel_c = state.reshape(-1)[1].cpu().numpy().tolist()
                lat_c,long_c = calculateLoc([lat_c,long_c],dir_c,vel_c)
                Loc.append([lat_c,long_c])
                state = state.reshape(1,-1)
                state = model(input, state).squeeze()
                pre.append(state.cpu().numpy())
                
            Loc = np.array(Loc)
            Locinput = pd.read_csv(test_final_list[ii]+'Location_input'+'.csv')
            targettime = Locinput.iloc[:,0]
            LEN = len(targettime)
            gt = Locinput.iloc[:,1]
            LEN10 = sum(np.isfinite(gt))
            output = pd.DataFrame(np.zeros((LEN,8)))
            output.columns = Locinput.columns
            pre = np.array(pre).reshape(-1,2)
            sourcetime=list(range(1,len(pre)+1))
            output.iloc[:,0]=targettime
            output.iloc[:,1]=align_data(targettime, sourcetime, Loc[:,0])
            output.iloc[:,2]=align_data(targettime, sourcetime, Loc[:,1])
            output.iloc[:,5]=align_data(targettime, sourcetime, pre[:,0])
            output.iloc[:LEN10,1]=Locinput.iloc[:LEN10,1]
            output.iloc[:LEN10,2]=Locinput.iloc[:LEN10,2]
            output.iloc[:LEN10,5]=Locinput.iloc[:LEN10,5]
            time_end = Time.time()
            output.to_csv(test_final_list[ii]+'Location_otuput.csv',index=None)
            print('time ts'+str(ii)+':',time_end-time_start,'s')


if __name__ =='__main__':
    #for path in test_final_list:
    #    pre_test_time_align(path)
    model = VelocityModel(inputsize=600, statesize=2).cuda()
    param = torch.load('./experiment/velocity/4.pkl')
    rnn2_param = torch.load('./experiment/rnn2_simple_loss/6.pkl')
    model.directionModel.load_state_dict(rnn2_param)
    test_dataset = ts_data(test_final_list,'test',4) # dir, loc , vel
    inferenceOutput(model, test_dataset)



