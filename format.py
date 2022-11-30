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
#test_final_list = ['./test/test1', './test/test2', './test/test3', './test/test5', './test/test6', './test/test7', './test/test8', './test/test9', './test/test10', './test/test11']
test_final_list = ['./data/test/test0/']

def pre_test_time_align(path):
    local_input = pd.read_csv(os.path.join(path,'Location_input.csv'))
    local_input.to_csv(os.path.join(path,'Location.csv'),index=None)
    prepro(path)
        

def dummy_model(x,state):
    return torch.tensor(torch.ones(x.shape[0],3))       

def outputformat(path):
    target = pd.read_csv(os.path.join(path,'Location_input.csv'))
    pd.DataFrame(pre).to_csv('./output')
    
def inferenceOutput(model,dataset):
    #model.eval()
    print('InferencingOutput')
    with torch.no_grad():
        for ii,(ts,states,dir) in enumerate(dataset):
            #print(ts.shape, dir.shape)
            pre = []
            time = ts.shape[0]
            state = dir[-1].type(torch.float32).cuda()
            for t in range(time):
                # state = states[t].type(torch.float32).cuda()
                
                if t<len(states):
                    pre.append(states[t].cpu().numpy())
                elif t==len(states):
                    pre.append(dir[t-1].cpu().numpy())
                else:
                    input = ts[t].reshape(1, -1).type(torch.float32).cuda()
                    state = model(input, state).squeeze()
                    pre.append(state.cpu().numpy())
            Locinput = pd.read_csv(test_final_list[ii]+'Location_input'+'.csv')
            targettime = Locinput.iloc[:,0]
            LEN = len(targettime)
            gt = Locinput.iloc[:,1]
            LEN10 = sum(np.isfinite(gt))
            output = pd.DataFrame(np.zeros((LEN,8)))
            output.columns = Locinput.columns
            pre = np.array(pre).reshape(-1,3)
            sourcetime=list(range(1,len(pre)+1))
            output.iloc[:,0]=targettime
            output.iloc[:,1]=align_data(targettime, sourcetime, pre[:,1])
            output.iloc[:,2]=align_data(targettime, sourcetime, pre[:,2])
            output.iloc[:,5]=align_data(targettime, sourcetime, pre[:,0])
            output.iloc[:LEN10,1]=Locinput.iloc[:LEN10,1]
            output.iloc[:LEN10,2]=Locinput.iloc[:LEN10,2]
            output.iloc[:LEN10,5]=Locinput.iloc[:LEN10,5]
            output.to_csv(test_final_list[ii]+'otuput.csv',index=None)
            
            
for path in test_final_list:
    pre_test_time_align(path)
test_dataset = ts_data(test_final_list,'test',3)
inferenceOutput(dummy_model, test_dataset)


