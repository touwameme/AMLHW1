import math
import numpy as np
from matplotlib  import pyplot as plt   
import pandas as pd

def align_data(target_time,source_time,vals):
    LEN=len(target_time)
    res = np.zeros(LEN)
    source_time = np.array(source_time)
    vals = np.array(vals)
    left=0
    right=left+1
    for i in range(LEN):
        xtime = target_time[i]
        if xtime<source_time[0]:
            res[i]=vals[-1]
        elif xtime>source_time[-1]:
            res[i]=vals[0]
        else:
            #Indx = xtime<source_time
            while(source_time[left+1]<xtime):
                left+=1
            right = left+1           
            right_time =source_time[right]
            right_val = vals[right]
            left_time = source_time[left]
            left_val = vals[left]
            res[i]=left_val+(right_val-left_val)/(right_time-left_time)*(xtime-left_time)
    return res

def align_by_time(data,end,Hz=50,start=0,):
    times=[i/Hz for i in range(int(round(-start+end))*Hz)]
    datatime = data.iloc[:,0]
    res=pd.DataFrame()
    res.insert(0,'time',times)
    for c in range(1,data.shape[1]):
        aligned_col = align_data(times,datatime,data.iloc[:,c])
        res.insert(c, str(c), aligned_col)
    return res    

#filenames=['Accelerometer']
filenames=['Accelerometer','Gyroscope','Linear Accelerometer','Magnetometer','Location']
locationdata = pd.read_csv('Location.csv')
endtime = locationdata.iloc[-1,0]
for file in filenames:
    try:
        data = pd.read_csv(file+'.csv')
    except:
        continue
    columns = data.columns
    aligned_data = align_by_time(data, endtime)
    aligned_data.to_csv(file+'.csv',index=None)
    print('Done:'+file)