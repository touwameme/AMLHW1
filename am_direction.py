import math
import numpy as np
from matplotlib  import pyplot as plt   
import pandas as pd
from period import *
from test import get_dir_error
from filter import *
def getYaw(accVals,magVals):
    roll =math.atan2(accVals[0], accVals[2])
    pitch = -math.atan(accVals[1] / (accVals[0] * math.sin(roll) + accVals[2] * math.sin(roll)))
    yaw = math.atan2(magVals[0] * math.sin(roll) * math.sin(pitch) + magVals[2] * math.cos(roll) * math.sin(pitch) + magVals[1] * math.cos(pitch),magVals[0] * math.cos(roll) - magVals[2] * math.sin(roll))
    return yaw

def getDirection(Yaw,mody):
    ftemp=Yaw*180/math.pi
    return ftemp
    if (ftemp>0):
        ftemp=-180.0 + (ftemp - 180.0)
    ftemp = -ftemp+90+mody
    if ftemp>360:
        ftemp = ftemp-360
    return ftemp
def getnearIndx(list,i):
    if i in list:
        return i
    if (i-1) in list:
        return i-1
    else:
        return getnearIndx(list, i-1)
path = '../test_case0/'
df_ac=pd.read_csv(path+'Accelerometer.csv')
df_mag=pd.read_csv(path+'Magnetometer.csv')
df_loc=pd.read_csv(path+'Location.csv')
direction  = df_loc[df_loc.keys()[5]].values
Ax = df_ac[df_ac.keys()[1]].values
Ay = df_ac[df_ac.keys()[2]].values
Az = df_ac[df_ac.keys()[3]].values
Mx = df_mag[df_mag.keys()[1]].values
My = df_mag[df_mag.keys()[2]].values
Mz = df_mag[df_mag.keys()[3]].values




period = min(findperiod(Ax),findperiod(Ay),findperiod(Az))

plt.plot(Mz)
plt.show()
'''
accVals=list(zip(Ax,Ay,Az))
magVals=list(zip(Mx,My,Mz))
Ltime=df_loc[df_loc.keys()[0]].values
Atime=df_ac[df_ac.keys()[0]].values
Mtime=df_ac[df_mag.keys()[0]].values
Indx=np.apply_along_axis(lambda x: np.round(x*50),axis=0,arr=Ltime).astype(int)
IndxA=np.apply_along_axis(lambda x: np.round(x*50),axis=0,arr=Atime).astype(int)
IndxM=np.apply_along_axis(lambda x: np.round(x*50),axis=0,arr=Mtime).astype(int) 

I2A=[getnearIndx(IndxA,i) for i in Indx]
I2A_idx=[int(np.argwhere(IndxA==i)) for i in I2A]
I2M=[getnearIndx(IndxM,i) for i in Indx]
I2M_idx=[int(np.argwhere(IndxM==i)) for i in I2M]
#preYaw=[getYaw(accVals[i], magVals[i]) for i in range(len(Indx))]
preYaw=[getYaw(accVals[I2A_idx[i]], magVals[I2M_idx[i]]) for i in range(len(Indx))]
preYaw = siimp_window_filter(50, preYaw)
preDir=[getDirection(preYaw[i], mody=0) for i in range(len(Indx))]


#


f1_preYaw=siimp_window_filter(50, preYaw)
print(sum(np.array(preYaw)>0))
#plt.scatter(list(range(len(preYaw))),preYaw)
#plt.scatter(list(range(len(preYaw))),siimp_window_filter(50, preYaw))
plt.plot(direction)
plt.plot(exponential_smoothing(1, preDir))
plt.show()


'''