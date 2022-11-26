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

    
path = './'
df_acg=pd.read_csv(path+'Accelerometer.csv')
df_ac=pd.read_csv(path+'Linear Accelerometer.csv')
df_mag=pd.read_csv(path+'Magnetometer.csv')
df_loc=pd.read_csv(path+'Location.csv')
direction  = df_loc[df_loc.keys()[5]].values
Ax = df_ac[df_ac.keys()[1]].values
Ay = df_ac[df_ac.keys()[2]].values
Az = df_ac[df_ac.keys()[3]].values
Agx = df_acg[df_acg.keys()[1]].values
Agy = df_acg[df_acg.keys()[2]].values
Agz = df_acg[df_acg.keys()[3]].values
Mx = df_mag[df_mag.keys()[1]].values
My = df_mag[df_mag.keys()[2]].values
Mz = df_mag[df_mag.keys()[3]].values

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


period = min(findperiod(Ax),findperiod(Ay),findperiod(Az))

x=(Ax[:500]-Agx[:500])
y=(Ay[:500]-Agy[:500])
z=(Az[:500]-Agz[:500])
gx=np.mean(x)
gy=np.mean(y)
gz=np.mean(z)

#plt.show()