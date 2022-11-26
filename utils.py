import torch
from torch import nn
from torch.utils.data import Dataset
import numpy as np
from torch.nn import functional as F
import math


#def getDirection(Yaw,err=0):
#    ftemp=Yaw*180/math.pi
#    ftemp= ftemp+err
#    return ftemp
def getDirection(Yaw,err=0):
    ftemp=Yaw*180/math.pi#+90#+err
    if ftemp<0:
        ftemp=ftemp+360
    return ftemp


def getYaw(magVals,gx,gy,gz):
    g = np.sqrt(gx**2+gy**2+gx**2)
    theta = math.asin(-gx/g)
    gamma = math.atan(gy/gz)
    mx = magVals[0]
    my = magVals[1]
    mz = magVals[2]
    yaw = math.atan(-(my*math.cos(gamma)-mz*math.sin(gamma))/(mx*math.cos(theta)+my*math.sin(theta)*math.sin(gamma)+mz*math.sin(theta)*math.cos(gamma)))
    return yaw
def getnearIndx(list,i):
    if i==0:
        return 0
    if i in list:
        try:
            a= int(np.argwhere(np.array(list)==i)[0])
            return a
        except:
            print(np.argwhere(np.array(list)==i)[0])
    else:
        return getnearIndx(list, i-1)
def getEstiDir(direction, Indx,start,l):
    idx=np.where(np.logical_and(Indx>=start,Indx<start+l))
    if len(idx[0])==0:
        return direction[Indx>=start+l][0]
    else:
        return np.mean(direction[idx])

def getMagangle(Mag):
    angle = math.atan2(Mag[1], Mag[0])*(180/math.pi)+180
    return angle

