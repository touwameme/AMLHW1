import numpy as np
import matplotlib.pyplot as plt
import torch
from dataset import ts_data
from RNN import *
from torch.autograd import Variable
from torch.utils.data import DataLoader
import pandas as pd
from geopy.distance import geodesic
import geopy

def calculateLoc(loc0,dir,vel):
    distance = vel*1/1000  #km
    distance = geopy.distance.geodesic(distance)
    next_loc = distance.destination(loc0,dir)
    return [next_loc.latitude,next_loc.longitude]

def get_dir_error(gt, pred):
    dir_list = []
    for i in range(len(gt)):
        dir = min(abs(gt[i] - pred[i]), 360 - abs(gt[i] - pred[i]))
        dir_list.append(dir)
    error = sum(dir_list) / len(dir_list)
    return error


def get_dist_error(gt, pred):
    print("local_error")
    dist_list = []
    for i in range(len(gt)):
        dist = geodesic((gt[i][0], gt[i][1]), (pred[i][0], pred[i][1])).meters
        dist_list.append(dist)
    error = sum(dist_list) / len(dist_list)
    return error

def inference(model,dataset,statesize=3):
    model.eval()
    print('Inferencing')
    path = './experiment/fig/velocity/test'
    loss_sum = 0
    #state = dataset[0][1]
    #dir = [state]
    #gt = [state]
    ts_loss = []
    loc_loss = []
    vel_loss=[]
    # loss_func = get_dir_error()
    with torch.no_grad():
        for ii,(ts,states,dir) in enumerate(dataset):
            # print(ts.shape, dir.shape)
            assert ts.shape[0] == dir.shape[0]
            pre = []
            gt = dir.numpy()
            Loc=[]
            # print(gt)
            time = ts.shape[0]
            state = states[0].type(torch.float32).cuda()
            if statesize==4:
                    dir_c = state.reshape(-1)[0]
                    lat_c = state.reshape(-1)[1].cpu().numpy().tolist()
                    long_c = state.reshape(-1)[2].cpu().numpy().tolist()
                    vel_c = state.reshape(-1)[3]
                    state = torch.tensor([dir_c,vel_c]).type(torch.float32).cuda()
            for t in range(time):
                # state = states[t].type(torch.float32).cuda()
                input = ts[t].reshape(1, -1).type(torch.float32).cuda()
                label = dir[t].type(torch.float32).cuda()
                if statesize==4:  #calculate next time location
                    dir_c = state.reshape(-1)[0].cpu().numpy().tolist()
                    vel_c = state.reshape(-1)[1].cpu().numpy().tolist()
                    lat_c,long_c = calculateLoc([lat_c,long_c],dir_c,vel_c)
                    Loc.append([lat_c,long_c])
                state = state.reshape(1,-1)
                state = model(input, state).squeeze()
                # print(label, state)
            #     loss = loss_function(state, label)
            #     loss_sum += loss.item()
                pre.append(state.cpu().numpy())
                
                # gt.append(label.cpu().numpy())

            # ts_loss.append(loss_sum/time)
            # print(pre, gt)
            print(len(pre), len(gt))
            if statesize==3:
                pre = np.concatenate(pre).reshape(-1,3)
                plt.plot((pre[:,0]+360)%360)
                plt.plot(gt[:,0])
                plt.title(dataset.path_list[ii])
                plt.savefig(path+'dir'+str(ii))
                plt.show()
                plt.cla()
                plt.plot(pre[:, 1])
                plt.plot(gt[:, 1])
                plt.title(dataset.path_list[ii])
                plt.savefig(path+'x'+str(ii))
                plt.show()
                plt.cla()
                plt.plot(pre[:, 2])
                plt.plot(gt[:, 2])
                plt.title(dataset.path_list[ii])
                plt.savefig(path+'y'+str(ii))
                plt.show()
                plt.cla()
                results = pd.DataFrame()
                loss = get_dir_error(gt[:,0], pre[:,0])
                ts_loss.append(loss)
                dist_loss = get_dist_error(gt[:,1:3], pre[:,1:3])
                loc_loss.append(dist_loss)
            elif statesize==2:
                pre = np.concatenate(pre).reshape(-1,2)
                plt.plot((pre[:,0]+360)%360)
                plt.plot(gt[:,0])
                plt.title(dataset.path_list[ii])
                plt.savefig(path+'dir'+str(ii))
                plt.show()
                plt.cla()
                plt.plot(pre[:, 1])
                plt.plot(gt[:, 1])
                plt.title(dataset.path_list[ii])
                plt.savefig(path+'vel'+str(ii))
                plt.show()
                plt.cla()
                results = pd.DataFrame()
                loss = get_dir_error(gt[:,0], pre[:,0])
                ts_loss.append(loss)
                vloss = np.sum(np.abs(gt[:,1]-pre[:,1]))/gt.shape[0]
                vel_loss.append(vloss)
                #dist_loss = get_dist_error(gt[:,-1:0:-1], pre[:,-1:0:-1])
                #loc_loss.append(dist_loss)
            elif statesize==4:
                Loc = np.array(Loc)
                pre = np.concatenate(pre).reshape(-1,2)
                plt.plot((pre[:,0]+360)%360)
                plt.plot(gt[:,0])
                plt.title(dataset.path_list[ii])
                plt.savefig(path+'dir'+str(ii))
                plt.show()
                plt.cla()
                plt.plot(Loc[:, 0])
                plt.plot(gt[:, 1])
                plt.title(dataset.path_list[ii])
                plt.savefig(path+'x'+str(ii))
                plt.show()
                plt.cla()
                plt.plot(Loc[:, 1])
                plt.plot(gt[:, 2])
                plt.title(dataset.path_list[ii])
                plt.savefig(path+'y'+str(ii))
                plt.show()
                plt.cla()
                plt.plot(pre[:,1])
                plt.plot(gt[:, 3])
                plt.title(dataset.path_list[ii])
                plt.savefig(path+'vel'+str(ii))
                plt.show()
                results = pd.DataFrame()
                print(gt.shape,pre.shape)
                loss = get_dir_error(gt[:,0], pre[:,0])
                ts_loss.append(loss)
                dist_loss = get_dist_error(gt[:,1:3], Loc[:,:])
                loc_loss.append(dist_loss)
            else:
                # plt.plot(pre)
                # plt.plot(gt)
                # plt.title(dataset.path_list[ii])
                # plt.savefig('./experiment/fig/rnn2_simple_loss/' + str(ii))
                # plt.show()
                # plt.cla()
                loss = get_dir_error(gt, pre)
                ts_loss.append(loss)
            # np.save('experiment/pre_dir.npy', np.array(pre))
            # np.save('experiment/pre_dir.npy', np.array(gt))

    print(ts_loss, np.mean(ts_loss))
    print(vel_loss,np.mean(vel_loss))
    print(loc_loss, np.mean(loc_loss))

# train_list = ['data/Hand/00/', 'data/Hand/01/', 'data/Hand/04/', 'data/Hand/08/', 'data/Hand/12/','data/Hand/21/','data/Hand/25/','data/Hand/31/','data/Pocket/01/','data/Pocket/02/', 'data/Pocket/04/','data/Pocket/08/','data/Pocket/12/','data/Bag/00/','data/Bag/01/']
# test_list = ['data/test_case0/']
train_list = ['data/Bag/00/', 'data/Bag/01/', 'data/Hand/03/', 'data/Hand/04/', 'data/Hand/05/', 'data/Hand/02/', 'data/Hand/20/', 'data/Hand/18/', 'data/Hand/27/', 'data/Hand/11/', 'data/Hand/29/', 'data/Hand/19/', 'data/Hand/26/', 'data/Hand/07/', 'data/Hand/00/',  'data/Hand/31/', 'data/Hand/30/', 'data/Hand/08/', 'data/Hand/01/', 'data/Hand/06/', 'data/Hand/24/', 'data/Hand/23/',  'data/Hand/12/', 'data/Hand/25/', 'data/Pocket/03/', 'data/Pocket/02/', 'data/Pocket/11/', 'data/Pocket/10/', 'data/Pocket/07/', 'data/Pocket/00/', 'data/Pocket/09/', 'data/Pocket/08/']

test_list = ['data/Bag/02/', 'data/Hand/10/', 'data/Hand/15/', 'data/Hand/21/', 'data/Hand/22/', 'data/Hand/30/', 'data/Pocket/04/', 'data/Pocket/01/', 'data/Pocket/12/', 'data/test_case0/']
test_final_list = ['./test/test1', './test/test2', './test/test3', './test/test5', './test/test6', './test/test7', './test/test8', './test/test9', './test/test10', './test/test11']



val_dataset = ts_data(train_list, 'val',4)
model = VelocityModel(inputsize=600, statesize=2).cuda()
param = torch.load('./experiment/velocity/4.pkl')
# val_dataset = ts_data(test_list, 'val')
# model = RNN2(inputsize=600,statesize=1).cuda()
# param = torch.load('./experiment/rnn2_simple_loss/5.pkl')
#val_dataset = ts_data(train_list, 'val',3)
#model = LocationModel(inputsize=600, statesize=3).cuda()
#param = torch.load('./experiment/loc_adaptive/4.pkl')

model.load_state_dict(param)
rnn2_param = torch.load('./experiment/rnn2_simple_loss/6.pkl')
model.directionModel.load_state_dict(rnn2_param)
print('the size of val_dataset is:', len(val_dataset))
inference(model, val_dataset,4)