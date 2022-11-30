import numpy as np
import matplotlib.pyplot as plt
import torch
from dataset import ts_data
from RNN import *
from torch.autograd import Variable
from torch.utils.data import DataLoader
import pandas as pd
from geopy.distance import geodesic

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
        dist = geodesic((gt[0][i], gt[1][i]), (pred[0][i], pred[1][i])).meters
        dist_list.append(dist)
    error = sum(dist_list) / len(dist_list)
    return error

def inference(model,dataset,statesize=3):
    model.eval()
    print('Inferencing')
    loss_sum = 0
    #state = dataset[0][1]
    #dir = [state]
    #gt = [state]
    ts_loss = []
    loc_loss = []
    # loss_func = get_dir_error()
    with torch.no_grad():
        for ii,(ts,states,dir) in enumerate(dataset):
            # print(ts.shape, dir.shape)
            assert ts.shape[0] == dir.shape[0]
            pre = []
            gt = dir.numpy()
            # print(gt)
            time = ts.shape[0]
            state = states[10].type(torch.float32).cuda()
            for t in range(10, time):
                # state = states[t].type(torch.float32).cuda()
                input = ts[t].reshape(1, -1).type(torch.float32).cuda()
                label = dir[t].type(torch.float32).cuda()
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
                plt.plot(pre[:,0])
                plt.plot(gt[10:,0])
                plt.title(dataset.path_list[ii])
                plt.savefig('./experiment/fig/loc/'+'dir'+str(ii))
                plt.show()
                plt.cla()
                plt.plot(pre[:, 1])
                plt.plot(gt[10:, 1])
                plt.title(dataset.path_list[ii])
                plt.savefig('./experiment/fig/loc/'+'x'+str(ii))
                plt.show()
                plt.cla()
                plt.plot(pre[:, 2])
                plt.plot(gt[10:, 2])
                plt.title(dataset.path_list[ii])
                plt.savefig('./experiment/fig/loc/'+'y'+str(ii))
                plt.show()
                plt.cla()
                results = pd.DataFrame()
                loss = get_dir_error(gt[10:,0], pre[:,0])
                ts_loss.append(loss)
                dist_loss = get_dist_error(gt[10:,1:], pre[:,1:])
                loc_loss.append(dist_loss)
            else:
                plt.plot(pre)
                plt.plot(gt[10:])
                plt.title(dataset.path_list[ii])
                plt.savefig('./experiment/fig/rnn1/' + str(ii))
                plt.show()
                plt.cla()
                loss = get_dir_error(gt[10:], pre)
                ts_loss.append(loss)
            # np.save('experiment/pre_dir.npy', np.array(pre))
            # np.save('experiment/pre_dir.npy', np.array(gt))

    print(ts_loss, np.mean(ts_loss))
    print(loc_loss, np.mean(loc_loss))

# train_list = ['data/Hand/00/', 'data/Hand/01/', 'data/Hand/04/', 'data/Hand/08/', 'data/Hand/12/','data/Hand/21/','data/Hand/25/','data/Hand/31/','data/Pocket/01/','data/Pocket/02/', 'data/Pocket/04/','data/Pocket/08/','data/Pocket/12/','data/Bag/00/','data/Bag/01/']
# test_list = ['data/test_case0/']
train_list = ['./Bag/00/', './Bag/01/', './Hand/03/', './Hand/04/', './Hand/05/', './Hand/02/', './Hand/20/', './Hand/18/', './Hand/27/', './Hand/11/', './Hand/29/', './Hand/19/', './Hand/26/', './Hand/07/', './Hand/00/',  './Hand/31/', './Hand/30/', './Hand/08/', './Hand/01/', './Hand/06/', './Hand/24/', './Hand/23/',  './Hand/12/', './Hand/25/', './Pocket/03/', './Pocket/02/', './Pocket/11/', './Pocket/10/', './Pocket/07/', './Pocket/00/', './Pocket/09/', './Pocket/08/']

test_list = ['./Bag/02/', './Hand/10/', './Hand/15/', './Hand/21/', './Hand/22/', './Hand/30/', './Pocket/04/', './Pocket/01/', './Pocket/12/', './test_case0/']
# val_dataset = ts_data(test_list, 'val')
# model = RNN(inputsize=600,statesize=0).cuda()
# param = torch.load('./experiment/dir_single.pkl')
val_dataset = ts_data(train_list, 'val',3)
model = LocationModel(inputsize=600, statesize=3).cuda()
# param = torch.load('./experiment/loc.pkl')
model.load_state_dict(param)
print('the size of val_dataset is:', len(val_dataset))
inference(model, val_dataset,0)