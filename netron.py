import torch
from torch import nn
from torch.nn import functional as F 
import netron

from torchviz import make_dot, make_dot_from_trace
class RNN2(nn.Module):
    def __init__(self,inputsize,statesize):
        super(RNN2, self).__init__()
        self.inputsize = inputsize
        self.statesize = statesize
        self.layer1 = [224, 256,128, 64]
        self.dropout = nn.Dropout(0.2)
        self.fc1=nn.Sequential(
            nn.Linear(inputsize, self.layer1[0]),
            nn.Sigmoid(),
            nn.Linear(self.layer1[0], self.layer1[1]),
            nn.Sigmoid(),
            nn.Linear(self.layer1[1], self.layer1[2]),
            nn.Sigmoid(),
            nn.Linear(self.layer1[2], self.layer1[3]),
            nn.Sigmoid()
        )
        self.layer2 = [32]
        self.fc2=nn.Sequential(
            nn.Linear(self.layer1[-1]+self.statesize, self.layer2[0]),
            nn.Sigmoid(),
            nn.Linear(self.layer2[0], 1)
        )
    def forward(self,x,state):
        batch_size = x.size(0)
        x =x.reshape(batch_size, -1).type(torch.float32)
        x =self.fc1(x)
        x = self.dropout(x)
        x = torch.cat([x, state.reshape(batch_size,1)], 1).type(torch.float32)
        x = self.fc2(x)
        return x
if __name__=="__main__":
    model = RNN2(600, 1)
    x = torch.randn(1,600)
    state =torch.randn(1)
    vis=make_dot(model(x,state),params=dict(model.named_parameters()))
    vis.view()
    #onnx = torch.onnx.export(net, (x,state), 'test.onnx')
    #netron.start(onnx)