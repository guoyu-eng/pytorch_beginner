import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch import nn
import torch
from torch.nn import Conv2d

import torch.nn.functional as F
import torch
from torch.nn import ReLU
from torch.nn import Sigmoid


input = torch.tensor([[1,-0.5],
                      [2,0]])

input = torch.reshape(input,(-1,1,2,2))


test_set  = torchvision.datasets.CIFAR10(root='../data', train=False,transform=torchvision.transforms.ToTensor(),download=True)
test_dataloder = DataLoader(dataset=test_set, batch_size=64)





class Nnmodule(nn.Module):
    def __init__(self):
        super(Nnmodule,self).__init__()
        self.relu1 = ReLU()
        self.sigmoid1 = Sigmoid()
        
    def forward(self, input):
        output = self.sigmoid1(input)
        return output
    
nnmodule = Nnmodule()
# output = nnmodule(input)
# print(output)
wirter = SummaryWriter('sig_logs')
step = 0
for data in test_dataloder:
    imgs, targets = data
    wirter.add_images('input', imgs, step)
    output = nnmodule(imgs)
    wirter.add_images('output', output, step)
    step+=1
wirter.close()