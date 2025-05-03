import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch import nn
import torch
from torch.nn import Conv2d

import torch.nn.functional as F
import torch



input = torch.tensor([[1,2,0,3,1],
                     [6,7,8,9,0],
                     [1,2,1,1,5],
                     [1,0,1,0,0],
                     [2,0,0,2,5]],dtype=torch.float32)

kernel = torch.tensor([[1,0,0],
                      [0,1,0],
                      [0,0,1]])
input = torch.reshape(input,(-1,1,5,5))
kernel = torch.reshape(kernel,(1,1,3,3))


class Nnmodule(nn.Module):
    def __init__(self):
        super(Nnmodule,self).__init__()
        # self.conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=0)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3,ceil_mode=True)
        
    def forward(self, input):
        output = self.maxpool1(input)
        return output
nnmodule = Nnmodule()
out = nnmodule(input)
print(out)