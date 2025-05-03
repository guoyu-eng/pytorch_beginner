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


input = torch.tensor([[1,-0.5],
                      [2,0]])

input = torch.reshape(input,(-1,1,2,2))

class Nnmodule(nn.module):
    def __init__(self):
        super(Nnmodule,self).__init__()
        self.relu1 = ReLU()
        
    def forward(self, input):
        output = self.relu1(input)
        return output
    
nnmodule = Nnmodule()
output = nnmodule(input)
print(output)