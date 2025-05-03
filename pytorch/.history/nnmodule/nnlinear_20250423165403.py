

import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn import Linear


data_set = torchvision.datasets.CIFAR10(root='../data', train=False,transform=torchvision.transforms.ToTensor(),download=True)

dataloder = DataLoader(dataset=data_set, batch_size=64)


class Nnmodule(torch.nn.Module):
    def __init__(self):
        super(Nnmodule,self).__init__()
        self.linear1 = Linear(196608, 10)
        
    def forward(self, input):
       
        output = self.linear1(input)
        return output
    
nnmodule = Nnmodule()
    
for data in dataloder:
    imgs, targets = data
    print(imgs.shape)
    output = torch.reshape(imgs, (1,1,1,-1))
    print(output.shape)
    output = nnmodule(output)
    print(output.shape)