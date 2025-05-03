import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch import nn
import torch
from torch.nn import Conv2d



test_set  = torchvision.datasets.CIFAR10(root='../data', train=False,transform=torchvision.transforms.ToTensor(),download=True)
test_dataloder = DataLoader(dataset=test_set, batch_size=64)



class Nnmodule(nn.Module):
    def __init__(self):
        super(Nnmodule,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)
        # self.conv2 = nn.Conv2d()
        
    def forward(self, input):
        output = self.conv1(input)
        return output
    
nnmodule = Nnmodule()
print(nnmodule)



