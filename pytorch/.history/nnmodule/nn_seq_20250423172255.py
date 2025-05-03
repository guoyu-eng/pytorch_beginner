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
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(1024, 64)
        self.linear2 = nn.Linear(64, 10)
        
        
    def forward(self, input):
        output = self.conv1(input)
        output = self.maxpool1(output)
        output = self.conv2(output)
        output = self.maxpool2(output)
        output = self.conv3(output)
        output = self.maxpool3(output)
        output = self.flatten(output)
        output = self.linear1(output)
        output = self.linear2(output)
        
        return output
    
nnmodule = Nnmodule()
print(nnmodule)