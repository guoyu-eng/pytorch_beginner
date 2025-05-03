import torchvision
# from torchvision import transforms
from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter
from torch import nn
# import torch
from torch.nn import Conv2d, Sequential, Linear, Flatten, MaxPool2d



test_set  = torchvision.datasets.CIFAR10(root='../data', train=False,transform=torchvision.transforms.ToTensor(),download=True)
test_dataloder = DataLoader(dataset=test_set, batch_size=1)


class Nnmodule(nn.Module):
    def __init__(self):
        super(Nnmodule, self).__init__()
        self.model1 = Sequential(
            Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2),
            MaxPool2d(kernel_size=2),
            Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2),
            MaxPool2d(kernel_size=2),
            Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
            MaxPool2d(kernel_size=2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
            # self.linear3 = nn.Linear()
        )
        # self.model1

    def forward(self,x):
        x = self.model1(x)

        return x



loss = nn.CrossEntropyLoss()

nnmodule = Nnmodule()


for data in test_dataloder:
    imgs,targets = data
    outputs = nnmodule(imgs)
    result = loss(outputs,targets)
    print(result)
    # back word
    result.backward()
    # print(targets)
