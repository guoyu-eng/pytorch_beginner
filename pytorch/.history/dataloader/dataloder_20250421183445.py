import torchvision
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

test_set  = torchvision.datasets.CIFAR10(root='../data', train=False,transform=torchvision.transforms.ToTensor)
test_dataloder = DataLoder(dataset=test_set, batch_size=4, shuffle=True, num_workers=0, drop_last=False)
