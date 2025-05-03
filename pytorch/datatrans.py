import torchvision
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
# used the data set
dataset_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    # torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_set  = torchvision.datasets.CIFAR10(root='./data', train=True,transform=dataset_transform, download=True)
test_set = torchvision.datasets.CIFAR10(root='./data', train=False,transform=dataset_transform, download=True)

print(test_set[0])
# print(test_set.classes)
# img, target = test_set[0]
# print(img)
# print(target)

writer =SummaryWriter('datatrans_logs')
# need use the tensor type
for i in range(10):
    img,target   = test_set[i]
    writer.add_image('test', img, i)
writer.close()
