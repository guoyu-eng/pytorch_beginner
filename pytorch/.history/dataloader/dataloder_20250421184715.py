import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# dataload is like a package
test_set  = torchvision.datasets.CIFAR10(root='../data', train=False,transform=torchvision.transforms.ToTensor())
test_dataloder = DataLoader(dataset=test_set, batch_size=4, shuffle=True, num_workers=0, drop_last=False)

img , target = test_set[0]

print(img.shape)
print(target)



writer = SummaryWriter('dataloder_logs')
step = 0
for data in test_dataloder:
    imgs, targets = data
    writer.add_images('dataloder_logs', imgs, step)
    step += 1

writer.close()