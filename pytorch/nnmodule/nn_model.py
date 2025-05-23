import torchvision.datasets
from torch import nn

# train_data = torchvision.datasets.ImageNet('./data_image_net,',split='train',download=True,
#                                            transform=torchvision.transforms.ToTensor())

vgg16_false = torchvision.models.vgg16(pretrained=False)
vgg16_true = torchvision.models.vgg16(pretrained=True)

print(vgg16_true)#


test_set  = torchvision.datasets.CIFAR10(root='../data', train=False,transform=torchvision.transforms.ToTensor(),
                                         download=True)
# test_dataloder = DataLoader(dataset=test_set, batch_size=1)

#  how to add some change on the pre-train model
vgg16_true.add_module('add_linear',nn.Linear(1000,10))
# vgg16_true.classifier.add_module('add_linear',nn.Linear(1000,10))


# change
vgg16_false.classifier[6] = nn.Linear(4096,10)