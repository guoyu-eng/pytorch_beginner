import torchvision

train_set  = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True)

print(test_set[0])
img, target = test_set[0]
print(img)
print(target)