import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import *

train_data = torchvision.datasets.CIFAR10(root='../data', train=True,transform=torchvision.transforms.ToTensor(),
                                          download=True)
test_data = torchvision.datasets.CIFAR10(root='../data', train=False,transform=torchvision.transforms.ToTensor(),
                                         download=True)
# length of the data
train_data_size = len(train_data)
test_data_size = len(test_data)

print('trian data lendgth : {}'.format(train_data_size))
print(test_data_size)


# use the dataloder
train_loader = DataLoader(train_data, batch_size=64)
test_loader =DataLoader(test_data, batch_size=64)


# creart
nn_model = Nnmodel()

# loss function
loss_fn = nn.CrossEntropyLoss()

# opti
optim = torch.optim.SGD(nn_model.parameters(), lr=0.01)

# train parament
total_train_step = 0

# test number
total_test_step = 0
# epoch
epoch = 10


#  add the tensboard
writer = SummaryWriter()

for epoch in range(epoch):
    print('epoch {}'.format(epoch + 1))

    nn_model.train()
    for data in train_loader:
        images, labels = data

        # loss funtion
        output = nn_model(images)
        loss = loss_fn(output, labels)

        # optimse
        optim.zero_grad()
        loss.backward()
        optim.step()
        total_train_step += 1

        if total_train_step % 100 == 0:
            print('train times : {} , Loss:{}'.format(total_train_step, loss.item()))
            writer.add_scalar('train_loss', loss.item(), total_train_step)

    nn_model.eval()
    total_test_loss = 0
    total_test_accuracy = 0
    # donnt use the grad , bacause do not use it optim
    with torch.no_grad() :
        for data in test_loader:
            imgs, labels = data
            output = nn_model(imgs)
            loss = loss_fn(output, labels)
            total_test_loss += loss.item()
            accuracy = (output.argmax(1) == labels).sum()
            total_test_accuracy += accuracy.item()

    print('test loss : {}'.format(total_test_loss ))
    print('test accuracy : {}'.format(total_test_accuracy / len(test_data)))
    writer.add_scalar('test_loss', total_test_loss , total_test_step)
    writer.add_scalar('test_accuracy', total_test_accuracy / len(test_data), total_test_step)
    total_test_step += 1

#     save the model for each epoch
#   torch.save(nn_module, ' nnmodule_{}.pth'.format(epoch))

writer.close()






