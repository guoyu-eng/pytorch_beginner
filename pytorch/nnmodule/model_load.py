import torch
import torchvision
from torch import nn
from torch.onnx.symbolic_opset9 import addcmul

# plan  1
model = torch.load('vgg16.pth')
# print(model)

# load2

model2 = torchvision.models.vgg16(pretrained=True)
model2.load_state_dict(torch.load('vgg16_model2.pth'))
# model2 = torch.load('vgg16_model2.pth')
# model.add_module('add_linear',nn.Linear(1000,10))
# del model.add_linear

print(model)