
import torch.nn.functional as F
import torch

input = torch.tensor([[1,2,3,4,5],
                     [6,7,8,9,10],
                     [11,12,13,14,15],
                     [16,0,18,0,20],
                     [21,22,23,20,25]])

kernel = torch.tensor([[1,0,0],
                      [0,1,0],
                      [0,0,1]])
input = torch.shape(input,(1,1,5,5))
kernel = torch.shape(kernel,(1,1,3,3))

# torch.nn.functional.conv2d(input.unsqueeze(0).unsqueeze(0), kernel.unsqueeze(0).unsqueeze(0), padding=1, stride=1)

output = F.conv2d(input, kernel, padding=1, stride=1)
print(output)