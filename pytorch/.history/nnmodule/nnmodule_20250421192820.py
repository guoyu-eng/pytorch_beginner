from torch import nn
import torch


class Nnmodule(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, input):
        output = input
        return output
    
    
nnmodule = Nnmodule()
x = torch.randn(2, 3)

output = nnmodule(x)
print(output)