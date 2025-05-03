from torch import nn
import torch


class Nnmodule(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, input):
        output = input +1
        return output
    
    
nnmodule = Nnmodule()
x = torch.randn(1.0)

output = nnmodule(x)
print(output)