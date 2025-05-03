import torch
import torchvision



vgg16 = torchvision.models.vgg16(pretrained=False)

# saving
torch.save(vgg16,'vgg16.pth')

# saving 2
torch.save(vgg16.state_dict(),'vgg16_model2.pth')