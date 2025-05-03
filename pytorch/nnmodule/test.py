# from transform.transforms import image_path
import torch
import torchvision
from PIL import Image
from torch import nn

image_path = '../hymenoptera_data/train/ants/0013035.jpg'
image =  Image.open(image_path)
print(image)

image = image.convert('RGB')

transform = torchvision.transforms.Compose([torchvision.transforms.Resize((64,64)),
                                            torchvision.transforms.ToTensor()])
image = transform(image)
print(image.shape)



class Nnmodel(torch.nn.Module):
    def __init__(self):
        super(Nnmodel, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(in_features=4 * 4 * 64, out_features=64),
            nn.Linear(in_features=64, out_features=10)
        )

    def forward(self, x):
        x = self.model(x)
        return x

model =torch.load('vgg16.pth', map_location=torch.device('cpu'), weights_only=False)
# print(model)
image = torch.reshape(image,(1, 3, 64, 64))

model.eval()
with torch.no_grad():
    out = model(image)
print(out.argmax(1))