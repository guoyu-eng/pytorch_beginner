from torchvision import transforms
from torchvision.transforms import functional as F
from PIL import Image
import random
import numpy as np
from torch.utils.tensorboard import SummaryWriter


#
image_path = r'..\hymenoptera_data\train\ants\0013035.jpg'

image_pil = Image.open(image_path)
image_narry = np.array(image_pil)
# how to use the transforms

# print(image_pil)
tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(image_pil)
# print(tensor_img)


wirter = SummaryWriter('logs')
wirter.add_image('tensor_img', tensor_img )
wirter.close()
# tensot type


