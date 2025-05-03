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


# nomalize
trans_norm = transforms.Normalize(mean=[3, 1, 4], std=[0.5, 0.5, 0.5])
img_norm = trans_norm(tensor_img)
wirter.add_image('img_norm', img_norm)

wirter.close()
# tensot type





