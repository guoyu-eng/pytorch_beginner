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
wirter.add_image('img_norm', img_norm,2)


# resize
tran_size = transforms.Resize((60  , 100))
img_size = tran_size(image_pil)
img_size = tensor_trans(img_size)
wirter.add_image('img_size', img_size, 0)

# compose
tran_resize_2 = transforms.Resize((512))

tran_compose = transforms.Compose([tran_resize_2,tensor_trans])
img_compose = tran_compose(image_pil)
wirter.add_image('img_size', img_compose, 1)

# random crop
tran_crop = transforms.RandomCrop((512))
img_crop = tran_compose([tran_crop,tensor_trans])
for i in range(10):
    img_crop2 = img_crop(image_pil)
    wirter.add_image('randomcrop', img_crop2, i)



wirter.close()
# tensot type





