from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import os
import numpy as np


wirter = SummaryWriter('logs')
image_path = r'..\hymenoptera_data\train\ants\0013035.jpg'

image_pil = Image.open(image_path)
image_narry = np.array(image_pil)


wirter.add_image('test', image_narry,2,dataformats='HWC')
for i in range(100):
    wirter.add_scalar('y=2x', 3*i, i)

# tensorboard --logdir=logs 
#  -- port=6006
# the tensorboad the graph log files
wirter.close()