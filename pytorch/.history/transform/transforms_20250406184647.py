from torchvision import transforms
from torchvision.transforms import functional as F
from PIL import Image
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import patches

#
image_path = r'..\hymenoptera_data\train\ants\0013035.jpg'

image_pil = Image.open(image_path)
image_narry = np.array(image_pil)
# how to use the transforms

print(image_pil)
# tensot type
