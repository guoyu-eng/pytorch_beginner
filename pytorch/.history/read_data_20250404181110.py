from torch.utils.data import Dataset
# import pandas as pd
import numpy as np
import torch
import os
from PIL import Image
class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        self.path = os.path.join(self.data, self.labels)
        self.img_path = os.listdir(self.path)

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
        img_name = self.img_path[idx]
        img_item_path = os.path.join(self.data, self.labels, img_name)
        image = Image.open(img_item_path)
        label = self.labels
        return image, label
    

root_dir = 'hymenoptera_data/train'
ants_label_dir  = 'ants'
bees_label_dir  = 'bees'
antsdata  = CustomDataset(root_dir, ants_label_dir)
beesdata  = CustomDataset(root_dir, bees_label_dir)

train_data = antsdata + beesdata
print(len(train_data))
print(antsdata[0])