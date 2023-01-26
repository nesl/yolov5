import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset
import torch
import numpy as np
from utils.augmentations import letterbox
from utils.general import cv2

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, img_size, stride, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.img_size = img_size
        self.stride = stride

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        #image = read_image(img_path)
        image = cv2.imread(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        else:
            
            im = letterbox(image, self.img_size, stride=self.stride, auto=True)[0]  # padded resize
            im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            image = np.ascontiguousarray(im)  # contiguous
            
        if self.target_transform:
            label = self.target_transform(label)
        return image, label, img_path
