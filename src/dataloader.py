import os
import glob
import torch
import cv2
import random
import pandas as pd
from skimage import io
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class OccupancyNetDataset(Dataset):
    """Occupancy Network dataset."""

    def __init__(self, root_dir, transform=None, num_points=1024):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.files = []
        self.num_points = num_points
        
        for sub in glob.glob(self.root_dir+'/*'):
            self.files.extend(glob.glob(sub+'/*'))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        
        req_path = self.files[idx]
        img_folder = req_path + '/img_choy2016'
        img_path = random.choice(glob.glob(img_folder+'/*.jpg'))
#         image = Image.open(img_path)
        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
#         image = np.asarray(image)
#         image = Image.fromarray(image.astype('uint8'), 'RGB')
        
        points_path = req_path + '/points.npz'
        data = np.load(points_path)
        
        points = data['points']
        
        occupancies = np.unpackbits(data['occupancies'])

        sample = {'image': image, 'points': points, 'occupancies': occupancies}

        if self.transform:
            sample = self.transform(sample)

        return sample


if __name__ == '__main__':
	dataset = OccupancyNetDataset( root_dir='/home/saiamrit/Documents/CV Project/data/subset/ShapeNet')
	print(len(dataset))
	dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0)
	print(len(dataloader))