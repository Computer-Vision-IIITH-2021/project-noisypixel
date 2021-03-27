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
            num_points (int): Number of points to sample in the object point cloud from the data
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.num_points = num_points
        self.files = []
        
        for sub in glob.glob(self.root_dir+'/*'):
            self.files.extend(glob.glob(sub+'/*'))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # Fetch the file path and setup image folder paths
        req_path = self.files[idx]
        img_folder = os.path.join(req_path, 'img_choy2016')
        img_path = random.choice(glob.glob(img_folder+'/*.jpg'))

        # Load the image with opencv and convert to RGB
        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        
        # Load the points data
        points_path = os.path.join(req_path, 'points.npz')
        data = np.load(points_path)
        
        # Get the actual point of the object
        points = data['points']
        # Unpack the occupancies of the object
        occupancies = np.unpackbits(data['occupancies'])

        # Sample n points from the data
        selected_idx = np.random.permutation(np.arange(points.shape[0]))[:self.num_points]

        # Use only the selected indices and pack everything up in a nice dictionary
        sample = {'image': image, 'points': points[selected_idx], 'occupancies': occupancies[selected_idx]}

        # Apply any transformation necessary
        if self.transform:
            sample = self.transform(sample)

        return sample


if __name__ == '__main__':
	dataset = OccupancyNetDataset( root_dir='/home/saiamrit/Documents/CV Project/data/subset/ShapeNet')
	print(len(dataset))
	dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0)
	print(len(dataloader))