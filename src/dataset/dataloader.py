import os
import glob
import torch
import h5py
import cv2
import random
import pandas as pd
from skimage import io
import numpy as np
from PIL import Image
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

        img_path = random.choice(glob.glob(img_folder + '/*.jpg'))

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
        sample = (
          torch.from_numpy(image).float().transpose(1, 2).transpose(0, 1), 
          torch.from_numpy(points[selected_idx]), 
          torch.from_numpy(occupancies[selected_idx]))

        # Apply any transformation necessary
        if self.transform:
            sample[0] = self.transform(sample[0])

        return sample
    

class OccupancyNetDatasetHDF(Dataset):
    """Occupancy Network dataset."""

    def __init__(self, root_dir, transform=None, num_points=1024, default_transform=True, mode="train", balance=False):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
            num_points (int): Number of points to sample in the object point cloud from the data
                on a sample.
            mode (str): Which data split do we want among train, test and val
        """
        self.root_dir = root_dir
        self.transform = transform
        self.num_points = num_points
        self.mode = mode
        self.files = []
        self.pos_neg_ratio = [0.2, 0.5]
        self.balance = balance
        
        # Save the files
        f = open(os.path.join(self.root_dir, "{}.lst".format(self.mode)), 'r')
        self.files = f.read().split()
        f.close()
            
        # If not transforms have been provided, apply default imagenet transform
        if transform is None and default_transform:
            self.transform = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                  std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.files)
    
    def get_prob(self):
        return self.pos_neg_ratio[0] + (np.random.random() * (self.pos_neg_ratio[1] - self.pos_neg_ratio[0]))

    def __getitem__(self, idx):
        # Fetch the file path and setup image folder paths
        req_path = self.files[idx]
        file_path = os.path.join(self.root_dir, req_path)

        # Load the h5 file
        hf = h5py.File(file_path, 'r')
        
        # [NOTE]: the notation [()] below is to extract the value from HDF5 file
        # get all images and randomly pick one
        all_imgs = hf['images'][()]
        random_idx = int(np.random.random()*all_imgs.shape[0])
        
        # Fetch the image we need
        image = all_imgs[random_idx]
        try:
            # Get the points and occupancies
            points = hf['points']['points'][()]
            occupancies = np.unpackbits(hf['points']['occupancies'][()])
            
            # Sample n points from the data
            if self.balance:
                # Create index list
                indices = np.arange(occupancies.shape[0])
                n_pos = min(int(self.num_points * self.get_prob()), (occupancies == 1).sum())
                n_neg = self.num_points - n_pos
                positive_idx = np.random.permutation(indices[occupancies == 1])[:n_pos]
                negative_idx = np.random.permutation(indices[occupancies == 0])[:n_neg]
                selected_idx = np.concatenate([positive_idx, negative_idx])
                
            else:
                selected_idx = np.random.permutation(np.arange(points.shape[0]))[:self.num_points]
            

            # Use only the selected indices and pack everything up in a nice dictionary
            final_image = torch.from_numpy(image).float().transpose(1, 2).transpose(0, 1) / image.max()
            final_points = torch.from_numpy(points[selected_idx]).float()
            final_gt = torch.from_numpy(occupancies[selected_idx]).float()
        except:
            print(idx, file_path)
        
        # Close the hdf file
        hf.close()
        
        # Apply any transformation necessary
        if self.transform:
            final_image = self.transform(final_image)

        return [final_image, final_points, final_gt]


if __name__ == '__main__':
	dataset = OccupancyNetDataset( root_dir='/home/saiamrit/Documents/CV Project/data/subset/ShapeNet')
	print(len(dataset))
	dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0)
	print(len(dataloader))
