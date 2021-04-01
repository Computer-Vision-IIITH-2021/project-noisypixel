import os
import glob
import cv2
import random
import pandas as pd
from skimage import io
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import h5py


# Network building stuff
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models
import pytorch_lightning as pl
import torchmetrics

class Config:
    def __init__(self):
        self.c_dim = 128
        self.h_dim = 128
        self.p_dim = 3
        self.data_root = "/content/drive/MyDrive/datasets/hdf_data/"
        self.batch_size = 64
        self.output_dir = "/content/drive/MyDrive/datasets/occ_artifacts/"
        self.exp_name = "initial"

        # optimizer related config
        self.lr = 3e-04

        os.makedirs(self.output_dir, exist_ok=True)


config = Config()

class OccupancyNetDatasetHDF(Dataset):
    """Occupancy Network dataset."""

    def __init__(self, root_dir, transform=None, num_points=1024, default_transform=True):
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
        
        for sub in os.listdir(self.root_dir):
            self.files.append(sub)
            
        # If not transforms have been provided, apply default imagenet transform
        if transform is None and default_transform:
            self.transform = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                  std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.files)

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
        
        # Get the points and occupancies
        points = hf['points']['points'][()]
        occupancies = np.unpackbits(hf['points']['occupancies'][()])

        # Sample n points from the data
        selected_idx = np.random.permutation(np.arange(points.shape[0]))[:self.num_points]

        # Use only the selected indices and pack everything up in a nice dictionary
        final_image = torch.from_numpy(image).float().transpose(1, 2).transpose(0, 1) / image.max()
        final_points = torch.from_numpy(points[selected_idx]).float()
        final_gt = torch.from_numpy(occupancies[selected_idx]).float()
        
        # Close the hdf file
        hf.close()
        
        # Apply any transformation necessary
        if self.transform:
            final_image = self.transform(final_image)

        return [final_image, final_points, final_gt]


class Resnet50(nn.Module):
''' ResNet-50 encoder network.
Args:
    c_dim (int): output dimension of the latent embedding
'''

def __init__(self, c_dim):
    super().__init__()
    self.features = models.resnet50(pretrained=True)
    self.features.fc = nn.Sequential()
    self.fc = nn.Linear(2048, c_dim)

def forward(self, x):
    x = self.features(x)
    out = self.fc(x)
    return out

class ResBlockFC(nn.Module):
    def __init__(self, in_dim, out_dim=None, h_dim=None):
        super().__init__()
        if out_dim is None:
            out_dim = in_dim
        if h_dim is None:
            h_dim = min(in_dim, out_dim)
        
        self.fc_0 = nn.Linear(in_dim, h_dim)
        self.fc_1 = nn.Linear(h_dim, out_dim)
        self.act = nn.ReLU()

        if in_dim == out_dim:
            self.skip = None
        else:
            self.skip = nn.Linear(in_dim, out_dim, bias=False)

        # Initialize weights to zero
        nn.init.zeros_(self.fc_1.weight)
    
    def forward(self, x):
        out_0 = self.act(self.fc_0(x))
        out = self.act(self.fc_1(x))

        if self.skip is not None:
            x_skip = self.skip(x)
        else:
            x_skip = x
        
        return x_skip + out

class DecoderFC(nn.Module):
    def __init__(self, p_dim=3, c_dim=128, h_dim=128):
        super().__init__()
        self.p_dim = p_dim
        self.c_dim = c_dim
        self.h_dim = h_dim

        self.fc_p = nn.Linear(p_dim, h_dim)
        self.fc_c = nn.Linear(c_dim, h_dim)

        self.blocks = nn.Sequential(
            ResBlockFC(h_dim),
            ResBlockFC(h_dim),
            ResBlockFC(h_dim),
            ResBlockFC(h_dim),
            ResBlockFC(h_dim)
        )

        self.fc = nn.Linear(h_dim, 1)
        self.act = nn.ReLU()
    
    def forward(self, p, c):
        # Get size (B, N, D)
        batch_size, n_points, dim = p.size()
        # print(p.shape)
        enc_p = self.fc_p(p) # (B, N, h_dim)
        enc_c = self.fc_c(c).unsqueeze(1) # (B, 1, h_dim)

        # Add the features now
        enc = enc_p + enc_c

        # Run through the res blocks
        enc = self.blocks(enc)
        out = self.fc(self.act(enc)).squeeze(-1)
        return out

class OccNetImg(nn.Module):
    """
    Wrapper for the overall occupancy network module. This will
    contain the encoder as well as the decoder and provide functionalities
    such as extraction of feature, decoding to compute occupancy, and an
    end-to-end forward pass over the encoder-decoder architectures.
    """
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
    

    def extract_features(self, x):
        return self.encoder(x)
    
    def forward(self, img, pts):
        # print(img.shape,  pts.shape)
        # Compute the image features
        c = self.extract_features(img)

        # print(c.shape)
        out = self.decoder(pts, c)

        return out