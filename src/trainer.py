import os
import glob
import cv2
import random
import pandas as pd
from skimage import io
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import h5py

# Network building stuff
import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl
import torchmetrics

from .models import *
from .dataset.dataloader import OccupancyNetDatasetHDF


class ONetLit(pl.LightningModule):
    def __init__(self, cfg=None):
        super().__init__()
        if cfg is None:
            cfg = Config()
        self.config = cfg

        self.build_model()

    def build_model(self):
        # First we create the encoder and decoder models
        self.encoder_model = Resnet50(self.config.c_dim)
        self.decoder_model = DecoderFC(self.config.p_dim, 
                                       self.config.c_dim, self.config.h_dim)
        
        # Now, we initialize the decoder model
        self.net = OccNetImg(self.encoder_model, self.decoder_model)
    
    def forward(self, img, pts):
        return self.net(img, pts)
    
    def training_step(self, batch, batch_idx):
        imgs, pts, gts = batch
        output = self(imgs, pts)

        loss = F.binary_cross_entropy_with_logits(output, gts)
        self.log("train_loss", loss.item())
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.config.lr)
    
    def setup(self, stage=None):
        self.train_dataset = OccupancyNetDatasetHDF(self.config.data_root)
    
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, 
                                           batch_size=self.config.batch_size, 
                                           shuffle=True,
                                           num_workers=2)