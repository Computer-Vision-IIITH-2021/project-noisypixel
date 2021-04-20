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

try:
    from .models import *
    from .dataset.dataloader import OccupancyNetDatasetHDF
    from .utils import Config
except:
    from models import *
    from dataset.dataloader import OccupancyNetDatasetHDF
    from utils import Config



class ONetLit(pl.LightningModule):
    def __init__(self, cfg=None):
        super().__init__()
        if cfg is None:
            cfg = Config()
        self.config = cfg

        self.build_model()

    def build_model(self):
        # First we create the encoder and decoder models
        encoder_model = build_encoder(self.config.encoder)(self.config.c_dim)
        decoder_model = build_decoder(self.config.decoder)(
            self.config.p_dim, self.config.c_dim, self.config.h_dim)
        
        # Now, we initialize the decoder model
        self.net = OccNetImg(encoder_model, decoder_model)
    
    def forward(self, img, pts):
        return self.net(img, pts)
    
    def training_step(self, batch, batch_idx):
        imgs, pts, gts = batch
        output = self(imgs, pts)

        loss = F.binary_cross_entropy_with_logits(output, gts, reduction='none').sum(-1).mean()
        self.log("train_loss", loss.item())
        return loss
    
    def validation_step(self, batch, batch_idx):
        imgs, pts, gts = batch
        output = self(imgs, pts)

        loss = F.binary_cross_entropy_with_logits(output, gts, reduction='none').mean()
        acc = ((output > 0.5) == (gts > 0.5)).sum() / gts.flatten().shape[0]
        self.log("val_loss", loss.item())
        self.log("acc_loss", acc.item())
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.config.lr)
    
    def setup(self, stage=None):
        self.train_dataset = OccupancyNetDatasetHDF(self.config.data_root, mode="subtrain", balance=True)
        self.val_dataset = OccupancyNetDatasetHDF(self.config.data_root, mode="val", balance=True)
    
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, 
                                           batch_size=self.config.batch_size, 
                                           shuffle=True,
                                           num_workers=4)
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, 
                                           batch_size=self.config.batch_size, 
                                           shuffle=False)