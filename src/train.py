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
from pytorch_lightning.loggers import TensorBoardLogger
import torchmetrics

from models import *
from dataset.dataloader import OccupancyNetDatasetHDF
from trainer import ONetLit
from utils import Config, count_parameters


if __name__ == "__main__":
    # Get the model configuration
    config = Config()
    
    # Define the lightning module
    onet = ONetLit(config)
    
    # Initialize tensorboard logger
    logger = TensorBoardLogger(
        save_dir=os.getcwd(),
        version=1,
        name="lightning_logs"
    )
    
    # Define the trainer object
    trainer = pl.Trainer(
        gpus=1,
        # auto_scale_batch_size='binsearch',
        logger=logger,
        min_epochs=1,
        max_epochs=5,
        default_root_dir=os.getcwd(),
        log_every_n_steps=10,
        progress_bar_refresh_rate=2,
        # precision=16,
        # stochastic_weight_avg=True,
        track_grad_norm=2,
    )
    
    # Start training
    trainer.fit(onet)
