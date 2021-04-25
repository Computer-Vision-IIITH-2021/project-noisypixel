import sys
sys.path.append("/home2/sdokania/all_projects/project-noisypixel/")

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
import torch.distributions as dist


#mesh
from src.utils.libmise.mise import  MISE
from src.utils.libmcubes.mcubes import marching_cubes
import trimesh
from src.evaluate import *

from src.models import *
from src.dataset.dataloader import OccupancyNetDatasetHDF
from src.trainer import ONetLit
from src.utils import Config, count_parameters
import datetime
import tqdm
import torch.distributions as dist
import pandas as pd
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Argument parser for training the model")
    default_ckpt = "../occ_artifacts/efficient_cbn_bs_64_full_data/lightning_logs/version_1/checkpoints/epoch=131-step=63359.ckpt"
    parser.add_argument('--cdim', action='store', type=int, default=128, help="feature dimension")
    parser.add_argument('--hdim', action='store', type=int, default=128, help="hidden size for decoder")
    parser.add_argument('--pdim', action='store', type=int, default=3, help="points input size for decoder")
    parser.add_argument('--data_root', action='store', type=str, default="/ssd_scratch/cvit/sdokania/processed_data/hdf_data/", help="location of the parsed and processed dataset")
    parser.add_argument('--batch_size', action='store', type=int, default=64, help="Training batch size")
    parser.add_argument('--output_path', action='store', type=str, default="/home2/sdokania/all_projects/occ_artifacts/", help="Model saving and checkpoint paths")
    parser.add_argument('--exp_name', action='store', type=str, default="initial", help="Name of the experiment. Artifacts will be created with this name")
    parser.add_argument('--encoder', action='store', type=str, default="efficientnet-b0", help="Name of the Encoder architecture to use")
    parser.add_argument('--decoder', action='store', type=str, default="decoder-cbn", help="Name of the decoder architecture to use")
    parser.add_argument('--checkpoint', action='store', type=str, default=default_ckpt, help="Checkpoint Path")
    
    args = parser.parse_args()
    # Get the model configuration
    config = Config(args)

    onet = ONetLit(config)
    net = ONetLit.load_from_checkpoint(args.checkpoint, cfg=config).eval()
    dataset = OccupancyNetDatasetHDF(config.data_root, num_points=2048, mode="test", point_cloud=True)

    empty_point_dict = {
        'completeness': np.sqrt(3),
        'accuracy': np.sqrt(3),
        'completeness2': 3,
        'accuracy2': 3,
        'chamfer': 6,
    }

    empty_normal_dict = {
        'normals completeness': -1.,
        'normals accuracy': -1.,
        'normals': -1.,
    }

    DEVICE="cuda:0"
    nux = 0
    start = datetime.datetime.now()
    result = []

    shuffled_idx = np.random.permutation(np.arange(len(dataset)))[:500]

    for ix in tqdm.tqdm(shuffled_idx):
        try:
            test_img, test_pts, test_gt, pcl_gt, norm_gt = dataset[ix][:]
            net.to(DEVICE)
            pred_pts = net(test_img.unsqueeze(0).to(DEVICE), test_pts.unsqueeze(0).to(DEVICE)).cpu()
            mesh, mesh_data, normals = get_mesh(net, (test_img.to(DEVICE), test_pts, test_gt), threshold_g=0.5, return_points=True)
            pred_occ = dist.Bernoulli(logits=pred_pts).probs.data.numpy().squeeze()
            result.append(eval_pointcloud(mesh_data[0], pcl_gt, normals, norm_gt, pred_occ, test_gt))
        except:
            pass
    print(datetime.datetime.now() - start)
    df = pd.DataFrame(result)
    print(df.mean())