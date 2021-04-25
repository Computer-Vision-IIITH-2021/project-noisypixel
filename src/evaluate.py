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
from pykdtree.kdtree import KDTree


def make_3d_grid(bb_min, bb_max, shape):
    ''' Makes a 3D grid.
    Args:
        bb_min (tuple): bounding box minimum
        bb_max (tuple): bounding box maximum
        shape (tuple): output shape
    '''
    size = shape[0] * shape[1] * shape[2]

    pxs = torch.linspace(bb_min[0], bb_max[0], shape[0])
    pys = torch.linspace(bb_min[1], bb_max[1], shape[1])
    pzs = torch.linspace(bb_min[2], bb_max[2], shape[2])

    pxs = pxs.view(-1, 1, 1).expand(*shape).contiguous().view(size)
    pys = pys.view(1, -1, 1).expand(*shape).contiguous().view(size)
    pzs = pzs.view(1, 1, -1).expand(*shape).contiguous().view(size)
    p = torch.stack([pxs, pys, pzs], dim=1)

    return p

def eval_points(net, p, c, points_batch_size=100000):
    """
    """
    p_split = torch.split(p, points_batch_size)
    # print(len(p_split))
    occ_hats = []

    for pi in p_split:
        pi = pi.unsqueeze(0) 
        with torch.no_grad():
            occ_hat = net.net.decoder(pi.to(net.device), c.to(net.device))

        occ_hats.append(occ_hat.squeeze(0).detach().cpu())

    occ_hat = torch.cat(occ_hats, dim=0)

    return occ_hat

def extract_mesh(occ_hat, padding=0.1, threshold_g=0.2):
    n_x, n_y, n_z = occ_hat.shape
    box_size = 1 + padding
    threshold = np.log( threshold_g) - np.log(1. - threshold_g)
    
    occ_hat_padded = np.pad(occ_hat, 1, 'constant', constant_values=-1e6)
    # print(threshold,occ_hat_padded.shape, np.min(occ_hat_padded), np.max(occ_hat_padded))
    vertices, triangles = marching_cubes(occ_hat_padded, threshold)
  
    vertices -= 0.5
    # Undo padding
    vertices -= 1
    # Normalize to bounding box
    vertices /= np.array([n_x-1, n_y-1, n_z-1])
    vertices = box_size * (vertices - 0.5)

    mesh = build_mesh(vertices, triangles)
    return mesh, (vertices, triangles)

def build_mesh(vertices, triangles, normals=None):
    mesh = trimesh.Trimesh(vertices, triangles, vertex_normals=normals, process=False)
    return mesh

def get_mesh(net, data, padding=0.1, resolution0=32, upsampling_steps=2, threshold_g=0.2, return_points=False):
    # Get the image, points, and the ground truth
    test_img, test_pts, test_gt = data
    
    # Get the threshold and the box padding
    threshold = np.log( threshold_g) - np.log(1. - threshold_g)
    box_size = 1 +  padding
    nx = 32
    pointsf = 2 * make_3d_grid((-0.5,)*3, (0.5,)*3, (nx,)*3    )
    c = net.net.encoder(test_img.unsqueeze(0)).detach()
    
    if(upsampling_steps==0): 
        values = eval_points(net, pointsf,c ).cpu().numpy()
        value_grid = values.reshape(nx, nx, nx)
    else:
        mesh_extractor = MISE(resolution0, upsampling_steps, threshold)
        points = mesh_extractor.query()
        while points.shape[0] != 0:
            # Query points
            pointsf = torch.FloatTensor(points) 
            # Normalize to bounding box
            pointsf = pointsf / mesh_extractor.resolution
            pointsf = box_size * (pointsf - 0.5)
            # Evaluate model and update
            # print(pointsf.shape, c.shape)
            values = eval_points(net, pointsf, c).cpu().numpy()
            values = values.astype(np.float64)
            mesh_extractor.update(points, values)
            points = mesh_extractor.query()
        value_grid = mesh_extractor.to_dense()     
    mesh, mesh_data = extract_mesh(value_grid, threshold_g=threshold_g)
    
    normals = get_normals(net, mesh_data[0], c)
    mesh = build_mesh(mesh_data[0], mesh_data[1], normals)
    
    if return_points:
        return mesh, mesh_data, normals
    return mesh

def get_normals(net, vertices, c):
    pts = torch.FloatTensor(vertices)
    vertices_split = torch.split(pts, 10000)

    normals = []
    for vi in vertices_split:
        # net.zero_grad()
        vi = vi.unsqueeze(0)
        vi.requires_grad_()
        occ_hat = net.net.decoder(vi.to(net.device), c.to(net.device))
        out = occ_hat.sum()
        out.backward()
        ni = -vi.grad
        ni = ni / torch.norm(ni, dim=-1, keepdim=True)
        ni = ni.squeeze(0).cpu().numpy()
        normals.append(ni)

    normals = np.concatenate(normals, axis=0)
    return normals

def compute_iou(occ1, occ2):
    ''' Computes the Intersection over Union (IoU) value for two sets of
    occupancy values.
    Args:
        occ1 (tensor): first set of occupancy values
        occ2 (tensor): second set of occupancy values
    '''
    occ1 = np.asarray(occ1)
    occ2 = np.asarray(occ2)

    # Put all data in second dimension
    # Also works for 1-dimensional data
    if occ1.ndim >= 2:
        occ1 = occ1.reshape(occ1.shape[0], -1)
    if occ2.ndim >= 2:
        occ2 = occ2.reshape(occ2.shape[0], -1)

    # Convert to boolean values
    occ1 = (occ1 >= 0.5)
    occ2 = (occ2 >= 0.5)

    # Compute IOU
    area_union = (occ1 | occ2).astype(np.float32).sum(axis=-1)
    area_intersect = (occ1 & occ2).astype(np.float32).sum(axis=-1)

    iou = (area_intersect / area_union)

    return iou

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

def compute_separation(points_src, normals_src, points_tgt, normals_tgt):
    ''' Computes minimal distances of each point in points_src to points_tgt.
    Args:
        points_src (numpy array): source points
        normals_src (numpy array): source normals
        points_tgt (numpy array): target points
        normals_tgt (numpy array): target normals
    '''
    kdtree = KDTree(points_tgt)
    sepr, ind = kdtree.query(points_src)

    if normals_src is not None and normals_tgt is not None:
        normals_src = normals_src / np.linalg.norm(normals_src, axis=-1, keepdims=True)
        normals_tgt = normals_tgt / np.linalg.norm(normals_tgt, axis=-1, keepdims=True)

        normals_dot_product = (normals_tgt[ind] * normals_src).sum(axis=-1)
        normals_dot_product = np.abs(normals_dot_product)
    else:
        normals_dot_product = np.array(
            [np.nan] * points_src.shape[0], dtype=np.float32)
    return sepr, normals_dot_product

def eval_pointcloud(pointcloud, pointcloud_gt,
                        normals, normals_gt, occ1, occ2):
        ''' 
        Evaluates a point cloud.
        Args:
            pointcloud (numpy array): predicted point cloud
            pointcloud_gt (numpy array): ground truth point cloud
            normals (numpy array): predicted normals
            normals_gt (numpy array): ground truth normals
        '''
        # Return maximum losses if pointcloud is empty
        if pointcloud.shape[0] == 0:
            print('Empty pointcloud / mesh detected!')
            # [ERR]: there's supposed to be a .copy() here
            out_dict = empty_point_dict.copy()
            if normals is not None and normals_tgt is not None:
                out_dict.update(empty_normal_dict)
            return out_dict

        pointcloud = np.asarray(pointcloud)
        pointcloud_gt = np.asarray(pointcloud_gt)

        # Completeness: how far are the points of the groundtruth point cloud
        # from the predicted point cloud
        completeness, normal_completeness = compute_separation(
            pointcloud_gt, normals_gt, pointcloud, normals
        )
        completeness_sq = completeness**2

        completeness = completeness.mean()
        completeness_sq = completeness_sq.mean()
        normal_completeness = normal_completeness.mean()

        # Accuracy: how far are the points of the predicted pointcloud
        # from the groundtruth pointcloud
        accuracy, normal_accuracy = compute_separation(
            pointcloud, normals, pointcloud_gt, normals_gt
        )
        accuracy_sq = accuracy**2

        accuracy = accuracy.mean()
        accuracy_sq = accuracy_sq.mean()
        normal_accuracy = normal_accuracy.mean()

        # Chamfer distance
        chamferL2 = 0.5 * (completeness_sq + accuracy_sq)
        normals_correction = (
            0.5 * normal_completeness + 0.5 * normal_accuracy
        )
        chamferL1 = 0.5 * (completeness + accuracy)
        
        occupancy_iou = compute_iou(occ1, occ2)

        out_dict = {
            'completeness': completeness,
            'accuracy': accuracy,
            'normals completeness': normal_completeness,
            'normals accuracy': normal_accuracy,
            'normals': normals_correction,
            'completeness_sq': completeness_sq,
            'accuracy_sq': accuracy_sq,
            'chamfer-L2': chamferL2,
            'chamfer-L1': chamferL1,
            'iou': occupancy_iou
        }

        return out_dict