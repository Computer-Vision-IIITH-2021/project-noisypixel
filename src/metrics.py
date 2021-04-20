from pykdtree.kdtree import KDTree
import numpy as np

pc_path1 = '/home/madhvi/Documents/CV Project/data/subset/ShapeNet/02691156/1ac29674746a0fc6b87697d3904b168b/pointcloud.npz'
pc_path2 = '/home/madhvi/Documents/CV Project/data/subset/ShapeNet/02691156/1ac29674746a0fc6b87697d3904b168b/pointcloud.npz'
point_path1 = '/home/madhvi/Documents/CV Project/data/subset/ShapeNet/02691156/1ac29674746a0fc6b87697d3904b168b/points.npz'
point_path2 = '/home/madhvi/Documents/CV Project/data/subset/ShapeNet/02691156/1ac29674746a0fc6b87697d3904b168b/points.npz'

pc_data1 = np.load(pc_path1)
pc_data2 = np.load(pc_path2)
points_data1 = np.load(point_path1)
points_data2 = np.load(point_path2)

pointcloud = pc_data1['points']
pointcloud_gt = pc_data2['points']
normals = pc_data1['normals']
normals_gt = pc_data2['normals']
occ_1 = points_data1['occupancies']
occ_2 = points_data2['occupancies']

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