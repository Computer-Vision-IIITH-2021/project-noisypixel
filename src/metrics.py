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