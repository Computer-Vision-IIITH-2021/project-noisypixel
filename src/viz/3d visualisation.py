#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import open3d as o3d
from open3d import JVisualizer


# In[20]:


#from shapenet dataset
def visualize(path):
    #load point cloud and normals
    pc_xyz = np.load(path+'points.npy')
    pc_normals = np.load(path+'normals.npy')
    
    
    num_points = 5000
    selected_idx = np.random.permutation(np.arange(pc_xyz.shape[0]))[:num_points]
    pc_xyz = pc_xyz[selected_idx]
    pc_normals = pc_normals[selected_idx]

 
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc_xyz)
    pcd.normals = o3d.utility.Vector3dVector(pc_normals)
    pcd.colors = o3d.utility.Vector3dVector(pc_normals)
# 
#     pcd.paint_uniform_color([1,1,1])

    #create mesh
    radii = [0.005, 0.01, 0.02, 0.04]
    rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector(radii))

    
    #visualize point cloud
    visualizer = JVisualizer()
    visualizer.add_geometry(pcd)
    visualizer.show()
    
    #visualize mesh
#     o3d.visualization.draw_geometries([rec_mesh])


# In[21]:


filename = '/home/shanthika/Documents/CV/project/subset(1)/subset/ShapeNet/02828884/1b0463c11f3cc1b3601104cd2d998272/pointcloud/'
visualize(filename)


# In[42]:





# In[ ]:




