#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import open3d as o3d
from open3d import JVisualizer


# In[14]:


#from shapenet dataset

def visualize(path,viz_type = 'pc', num_points = 5000,radii = [0.005, 0.01, 0.02, 0.04]):
    
    #load point cloud and normals
    pc_xyz = np.load(path+'points.npy')
    pc_normals = np.load(path+'normals.npy')
    
    
    #sample points or visualisation
    
    selected_idx = np.random.permutation(np.arange(pc_xyz.shape[0]))[:num_points]
    pc_xyz = pc_xyz[selected_idx]
    pc_normals = pc_normals[selected_idx]

 
    #create point cloud dataset using open3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc_xyz)
    pcd.normals = o3d.utility.Vector3dVector(pc_normals)
    pcd.colors = o3d.utility.Vector3dVector(pc_normals)

    if(viz_type=='pc'):
        #visualize point cloud
        
        visualizer = JVisualizer()
        visualizer.add_geometry(pcd)
        visualizer.show()

        
    else:
        
        #create mesh
        rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector(radii))
    
        #visualize mesh
        o3d.visualization.draw_geometries([rec_mesh])

    
    


# In[15]:


path = '/home/shanthika/Documents/CV/project/subset(1)/subset/ShapeNet/'
class_id = '02828884/'
file_id = '1b0463c11f3cc1b3601104cd2d998272/'
filename = path+class_id+file_id+'pointcloud/'
visualize(filename,viz_type='mesh')

