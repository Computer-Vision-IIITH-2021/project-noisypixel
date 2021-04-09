import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import transforms, gridspec
import cv2

cam_path = '/home/saiamrit/Documents/CV Project/data/subset/ShapeNet/02691156/1ac29674746a0fc6b87697d3904b168b/img_choy2016/cameras.npz'
points_path = '/home/saiamrit/Documents/CV Project/data/subset/ShapeNet/02691156/1ac29674746a0fc6b87697d3904b168b/pointcloud/points.npy'
img_path = '/home/saiamrit/Documents/CV Project/data/subset/ShapeNet/02691156/1ac29674746a0fc6b87697d3904b168b/img_choy2016'

cameras = np.load(cam_path)
points = np.load(points_path)[3000:5000]
for i in range(10):
    fig = plt.figure(figsize=(16, 8))
    print('Using Camera matrix: world_mat_{}'.format(i))
    p = cameras['world_mat_{}'.format(i)]
    for j in range(points.shape[0]):
    	proj = p @ np.append(points[j],1).T
        proj = proj/proj[2]
        ax0 = plt.subplot(gs[0])
        ax0.plot(proj[1], proj[0], 'r*', markersize=3)
        ax0.set_title("Projected Point Cloud")
        ax0.axis('off')
    plt.show()    