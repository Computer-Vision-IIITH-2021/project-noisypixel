import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import transforms, gridspec
import cv2

def plotProjection(img_path, points, cameras):
	'''
		- Projects randomly sampled points from the point cloud and projects them to 
			image space using the corresponding projection matrix.
		- Plots the projected point clouds and corresponding images

		Inputs:
			img_path: Path to the image folder of the object
			points: sampled points from the point cloud for the object
			cameras: Camera matrices for the corresponding images
	'''
	for i in range(10):
		# initializing grid spec object that can control the sub-plots widths
	    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 3]) 
	    # inititializing figure
	    fig = plt.figure(figsize=(16, 8))

	    # Loading the image to be plotted and performing rotation
	    print('Loading image: 00{}.jpg'.format(i))
	    img = cv2.imread(os.path.join(img_path,'00{}.jpg'.format(i)))
	    im = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

	    # Loading the camera matrix corresponding to the image
	    print('Using Camera matrix: world_mat_{}'.format(i))
	    p = cameras['world_mat_{}'.format(i)]

	    # Projecting the sampled points using the camera projection matrix
	    for j in range(points.shape[0]):
	        proj = p @ np.append(points[j],1).T
	        proj = proj/proj[2]
	        ax0 = plt.subplot(gs[0])
	        ax0.plot(proj[1], proj[0], 'r*', markersize=3)
	        ax0.set_title("Projected Point Cloud")
	        ax0.axis('off')
	        
	        ax1 = plt.subplot(gs[1])
	        ax1.imshow(im)
	        ax1.set_title("Corresponding Image")
	        ax1.axis('off')
	    plt.show()

if __name__ == '__main__':

	# Initializing the paths
	cam_path = '../ShapeNet/02691156/1ac29674746a0fc6b87697d3904b168b/img_choy2016/cameras.npz'
	points_path = '../ShapeNet/02691156/1ac29674746a0fc6b87697d3904b168b/pointcloud/points.npy'
	img_path = '../02691156/1ac29674746a0fc6b87697d3904b168b/img_choy2016'

	# Loading the camera matrices and pointclouds
	cameras = np.load(cam_path)
	pointcloud = np.load(points_path)[3000:5000]

	# Calling the plotter function
	plotProjection(img_path, points, cameras)