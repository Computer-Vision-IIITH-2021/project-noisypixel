import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import transforms, gridspec
import cv2

cam_path = '/home/saiamrit/Documents/CV Project/data/subset/ShapeNet/02691156/1ac29674746a0fc6b87697d3904b168b/img_choy2016/cameras.npz'
points_path = '/home/saiamrit/Documents/CV Project/data/subset/ShapeNet/02691156/1ac29674746a0fc6b87697d3904b168b/pointcloud/points.npy'
img_path = '/home/saiamrit/Documents/CV Project/data/subset/ShapeNet/02691156/1ac29674746a0fc6b87697d3904b168b/img_choy2016'

cameras = np.load(cam_path)
points = np.load(points_path)[5000:10000]