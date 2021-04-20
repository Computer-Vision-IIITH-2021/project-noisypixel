import os
import torch
import numpy as np

def count_parameters(network):
    """
    Function to count the number of parameters in an network
    """
    tot = 0
    for ix in network.parameters():
        tot += ix.flatten().shape[0]
    print("Parameters: {}M".format(np.round(tot/1e06, 3)))
    

class Config:
    def __init__(self):
        self.c_dim = 128
        self.h_dim = 128
        self.p_dim = 3
        self.data_root = "/ssd_scratch/"
        self.batch_size = 64
        self.output_dir = "/home2/sdokania/all_projects/occ_artifacts/"
        self.exp_name = "initial"

        # optimizer related config
        self.lr = 3e-04
        os.makedirs(self.output_dir, exist_ok=True)