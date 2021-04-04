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
    def __init__(self, args=None):
        if args is None:
            self.set_default_data()
        else:
            self.c_dim = args.cdim
            self.h_dim = args.hdim
            self.p_dim = args.pdim
            self.data_root = args.data_root
            self.batch_size = args.batch_size
            self.output_dir = args.output_path
            self.exp_name = args.exp_name
        
        self.exp_path = os.path.join(self.output_dir, self.exp_name)
        # optimizer related config
        self.lr = 3e-04
        
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.exp_path, exist_ok=True)
    
    def set_default_data(self):
        self.c_dim = 128
        self.h_dim = 128
        self.p_dim = 3
        self.data_root = "/ssd_scratch/"
        self.batch_size = 64
        self.output_dir = "/home2/sdokania/all_projects/occ_artifacts/"
        self.exp_name = "initial"
    
    def print_config(self):
        # Print as a dictionary
        print(vars(self))
    
    def export_config(self):
        return vars(self)