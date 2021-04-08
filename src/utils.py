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