import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class OccNetImg(nn.Module):
    """
    Wrapper for the overall occupancy network module. This will
    contain the encoder as well as the decoder and provide functionalities
    such as extraction of feature, decoding to compute occupancy, and an
    end-to-end forward pass over the encoder-decoder architectures.
    """
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
    

    def extract_features(self, x):
        return self.encoder(x)
    
    def forward(self, img, pts):
        # print(img.shape,  pts.shape)
        # Compute the image features
        c = self.extract_features(img)

        # print(c.shape)
        out = self.decoder(pts, c)

        return out