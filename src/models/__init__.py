import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from .decoder import DecoderFC, DecoderCBN
from .efficientnet import EfficientNetB0, EfficientNetB1, EfficientNetB5, EfficientNetB7
from .resnet import Resnet50, Resnet18


encoder_models = {
    "resnet-50": Resnet50,
    "resnet-18": Resnet18,
    "efficientnet-b0": EfficientNetB0,
    "efficientnet-b1": EfficientNetB1,
    "efficientnet-b5": EfficientNetB5,
    "efficientnet-b7": EfficientNetB7,
}

decoder_models = {
    "decoder-fc": DecoderFC,
    "decoder-cbn": DecoderCBN
}


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