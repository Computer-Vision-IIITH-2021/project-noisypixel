import torch.nn as nn
from torchvision import models
from efficientnet_pytorch import EfficientNet


class EfficientNetB0(nn.Module):
    ''' EfficientNet-b0 encoder network for image input.
    Args:
        c_dim (int): output dimension of the latent embedding
    '''

    def __init__(self, c_dim):
        super().__init__()
        self.features = EfficientNet.from_pretrained('efficientnet-b0', include_top=False)
        self.features.fc = nn.Sequential()
        self.fc = nn.Linear(1280, c_dim)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        out = self.fc(x)
        return out

    

class EfficientNetB1(nn.Module):
    ''' EfficientNet-b1 encoder network for image input.
    Args:
        c_dim (int): output dimension of the latent embedding
    '''

    def __init__(self, c_dim):
        super().__init__()
        self.features = EfficientNet.from_pretrained('efficientnet-b1', include_top=False)
        self.features.fc = nn.Sequential()
        self.fc = nn.Linear(1280, c_dim)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        out = self.fc(x)
        return out


class EfficientNetB5(nn.Module):
    ''' EfficientNet-b5 encoder network for image input.
    Args:
        c_dim (int): output dimension of the latent embedding
    '''

    def __init__(self, c_dim):
        super().__init__()
        self.features = EfficientNet.from_pretrained('efficientnet-b5', include_top=False)
        self.features.fc = nn.Sequential()
        self.fc = nn.Linear(2048, c_dim)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        out = self.fc(x)
        return out


class EfficientNetB7(nn.Module):
    ''' EfficientNet-b7 encoder network for image input.
    Args:
        c_dim (int): output dimension of the latent embedding
    '''

    def __init__(self, c_dim):
        super().__init__()
        self.features = EfficientNet.from_pretrained('efficientnet-b7', include_top=False)
        self.features.fc = nn.Sequential()
        self.fc = nn.Linear(2560, c_dim)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        out = self.fc(x)
        return out