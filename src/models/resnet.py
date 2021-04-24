import torch.nn as nn
from torchvision import models


class Resnet18(nn.Module):
    ''' ResNet-18 encoder network for image input.
    Args:
        c_dim (int): output dimension of the latent embedding
    '''

    def __init__(self, c_dim):
        super().__init__()
        self.features = models.resnet18(pretrained=True)
        self.features.fc = nn.Sequential()
        self.fc = nn.Linear(512, c_dim)

    def forward(self, x):
        x = self.features(x)
        out = self.fc(x)
        return out


class Resnet50(nn.Module):
    ''' ResNet-50 encoder network.
    Args:
        c_dim (int): output dimension of the latent embedding
    '''

    def __init__(self, c_dim):
        super().__init__()
        self.features = models.resnet50(pretrained=True)
        self.features.fc = nn.Sequential()
        self.fc = nn.Linear(2048, c_dim)

    def forward(self, x):
        x = self.features(x)
        out = self.fc(x)
        return out
