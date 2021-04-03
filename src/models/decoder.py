import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

class ResBlockFC(nn.Module):
    def __init__(self, in_dim, out_dim=None, h_dim=None):
        super().__init__()
        if out_dim is None:
            out_dim = in_dim
        if h_dim is None:
            h_dim = min(in_dim, out_dim)
        
        self.fc_0 = nn.Linear(in_dim, h_dim)
        self.fc_1 = nn.Linear(h_dim, out_dim)
        self.act = nn.ReLU()

        if in_dim == out_dim:
            self.skip = None
        else:
            self.skip = nn.Linear(in_dim, out_dim, bias=False)

        # Initialize weights to zero
        nn.init.zeros_(self.fc_1.weight)
    
    def forward(self, x):
        out_0 = self.act(self.fc_0(x))
        out = self.act(self.fc_1(x))

        if self.skip is not None:
            x_skip = self.skip(x)
        else:
            x_skip = x
        
        return x_skip + out

class DecoderFC(nn.Module):
    def __init__(self, p_dim=3, c_dim=128, h_dim=128):
        super().__init__()
        self.p_dim = p_dim
        self.c_dim = c_dim
        self.h_dim = h_dim

        self.fc_p = nn.Linear(p_dim, h_dim)
        self.fc_c = nn.Linear(c_dim, h_dim)

        self.blocks = nn.Sequential(
            ResBlockFC(h_dim),
            ResBlockFC(h_dim),
            ResBlockFC(h_dim),
            ResBlockFC(h_dim),
            ResBlockFC(h_dim)
        )

        self.fc = nn.Linear(h_dim, 1)
        self.act = nn.ReLU()
    
    def forward(self, p, c):
        # Get size (B, N, D)
        batch_size, n_points, dim = p.size()
        # print(p.shape)
        enc_p = self.fc_p(p) # (B, N, h_dim)
        enc_c = self.fc_c(c).unsqueeze(1) # (B, 1, h_dim)

        # Add the features now
        enc = enc_p + enc_c

        # Run through the res blocks
        enc = self.blocks(enc)
        out = self.fc(self.act(enc)).squeeze(-1)
        return out