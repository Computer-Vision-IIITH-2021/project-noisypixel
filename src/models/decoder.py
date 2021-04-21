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

class CondBatchNorm(nn.Module):
    ''' Conditional batch normalization layer class.
    Args:
        c_dim: dimension of latent conditioned code c
        p_dim: points feature dimension
        norm: normalization method
    '''

    def __init__(self, c_dim, p_dim, norm = 'batch_norm'):
        super().__init__()
        self.c_dim = c_dim
        self.p_dim = p_dim
        self.norm = norm
        
        # computing the gamma and beta values
        self.gamma = nn.Linear(c_dim, p_dim, 1)
        self.beta = nn.Linear(c_dim, p_dim, 1)

        if self.norm == 'batch_norm':
            self.batchnorm = nn.BatchNorm1d(p_dim, affine=False)
        elif self.norm == 'instance_norm':
            self.batchnorm = nn.InstanceNorm1d(p_dim, affine=False)
        elif self.norm == 'group_norm':
            self.batchnorm = nn.GroupNorm1d(p_dim, affine=False)
        else:
            raise ValueError('Invalid normalization method!')
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.beta.weight)
        nn.init.ones_(self.gamma.bias)
        nn.init.zeros_(self.beta.bias)

    def forward(self, x, c):
        assert(x.size(0) == c.size(0))
        assert(c.size(1) == self.c_dim)

        # c should be of size batch_size x c_dim x T
        if len(c.size()) == 2:
            c = c.unsqueeze(2)

        # Affine mapping
        gamma = self.gamma(c)
        beta = self.beta(c)

        # Batchnorm
        net = self.batchnorm(x)
        out = gamma * net + beta

        return out

class CondResBlock(nn.Module):
    ''' Conditional batch normalization-based Resnet block class.
    Args:
        c_dim (int): dimension of latent conditioned code c
        in_dim (int): input dimension
        out_dim (int): output dimension
        h_dim (int): hidden dimension
        norm (str): normalization method
    '''

    def __init__(self, c_dim, in_dim, h_dim=None, out_dim=None,
                 norm = 'batch_norm'):
        super().__init__()
        # Attributes
        if h_dim is None:
            h_dim = in_dim
        if out_dim is None:
            out_dim = in_dim

        self.in_dim = in_dim
        self.h_dim = h_dim
        self.out_dim = out_dim
        
        self.batchnorm_0 = CondBatchNorm(
            c_dim, in_dim, norm = norm)
        self.batchnorm_1 = CondBatchNorm(
            c_dim, h_dim, norm = norm)
            
        self.fc_0 = nn.Conv1d(in_dim, h_dim, 1)
        self.fc_1 = nn.Conv1d(h_dim, out_dim, 1)
        self.act = nn.ReLU()

        if in_dim == out_dim:
            self.skip = None
        else:
            self.skip = nn.Conv1d(in_dim, out_dim, 1, bias=False)
        # Initialization
        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x, c):
        out = self.fc_0(self.act(self.batchnorm_0(x, c)))
        out = self.fc_1(self.act(self.batchnorm_1(out, c)))

        if self.skip is not None:
            skip = self.skip(x)
        else:
            skip = x

        return skip + out

class Decoder(nn.Module):
    ''' Decoder with conditional batch normalization (CBN) class.
    Args:
        in_dim (int): input dimension
        c_dim (int): dimension of latent conditioned code c
        h_dim (int): hidden size of Decoder network
    '''

    def __init__(self, in_dim=3, c_dim=128,
                 h_dim=256):
        super().__init__()
        self.z_dim = z_dim
      
        # self.fc_z = nn.Linear(z_dim, h_dim)

        self.fc_p = nn.Conv1d(in_dim, h_dim, 1)
        self.blocks = nn.Sequential(
            CondResBlock(c_dim, h_dim),
            CondResBlock(c_dim, h_dim),
            CondResBlock(c_dim, h_dim),
            CondResBlock(c_dim, h_dim),
            CondResBlock(c_dim, h_dim)
        )

        self.bn = CondBatchNorm(c_dim, h_dim)

        self.fc_out = nn.Conv1d(h_dim, 1, 1)

        self.act = F.relu

    def forward(self, p, c, **kwargs):
        # p = p.transpose(1, 2)
        batch_size, D, T = p.size()
        enc_p = self.fc_p(p)

        # enc_z = self.fc_z(z).unsqueeze(2)
        enc = enc_p #+ enc_z

        enc = self.blocks(enc, c)
        # net = self.block1(net, c)
        # net = self.block2(net, c)
        # net = self.block3(net, c)
        # net = self.block4(net, c)

        out = self.fc_out(self.act(self.bn(enc, c)))
        out = out.squeeze(1)
        return out