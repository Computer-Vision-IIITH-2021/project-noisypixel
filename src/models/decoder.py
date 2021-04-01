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