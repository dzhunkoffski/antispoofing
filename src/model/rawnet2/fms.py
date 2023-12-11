import torch
from torch import nn
import torch.nn.functional as F

class FMS(nn.Module):
    def __init__(self, n_channels: int) -> None:
        super().__init__()
        self.n_channels = n_channels
        self.fc = nn.Linear(in_features=n_channels, out_features=n_channels)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        '''
        input: (batch_size, n_channels, time_len)
        '''
        s = torch.mean(x, dim=-1)
        s = self.fc(s)
        s = self.sigmoid(s)
        s = s.unsqueeze(-1)

        x = x * s + s

        return x
