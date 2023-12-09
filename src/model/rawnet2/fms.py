import torch
from torch import nn
import torch.nn.functional as F

class FMS(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(in_features=1, out_features=1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        '''
        input: (batch_size, n_channels, time_len)
        '''
        s = torch.mean(x, dim=-1).unsqueeze(-1)
        s = self.fc(s)
        s = self.sigmoid(s)

        x = x * s + s

        return x
