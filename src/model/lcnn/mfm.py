import torch
from torch import nn
import torch.nn.functional as F

class MaxFeatureMap(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        N = x.size()[1]
        group1 = x[:, :N//2]
        group2 = x[:, N//2:]

        output = torch.maximum(group1, group2)
        return output