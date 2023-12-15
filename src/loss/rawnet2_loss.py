import torch
from torch import nn
import torch.nn.functional as F

class RawNet2Loss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss(weight=torch.tensor([0.9, 0.1]))
    
    def forward(self, logits, label, **batch):
        return self.cross_entropy(logits, label)