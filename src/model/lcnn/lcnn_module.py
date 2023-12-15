import torch
from torch import nn
import torch.nn.functional as F

from src.model.lcnn.mfm import MaxFeatureMap

class LighCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5, stride=1, padding='same'),
            MaxFeatureMap()
        )
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1, stride=1, padding='same'),
            MaxFeatureMap(),
            nn.BatchNorm2d(num_features=32),
            nn.Conv2d(in_channels=32, out_channels=96, kernel_size=3, stride=1, padding='same'),
            MaxFeatureMap()
        )
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.batch_norm1 = nn.BatchNorm2d(num_features=48)
        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=96, kernel_size=1, stride=1, padding='same'),
            MaxFeatureMap(),
            nn.BatchNorm2d(num_features=48),
            nn.Conv2d(in_channels=48, out_channels=128, kernel_size=3, stride=1, padding='same'),
            MaxFeatureMap()
        )
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.block4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1, stride=1, padding='same'),
            MaxFeatureMap(),
            nn.BatchNorm2d(num_features=64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding='same'),
            MaxFeatureMap(),
            nn.BatchNorm2d(num_features=32),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1, stride=1),
            MaxFeatureMap(),
            nn.BatchNorm2d(num_features=32),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1),
            MaxFeatureMap()
        )
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.block5 = nn.Sequential(
            nn.Linear(in_features=32 * 52 * 45, out_features=160),
            MaxFeatureMap(),
            nn.BatchNorm1d(num_features=80)
        )
        self.clf = nn.Linear(in_features=80, out_features=2, bias=False)
    
    def forward(self, audio, **batch):
        x = self.block1(audio)
        x = self.maxpool1(x)
        x = self.block2(x)
        x = self.maxpool2(x)
        x = self.batch_norm1(x)
        x = self.block3(x)
        x = self.maxpool3(x)
        x = self.block4(x)
        x = self.maxpool4(x)
        x = torch.flatten(x, start_dim=1)
        x = self.block5(x)
        x = self.clf(x)
        return x