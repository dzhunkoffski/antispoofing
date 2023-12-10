import torch
from torch import nn
import torch.nn.functional as F

from src.model.rawnet2.sinc_layer import SincConv_fast
from src.model.rawnet2.fms import FMS

from src.base.base_model import BaseModel

class ResBlockInitial(BaseModel):
    def __init__(self, in_channels:int, out_channels: int) -> None:
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding='same'),
            nn.BatchNorm1d(num_features=out_channels),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding='same')
        )
        self.conv_pad_channels = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, padding='same')
        self.maxpool = nn.MaxPool1d(kernel_size=3)
        self.fms = FMS()

    def forward(self, x):
        res = self.layer(x)
        x = self.conv_pad_channels(x) + res
        x = self.maxpool(x)
        x = self.fms(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layer = nn.Sequential(
            nn.BatchNorm1d(num_features=in_channels),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding='same'),
            nn.BatchNorm1d(num_features=out_channels),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding='same')
        )

        self.pad_channels = False
        if in_channels != out_channels:
            self.conv_pad_channels = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, padding='same')
            self.pad_channels = True

        self.maxpool = nn.MaxPool1d(kernel_size=3)
        self.fms = FMS()
    
    def forward(self, x):
        res = self.layer(x)
        if self.pad_channels:
            x = self.conv_pad_channels(x) + res
        else:
            x = x + res
        x = self.maxpool(x)
        x = self.fms(x)
        return x
    
class AbsoluteWrapper(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        x = torch.absolute(x)
        return x

class RawNet2(nn.Module):
    def __init__(self, ) -> None:
        super().__init__()
        self.fixed_sync_filters = nn.Sequential(
            SincConv_fast(out_channels=128, kernel_size=1024, in_channels=1, stride=1, min_low_hz=0, min_band_hz=0),
            AbsoluteWrapper(),
            nn.MaxPool1d(kernel_size=3),
            nn.BatchNorm1d(num_features=128),
            nn.LeakyReLU()
        )
        self.resblock1 = nn.Sequential(
            ResBlockInitial(in_channels=128, out_channels=20),
            ResBlock(in_channels=20, out_channels=20)
        )
        self.resblock2 = nn.Sequential(
            ResBlock(in_channels=20, out_channels=128),
            ResBlock(in_channels=128, out_channels=128),
            ResBlock(in_channels=128, out_channels=128),
            ResBlock(in_channels=128, out_channels=128)
        )
        self.batch_norm = nn.BatchNorm1d(num_features=128)
        self.gru = nn.GRU(input_size=128, hidden_size=1024, num_layers=3)
        self.fc = nn.Linear(in_features=1024, out_features=1024)
        self.activasion = nn.LeakyReLU()
        self.clf = nn.Linear(in_features=1024, out_features=2)

    
    def forward(self, audio, **batch):
        x = self.fixed_sync_filters(audio)
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.batch_norm(x)
        x = self.activasion(x)
        x = torch.permute(x, (0, 2, 1))
        x, _ = self.gru(x)
        x = x[:, -1, :]
        x = self.fc(x)
        x = self.activasion(x)
        x = self.clf(x)
        return {"logits": x}
    