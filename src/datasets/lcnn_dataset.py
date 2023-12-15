import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchaudio

import pandas as pd
import numpy as np

class LCNNDataset(Dataset):
    def __init__(self, flac_path: str, labels_path: str, n_fft: int) -> None:
        super().__init__()
        self.data = pd.read_csv(
            labels_path, sep=' ', 
            names=['SpeakerID', 'UtteranceID', 'UtteranceType', 'SpoofAlgoId', 'IsSpoofed'], header=None
        )
        self.flac_path = flac_path
        self.n_fft = n_fft
    
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, index):
        audio_id = self.data.iloc[index]['UtteranceID']
        label = self.data.iloc[index]['IsSpoofed']
        label = 0 if label == 'bonafide' else 1
        waveform, sample_rate = torchaudio.load(f'{self.flac_path}/{audio_id}.flac')
        spec = torchaudio.transforms.Spectrogram(
            n_fft=self.n_fft, 
            win_length=1724,
            hop_length=int(0.0081 * sample_rate),
            window_fn=torch.blackman_window
        )(waveform)
        t = spec.size()[-1]
        if t >= 750:
            spec = spec[:, :, :750]
        else:
            spec = F.pad(spec, (0, 750-t), 'constant', 0)
        return spec, label