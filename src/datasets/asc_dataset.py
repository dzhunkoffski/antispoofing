import torch
from torch import nn
from torch.utils.data import Dataset
import torchaudio

import pandas as pd

class ASVspoofDataset(Dataset):
    def __init__(self, flac_path: str, labels_path: str) -> None:
        super().__init__()
        self.data = pd.read_csv(
            labels_path, sep=' ', 
            names=['SpeakerID', 'UtteranceID', 'UtteranceType', 'SpoofAlgoId', 'IsSpoofed'], header=None
        )
        self.flac_path = flac_path
    
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, index):
        audio_id = self.data.iloc[index]['UtteranceID']
        label = self.data.iloc[index]['IsSpoofed']
        label = 0 if label == 'bonafide' else 1
        waveform, sample_rate = torchaudio.load(f'{self.flac_path}/{audio_id}.flac')
        assert sample_rate == 16000, 'Sample rate should be 16k'
        if waveform.size()[-1] >= 64000:
            waveform = waveform[:, :64000]
        else:
            buffer = torch.zeros(1, 64000)
            pad_size = 64000 - waveform.size()[-1]
            buffer[:, :waveform.size()[-1]] = waveform
            buffer[:, waveform.size()[-1]:] = waveform[:, pad_size]
            waveform = buffer
        return waveform, label
