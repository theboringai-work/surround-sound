"""Log-mel front end and the SED model (mel -> CNN -> Transformer -> per-frame sigmoid, 40 ms tokens)."""
import math

import torch
import torch.nn as nn
import torchaudio

SR, N_MELS, FRAME, DUR = 16000, 64, 0.04, 8.0     # sample rate, mel bins, seconds per output token, window


class LogMel(nn.Module):
    def __init__(self):
        super().__init__()
        self.m = torchaudio.transforms.MelSpectrogram(
            sample_rate=SR, n_fft=512, win_length=400, hop_length=160,
            n_mels=N_MELS, f_min=50, f_max=8000, power=2.0)
    def forward(self, wav):                      # (B, T) -> (B, M, T'), always fp32
        x = 10 * torch.log10(self.m(wav.float()) + 1e-9)
        return ((x + 60) / 20).clamp(-1, 3)

class ConvBlock(nn.Module):
    def __init__(self, cin, cout, stride=(1, 1)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, stride=stride, padding=1), nn.BatchNorm2d(cout), nn.ReLU(),
            nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU())
    def forward(self, x): return self.net(x)

class SED(nn.Module):
    def __init__(self, n_classes, n_mels=N_MELS, base=48, d=256, heads=8, layers=6):
        super().__init__()
        self.front = nn.Sequential(                    # stride = (freq, time); input (B, 1, M, T) at 10 ms
            ConvBlock(1, base),
            ConvBlock(base, base,         stride=(2, 1)),   # freq /2
            ConvBlock(base, 2 * base,     stride=(2, 1)),   # freq /4
            ConvBlock(2 * base, 2 * base, stride=(2, 2)),   # freq /8,  time /2 -> 20 ms
            ConvBlock(2 * base, 4 * base, stride=(2, 2)))   # freq /16, time /4 -> 40 ms
        self.proj = nn.Linear(4 * base * (n_mels // 16), d)
        pe = torch.zeros(4000, d)
        pos = torch.arange(4000).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(1e4) / d))
        pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.1,
                                           batch_first=True, norm_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.head = nn.Linear(d, n_classes)
    def forward(self, mel):                    # (B, M, T')
        f = self.front(mel.unsqueeze(1))       # (B, C, M', T'/4)
        B, Ch, M, T = f.shape
        h = self.proj(f.permute(0, 3, 1, 2).reshape(B, T, Ch * M)) + self.pe[:T]
        return self.head(self.enc(h))
