from __future__ import annotations
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

class Residual(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 1),
        )
    
    def forward(self, x): return x + self.block(x)

def conv_block(ci: int, co: int, ks: int = 3, stride: int = 1, pad: int = 1):
    return nn.Sequential(
        nn.Conv2d(ci, co, ks, stride=stride, padding=pad),
        nn.BatchNorm2d(co),
        nn.ReLU(inplace=True),
    )