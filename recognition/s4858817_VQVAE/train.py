from __future__ import annotations
from pathlib import Path
import argparse
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from dataset import get_dataloader
from modules import VQVAE

# defining a function to do per image mini and max scale to [0, 1] for SSIM
def norm_minmax(x: torch.Tensor, eps: float = 1e-6):
    x_min = x.amin(dim=(-2, -1), keepdim=True)
    x_max = x.amax(dim=(-2, -1), keepdim=True)
    return (x - x_min) / (x_max - x_min + eps)
