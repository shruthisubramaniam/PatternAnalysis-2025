from __future__ import annotations
from pathlib import Path
import argparse
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.optim import Adam
from tqdm import tqdm
from torchmetrics.image import StructuralSimilarityIndexMeasure
from dataset import get_dataloader
from modules import VQVAE

# Creating a function that will obtain the plots and saves the training and 
# validation metrics. 

def plot_metrics(save_dir: Path, train_losses, val_losses, val_ssims, val_perplexities):
    epochs = range(1, len(train_losses) + 1)

    # Plotting the total loss 
    plt.figure(figsize = (10,5))
    plt.plot(epochs, train_losses, label='Total loss for training')
    plt.plot(epochs, val_losses, label='Total loss for validation')
    plt.title('Loss for training and validation')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_dir / "lost_plot.png")

    # Plotting the validation SSIM
    plt.figure(figsize = (10,5))
    plt.plot(epochs, val_ssims, label='Validation SSIM', color = 'red')
    plt.title('Validation SSIM')
    plt.xlabel('Epochs')
    plt.ylabel('SSIM Scores')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_dir / "validation_ssim_plot.png")
    plt.close()

    

    