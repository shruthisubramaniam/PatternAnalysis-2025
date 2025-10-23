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

    # Plotting the validation perplexity 
    plt.figure(figsize = (10,5))
    plt.plot(epochs, val_perplexities, label='Validation perplexity', color = 'green')
    plt.title('Validation perplexity')
    plt.xlabel('Epochs')
    plt.ylabel('Perplexity')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_dir / "validation_perplexity_plot.png")
    plt.close()

    print (f"Plots saved to {save_dir}")

# Creating a fucntion that will help to compare the original images versus the reconstructed images
def reconstruction_vs_original(model, dataloader, device, save_path: Path, num_images = 8):
    model.eval()
    images, _ = next(iter(dataloader))
    images = images[:num_images].to(device)

    with torch.no_grad():
        reconstructions, _, _ = model(images)

    images = images.cpu().numpy()

    fig, axes = plt.subplots(2, num_images, figsize = (num_images * 2, 4))
    for i in range (num_images):
        # Displaying the original images
        axes[0, i].imshow(images[i, 0], cmap='gray')
        axes[0, i].set_title("Original")
        axes[0, i].axis('off')

        # Displaying the reconstructed images 
        axes[1, i].imshow(reconstructions[i, 0], cmpa='gray')
        axes[1, i].set_title("Reconstruction")
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Reconstruction examples vs originals saved to {save_path}")
    plt.close()

# Creating the main fucntion 
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Getting the dataloaders 
    train_dl, val_dl, test_dl = get_dataloader(repo_root=args.repo_root, batch_size=args.batch_size, num_workers=args.num_workers)

    # Initialising the model, optimiser and metrics 
    model = VQVAE(in_channels=1, base=args.model_base_channels, z_dim=args.z_dim, n_codes=args.n_code).to(device)

    optimiser = Adam(model.parameters(), lr=args.lr)

    