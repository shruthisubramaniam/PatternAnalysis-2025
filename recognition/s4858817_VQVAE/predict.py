# Command line to run predict.py below 
# predict.py --model_path /Users/shruthisubramaniam/Desktop/best_model.pth

import argparse
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchmetrics.image import StructuralSimilarityIndexMeasure

from modules import VQVAE
from dataset import SliceDataset 

def reconstruction_function(model, dataloader, device, output_dir, num_images = 8):
    print("Running Reconstruction Demonstration")
    model.eval()
    # Obtaining from the test set one batch of data 
    images, paths = next(iter(dataloader))
    images = images[:num_images].to(device)
    with torch.no_grad():
        reconstructions, _, _ = model(images)
    
    # Obtaining the SSIM for the batch 
    data_range = float(images.max() - images.min())
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=data_range).to(device)
    batch_ssim = ssim_metric(reconstructions, images).item()
    print(f"SSIM for batch of {num_images} images: {batch_ssim:.4f}")

    # Pulling images for plotting to CPU or Numpy 
    images_np = images.cpu().numpy()
    reconstructions_np = reconstructions.cpu().numpy()

    # Plots
    fig, axes = plt.subplots(2, num_images, figsize=(num_images * 2, 4.5))
    fig.suptitle(f'Reconstruction Examples (SSIM: {batch_ssim:.4f})', fontsize=16)

    for i in range(num_images):
        axes[0, i].imshow(images_np[i, 0], cmap='gray')
        axes[0, i].set_title(f"Original {i+1}")
        axes[0, i].axis('off')

        axes[1, i].imshow(reconstructions_np[i, 0], cmap='gray')
        axes[1, i].set_title(f"Recon {i+1}")
        axes[1, i].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_path = output_dir / "prediction_reconstruction_examples.png"
    plt.savefig(save_path)
    print(f"Reconstructions saved to: {save_path}")
    plt.close()


def generation_function(model, dataloader, device, output_dir, num_images=8):
    print("Running Generative Demonstration")
    model.eval()

    # Obtaining latent space shape by encoding an image that is real
    sample_image, _ = next(iter(dataloader))
    sample_image = sample_image[:1].to(device)
    with torch.no_grad():
        z_e = model.encoder(sample_image)
        _, _, _, latent_indices = model.quantizer(z_e)
    
    B, W, H = latent_indices.shape
    latent_height, latent_width = H, W
    print(f"Latent space grid size: {latent_height}x{latent_width}")




