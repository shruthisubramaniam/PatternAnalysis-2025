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

    



