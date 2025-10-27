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
        reconstruction, _, _ = model(images)

    



