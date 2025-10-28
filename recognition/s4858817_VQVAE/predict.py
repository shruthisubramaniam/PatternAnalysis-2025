"""
This Python program uses a trained VQ-VAE to do reconstruction inference 
on 2D slices of hipMRI prostate cancer.

Purpose of predict.py
- Load the trained VQVAE at the best checkpoint.
- Draw a small batch from the test split and reconstruct it.
- Compute a single batch SSIM.
- Save a grid image comparing originals versus reconstructions.

What this code does:
1) Reconstruction function 
Gets one test batch (up to num_examples), runs model forward,
computes SSIM for the batch, and saves a 2xN PNG grid.

2) Main function 
It loads the checkpoint weights, creates the test loader, executes the 
reconstruction function, parses CLI arguments, and builds the model using 
training hyperparameters.

Shapes:
Input/Output tensors are [B, 1, H, W] (float32)

CLI:
python predict.py --model_path /path/to/best_model.pth --input_data_dir 
/path/to/repo --num_examples 8

Dependencies:
PyTorch, torchmetrics (SSIM), matplotlib, modules.py (VQVAE), 
dataset.py (SliceDataset)

Note:
ChatGPT was used to aid in the development of this file
Prompt: "Here is my dataset.py, modules.py and train.py, 
based on this can you help me reconstruct images based on the test set"
"""

import argparse
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchmetrics.image import StructuralSimilarityIndexMeasure

from modules import VQVAE
from dataset import SliceDataset 

def reconstruction_function(model, dataloader, device, output_dir, num_images = 8):
    """
    Reconstruct a small batch from the test set. Takes the first batch from `dataloader`, reconstructs up to `num_images`
    images with a trained VQ-VAE, computes a single SSIM score for that subset,
    and saves a 2xN grid comparing originals (top) vs reconstructions (bottom).

    Parameters:
    model (torch.nn.Module):
        Trained VQ-VAE model. This function switches it to eval mode and runs inference
        under `torch.no_grad()`.
    dataloader (torch.utils.data.DataLoader):
        DataLoader for the **test** split. Must yield `(images, paths)` where `images`
        is a float tensor of shape `[B, 1, H, W]` (z-scored as in dataset.py).
    device (torch.device or str):
        Device for inference, e.g. `'cuda'` or `'cpu'`.
    output_dir (str or pathlib.Path):
        Directory to write the output PNG (`prediction_reconstruction_examples.png`).
    num_images (int, optional):
        Number of examples (columns) to include from the first batch. Defaults to 8.

    Returns:
    None
        Saves `prediction_reconstruction_examples.png` in `output_dir`, and prints a
        single batch SSIM computed using the batch’s dynamic range
        (`images.max() - images.min()`).

    Raises:
    StopIteration:
        If `dataloader` is empty (no batches available).

    """
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

def main(args):
    """
    This function will load a trained VQ-VAE, reconstruct a small test batch,
    compute SSIM for that subset, and save a comparison grid.

        Parameters:
        args (argparse.Namespace):
            Parsed CLI arguments with the following fields:
              - model_path (str): Path to the trained model file.
              - input_data_dir (str): Path to the repository root containing the dataset folder.
              - output_dir (str): Directory to save the output PNG grid.
              - num_examples (int): Number of images to reconstruct and plot.
              - device (str): Requested device string, 'cuda' or 'cpu'. If 'cuda'
                is requested but not available, CPU is used automatically.
              - model_base_channels (int): Base channel width used for encoder/decoder.
              - z_dim (int): Latent channel width/embedding dimension.
              - n_codes (int): Size of the VQ codebook.
        
        Returns:
        None
            Loads weights, builds a test DataLoader, runs reconstruction_function, and exits.
    """

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"Error: Model file not found at {model_path}")
        return
    
    # Producing the model with the same training hyper-parameters 
    model = VQVAE(in_channels=1,base=args.model_base_channels,z_dim=args.z_dim,n_codes=args.n_codes).to(device)

    # Weights (trained) to load 
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Model loaded successfully from {model_path}")

    # Dataset/loader (Test)
    test_ds = SliceDataset(args.input_data_dir, split="test")
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=args.num_examples, shuffle=True)

    # Calling the reconstruction function 
    reconstruction_function(model, test_dl, device, output_dir, num_images=args.num_examples)
    print("Predictions finished")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Use a trained VQ-VAE for reconstruction and generation.")

    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model (.pth file).")
    parser.add_argument("--input_data_dir", type=str, default=".", help="Path to the root of the dataset repository.")
    parser.add_argument("--output_dir", type=str, default="./predictions", help="Directory to save the output visualizations.")

    parser.add_argument("--num_examples", type=int, default=8, help="Number of images to reconstruct/generate.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use ('cuda' or 'cpu').")

    parser.add_argument("--model_base_channels", type=int, default=64, help="Number of base channels in the VQ-VAE.")
    parser.add_argument("--z_dim", type=int, default=128, help="Dimension of the latent embeddings.")
    parser.add_argument("--n_codes", type=int, default=512, help="Number of codes in the VQ codebook.")

    args = parser.parse_args()
    main(args)



