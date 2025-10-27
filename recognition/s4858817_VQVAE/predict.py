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

    # Producing a batch of random latent indices
    num_codes = model.quantizer.num_embeddings
    random_indices = torch.randint(low=0, high=num_codes, 
                                   size=(num_images, latent_height, latent_width),
                                   device=device)
    with torch.no_grad():
        z_q = model.quantizer.embedding(random_indices) # Shape: (B, H, W, D)
        z_q = z_q.permute(0, 3, 1, 2) # Reshape to (B, D, H, W) for the decoder
        # To get the generative images I am decoding the random latent codes 
        generated_images = model.decoder(z_q) 

    # Pulling images for plotting to CPU or Numpy 
    generated_images_np = generated_images.cpu().numpy()

    # Plots
    fig, axes = plt.subplots(1, num_images, figsize=(num_images * 2, 2.5))
    fig.suptitle('Generative Examples from Random Latent Codes', fontsize=16)
    for i in range(num_images):
        axes[i].imshow(generated_images_np[i, 0], cmap='gray')
        axes[i].set_title(f"Generated {i+1}")
        axes[i].axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_path = output_dir / "prediction_generative_examples.png"
    plt.savefig(save_path)
    print(f"Generations saved to: {save_path}")
    plt.close()

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    output_dir = Path(args.model_path)
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

    test_ds = SliceDataset(args.input_data_dir, split="test")
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=args.num_examples, shuffle=True)

    # Calling the two functions 
    reconstruction_function(model, test_dl, device, output_dir, num_images=args.num_examples)
    generation_function(model, test_dl, device, output_dir, num_images=args.num_examples)

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



