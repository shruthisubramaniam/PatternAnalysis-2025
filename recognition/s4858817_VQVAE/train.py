"""
This python file trains/evaluates a VQ-VAE to examine HipMRI Prostate Cancer.

Purpose of train.py:
- Construct train/validation/test DataLoaders (from dataset.py).
- Call VQVAE (from modules.py) and optimize with Adam.
- Keep track of losses, SSIM, and codebook perplexity; save best model by validation SSIM.
- Export plots and a CSV log; visualize reconstructions on the test set.

What the code does:
1) Metrics and plotting utilities
Creates plots such as validation SSIM plot, loss curves and reconstruction vs original plots

2) Training loop 
- Forward pass: x passes into VQVAE to then produce (x_hat, vq_loss, perplexity).
- Loss = MSE reconstruction + VQ loss; optimize with Adam
- Aggregates epoch averages and logs per-batch totals using tqdm.

3) Validation loop
Calculates validation loss, mean perplexity, and SSIM.

4) Model selection and early stopping
- Tracks best Val SSIM; saves best_model.pth when improved.
- Early stopping on SSIM with patience = 8; periodic checkpoints every 10 epochs.

5) Exporting results
- Saves metric results to training_metrics.csv and plots to the save directory.
- A test reconstruction panel is saved after training, best_model.pth is reloaded, 
and a final test loop (which returns loss and perplexity) is executed.

Shapes:
Input tensors from dataset.py - [B, 1, H, W] (float32; z-scored per slice).
Model latent - [B, z_dim, H/8, W/8]; indices: [B, H/8, W/8].

CLI:
python train.py --repo_root /path/to/repo --save_dir ./results --epochs 50 --batch_size 32

Dependencies:
PyTorch, torchmetrics (SSIM), tqdm, matplotlib, pandas, dataset.py, modules.py.

Note:
ChatGPT was used to aid in the development of this file
Prompt: "Here is my dataset.py file an my modules.py file. Based on these files please train the VQVAE 
such that we satify the criteria, "“train.py" containing the source code for training, validating, 
testing and saving your model. The model should be imported from “modules.py” and the data 
loader should be imported from “dataset.py”. Make sure to plot the losses and metrics during training"
"""
# Obtaining imports
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
import pandas as pd 

 
def plot_metrics(save_dir: Path, train_losses, val_losses, val_ssims, val_perplexities):
    """
    Creating a function that will obtain the plots and saves the training and 
    validation metrics.

    Parameters:
    save_dir: pathlib.Path
        Directory where plots will be saved.
    train_losses: list[float]
        Per-epoch average training loss (MSE + VQ loss).
    val_losses: list[float]
        Per-epoch average validation loss (MSE + VQ loss).
    val_ssims: list[float]
        Per-epoch validation SSIM scores.
    val_perplexities: list[float]
        Per-epoch average codebook perplexity.

    Returns:
    None
    Saves three PNGs into save_dir:
        - lost_plot.png (total loss curves) 
        - validation_ssim_plot.png
        - validation_perplexity_plot.png
    """
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

def reconstruction_vs_original(model, dataloader, device, save_path: Path, num_images = 8):
    """
    Creating a fucntion that will help to compare the original images 
    versus the reconstructed images

    Parameters:
    model: torch.nn.Module
        Trained VQ-VAE model; this function sets it to eval() and runs inference under no-grad.
    dataloader: torch.utils.data.DataLoader
        DataLoader providing a test (or val) split; must yield (images, paths) where
        images are float tensors of shape [B, 1, H, W].
    device: torch.device or str
        Whether using device type cuda or cpu.
    save_path : pathlib.Path
        Destination PNG path for the comparison grid.
    num_images : int
        Number of examples (columns) to visualize from the first batch.

    Returns:
    None
        Writes a PNG to save_path and prints its location.
    """
    model.eval()
    images, _ = next(iter(dataloader)) # Taking one batch 
    images = images[:num_images].to(device) # limiting to num_images columns

    # Forward without grad to get reconstructions
    with torch.no_grad():
        reconstructions, _, _ = model(images)

    # Moving to CPU numpy for matplotlib
    images = images.detach().cpu().numpy()
    reconstructions = reconstructions.detach().cpu().numpy()

    # Producing two rows where the originals are the top and the 
    # reconstructions are at the bottom
    fig, axes = plt.subplots(2, num_images, figsize = (num_images * 2, 4))
    for i in range (num_images):
        # Displaying the original images
        axes[0, i].imshow(images[i, 0], cmap='gray')
        axes[0, i].set_title("Original")
        axes[0, i].axis('off')

        # Displaying the reconstructed images 
        axes[1, i].imshow(reconstructions[i, 0], cmap='gray')
        axes[1, i].set_title("Reconstruction")
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Reconstruction examples vs originals saved to {save_path}")
    plt.close()

# Creating the main fucntion 
def main(args):
    """
    Training/validation/testing for a VQ-VAE on HipMRI slices.

    Parameters:
    args: argparse.Namespace
        Parsed CLI args:
          - repo_root (str): Repository root containing the dataset folder.
          - save_dir (str): Output directory for checkpoints, CSV, and plots.
          - epochs (int): Number of training epochs.
          - batch_size (int): Batch size for train/val.
          - lr (float): Learning rate for Adam.
          - num_workers (int): DataLoader workers.
          - model_base_channels (int): Base channels for encoder/decoder.
          - z_dim (int): Latent channel width / embedding dimension.
          - n_codes (int): Codebook size for quantizer.

    Returns:
    None
        Writes checkpoints, CSV/plots, and prints metrics. Also saves a test
        reconstruction panel
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Getting the dataloaders 
    train_dl, val_dl, test_dl = get_dataloader(repo_root=args.repo_root, batch_size=args.batch_size, num_workers=args.num_workers)

    # Initialising the model, optimiser and metrics 
    model = VQVAE(in_channels=1, base=args.model_base_channels, z_dim=args.z_dim, n_codes=args.n_codes).to(device)

    optimiser = Adam(model.parameters(), lr=args.lr)

    # z-score normalisation of the data is done in dataset.py so the data_range is not fixed to 1.0.
    # By doing the below we can estimate it from a batch of data to calculate the SSIM
    first_batch, _ = next(iter(val_dl))
    data_range = float(first_batch.max() - first_batch.min())
    print(f"Estimated data range for SSIM: {data_range:.4f}")
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=data_range).to(device)

    # Tracking the training 
    history = {'train_loss': [], 'val_loss': [], 'val_ssim': [], 'val_perplexity': []}
    best_ssim = -1.0

    # patience tracker
    patience = 8
    patience_counter = 0

    # Defining the training loop
    print("Starting training")
    for epoch in range (args.epochs):
        model.train()
        train_loss_epoch = 0.0
        # Train loop with progress bar and per-batch logging
        pbar = tqdm(train_dl, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        for batch, _ in pbar:
            batch = batch.to(device)
            optimiser.zero_grad()

            # Forward through VQVAE giving reconstructions, VQ loss
            x_hat, vq_loss, _ = model(batch)

            # Total loss = reconstruction + VQ loss
            recon_loss = F.mse_loss(x_hat, batch)
            loss = recon_loss + vq_loss

            # Backprop + update
            loss.backward()
            optimiser.step()
            
            # Accumulating loss for epoch average and show in progress bar
            train_loss_epoch += loss.item()
            pbar.set_postfix(total_loss=loss.item(), recon_loss=recon_loss.item(), vq_loss=vq_loss.item())

        # Average training loss per epoch
        avg_train_loss = train_loss_epoch / len(train_dl)
        history['train_loss'].append(avg_train_loss)

        # Creating the validation loop 
        model.eval()
        val_loss_epoch = 0.0
        val_perplexity_epoch = 0.0
        ssim_metric.reset()  # Resetting running SSIM metric per epoch

        pbar_val = tqdm(val_dl, desc=f"Epoch {epoch+1}/{args.epochs} [Val]")
        with torch.no_grad():
            for batch, _ in pbar_val:
                batch = batch.to(device)
                x_hat, vq_loss, perplexity = model(batch)

                # Validation total loss
                recon_loss = F.mse_loss(x_hat, batch)
                loss = recon_loss + vq_loss

                val_loss_epoch += loss.item()
                val_perplexity_epoch += perplexity.item()

                # Accumulating SSIM across the whole val set
                ssim_metric.update(x_hat, batch)
                pbar_val.set_postfix(val_loss=loss.item(), perplexity=perplexity.item())
        
        avg_val_loss = avg_val_loss = val_loss_epoch / len(val_dl) # Average validation loss per epoch
        avg_val_perplexity = val_perplexity_epoch / len(val_dl) # Average perplexity per epoch
        epoch_ssim = ssim_metric.compute().item() # SSIM for the epoch 
        
        # Record metrics for plotting/CSV
        history['val_loss'].append(avg_val_loss)
        history['val_ssim'].append(epoch_ssim)
        history['val_perplexity'].append(avg_val_perplexity)

        # Printing the train loss, val loss, val ssim and val perplexity
        print(f"Epoch {epoch+1}/{args.epochs} Summary: "
              f"Train Loss: {avg_train_loss:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}, "
              f"Val SSIM: {epoch_ssim:.4f}, "
              f"Val Perplexity: {avg_val_perplexity:.2f}")
        
        # Using the validation ssim to save the best model 
        if epoch_ssim > best_ssim:
            best_ssim = epoch_ssim
            model_path = save_dir / "best_model.pth"
            torch.save(model.state_dict(), model_path)
            print(f"SSIM and saved new best model: {best_ssim:.4f} at {model_path}")

            patience_counter = 0

        else:
            patience_counter += 1
            print(f"No improvement in Val SSIM for {patience_counter} epoch(s). Patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print(f"EARLY STOPPING: Stopping training after {patience} epochs with no improvement.")
                break
        # Checkpoint for every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = save_dir / f"model_epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved periodic checkpoint at epoch {epoch+1} to {checkpoint_path}")

    print("\nTraining complete.")

    # Saving metrics to CSV
    history_df = pd.DataFrame(history)
    history_df.insert(0, 'epoch', range(1, len(history_df) + 1))
    csv_path = save_dir / "training_metrics.csv"
    history_df.to_csv(csv_path, index=False)
    print(f"Training metrics saved to {csv_path}")

    # Calling the plots
    plot_metrics(
        save_dir, 
        history['train_loss'], 
        history['val_loss'], 
        history['val_ssim'],
        history['val_perplexity']
    )

    # loading the best model for final evaluation so that it can be used in testing 
    print("Loading best model for testing and visualization")
    model.load_state_dict(torch.load(save_dir / "best_model.pth"))

    # Visualising some reconstructions on test data 
    reconstruction_vs_original(model, test_dl, device, save_dir / "test_reconstructions.png")

    # Creating the testing loop 
    model.eval()
    test_loss = 0.0
    test_perplexity = 0.0
    ssim_metric.reset()
    pbar_test = tqdm(test_dl, desc="[Test]")
    with torch.no_grad():
        for batch, _ in pbar_test:
            batch = batch.to(device)
            x_hat, vq_loss, perplexity = model(batch)
            recon_loss = F.mse_loss(x_hat, batch)
            loss = recon_loss + vq_loss
            test_loss += loss.item()
            test_perplexity += perplexity.item()
            ssim_metric.update(x_hat, batch)
    
    avg_test_loss = test_loss / len(test_dl)
    avg_test_perplexity = test_perplexity / len(test_dl)
    final_test_ssim = ssim_metric.compute().item()

    print("\n--- Final Test Results (using best model) ---")
    print(f"Test Loss: {avg_test_loss:.4f}")
    print(f"Test Perplexity: {avg_test_perplexity:.2f}")

# CLI entry point and args 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a VQ-VAE model on 2D MRI slices.")
    parser.add_argument("--repo_root", type=str, default=".", help="Root directory of the repository containing the 'dataset' folder.")
    parser.add_argument("--save_dir", type=str, default="./results", help="Directory to save model checkpoints and plots.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training and validation.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate for the Adam optimizer.")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of workers for the DataLoader.")
    parser.add_argument("--model_base_channels", type=int, default=64, help="Number of base channels in the VQ-VAE encoder/decoder.")
    parser.add_argument("--z_dim", type=int, default=128, help="Dimension of the latent embeddings.")
    parser.add_argument("--n_codes", type=int, default=512, help="Number of codes in the VQ codebook.")

    args = parser.parse_args()
    main(args)





