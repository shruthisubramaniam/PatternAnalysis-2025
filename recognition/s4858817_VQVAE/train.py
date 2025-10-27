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
        axes[1, i].imshow(reconstructions[i, 0], cmap='gray')
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
        pbar = tqdm(train_dl, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        for batch, _ in pbar:
            batch = batch.to(device)
            optimiser.zero_grad()

            x_hat, vq_loss, _ = model(batch)

            recon_loss = F.mse_loss(x_hat, batch)
            loss = recon_loss + vq_loss

            loss.backward()
            optimiser.step()

            train_loss_epoch += loss.item()
            pbar.set_postfix(total_loss=loss.item(), recon_loss=recon_loss.item(), vq_loss=vq_loss.item())

        avg_train_loss = train_loss_epoch / len(train_dl)
        history['train_loss'].append(avg_train_loss)

        # Creating the validation loop 
        model.eval()
        val_loss_epoch = 0.0
        val_perplexity_epoch = 0.0
        ssim_metric.reset()

        pbar_val = tqdm(val_dl, desc=f"Epoch {epoch+1}/{args.epochs} [Val]")
        with torch.no_grad():
            for batch, _ in pbar_val:
                batch = batch.to(device)
                x_hat, vq_loss, perplexity = model(batch)

                recon_loss = F.mse_loss(x_hat, batch)
                loss = recon_loss + vq_loss

                val_loss_epoch += loss.item()
                val_perplexity_epoch += perplexity.item()
                ssim_metric.update(x_hat, batch)
                pbar_val.set_postfix(val_loss=loss.item(), perplexity=perplexity.item())
        
        avg_val_loss = avg_val_loss = val_loss_epoch / len(val_dl)
        avg_val_perplexity = val_perplexity_epoch / len(val_dl)
        epoch_ssim = ssim_metric.compute().item()

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
        
        if (epoch + 1) % 10 == 0:
            checkpoint_path = save_dir / f"model_epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved periodic checkpoint at epoch {epoch+1} to {checkpoint_path}")

    print("\nTraining complete.")

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

# Calling the main function 
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





