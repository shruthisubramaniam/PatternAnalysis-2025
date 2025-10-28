"""
The model modules (VQ-VAE) for analysing HipMRI Prostate Cancer are 
included in this Python file.

Purpose of modules.py:
- Define CNN building blocks (Residual, conv_block).
- Implement vector quantization (codebook) with a straight-through estimator.
- Provide an Encoder/Decoder backbone with x8 down/upsampling.
- Create VQVAE model that returns reconstruction, VQ loss, and perplexity.

What this code does:
1) Residual block:
ReLU to 3x3 Conv to ReLU to 1x1 Conv is the pre-activation residual unit. 
Next, a skip connection is added. This is to promote steady training with few parameters.

2) conv_block
Conv2d to BatchNorm2d to ReLU is the convenience layer for fast feature extraction.

3) VectorQuantizer
- Calculates L2 distances to a KxD codebook and flattens latents.
- Each latent vector is assigned to the closest code (argmin).
- To ensure that gradients pass to the encoder, the straight-through technique is used.
- VQ loss is calculated by adding codebook loss and β * commitment loss.

4) Encoder (downsampling x8)
The input is compressed to z_e ∈ R^{BxDxH/8xW/8} via the stacked conv/BN/ReLU and
stride-2 stages, and then it is projected 1x1 to z_dim.

5) Decoder (upsample x8)
After restoring spatial size by a 1x1 projection from z_dim to deconvolution 
(ConvTranspose2d) steps, a final 1x1 Conv is applied to the output channels.

6) VQVAE Wrapper 
- x to encoder to quantiser to decoder is the forward pass.
- Reconstructed x_hat, vq_loss, and perplexity are the returns.

Shapes:
Input/Output  -  [B, 1, H, W]
Latent - [B, z_dim, H/8, W/8]
Code indices - [B, H/8, W/8]

Dependencies:
PyTorch (torch, torch.nn, torch.nn.functional)

Note:
ChatGPT was used to aid in the development of this file
Prompt: "Here is my dataset.py file. Based on this please build PyTorch modules 
for a VQ-VAE suitable for 2D HipMRI slices (encoder/decoder, vector quantizer 
with straight-through estimator) that returns reconstruction, VQ loss, and perplexity."
"""

# Obtaining imports 
from __future__ import annotations
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

class Residual(nn.Module):
    """
    Pre-activation residual block. Skip (x + block(x)) is added after ReLU to 3x3 Conv to 
    ReLU to 1x1 Conv.

    Parameters:
    ch: int
        Number of input/output channels.
    """
    def __init__(self, ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 1), # 1x1 for channel mixing
        )
    
    def forward(self, x): 
        """
        Forward pass where # Residual connection where 
        output = input + transformed(input)

        Parameters:
        x: torch.Tensor
            Input feature map of shape [B, ch, H, W].
        
        Returns:
        torch.Tensor
            Output feature map of shape [B, ch, H, W] with residual skip added.
        """
        return x + self.block(x)

def conv_block(ci: int, co: int, ks: int = 3, stride: int = 1, pad: int = 1):
    """Convenience conv block where it is BatchNorm2d to ReLU to Conv2d.

    Returns:
    ci: int
        Number of input channels.
    co: int
        Number of output channels.
    ks: int
        Kernel size.
    stride: int
        Convolution stride.
    pad: int
        Zero padding.
    
    Returns:
    nn.Sequential
        A sequence: Conv2d(ci to co, ks, stride, pad) to BN(co) to ReLU.
    """
    return nn.Sequential(
        nn.Conv2d(ci, co, ks, stride=stride, padding=pad),
        nn.BatchNorm2d(co),
        nn.ReLU(inplace=True),
    )

class VectorQuantizer(nn.Module):
    """
    Codebook-based vector quantizer with a straight-through estimator

    Parameters:
    num_embeddings (K): codebook size
    embedding_dim (D): code vector dimension
    beta: commitment loss weight
    
    Forward:
    z_e: (B, D, H, W) continuous latents

    Returns:
    z_q - (B, D, H, W) quantized latents 
    vq_loss 
    perplexity
    indices - (B, H, W) integer code indices
    """
    def __init__(self, num_embeddings: int = 512, embedding_dim: int = 128, beta: float = 0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.beta = beta
        # codebook matrix E ∈ R^{K×D}
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        # Uniform init to help avoid early code collapse
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

    def forward(self, z_e: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, D, H, W = z_e.shape
        # Move channels to last for vector distance computation: (B,H,W,D)
        z = z_e.permute(0, 2, 3, 1).contiguous() # (B, H, W, D)
        flat = z.view(-1, D) # (BHW, D)
        emb = self.embedding.weight # (K, D)

        # L2 sqaured distances calculated between the embeddings and the latent
        dist = (
            flat.pow(2).sum(dim=1, keepdim=True) # (BHW, 1)
            - 2 * flat @ emb.t()
            + emb.pow(2).sum(dim=1) 
        ) # (BHW, K)
        # nearest code index per vector
        indices = torch.argmin(dist, dim=1) # (BHW, )
        z_q = self.embedding(indices).view(B, H, W, D).permute(0, 3, 1, 2)

        # Obtaining the loss
        loss_code = F.mse_loss(z_q, z_e.detach())
        loss_commit = F.mse_loss(z_e, z_q.detach())
        vq_loss = loss_code + self.beta * loss_commit

        # Estimator 
        # Forward uses z_q, backward uses gradients of z_e
        z_q = z_e + (z_q - z_e).detach()

        onehot = F.one_hot(indices, self.num_embeddings).float() # (BWH, K)
        avg_probs = onehot.mean(dim=0) # (K, )
        perplexity = torch.exp(-(avg_probs * (avg_probs + 1e-10).log()).sum())
        return z_q, vq_loss, perplexity, indices.view(B, H, W)
 
class Encoder(nn.Module):
    """
    Encoder that projects to z_dim channels after downsampling spatially by x8.
    
    Structure:
    base = 64
    - conv_block(in_ch, base)
    - conv_block(base, base)
    - conv_block(base, base, stride=2)
    - The residual (base) - downsampled by 2
    - The function conv_block(base, base*2)
    - conv_block(base*2, stride=2, base*2)  - downsampled by 4
    - Residual(base*2)
    - conv_block(base*2, base*4)
    - conv_block(base*4, base*4, stride=2) - downsampled by 8
    - Residual(base*4)
    - 1x1 Conv to z_dim

    Parameters:
    in_ch: int
        Number of input channels (1 for grayscale).
    base: int
        Base channel width multiplier.
    z_dim: int
        Latent channel width/embedding dimension.
    
    Forward:
    x: torch.Tensor
        Input tensor [B, in_ch, H, W].
    
    Returns:
    torch.Tensor
        Latent tensor z_e of shape [B, z_dim, H/8, W/8].
    """
    def __init__(self, in_ch: int = 1, base: int = 64, z_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            conv_block(in_ch, base),
            conv_block(base, base),
            conv_block(base, base, stride=2), # spatial /2
            Residual(base),
            conv_block(base, base*2),
            conv_block(base*2, base*2, stride=2), # spatial /4
            Residual(base*2),
            conv_block(base*2, base*4),
            conv_block(base*4, base*4, stride=2), # spatial /8
            Residual(base*4),
            nn.Conv2d(base*4, z_dim, 1),
        )
    
    def forward(self, x): 
        return self.net(x)
            
class Decoder(nn.Module):
    """
    A decoder that returns to the input resolution after upsampling spatially by x8.

    Structure:
    - 1x1 projection: z_dim to base*4
    - Residual(base*4)
    - ConvTranspose2d: base*4 to base*2 (stride=2)
    - Residual(base*2)
    - ConvTranspose2d: base*2 to base (stride=2)
    - Residual(base)
    - ConvTranspose2d: base -> base (stride=2) 
    - ReLU
    - 1x1 Conv: base to out_ch

    Parameters:
    out_ch: int
        Number of output channels (1 for grayscale)
    base: int
        Base channel width multiplier.
    z_dim: int
        Latent channel width/embedding dimension.
    
    Forward:
    z: torch.Tensor
        Quantized latent tensor [B, z_dim, H/8, W/8].
    
    Returns:
    torch.Tensor
        Reconstruction x̂ of shape [B, out_ch, H, W].
    """
    def __init__(self, out_ch: int = 1, base: int = 64, z_dim: int = 128):
        super().__init__()
        self.in_proj = nn.Conv2d(z_dim, base*4, 1)
        self.net = nn.Sequential(
            Residual(base*4),
            nn.ConvTranspose2d(base*4, base*2, 4, stride=2, padding=1), # x2
            Residual(base*2),
            nn.ConvTranspose2d(base*2, base, 4, stride=2, padding=1), # x4
            Residual(base),
            nn.ConvTranspose2d(base, base, 4, stride=2, padding=1), # x8
            nn.ReLU(inplace=True),
            nn.Conv2d(base, out_ch, 1), # Final projection
        )

    def forward(self, z):
        z = self.in_proj(z)
        return self.net(z)

class VQVAE(nn.Module):
    """
    A class that builds the VQVAE

    Parameters:
    in_channels: int
        input channels (1 for grayscale HipMRI slices)
    base: int
        base channel width (encoder/decoder)
    z_dim: int 
        Latent channel width
    n_codes: int
        Codebook size K
    beta: float 
        commitment weight for VQ loss 

    Forward:
    x: torch.Tensor
        Input tensor [B, in_channels, H, W].
    
    Returns:
    x_hat: torch.Tensor
        reconstructed image (B, in_channels, H, W)
    vq_loss: torch.Tensor
        Scalar VQ loss. 
    perplexity: torch.Tensor
        code usage perplexity
    """
    def __init__(self, in_channels: int = 1, base: int = 64, z_dim: int = 128, n_codes: int = 512, beta: float = 0.25):
        super().__init__()
        self.encoder = Encoder(in_ch=in_channels, base=base, z_dim=z_dim)
        self.quantizer = VectorQuantizer(num_embeddings=n_codes, embedding_dim=z_dim, beta=beta)
        self.decoder = Decoder(out_ch=in_channels, base=base, z_dim=z_dim)
    
    def forward(self, x):
        # x: (B, C=1, H, W)
        z_e = self.encoder(x) # (B, z_dim, H/8, W/8)
        z_q, vq_loss, perplexity, _ = self.quantizer(z_e)
        x_hat = self.decoder(z_q) # (B, 1, H, W)
        return x_hat, vq_loss, perplexity

