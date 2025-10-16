from __future__ import annotations
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

class Residual(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 1),
        )
    
    def forward(self, x): return x + self.block(x)

def conv_block(ci: int, co: int, ks: int = 3, stride: int = 1, pad: int = 1):
    return nn.Sequential(
        nn.Conv2d(ci, co, ks, stride=stride, padding=pad),
        nn.BatchNorm2d(co),
        nn.ReLU(inplace=True),
    )

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings: int = 512, embedding_dim: int = 128, beta: float = 0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.beta = beta
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

def forward(self, z_e: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    B, D, H, W = z_e.shape
    z = z_e.permute(0, 2, 3, 1).contiguous() # (B, H, W, D)
    flat = z.view(-1, D) # (BHW, D)
    emb = self.embedding.weight # (K, D)

    # L2 sqaured distances calculated between the embeddings and the latent
    dist = (
        flat.pow(2).sum(dim=1, keepdim=True)
        - 2 * flat @ emb.t()
        + emb.pow(2).sum(dim=1)
    ) # (BHW, K)
    indices = torch.argmin(dist, dim=1) # (BHW, )
    z_q = self.embedding(indices).view(B, H, W, D).permute(0, 3, 1, 2)

    # Obtaining the loss
    loss_code = F.mse_loss(z_q, z_e.detach())
    loss_commit = F.mse_loss(z_e, z_q.detach())
    vq_loss = loss_code + self.beta * loss_commit

    # Estimator 
    z_q = z_e + (z_q - z_e).detach()

    onehot = F.one_hot(indices, self.num_embeddings).float()
    avg_probs = onehot.mean(dim=0)
    perplexity = torch.exp(-(avg_probs * (avg_probs + 1e-10).log()).sum())
    return z_q, vq_loss, perplexity, indices.view(B, H, W)
 

