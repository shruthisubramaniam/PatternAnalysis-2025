"""
This python file is the dataset utilities for building a VQVAE to examine
HipMRI Prostate Cancer 

Project Objective:
Develop a generative VQVAE or VQVAE2 model for the HipMRI Study on Prostate Cancer 
using processed 2D slices. 

Goal:
Produce "reasonably clear images" and a structured Similarity (SSIM) of over 0.6

Purpose of dataset.py:
- Discover, load, optionally z-score, and batch variable-sized 2D slices
from NIfTI (.nii/.nii.gz) or NumPy (.npy). 
- 3D volumes and takes middle slices.
- Batching uses right/bottom padding so shapes align.

What this code does:
1) Discover per-split files:
List *.nii/*.nii.gz/*.npy; sort sequentially; map split -> subfolder. This is so that 
there will be a clean separation of train, val, and test, as well as reproducible ordering.

2) Use the canonical orientation while loading NIfTI/NumPy.
`nib.load` + `nib.as_closest_canonical` for NIfTI; `np.load` for.npy. This is because
different axis orders may be used to store medical pictures; canonicalizing prevents unintentional
flips and rotations.

3) Select the centre slice or squeeze channels to enforce 2D.
Squeeze if HxWx1, remove the central slice if HxWxD, and leave as is if HxW. This is because,  
in order to maintain sample comparability, the model is trained on 2D slices.

4) Conduct z-score normalisation 
(x − mean) / (std + 1e-6); if std==0, just center. This is so that the 
learning and optimisation of the VQ-VAE codebook are stabilised by standardised inputs (mean≈0, std≈1).

5) Return tensors as [1, H, W]
Start by adding a channel dimension (grayscale=1). This is so that the images 
comply to downstream convolution layers and standard PyTorch (N, C, H, and W) norms.

6) Batch mixed sizes with padding at the top and bottom to [B, 1, Hmax, Wmax].
Calculate the batch's maximum H/W in the collate fn and pad the others on the right and bottom.
This is to allow batching while maintaining the original pixel coordinates (top-left anchored) and avoiding 
scaling or blurring thin structures.

7) Safe fallback on load errors
Return a 1x1 zero array rather than raising on any I/O/format error. This is to 
stop a single faulty file from causing training crashes; you may log or filter afterward.

Expected layout (relative to `repo_root`):
repo_root/
  └─ dataset/
     ├─ keras_slices_train/
     ├─ keras_slices_validate/
     └─ keras_slices_test/      # files: *.nii | *.nii.gz | *.npy

Shapes:
Sample tensor - [1, H, W] (float32)
Batch tensor -  [B, 1, Hmax, Wmax] (right/bottom padded within batch)

CLI:
python dataset.py --repo_root /path/to/repo 
This is to see number of samples in each set

Dependencies:
Python 3.10, numpy, nibabel, torch, matplotlib

Note:
ChatGPT was used to aid in the development of this file
Prompt: "My dataset is the HipMRI processed 2D slices found on my desktop. 
Please develop code that will contain the data loader for loading and preprocessing the data"
"""
# Obtaining imports  
from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple, List

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

# Defining how the images exist
IMG_EXTS = (".nii", ".nii.gz", ".npy") 

# Directory that splits names for the subfolders 
SPLIT2SUBDIR = {
    "train": "keras_slices_train",
    "val":   "keras_slices_validate",
    "test":  "keras_slices_test",
}

# Finding all files for a split directory. 
def _discover(split_dir: Path) -> List[Path]:
    """
    Under split_dir, return a sorted list of files with permitted extensions.
    Deterministic ordering between runs is guaranteed via sorting.
    """
    files: List[Path] = []
    for ext in IMG_EXTS:
        files += list(split_dir.glob(f"*{ext}"))
    files.sort()
    return files

# For training it is important to have every sample in the training set to be consistent. 
# Every sample should be in a single 2D array (H x W) of float32 because right now
# files are stores as .nii/.nii.gz. 

# Function that does z-score normalisation so that it recenters each slice to 
# a mean of 0 and a variance of 1 so that the intensities are on a comparable scale. 
def _zscore(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Per-slice z-score normalization: (x - mean) / (std + eps).
    If std == 0, we just mean-center to avoid division by zero.
    """
    m, s = float(x.mean()), float(x.std())
    return (x - m) / (s + eps) if s > 0 else x - m

def _load_slice(p: Path) -> np.ndarray:
    """
    Handle both.npy and.nii/.nii.gz files by loading a single 2D slice as float32. -.npy: 
    squeeze singleton dims; if still >2D, retain the first channel.
    """
    try:
        if p.suffix == ".npy":
            arr = np.squeeze(np.load(p)).astype(np.float32)
            if arr.ndim > 2:
                # If shape like (H,W,C), take first channel to enforce 2D.
                arr = arr[..., 0]
            return arr
        # NIfTI path: load and canonicalize orientation for consistency.
        nimg = nib.load(str(p))
        nimg = nib.as_closest_canonical(nimg)
        arr = np.squeeze(np.asarray(nimg.get_fdata(), dtype=np.float32))
        if arr.ndim == 2:
            # Already a single slice (H,W)
            return arr
        if arr.ndim == 3:
            # (H,W,D): take the middle slice along depth
            mid = arr.shape[2] // 2
            return arr[..., mid].astype(np.float32)
        # flatten to a 2D view so downstream code still works.
        flat = arr.reshape(arr.shape[0], -1) if arr.ndim >= 2 else np.atleast_2d(arr)
        return flat.astype(np.float32)
    except Exception:
        # Safety fall back just in case a single bad file does not cause a full crash 
        return np.zeros((1, 1), dtype=np.float32)
    
def pad_collate(batch):
    """
    Use a custom collate method to aggregate slices of varying sizes.
    """
    # batch: list[(tensor[1,H,W], path)]
    xs, paths = zip(*batch)
    # Compute target batch size (max height and width across samples)
    Hmax = max(x.shape[-2] for x in xs)
    Wmax = max(x.shape[-1] for x in xs)

    padded = []
    for x in xs:
        dh = Hmax - x.shape[-2]
        dw = Wmax - x.shape[-1]
        # Pad (left, right, top, bottom). 
        padded.append(F.pad(x, (0, dw, 0, dh)))
    return torch.stack(padded, 0), list(paths)

class SliceDataset(Dataset):
    """
    PyTorch 2D slice dataset.
    Points to the dataset at repo_root/\split_subdir>
    Valid files are found and loaded as 2D float32 
    Returns path_string and tensor[1,H,W].
    """

    def __init__(self, repo_root: str | Path, split: str = "train", normalization: str = "zscore"):
        assert split in SPLIT2SUBDIR, f"split must be one of {list(SPLIT2SUBDIR)}"
        self.root = Path(repo_root) / "dataset" / SPLIT2SUBDIR[split]
        self.files = _discover(self.root)
        self.normalization = normalization
    
    def __len__(self) -> int:
        # Number of discovered files in this split
        return len(self.files)

    def __getitem__(self, idx: int):
        # Loading array as HxW float32
        p = self.files[idx]
        arr = _load_slice(p) # np.ndarray HxW float32
        if self.normalization == "zscore":
            arr = _zscore(arr)
        # Converting to tensor with channel dimension first: [1, H, W]
        x = torch.from_numpy(arr[None, ...]) # float32
        return x, str(p)

def get_dataloader(
    repo_root: str | Path,
    batch_size: int = 16,
    num_workers: int = 0,
    normalization: str = "zscore",
): 
    """
    Building DataLoaders for train/val/test splits
    """
    train_ds = SliceDataset(repo_root, "train", normalization)
    val_ds   = SliceDataset(repo_root, "val",   normalization)
    test_ds  = SliceDataset(repo_root, "test",  normalization)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                      num_workers=num_workers, pin_memory=True,
                      collate_fn=pad_collate)

    val_dl   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=True,
                      collate_fn=pad_collate)

    test_dl  = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=True,
                      collate_fn=pad_collate)
    return train_dl, val_dl, test_dl

# Calling all functions under main 
if __name__ == "__main__":
    import argparse
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_root", type=str, default=".", help="Repo root containing /dataset")
    ap.add_argument("--split", choices=["train", "val", "test"], default="train")
    args = ap.parse_args()

    ds = SliceDataset(args.repo_root, args.split)
    print(f"{args.split} samples:", len(ds))
    x, p = ds[0]
    print("first sample shape:", tuple(x.shape), "path:", p)

    plt.imshow(x[0].numpy(), cmap="gray")
    plt.title(p)
    plt.axis("off")
    plt.show()