from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple, List

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset

IMG_EXTS = (".nii", ".nii.gz", ".npy") # Defining how the images exist

# Directory that splits names for the subfolders 
SPLIT2SUBDIR = {
    "train": "keras_slices_train",
    "val":   "keras_slices_val",
    "test":  "keras_slices_test",
}

# Finding all files for a split directory. 
def _discover(split_dir: Path) -> List[Path]:
    files: List[Path] = []
    for ext in IMG_EXTS:
        files += list(split_dir.glob(f"*{ext}"))
        files.sort()
    if not files:
        files.sort()
        return files

# For training it is important to have every sample in the training 
# set to be consistent. 
# Every sample should be in a single 2D array (H x W) of float32 because right now
# files are stores as .nii/.nii.gz. 
# So the function below accounts for this. It also cleans the data in a way by 
# making sure that the smples are not in this format (H x W x 1) or even full 3D volumes (H x W x D). 


class SliceDataset(Dataset):

    def __init__(self, repo_root: str | Path, split: str = "train", normalization: str = "zscore"):
        assert split in SPLIT2SUBDIR, f"split must be one of {list(SPLIT2SUBDIR)}"
        self.root = Path(repo_root) / "dataset" / SPLIT2SUBDIR[split]
        self.files = _discover(self.root)
        self.normalization = normalization
    
    



def _load_slice(p: Path) -> np.ndarray:
    try:
        if p.suffix == ".npy":
            arr = np.squeeze(np.load(p)).astype(np.float32)
            if arr.ndim > 2:
                arr = arr[..., 0]
            return arr
        nimg = nib.load(str(p))
        arr = np.squeeze(np.asarray(nimg.get_fdata(), dtype=np.float32))
        if arr.ndim == 2:
            return arr
        if arr.ndim == 3:
            mid = arr.shape[2] // 2
            return arr[..., mid].astype(np.float32)
        flat = arr.reshape(arr.shape[0], -1) if arr.ndim >= 2 else np.atleast_2d(arr)
        return flat.astype(np.float32)
    except Exception:
        return np.zeros((1, 1), dtype=np.float32)
    
# Function that does z-score normalisation so that it recenters each slice to 
# a mean of 0 and a variance of 1 so that the intensities are on a comparable scale. 
def _zscore(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    m, s = float(x.mean()), float(x.std())
    return (x - m) / (s + eps) if s > 0 else x - m