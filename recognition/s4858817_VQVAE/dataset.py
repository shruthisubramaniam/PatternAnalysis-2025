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