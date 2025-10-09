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
