# Command line to run predict.py below 
# predict.py --model_path /Users/shruthisubramaniam/Desktop/best_model.pth

import argparse
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchmetrics.image import StructuralSimilarityIndexMeasure