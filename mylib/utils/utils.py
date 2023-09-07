import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from utils import torch_utils

def init_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch_utils.init_seeds(seed=seed)