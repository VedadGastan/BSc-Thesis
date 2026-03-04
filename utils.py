import random
import numpy as np
import torch

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def binarize_hash(x):
    """Converts continuous network output (-1 to 1) to binary bits (0 or 1)."""
    return (x > 0).float()

def calculate_hamming_distance(hash1, hash2):
    """Calculates the number of differing bits between two hashes."""
    return (hash1 != hash2).sum(dim=1)