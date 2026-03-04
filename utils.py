import random
import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def binarize_hash(x):
    #Converts continuous Tanh output to binary bits: positive → 1, non-positive → 0
    return (x > 0).float()


def calculate_hamming_distance(hash1, hash2):
    #Number of bit positions that differ between two binary hashes
    return (hash1 != hash2).sum(dim=1)