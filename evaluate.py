import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from utils import binarize_hash, calculate_hamming_distance


def evaluate_signature_robustness(model, dataloader, device):
    model.eval()
    distances_genuine = []
    distances_forged  = []

    with torch.no_grad():
        for anchor, positive, negative in dataloader:
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)

            hash_a = binarize_hash(model(anchor))
            hash_p = binarize_hash(model(positive))
            hash_n = binarize_hash(model(negative))

            distances_genuine.extend(calculate_hamming_distance(hash_a, hash_p).cpu().numpy())
            distances_forged.extend(calculate_hamming_distance(hash_a, hash_n).cpu().numpy())

    labels = [1] * len(distances_genuine) + [0] * len(distances_forged)
    scores = [-d for d in distances_genuine] + [-d for d in distances_forged]
    auc = roc_auc_score(labels, scores)

    tar, thresh = _tar_at_far(distances_genuine, distances_forged, target_far=0.01)

    print("\n--- Verification Results ---")
    print(f"  Avg Hamming (Genuine):  {np.mean(distances_genuine):.2f} bits")
    print(f"  Avg Hamming (Tampered): {np.mean(distances_forged):.2f} bits")
    print(f"  AUC:                    {auc:.4f}")
    print(f"  TAR @ FAR=1%:           {tar:.4f}  (threshold = {thresh:.0f} bits)")

    return auc


def _tar_at_far(distances_genuine, distances_forged, target_far=0.01):
    threshold = np.percentile(distances_forged, target_far * 100)
    tar = np.mean(np.array(distances_genuine) <= threshold)
    return float(tar), float(threshold)