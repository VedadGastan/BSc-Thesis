import torch
from utils import binarize_hash, calculate_hamming_distance
from sklearn.metrics import roc_auc_score

def evaluate_signature_robustness(model, dataloader, device):
    """
    Evaluates if the model can successfully distinguish between benign modifications
    (Positive pairs) and malicious tampering (Negative pairs).
    """
    model.eval()
    distances_genuine = [] # Distances between original and benignly altered (should be small)
    distances_forged = []  # Distances between original and tampered (should be large)

    with torch.no_grad():
        for anchor, positive, negative in dataloader:
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

            # Extract continuous hashes and binarize them
            hash_a = binarize_hash(model(anchor))
            hash_p = binarize_hash(model(positive))
            hash_n = binarize_hash(model(negative))

            # Calculate Hamming Distances
            dist_pos = calculate_hamming_distance(hash_a, hash_p)
            dist_neg = calculate_hamming_distance(hash_a, hash_n)

            distances_genuine.extend(dist_pos.cpu().numpy())
            distances_forged.extend(dist_neg.cpu().numpy())

    # Calculate standard metrics
    labels = [1] * len(distances_genuine) + [0] * len(distances_forged)
    # We invert distances because a smaller distance = higher similarity (closer to label 1)
    scores = [-d for d in distances_genuine] + [-d for d in distances_forged]
    
    auc = roc_auc_score(labels, scores)

    print("\n--- Digital Signature Verification Results ---")
    print(f"Average Bit Error (Genuine/Benign): {sum(distances_genuine)/len(distances_genuine):.2f} bits")
    print(f"Average Bit Error (Tampered/Forged): {sum(distances_forged)/len(distances_forged):.2f} bits")
    print(f"Separation AUC Score: {auc:.4f} (1.0 is perfect separation)")
    
    return auc