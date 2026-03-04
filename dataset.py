import os
import random
from torch.utils.data import Dataset
from PIL import Image

class SignatureTripletDataset(Dataset):
    """
    Provides triplets: (Anchor, Positive, Negative)
    Anchor: The original image.
    Positive: The same original image (transforms simulate benign compression/edits).
    Negative: A tampered version of the original image (malicious edit).
    """
    def __init__(self, original_dir, tampered_dir, transform):
        self.transform = transform
        self.originals = {os.path.splitext(f)[0]: os.path.join(original_dir, f) for f in os.listdir(original_dir)}
        
        self.tampered = {}
        for f in os.listdir(tampered_dir):
            key = f.split("_tamp")[0]
            self.tampered.setdefault(key, []).append(os.path.join(tampered_dir, f))

        self.keys = list(set(self.originals.keys()) & set(self.tampered.keys()))

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        k = self.keys[idx]
        
        orig_img = Image.open(self.originals[k]).convert("RGB")
        tamp_img = Image.open(random.choice(self.tampered[k])).convert("RGB")

        # By passing the original image twice to transform, we get two differently 
        # augmented versions of the same image (Anchor and Positive).
        anchor = self.transform(orig_img)
        positive = self.transform(orig_img) 
        negative = self.transform(tamp_img)

        return anchor, positive, negative