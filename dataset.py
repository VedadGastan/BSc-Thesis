import os
import random
from torch.utils.data import Dataset
from PIL import Image


class SignatureTripletDataset(Dataset):
    def __init__(self, original_dir, tampered_dir, transform):
        self.transform = transform
        self.originals = {
            os.path.splitext(f)[0]: os.path.join(original_dir, f)
            for f in os.listdir(original_dir)
        }
        self.tampered = {}
        for f in os.listdir(tampered_dir):
            key = f.split("_tamp")[0]
            self.tampered.setdefault(key, []).append(os.path.join(tampered_dir, f))

        self.keys = list(set(self.originals) & set(self.tampered))

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        k = self.keys[idx]
        orig = Image.open(self.originals[k]).convert("RGB")
        tamp = Image.open(random.choice(self.tampered[k])).convert("RGB")
        return self.transform(orig), self.transform(orig), self.transform(tamp)