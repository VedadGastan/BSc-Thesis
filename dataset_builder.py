import os
import time
import random
import requests
import hashlib
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance

class MediaSignatureDatasetBuilder:
    """
    Downloads original images and generates tampered versions.
    In a digital signature context, the originals represent 'signed' documents,
    and the tampered versions represent malicious forgeries.
    """
    def __init__(self, orig_root="dataset/originals", tamp_root="dataset/tampered"):
        self.orig_root = orig_root
        self.tamp_root = tamp_root
        for s in ["train", "val", "test"]:
            os.makedirs(os.path.join(orig_root, s), exist_ok=True)
            os.makedirs(os.path.join(tamp_root, s), exist_ok=True)

    def download_originals(self, total=1000, splits=(0.7, 0.15, 0.15)):
        print("Downloading original images...")
        targets = {
            "train": int(total * splits[0]),
            "val": int(total * splits[1]),
            "test": total - int(total * splits[0]) - int(total * splits[1])
        }
        
        session = requests.Session()
        for split_name, target in targets.items():
            out_dir = os.path.join(self.orig_root, split_name)
            saved = len(os.listdir(out_dir))
            
            while saved < target:
                try:
                    r = session.get(f"https://picsum.photos/800/600", timeout=5)
                    if r.status_code == 200:
                        path = os.path.join(out_dir, f"img_{saved:05d}.jpg")
                        with open(path, "wb") as f:
                            f.write(r.content)
                        saved += 1
                except Exception as e:
                    time.sleep(1)

    def create_forgeries(self, per_image=2):
        print("Generating tampered forgeries...")
        methods = [self._copy_move, self._blur_region, self._color_shift]
        
        for split in ["train", "val", "test"]:
            orig_dir = os.path.join(self.orig_root, split)
            tamp_dir = os.path.join(self.tamp_root, split)
            files = [os.path.join(orig_dir, f) for f in os.listdir(orig_dir) if f.endswith('.jpg')]
            
            for path in files:
                base_name = os.path.splitext(os.path.basename(path))[0]
                img = Image.open(path).convert("RGB")
                
                for i in range(per_image):
                    method = random.choice(methods)
                    forgery = method(img.copy())
                    forgery.save(os.path.join(tamp_dir, f"{base_name}_tamp{i}.jpg"))

    # --- Tampering Methods ---
    def _copy_move(self, img):
        w, h = img.size
        pw, ph = random.randint(60, 150), random.randint(60, 150)
        x1, y1 = random.randint(0, w - pw), random.randint(0, h - ph)
        x2, y2 = random.randint(0, w - pw), random.randint(0, h - ph)
        patch = img.crop((x1, y1, x1 + pw, y1 + ph))
        img.paste(patch, (x2, y2))
        return img

    def _blur_region(self, img):
        w, h = img.size
        pw, ph = random.randint(80, 200), random.randint(80, 200)
        x, y = random.randint(0, w - pw), random.randint(0, h - ph)
        region = img.crop((x, y, x + pw, y + ph)).filter(ImageFilter.GaussianBlur(5))
        img.paste(region, (x, y))
        return img

    def _color_shift(self, img):
        enhancer = ImageEnhance.Color(img)
        return enhancer.enhance(random.uniform(0.3, 1.7))

if __name__ == "__main__":
    builder = MediaSignatureDatasetBuilder()
    builder.download_originals(total=1000) # Reduced to 1000 for faster local testing
    builder.create_forgeries(per_image=2)
    print("Dataset built successfully!")