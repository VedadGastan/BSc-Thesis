import io
import os
import time
import random
import requests
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance


class MediaSignatureDatasetBuilder:
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
            "val":   int(total * splits[1]),
            "test":  total - int(total * splits[0]) - int(total * splits[1])
        }
        session = requests.Session()
        for split, target in targets.items():
            out_dir = os.path.join(self.orig_root, split)
            saved = len(os.listdir(out_dir))
            while saved < target:
                try:
                    r = session.get("https://picsum.photos/800/600", timeout=5)
                    if r.status_code == 200:
                        with open(os.path.join(out_dir, f"img_{saved:05d}.jpg"), "wb") as f:
                            f.write(r.content)
                        saved += 1
                except Exception:
                    time.sleep(1)
        print("Done.")

    def create_forgeries(self, per_image=2):
        
        print("Generating forgeries...")
        for split in ["train", "val", "test"]:
            orig_dir = os.path.join(self.orig_root, split)
            tamp_dir = os.path.join(self.tamp_root, split)
            all_files = [os.path.join(orig_dir, f) for f in sorted(os.listdir(orig_dir)) if f.endswith(".jpg")]

            methods = [
                self._copy_move,
                self._blur_region,
                self._color_shift,
                self._jpeg_compress,
                self._gaussian_noise,
                self._crop_resize,
                lambda img: self._splice(img, all_files),
            ]

            for path in all_files:
                base = os.path.splitext(os.path.basename(path))[0]
                img = Image.open(path).convert("RGB")
                for i, method in enumerate(random.sample(methods, min(per_image, len(methods)))):
                    method(img.copy()).save(os.path.join(tamp_dir, f"{base}_tamp{i}.jpg"))
        print("Done.")

    # Structural attacks
    def _copy_move(self, img):
        w, h = img.size
        pw, ph = random.randint(60, 150), random.randint(60, 150)
        x1, y1 = random.randint(0, w - pw), random.randint(0, h - ph)
        x2, y2 = random.randint(0, w - pw), random.randint(0, h - ph)
        img.paste(img.crop((x1, y1, x1 + pw, y1 + ph)), (x2, y2))
        return img

    def _splice(self, img, all_files):
        donor = Image.open(random.choice(all_files)).convert("RGB")
        w, h = img.size
        dw, dh = donor.size
        pw = random.randint(80, min(200, dw, w))
        ph = random.randint(80, min(200, dh, h))
        sx, sy = random.randint(0, dw - pw), random.randint(0, dh - ph)
        dx, dy = random.randint(0, w - pw), random.randint(0, h - ph)
        img.paste(donor.crop((sx, sy, sx + pw, sy + ph)), (dx, dy))
        return img

    # Degradation attacks
    def _blur_region(self, img):
        w, h = img.size
        pw, ph = random.randint(80, 200), random.randint(80, 200)
        x, y = random.randint(0, w - pw), random.randint(0, h - ph)
        region = img.crop((x, y, x + pw, y + ph)).filter(ImageFilter.GaussianBlur(6))
        img.paste(region, (x, y))
        return img

    def _jpeg_compress(self, img):
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=random.randint(5, 20))
        buf.seek(0)
        return Image.open(buf).copy()

    def _gaussian_noise(self, img):
        arr = np.array(img, dtype=np.float32)
        arr += np.random.normal(0, random.uniform(20, 50), arr.shape)
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

    # Geometric attack

    def _crop_resize(self, img):
        w, h = img.size
        m = random.uniform(0.10, 0.25)
        left   = int(w * random.uniform(0, m))
        top    = int(h * random.uniform(0, m))
        right  = int(w * (1 - random.uniform(0, m)))
        bottom = int(h * (1 - random.uniform(0, m)))
        return img.crop((left, top, right, bottom)).resize((w, h), Image.LANCZOS)

    # Photometric attack

    def _color_shift(self, img):
        return ImageEnhance.Color(img).enhance(random.uniform(0.0, 0.4))


if __name__ == "__main__":
    builder = MediaSignatureDatasetBuilder()
    builder.download_originals(total=1000)
    builder.create_forgeries(per_image=2)
    print("Dataset ready.")