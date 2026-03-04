import os
import torch
from torchvision import transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

HASH_BITS = 64
SEED = 42
LR = 1e-4
BATCH_SIZE = 32
EPOCHS = 20

TRAIN_ORIG = "dataset/originals/train"
TRAIN_TAMP = "dataset/tampered/train"
VAL_ORIG   = "dataset/originals/val"
VAL_TAMP   = "dataset/tampered/val"
TEST_ORIG  = "dataset/originals/test"
TEST_TAMP  = "dataset/tampered/test"

TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])