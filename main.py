import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from config import *
from dataset import SignatureTripletDataset
from evaluate import evaluate_signature_robustness
from loss import HashingLoss
from model import PerceptualHashNet
from utils import set_seed


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total = 0.0
    for anchor, positive, negative in loader:
        anchor = anchor.to(device)
        positive = positive.to(device)
        negative = negative.to(device)

        optimizer.zero_grad()
        loss = criterion(model(anchor), model(positive), model(negative))
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / len(loader)


def main():
    set_seed(SEED)
    print(f"Device: {DEVICE}")

    model = PerceptualHashNet(HASH_BITS).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = HashingLoss(margin=1.0, quant_weight=QUANT_WEIGHT)

    train_ds = SignatureTripletDataset(TRAIN_ORIG, TRAIN_TAMP, TRAIN_TRANSFORM)
    val_ds = SignatureTripletDataset(VAL_ORIG, VAL_TAMP, EVAL_TRANSFORM)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    best_auc = 0.0
    print("Training...\n")
    for epoch in range(EPOCHS):
        loss = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        scheduler.step()
        print(f"Epoch [{epoch+1:02d}/{EPOCHS}]  Loss: {loss:.4f}  LR: {scheduler.get_last_lr()[0]:.2e}")

        if (epoch + 1) % 5 == 0 or epoch == EPOCHS - 1:
            auc = evaluate_signature_robustness(model, val_loader, DEVICE)
            if auc > best_auc:
                best_auc = auc
                torch.save(model.state_dict(), "model.pth")
                print("  >> Saved new best model.")

    print(f"\nBest Validation AUC: {best_auc:.4f}")


if __name__ == "__main__":
    main()