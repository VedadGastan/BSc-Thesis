import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from config import *
from model import PerceptualHashNet
from dataset import SignatureTripletDataset
from evaluate import evaluate_signature_robustness
from utils import set_seed

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for anchor, positive, negative in dataloader:
        anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
        
        optimizer.zero_grad()
        
        # 1. Forward pass
        hash_a = model(anchor)
        hash_p = model(positive)
        hash_n = model(negative)
        
        # 2. Calculate standard Triplet Margin Loss
        loss = criterion(hash_a, hash_p, hash_n)
        
        # 3. Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    return total_loss / len(dataloader)

def main():
    set_seed(SEED)
    print(f"Running on device: {DEVICE}")

    # Initialize components
    model = PerceptualHashNet(HASH_BITS).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    # Standard PyTorch Triplet Loss (Margin ensures tampered images are pushed away)
    criterion = nn.TripletMarginLoss(margin=1.0) 

    # Load Data
    train_ds = SignatureTripletDataset(TRAIN_ORIG, TRAIN_TAMP, TRAIN_TRANSFORM)
    val_ds = SignatureTripletDataset(VAL_ORIG, VAL_TAMP, EVAL_TRANSFORM)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    # Training Loop
    best_auc = 0.0
    print("Starting Model Training...")
    for epoch in range(EPOCHS):
        loss = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {loss:.4f}")
        
        # Evaluate separation every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == EPOCHS - 1:
            auc = evaluate_signature_robustness(model, val_loader, DEVICE)
            if auc > best_auc:
                best_auc = auc
                torch.save(model.state_dict(), "model.pth")
                print(">> Saved new best model!")

if __name__ == "__main__":
    main()