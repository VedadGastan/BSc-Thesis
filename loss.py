import torch.nn as nn

class HashingLoss(nn.Module):
    def __init__(self, margin=1.0, quant_weight=0.05):
        super().__init__()
        self.triplet = nn.TripletMarginLoss(margin=margin)
        self.quant_weight = quant_weight

    def forward(self, anchor, positive, negative):
        triplet_loss = self.triplet(anchor, positive, negative)
        quant_loss = (anchor.abs() - 1).pow(2).mean()
        return triplet_loss + self.quant_weight * quant_loss