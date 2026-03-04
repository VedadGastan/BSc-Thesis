import torch.nn as nn
from torchvision import models

class PerceptualHashNet(nn.Module):
    """
    Extracts a robust binary-ready hash from media.
    Acts as the hashing layer before digital signing.
    """
    def __init__(self, hash_bits=128):
        super().__init__()
        # Use a pre-trained ResNet to extract deep features
        base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.feature_extractor = nn.Sequential(*list(base.children())[:-1])

        # Freeze early layers to speed up training
        for param in list(self.feature_extractor.parameters())[:-20]:
            param.requires_grad = False

        # Hash projection head
        self.hash_head = nn.Sequential(
            nn.Linear(base.fc.in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, hash_bits),
            nn.Tanh() # Tanh forces outputs between -1 and 1 (easy to binarize)
        )

    def forward(self, x):
        features = self.feature_extractor(x).flatten(1)
        return self.hash_head(features)