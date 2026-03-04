import torch.nn as nn
from torchvision import models


class PerceptualHashNet(nn.Module):
    def __init__(self, hash_bits=128):
        super().__init__()
        base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

        # Strip the classification head
        self.feature_extractor = nn.Sequential(*list(base.children())[:-1])

        for child in list(self.feature_extractor.children())[:6]:
            for param in child.parameters():
                param.requires_grad = False

        self.hash_head = nn.Sequential(
            nn.Linear(base.fc.in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, hash_bits),
            nn.Tanh()
        )

    def forward(self, x):
        return self.hash_head(self.feature_extractor(x).flatten(1))