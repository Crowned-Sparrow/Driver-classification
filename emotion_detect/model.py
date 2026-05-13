from torchvision import models
import torch
import torch.nn as nn
from attention import CBAM

class EmotionResNetCBAM(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()

        base = models.resnet18(pretrained=True)

        self.features = nn.Sequential(
            base.conv1,
            base.bn1,
            base.relu,
            base.maxpool,

            base.layer1,
            CBAM(64),

            base.layer2,
            CBAM(128),

            base.layer3,
            CBAM(256),

            base.layer4,
            CBAM(512),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)

        x = self.pool(x)

        x = torch.flatten(x, 1)

        logits = self.classifier(x)

        return logits
