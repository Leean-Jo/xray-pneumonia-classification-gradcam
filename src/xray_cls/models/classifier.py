import torch.nn as nn
from torchvision import models


class XRayClassifier(nn.Module):
    def __init__(
        self,
        model_name: str = "resnet18",
        pretrained: bool = True,
        dropout: float = 0.2,
    ):
        super().__init__()

        if model_name == "resnet18":
            self.backbone = models.resnet18(
                weights=models.ResNet18_Weights.DEFAULT if pretrained else None
            )
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_features, 1),
            )
        else:
            raise ValueError(f"Unsupported model: {model_name}")

    def forward(self, x):
        return self.backbone(x)