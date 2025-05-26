import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import (
    DenseNet121_Weights, DenseNet169_Weights,
    DenseNet201_Weights, DenseNet161_Weights,
)


class DenseNetClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        variant: str = "121",
        pretrained: bool = True,
        freeze_backbone: bool = False,
        mc_dropout: bool = False,
        mc_p: float = 0.3,
    ):
        super().__init__()
        self.mc_dropout = mc_dropout

        # ----- Choose variant ----------------------------------------------
        weight_map = {
            "121": DenseNet121_Weights.IMAGENET1K_V1,
            "169": DenseNet169_Weights.IMAGENET1K_V1,
            "201": DenseNet201_Weights.IMAGENET1K_V1,
            "161": DenseNet161_Weights.IMAGENET1K_V1,
        }
        if variant not in weight_map:
            raise ValueError("variant must be one of 121/169/201/161")

        weights = weight_map[variant] if pretrained else None
        self.net = getattr(models, f"densenet{variant}")(weights=weights)

        # Replace classifier
        in_f = self.net.classifier.in_features
        if mc_dropout:
            self.net.classifier = nn.Sequential(
                nn.Dropout(mc_p), nn.Linear(in_f, num_classes)
            )
        else:
            self.net.classifier = nn.Linear(in_f, num_classes)

        # Freeze backbone if asked
        if freeze_backbone:
            for p in self.net.features.parameters():
                p.requires_grad = False
            for p in self.net.classifier.parameters():
                p.requires_grad = True

    def forward(self, x): return self.net(x)

    def _enable_mc_dropout(self):
        for m in self.modules():
            if isinstance(m, nn.Dropout): m.train()

    def predict(self, x):
        self.eval();  self._enable_mc_dropout() if self.mc_dropout else None
        with torch.no_grad(): return self.forward(x)

    def predict_mc(self, x, T=20):
        self.eval();  self._enable_mc_dropout()
        with torch.no_grad():
            outs = [self.forward(x).softmax(1) for _ in range(T)]
        outs = torch.stack(outs)
        return outs.mean(0), outs.std(0)
