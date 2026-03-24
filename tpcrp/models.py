"""Model definitions for TPCRP on CIFAR-10."""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import resnet18


def _build_cifar_resnet18() -> nn.Module:
    """ResNet-18 adapted to 32x32 CIFAR inputs."""
    backbone = resnet18(weights=None)
    backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    backbone.maxpool = nn.Identity()
    return backbone


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int = 512, hidden_dim: int = 512, out_dim: int = 128) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class SimCLRModel(nn.Module):
    """ResNet-18 encoder with a SimCLR-style projection head."""

    def __init__(self, projection_dim: int = 128) -> None:
        super().__init__()
        self.backbone = _build_cifar_resnet18()
        feature_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.projector = ProjectionHead(in_dim=feature_dim, out_dim=projection_dim)
        self.feature_dim = feature_dim

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encode(x)
        projections = self.projector(features)
        return features, projections


class LinearClassifier(nn.Module):
    """Linear probe or supervised head on top of frozen embeddings."""

    def __init__(self, in_dim: int = 512, num_classes: int = 10, dropout_p: float = 0.0) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_p) if dropout_p > 0 else nn.Identity()
        self.linear = nn.Linear(in_dim, num_classes)
        self.feature_dim = in_dim

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def forward_from_features(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear(self.dropout(features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encode(x)
        return self.forward_from_features(features)


class ResNet18Classifier(nn.Module):
    """Supervised ResNet-18 classifier used for evaluation on queried labels."""

    def __init__(self, num_classes: int = 10, dropout_p: float = 0.0) -> None:
        super().__init__()
        self.backbone = _build_cifar_resnet18()
        self.feature_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.dropout = nn.Dropout(p=dropout_p) if dropout_p > 0 else nn.Identity()
        self.classifier = nn.Linear(self.feature_dim, num_classes)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward_from_features(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.dropout(features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encode(x)
        return self.forward_from_features(features)


def reset_parameters(module: nn.Module) -> None:
    """Reset a module tree where child modules expose reset_parameters."""
    def _reset(child: nn.Module) -> None:
        reset = getattr(child, "reset_parameters", None)
        if callable(reset):
            reset()

    module.apply(_reset)
