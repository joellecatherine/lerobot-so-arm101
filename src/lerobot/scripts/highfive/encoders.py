"""
Custom dual-camera encoder for the High-Five environment.

Provides separate ResNet-18 backbones per camera (no weight sharing),
which is better for views with very different perspectives.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from torchvision import models
from torchvision.models import ResNet18_Weights

# Keys used by LeRobot
OBS_IMAGES = "observation.images"
OBS_STATE = "observation.state"


class DualCameraEncoder(nn.Module):
    """
    Dual-camera encoder with separate frozen ResNet-18 backbones.

    - Birdseye camera: far view, scene layout
    - Wrist camera: close-up, approach guidance

    Each camera has its own ResNet-18 (no weight sharing).
    Features are concatenated after projection to latent space.

    Args:
        latent_dim: Output dimension per stream (default: 256)
        state_dim: Joint positions dimension (default: 5)
        pretrained: Use ImageNet weights (default: True)
        freeze_backbones: Freeze ResNet weights (default: True)
    """

    def __init__(
        self,
        latent_dim: int = 256,
        state_dim: int = 5,
        pretrained: bool = True,
        freeze_backbones: bool = True,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.state_dim = state_dim
        self.frozen = freeze_backbones

        # Separate ResNet-18 per camera
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.encoder_birdseye = self._make_resnet_backbone(weights)
        self.encoder_wrist = self._make_resnet_backbone(weights)

        if freeze_backbones:
            self._freeze(self.encoder_birdseye)
            self._freeze(self.encoder_wrist)

        # ResNet-18 outputs 512-dim after avgpool
        resnet_out = 512

        # Project each stream to latent_dim
        self.proj_birdseye = nn.Sequential(
            nn.Linear(resnet_out, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ReLU(),
        )
        self.proj_wrist = nn.Sequential(
            nn.Linear(resnet_out, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ReLU(),
        )
        self.proj_state = nn.Sequential(
            nn.Linear(state_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.Tanh(),
        )

        self._output_dim = latent_dim * 3

    def _make_resnet_backbone(self, weights) -> nn.Module:
        """ResNet-18 without final FC layer."""
        resnet = models.resnet18(weights=weights)
        return nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            resnet.avgpool,
            nn.Flatten(),
        )

    def _freeze(self, module: nn.Module):
        for p in module.parameters():
            p.requires_grad = False

    def unfreeze_backbones(self):
        """Unfreeze both ResNets for fine-tuning."""
        for p in self.encoder_birdseye.parameters():
            p.requires_grad = True
        for p in self.encoder_wrist.parameters():
            p.requires_grad = True
        self.frozen = False

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(
        self,
        birdseye: torch.Tensor,
        wrist: torch.Tensor,
        state: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            birdseye: (B, 3, H, W)
            wrist: (B, 3, H, W)
            state: (B, state_dim)

        Returns:
            (B, output_dim) concatenated features
        """
        # Encode images
        f_bird = self.encoder_birdseye(birdseye)
        f_wrist = self.encoder_wrist(wrist)

        # Project to latent space
        f_bird = self.proj_birdseye(f_bird)
        f_wrist = self.proj_wrist(f_wrist)
        f_state = self.proj_state(state)

        return torch.cat([f_bird, f_wrist, f_state], dim=-1)


class SACEncoderAdapter(nn.Module):
    """
    Adapter that wraps DualCameraEncoder to match SAC's encoder interface.

    This allows injecting our custom encoder into SAC without modifying
    LeRobot's code.
    """

    def __init__(
        self,
        latent_dim: int = 256,
        state_dim: int = 5,
        pretrained: bool = True,
        freeze_backbones: bool = True,
    ):
        super().__init__()
        self.encoder = DualCameraEncoder(
            latent_dim=latent_dim,
            state_dim=state_dim,
            pretrained=pretrained,
            freeze_backbones=freeze_backbones,
        )
        self._output_dim = self.encoder.output_dim

        # For SAC compatibility
        self.has_images = True
        self.has_state = True
        self.has_env = False
        self.image_keys = [f"{OBS_IMAGES}.birdseye", f"{OBS_IMAGES}.wrist"]

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(
        self,
        obs: dict[str, Tensor],
        cache: dict[str, Tensor] | None = None,
        detach: bool = False,
    ) -> Tensor:
        """
        Forward pass matching SAC's SACObservationEncoder interface.

        Args:
            obs: Dictionary with image and state observations
            cache: Unused (for interface compatibility)
            detach: Whether to detach encoder output

        Returns:
            Encoded features (B, output_dim)
        """
        birdseye = obs[f"{OBS_IMAGES}.birdseye"]
        wrist = obs[f"{OBS_IMAGES}.wrist"]
        state = obs[OBS_STATE]

        features = self.encoder(birdseye, wrist, state)

        if detach:
            features = features.detach()

        return features

    def get_cached_image_features(self, obs: dict[str, Tensor]) -> dict[str, Tensor]:
        """
        For SAC compatibility - returns dummy cache since we don't use caching.
        """
        # We process both images together, so just return empty cache
        # The actual encoding happens in forward()
        return {}

    def unfreeze_backbones(self):
        """Unfreeze ResNet backbones for fine-tuning."""
        self.encoder.unfreeze_backbones()
