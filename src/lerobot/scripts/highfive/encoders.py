"""
Custom camera encoders for the High-Five environment.

Supports:
- Single or dual camera configurations
- RGB or RGBD (with depth) input
- Concatenation or cross-attention fusion
- Frozen or trainable ResNet backbones
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision import models
from torchvision.models import ResNet18_Weights

# Keys used by LeRobot
OBS_IMAGES = "observation.images"
OBS_STATE = "observation.state"


class CrossAttentionFusion(nn.Module):
    """
    Cross-attention module for fusing features from two camera views.

    Birdseye features attend to wrist features, allowing the model to
    learn spatial correspondences between the global and local views.
    """

    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim, "dim must be divisible by num_heads"

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm1 = nn.LayerNorm(dim)
        self.layer_norm2 = nn.LayerNorm(dim)

        # FFN after attention
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

    def forward(self, query: Tensor, key_value: Tensor) -> Tensor:
        """
        Args:
            query: Features to be updated (B, dim) - e.g., birdseye
            key_value: Features to attend to (B, dim) - e.g., wrist

        Returns:
            Updated query features (B, dim)
        """
        B = query.shape[0]

        # Add sequence dimension for attention (B, 1, dim)
        q = query.unsqueeze(1)
        kv = key_value.unsqueeze(1)

        # Project and reshape for multi-head attention
        Q = self.q_proj(q).view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(kv).view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(kv).view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scale = self.head_dim ** -0.5
        attn = (Q @ K.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Combine heads
        out = (attn @ V).transpose(1, 2).contiguous().view(B, 1, self.dim)
        out = self.out_proj(out).squeeze(1)

        # Residual + LayerNorm
        query = self.layer_norm1(query + out)

        # FFN with residual
        query = self.layer_norm2(query + self.ffn(query))

        return query


class FlexibleCameraEncoder(nn.Module):
    """
    Flexible camera encoder supporting various configurations.

    Args:
        latent_dim: Output dimension per stream (default: 256)
        state_dim: Joint positions dimension (default: 5)
        pretrained: Use ImageNet weights (default: True)
        freeze_backbones: Freeze ResNet weights (default: True)
        fusion: Fusion method - "concat" or "cross_attention"
        use_depth: Whether input has depth channel (4 channels vs 3)
        single_camera: Use only birdseye camera
    """

    def __init__(
        self,
        latent_dim: int = 256,
        state_dim: int = 5,
        pretrained: bool = True,
        freeze_backbones: bool = True,
        fusion: str = "concat",
        use_depth: bool = False,
        single_camera: bool = False,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.state_dim = state_dim
        self.frozen = freeze_backbones
        self.fusion = fusion
        self.use_depth = use_depth
        self.single_camera = single_camera

        # Input channels: 3 for RGB, 4 for RGBD
        in_channels = 4 if use_depth else 3

        # Create ResNet backbone(s)
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.encoder_birdseye = self._make_resnet_backbone(weights, in_channels)

        if not single_camera:
            self.encoder_wrist = self._make_resnet_backbone(weights, in_channels)

        if freeze_backbones:
            self._freeze(self.encoder_birdseye)
            if not single_camera:
                self._freeze(self.encoder_wrist)

        # ResNet-18 outputs 512-dim after avgpool
        resnet_out = 512

        # Projection layers
        self.proj_birdseye = nn.Sequential(
            nn.Linear(resnet_out, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ReLU(),
        )

        if not single_camera:
            self.proj_wrist = nn.Sequential(
                nn.Linear(resnet_out, latent_dim),
                nn.LayerNorm(latent_dim),
                nn.ReLU(),
            )

            # Cross-attention fusion if selected
            if fusion == "cross_attention":
                self.cross_attn = CrossAttentionFusion(latent_dim)
                self._output_dim = latent_dim * 2 + latent_dim  # bird + wrist + state
            else:
                self._output_dim = latent_dim * 3  # bird + wrist + state (concat)
        else:
            self._output_dim = latent_dim * 2  # bird + state only

        self.proj_state = nn.Sequential(
            nn.Linear(state_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.Tanh(),
        )

    def _make_resnet_backbone(self, weights, in_channels: int = 3) -> nn.Module:
        """ResNet-18 without final FC layer, optionally with modified input channels."""
        resnet = models.resnet18(weights=weights)

        # Modify first conv layer if using depth (4 channels instead of 3)
        if in_channels != 3:
            old_conv = resnet.conv1
            resnet.conv1 = nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
            # Initialize new channels with mean of RGB weights
            with torch.no_grad():
                resnet.conv1.weight[:, :3] = old_conv.weight
                if in_channels > 3:
                    # Initialize depth channel with mean of RGB
                    resnet.conv1.weight[:, 3:] = old_conv.weight.mean(dim=1, keepdim=True)

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
        """Unfreeze ResNet backbones for fine-tuning."""
        for p in self.encoder_birdseye.parameters():
            p.requires_grad = True
        if not self.single_camera:
            for p in self.encoder_wrist.parameters():
                p.requires_grad = True
        self.frozen = False

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(
        self,
        birdseye: torch.Tensor,
        wrist: torch.Tensor | None,
        state: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            birdseye: (B, C, H, W) where C=3 (RGB) or C=4 (RGBD)
            wrist: (B, C, H, W) or None if single_camera
            state: (B, state_dim)

        Returns:
            (B, output_dim) fused features
        """
        # Encode birdseye
        f_bird = self.encoder_birdseye(birdseye)
        f_bird = self.proj_birdseye(f_bird)

        # Encode state
        f_state = self.proj_state(state)

        if self.single_camera:
            return torch.cat([f_bird, f_state], dim=-1)

        # Encode wrist
        f_wrist = self.encoder_wrist(wrist)
        f_wrist = self.proj_wrist(f_wrist)

        # Fuse camera features
        if self.fusion == "cross_attention":
            # Birdseye attends to wrist (what in the global view relates to the close-up?)
            f_bird_attn = self.cross_attn(f_bird, f_wrist)
            return torch.cat([f_bird_attn, f_wrist, f_state], dim=-1)
        else:
            # Simple concatenation
            return torch.cat([f_bird, f_wrist, f_state], dim=-1)


class SACEncoderAdapter(nn.Module):
    """
    Adapter that wraps FlexibleCameraEncoder to match SAC's encoder interface.
    """

    def __init__(
        self,
        latent_dim: int = 256,
        state_dim: int = 5,
        pretrained: bool = True,
        freeze_backbones: bool = True,
        fusion: str = "concat",
        use_depth: bool = False,
        single_camera: bool = False,
    ):
        super().__init__()
        self.single_camera = single_camera

        self.encoder = FlexibleCameraEncoder(
            latent_dim=latent_dim,
            state_dim=state_dim,
            pretrained=pretrained,
            freeze_backbones=freeze_backbones,
            fusion=fusion,
            use_depth=use_depth,
            single_camera=single_camera,
        )
        self._output_dim = self.encoder.output_dim

        # For SAC compatibility
        self.has_images = True
        self.has_state = True
        self.has_env = False

        if single_camera:
            self.image_keys = [f"{OBS_IMAGES}.birdseye"]
        else:
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
        """
        birdseye = obs[f"{OBS_IMAGES}.birdseye"]
        wrist = None if self.single_camera else obs[f"{OBS_IMAGES}.wrist"]
        state = obs[OBS_STATE]

        features = self.encoder(birdseye, wrist, state)

        if detach:
            features = features.detach()

        return features

    def get_cached_image_features(self, obs: dict[str, Tensor]) -> dict[str, Tensor]:
        """For SAC compatibility - returns dummy cache."""
        return {}

    def unfreeze_backbones(self):
        """Unfreeze ResNet backbones for fine-tuning."""
        self.encoder.unfreeze_backbones()


# Keep old class name for backwards compatibility
DualCameraEncoder = FlexibleCameraEncoder
