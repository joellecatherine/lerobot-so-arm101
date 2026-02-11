"""
Custom camera encoders for the High-Five environment.

Supports:
- Single or dual camera configurations
- RGB or RGBD (with depth) input
- Concatenation or cross-attention fusion
- Frozen or trainable ResNet backbones
- Dedicated DepthCNN for depth-only inputs (no pretrained weights)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision import models
from torchvision.models import ResNet18_Weights


class SmallCNN(nn.Module):
    """
    Lightweight CNN for image encoding, trained from scratch.

    Standard approach for image-based RL (DrQ, SAC-AE, CURL).
    Much better suited for MuJoCo scenes than frozen ImageNet ResNet,
    since MuJoCo visuals (flat colors, simple geometry) are nothing
    like natural images.

    Args:
        in_channels: Number of input channels (1 for depth, 3 for RGB, 12 for 4-frame stack)
        output_dim: Output feature dimension (default: 512, matches ResNet-18 output)
    """

    def __init__(self, in_channels: int = 3, output_dim: int = 512):
        super().__init__()
        self.output_dim = output_dim

        # 4-layer CNN, trained from scratch
        self.features = nn.Sequential(
            # Layer 1: (C, 84, 84) -> (32, 40, 40)
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # Layer 2: (32, 40, 40) -> (64, 19, 19)
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # Layer 3: (64, 19, 19) -> (128, 9, 9)
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # Layer 4: (128, 9, 9) -> (256, 4, 4)
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # Global pooling: (256, 4, 4) -> (256, 1, 1)
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

        # Final projection to output_dim
        self.fc = nn.Linear(256, output_dim)

        # Initialize weights (He initialization for ReLU)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        x = x.contiguous()
        x = self.features(x)
        x = self.fc(x)
        return x


# Backwards compatibility alias
DepthCNN = SmallCNN


class StateEncoder(nn.Module):
    """Simple MLP encoder for state-only observations (no images).

    Takes raw state vector (joint positions + hand position) and produces
    a latent feature vector. Used to verify the environment is solvable
    without vision.

    Args:
        state_dim: Input state dimension (default: 8 = 5 joints + 3 hand pos)
        output_dim: Output feature dimension (default: 256)
    """

    def __init__(self, state_dim: int = 8, output_dim: int = 256):
        super().__init__()
        self.output_dim = output_dim
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
            nn.ReLU(),
        )

    def forward(self, state: Tensor) -> Tensor:
        return self.net(state)

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
        bev_depth_wrist_rgb: Asymmetric mode - BEV depth-only (1ch) + Wrist RGB (3ch)
        encoder_type: "resnet" (default) or "small_cnn" (lightweight, trained from scratch)
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
        bev_depth_wrist_rgb: bool = False,
        birdseye_channels: int | None = None,
        wrist_channels: int | None = None,
        encoder_type: str = "resnet",
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.state_dim = state_dim
        self.frozen = freeze_backbones
        self.fusion = fusion
        self.use_depth = use_depth
        self.single_camera = single_camera
        self.bev_depth_wrist_rgb = bev_depth_wrist_rgb
        self.encoder_type = encoder_type

        # Determine input channels for each camera
        # If explicit channels provided (e.g., from frame stacking), use those
        if birdseye_channels is not None:
            self._birdseye_channels = birdseye_channels
            self._wrist_channels = wrist_channels if wrist_channels is not None else 3
        elif bev_depth_wrist_rgb:
            # Asymmetric: BEV gets depth-only (1 channel), Wrist gets RGB (3 channels)
            self._birdseye_channels = 1
            self._wrist_channels = 3
        else:
            # Symmetric: both cameras get same channels
            self._birdseye_channels = 4 if use_depth else 3
            self._wrist_channels = 4 if use_depth else 3

        # Create encoder backbones
        # SmallCNN output dim matches ResNet-18 (512) so projection layers work for both
        backbone_out = 512

        if encoder_type == "small_cnn":
            # Lightweight CNN trained from scratch — better for MuJoCo scenes
            self.encoder_birdseye = SmallCNN(in_channels=self._birdseye_channels, output_dim=backbone_out)
            self._use_depth_cnn = False
            if not single_camera:
                self.encoder_wrist = SmallCNN(in_channels=self._wrist_channels, output_dim=backbone_out)
            # SmallCNN is always trainable, freeze_backbones is ignored
        elif bev_depth_wrist_rgb:
            # Use dedicated SmallCNN for BEV depth (no pretrained weights)
            self.encoder_birdseye = SmallCNN(in_channels=self._birdseye_channels, output_dim=backbone_out)
            self._use_depth_cnn = True
            if not single_camera:
                weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
                self.encoder_wrist = self._make_resnet_backbone(weights, self._wrist_channels)
                if freeze_backbones:
                    self._freeze(self.encoder_wrist)
        else:
            weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            self.encoder_birdseye = self._make_resnet_backbone(weights, self._birdseye_channels)
            self._use_depth_cnn = False
            if not single_camera:
                self.encoder_wrist = self._make_resnet_backbone(weights, self._wrist_channels)
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
        """ResNet-18 without final FC layer, optionally with modified input channels.

        Handles arbitrary input channels (e.g., from frame stacking: 12 = 3 RGB * 4 frames).
        Initializes new channels by tiling/averaging pretrained RGB weights.
        """
        resnet = models.resnet18(weights=weights)

        # Modify first conv layer if not standard 3 channels
        if in_channels != 3:
            old_conv = resnet.conv1
            resnet.conv1 = nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )

            # Initialize weights for arbitrary channel counts
            with torch.no_grad():
                if in_channels < 3:
                    # Fewer channels (e.g., depth-only): average RGB weights
                    resnet.conv1.weight[:, :in_channels] = old_conv.weight[:, :in_channels]
                else:
                    # More channels (e.g., frame stacking): tile RGB weights
                    # For 12 channels (4 frames of RGB): repeat RGB weights 4 times
                    n_repeats = (in_channels + 2) // 3  # ceiling division
                    tiled = old_conv.weight.repeat(1, n_repeats, 1, 1)[:, :in_channels]
                    # Scale down to maintain similar activation magnitudes
                    resnet.conv1.weight.copy_(tiled / n_repeats)

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
        """Unfreeze ResNet backbones for fine-tuning (DepthCNN is always trainable)."""
        if not self._use_depth_cnn:
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
        # Ensure contiguous tensors (replay buffer fancy indexing can return non-contiguous)
        birdseye = birdseye.contiguous()
        state = state.contiguous()

        # Encode birdseye
        f_bird = self.encoder_birdseye(birdseye)
        f_bird = self.proj_birdseye(f_bird)

        # Encode state
        f_state = self.proj_state(state)

        if self.single_camera:
            return torch.cat([f_bird, f_state], dim=-1)

        # Encode wrist
        wrist = wrist.contiguous()
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
    Adapter that wraps FlexibleCameraEncoder (or StateEncoder) to match
    SAC's encoder interface.
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
        bev_depth_wrist_rgb: bool = False,
        birdseye_channels: int | None = None,
        wrist_channels: int | None = None,
        encoder_type: str = "resnet",
    ):
        super().__init__()
        self.single_camera = single_camera
        self.bev_depth_wrist_rgb = bev_depth_wrist_rgb
        self._encoder_type = encoder_type

        if encoder_type == "state":
            # State-only encoder: no images, just MLP on state vector
            self.encoder = StateEncoder(state_dim=state_dim, output_dim=latent_dim)
            self._output_dim = latent_dim
            self.has_images = False
            self.has_state = True
            self.has_env = False
            self.image_keys = []
        else:
            self.encoder = FlexibleCameraEncoder(
                latent_dim=latent_dim,
                state_dim=state_dim,
                pretrained=pretrained,
                freeze_backbones=freeze_backbones,
                fusion=fusion,
                use_depth=use_depth,
                single_camera=single_camera,
                bev_depth_wrist_rgb=bev_depth_wrist_rgb,
                birdseye_channels=birdseye_channels,
                wrist_channels=wrist_channels,
                encoder_type=encoder_type,
            )
            self._output_dim = self.encoder.output_dim
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
        if self._encoder_type == "state":
            features = self.encoder(obs[OBS_STATE])
        else:
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
        if self._encoder_type != "state":
            self.encoder.unfreeze_backbones()


# Keep old class name for backwards compatibility
DualCameraEncoder = FlexibleCameraEncoder
