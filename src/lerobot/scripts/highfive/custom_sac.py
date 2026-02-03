"""
Custom SAC policy with dual-camera ResNet-18 encoder for High-Five task.
"""

from __future__ import annotations

from lerobot.policies.sac.modeling_sac import SACPolicy
from lerobot.scripts.highfive.encoders import SACEncoderAdapter


class HighFiveSACPolicy(SACPolicy):
    """
    SAC policy with separate ResNet-18 encoders per camera.

    Overrides the default encoder initialization to use our
    DualCameraEncoder (separate frozen ResNet-18 per camera).
    """

    def __init__(self, config, encoder_config: dict | None = None):
        """
        Args:
            config: SACConfig
            encoder_config: Optional dict with encoder settings:
                - latent_dim: int (default 256)
                - state_dim: int (default 5)
                - pretrained: bool (default True)
                - freeze_backbones: bool (default True)
        """
        self.encoder_config = encoder_config or {}
        super().__init__(config)

    def _init_encoders(self):
        """Override to use our dual-camera encoder."""
        self.shared_encoder = self.config.shared_encoder

        # Get encoder settings
        latent_dim = self.encoder_config.get("latent_dim", 256)
        state_dim = self.encoder_config.get("state_dim", 5)
        pretrained = self.encoder_config.get("pretrained", True)
        freeze_backbones = self.encoder_config.get("freeze_backbones", True)

        # Create our custom encoder (used by both actor and critic)
        self.encoder_critic = SACEncoderAdapter(
            latent_dim=latent_dim,
            state_dim=state_dim,
            pretrained=pretrained,
            freeze_backbones=freeze_backbones,
        )

        # For shared_encoder=True, actor uses same encoder as critic
        # For shared_encoder=False, actor gets its own encoder
        if self.shared_encoder:
            self.encoder_actor = self.encoder_critic
        else:
            self.encoder_actor = SACEncoderAdapter(
                latent_dim=latent_dim,
                state_dim=state_dim,
                pretrained=pretrained,
                freeze_backbones=freeze_backbones,
            )

    def unfreeze_backbones(self):
        """Unfreeze ResNet backbones for fine-tuning."""
        self.encoder_critic.unfreeze_backbones()
        if not self.shared_encoder:
            self.encoder_actor.unfreeze_backbones()
