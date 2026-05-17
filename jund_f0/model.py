"""
JUND-F0 Model Architecture

Joint Unvoiced/Voiced Detection and F0 Estimation framework.

Architecture overview:
- Shared Encoder: Multi-scale CNN backbone that extracts hierarchical features
  from mel spectrograms. Uses depthwise separable convolutions for efficiency.
- V/UV Head: Binary classification head for voiced/unvoiced detection
- F0 Head: Regression head for F0 estimation (only active for voiced frames)
- Joint Training: Combined loss function that optimizes both tasks simultaneously

The key innovation is that the shared encoder learns representations useful for
both voicing detection and pitch estimation, leading to better performance on
both tasks compared to training them separately.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class JUNDF0Config:
    """Configuration for JUND-F0 model."""

    # Mel spectrogram parameters
    sample_rate: int = 16000
    n_fft: int = 1024
    hop_length: int = 160  # 10ms at 16kHz
    n_mels: int = 80
    f_min: float = 50.0
    f_max: float = 800.0

    # Model parameters
    encoder_channels: int = 64
    encoder_blocks: int = 4
    encoder_kernel_sizes: tuple = (3, 5, 5, 5)
    encoder_dilation_rates: tuple = (1, 1, 2, 4)
    use_depthwise_separable: bool = True

    # Attention parameters
    use_self_attention: bool = True
    num_attention_heads: int = 4
    attention_dropout: float = 0.1

    # V/UV head parameters
    vuv_hidden_dim: int = 128
    vuv_dropout: float = 0.2

    # F0 head parameters
    f0_hidden_dim: int = 256
    f0_dropout: float = 0.2

    # F0 range for output
    f0_min: float = 50.0
    f0_max: float = 800.0

    # Training parameters
    input_frames: int = 32  # Number of consecutive frames for context

    # Loss weights
    vuv_loss_weight: float = 1.0
    f0_loss_weight: float = 1.0
    ffe_loss_weight: float = 0.5

    def __post_init__(self):
        assert len(self.encoder_kernel_sizes) == self.encoder_blocks
        assert len(self.encoder_dilation_rates) == self.encoder_blocks


class DepthwiseSeparableConv1d(nn.Module):
    """Depthwise separable 1D convolution for efficient computation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
        bias: bool = False,
    ):
        super().__init__()
        self.depthwise = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=dilation * (kernel_size - 1) // 2,
            groups=in_channels,
            bias=bias,
        )
        self.pointwise = nn.Conv1d(
            in_channels, out_channels, kernel_size=1, bias=bias
        )
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return x


class ResidualBlock(nn.Module):
    """Residual block with optional depthwise separable convolution."""

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilation: int,
        use_depthwise_separable: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        if use_depthwise_separable:
            self.conv1 = DepthwiseSeparableConv1d(
                channels, channels, kernel_size, dilation
            )
            self.conv2 = DepthwiseSeparableConv1d(
                channels, channels, kernel_size, dilation
            )
        else:
            self.conv1 = nn.Conv1d(
                channels,
                channels,
                kernel_size=kernel_size,
                dilation=dilation,
                padding=dilation * (kernel_size - 1) // 2,
                bias=False,
            )
            self.conv2 = nn.Conv1d(
                channels,
                channels,
                kernel_size=kernel_size,
                dilation=dilation,
                padding=dilation * (kernel_size - 1) // 2,
                bias=False,
            )

        self.bn1 = nn.BatchNorm1d(channels)
        self.bn2 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.dropout(F.gelu(self.bn1(self.conv1(x))))
        out = self.dropout(F.gelu(self.bn2(self.conv2(out))))
        return residual + out


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention for capturing long-range temporal dependencies."""

    def __init__(
        self,
        d_model: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape

        # Linear projections
        q = self.w_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        k = self.w_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)

        output = self.w_o(context)
        output = self.dropout(output)
        return self.layer_norm(output + x)


class SharedEncoder(nn.Module):
    """
    Shared encoder backbone that extracts hierarchical features from mel spectrograms.

    Architecture:
    1. Input projection from mel bins to encoder channels
    2. Multi-scale residual blocks with increasing dilation rates
    3. Optional self-attention layer for long-range context
    """

    def __init__(self, config: JUNDF0Config):
        super().__init__()
        self.config = config

        # Input projection: (batch, n_mels, time) -> (batch, channels, time)
        self.input_proj = nn.Sequential(
            nn.Conv1d(config.n_mels, config.encoder_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(config.encoder_channels),
            nn.GELU(),
        )

        # Multi-scale residual blocks
        self.residual_blocks = nn.ModuleList()
        for i in range(config.encoder_blocks):
            self.residual_blocks.append(
                ResidualBlock(
                    channels=config.encoder_channels,
                    kernel_size=config.encoder_kernel_sizes[i],
                    dilation=config.encoder_dilation_rates[i],
                    use_depthwise_separable=config.use_depthwise_separable,
                    dropout=0.1,
                )
            )

        # Self-attention for long-range context
        if config.use_self_attention:
            self.attention = MultiHeadSelfAttention(
                d_model=config.encoder_channels,
                num_heads=config.num_attention_heads,
                dropout=config.attention_dropout,
            )

        self.output_norm = nn.LayerNorm(config.encoder_channels)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel: (batch, n_mels, time) mel spectrogram

        Returns:
            features: (batch, time, channels) shared features
        """
        # Project input
        x = self.input_proj(mel)  # (batch, channels, time)

        # Residual blocks
        for block in self.residual_blocks:
            x = block(x)

        # Transpose for attention: (batch, time, channels)
        x = x.transpose(1, 2)

        # Self-attention
        if self.config.use_self_attention:
            x = self.attention(x)

        # Normalize
        x = self.output_norm(x)

        return x


class VUVHead(nn.Module):
    """
    Voiced/Unvoiced classification head.

    Predicts the probability of each frame being voiced.
    Uses a lightweight MLP with GELU activation.
    """

    def __init__(self, config: JUNDF0Config):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(config.encoder_channels, config.vuv_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.vuv_dropout),
            nn.Linear(config.vuv_hidden_dim, config.vuv_hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(config.vuv_dropout),
            nn.Linear(config.vuv_hidden_dim // 2, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (batch, time, channels) shared encoder features

        Returns:
            vuv_logits: (batch, time, 1) voiced/unvoiced logits
        """
        return self.classifier(features)


class F0Head(nn.Module):
    """
    F0 estimation regression head.

    Predicts the fundamental frequency for voiced frames.
    Uses a deeper MLP than VUV head with residual connections.

    Output is in Hz, using sigmoid activation scaled to [f0_min, f0_max].
    """

    def __init__(self, config: JUNDF0Config):
        super().__init__()
        self.f0_min = config.f0_min
        self.f0_max = config.f0_max

        self.regressor = nn.Sequential(
            nn.Linear(config.encoder_channels, config.f0_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.f0_dropout),
            nn.Linear(config.f0_hidden_dim, config.f0_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.f0_dropout),
            nn.Linear(config.f0_hidden_dim, config.f0_hidden_dim // 2),
            nn.GELU(),
            nn.Linear(config.f0_hidden_dim // 2, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (batch, time, channels) shared encoder features

        Returns:
            f0_pred: (batch, time, 1) predicted F0 in Hz
        """
        # Raw output
        x = self.regressor(features)

        # Sigmoid scaled to [f0_min, f0_max]
        f0_pred = torch.sigmoid(x) * (self.f0_max - self.f0_min) + self.f0_min

        return f0_pred


class JUNDF0(nn.Module):
    """
    JUND-F0: Joint Unvoiced/Voiced Detection and F0 Estimation

    A novel deep learning framework that jointly solves voiced/unvoiced detection
    and fundamental frequency estimation in a single unified model.

    Key innovations:
    1. Shared encoder learns representations useful for both tasks
    2. V/UV information guides F0 estimation (F0 loss only on voiced frames)
    3. Joint optimization leads to better performance on both tasks
    4. F0 Frame Error (FFE) metric combines GPE and VDE for evaluation
    """

    def __init__(self, config: Optional[JUNDF0Config] = None):
        super().__init__()
        self.config = config or JUNDF0Config()

        # Shared encoder
        self.encoder = SharedEncoder(self.config)

        # Task-specific heads
        self.vuv_head = VUVHead(self.config)
        self.f0_head = F0Head(self.config)

        # Initialize weights
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module):
        """Initialize weights using Kaiming initialization for Conv and Xavier for Linear."""
        if isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(
        self,
        mel: torch.Tensor,
        vuv_label: Optional[torch.Tensor] = None,
        f0_label: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Forward pass with optional loss computation.

        Args:
            mel: (batch, n_mels, time) mel spectrogram
            vuv_label: (batch, time, 1) voiced/unvoiced ground truth (0 or 1)
            f0_label: (batch, time, 1) F0 ground truth in Hz (0 for unvoiced)

        Returns:
            Dictionary containing:
                - vuv_logits: (batch, time, 1) V/UV classification logits
                - vuv_prob: (batch, time, 1) V/UV probability
                - f0_pred: (batch, time, 1) Predicted F0 in Hz
                - loss: Total loss (if labels provided)
                - vuv_loss: V/UV classification loss (if labels provided)
                - f0_loss: F0 regression loss (if labels provided)
                - ffe: F0 Frame Error (if labels provided)
        """
        # Shared features
        features = self.encoder(mel)  # (batch, time, channels)

        # V/UV prediction
        vuv_logits = self.vuv_head(features)  # (batch, time, 1)
        vuv_prob = torch.sigmoid(vuv_logits)

        # F0 prediction
        f0_pred = self.f0_head(features)  # (batch, time, 1)

        outputs = {
            "vuv_logits": vuv_logits,
            "vuv_prob": vuv_prob,
            "f0_pred": f0_pred,
        }

        # Compute losses if labels are provided
        if vuv_label is not None and f0_label is not None:
            losses = self._compute_losses(vuv_logits, vuv_prob, f0_pred, vuv_label, f0_label)
            outputs.update(losses)

        return outputs

    def _compute_losses(
        self,
        vuv_logits: torch.Tensor,
        vuv_prob: torch.Tensor,
        f0_pred: torch.Tensor,
        vuv_label: torch.Tensor,
        f0_label: torch.Tensor,
    ) -> dict:
        """
        Compute joint losses for V/UV detection and F0 estimation.

        Loss components:
        1. V/UV loss: Weighted binary cross-entropy (handles class imbalance)
        2. F0 loss: L1 loss on voiced frames only, in log-Hz space for better
           perceptual scaling
        3. FFE auxiliary loss: Encourages consistency between V/UV and F0 predictions
        """
        # === V/UV Loss ===
        # Weighted BCE to handle class imbalance (unvoiced frames are often more frequent)
        n_voiced = vuv_label.sum().clamp(min=1.0)
        n_unvoiced = (1 - vuv_label).sum().clamp(min=1.0)
        total = n_voiced + n_unvoiced
        weight_voiced = total / (2 * n_voiced)
        weight_unvoiced = total / (2 * n_unvoiced)

        bce_weight = vuv_label * weight_voiced + (1 - vuv_label) * weight_unvoiced
        vuv_loss = F.binary_cross_entropy_with_logits(
            vuv_logits, vuv_label, weight=bce_weight
        )

        # === F0 Loss ===
        # Only compute on voiced frames
        voiced_mask = (vuv_label > 0.5).float()  # (batch, time, 1)

        if voiced_mask.sum() > 0:
            # Log-Hz space for perceptual scaling
            f0_pred_log = torch.log1p(f0_pred)
            f0_label_log = torch.log1p(f0_label)

            # L1 loss on voiced frames
            f0_diff = (f0_pred_log - f0_label_log).abs() * voiced_mask
            f0_loss = f0_diff.sum() / voiced_mask.sum().clamp(min=1.0)

            # Also add coarse loss: penalize octave errors more
            f0_ratio = f0_pred / f0_label.clamp(min=1.0)
            f0_ratio = f0_ratio * voiced_mask
            # Deviation from 1.0 in log space captures octave errors
            octave_loss = (torch.log2(f0_ratio.clamp(0.5, 2.0)).abs() * voiced_mask).sum()
            octave_loss = octave_loss / voiced_mask.sum().clamp(min=1.0)
            f0_loss = f0_loss + 0.3 * octave_loss
        else:
            f0_loss = torch.tensor(0.0, device=f0_pred.device)

        # === FFE Auxiliary Loss ===
        # F0 Frame Error: A frame is erroneous if V/UV is wrong OR
        # F0 error exceeds 20% (Gross Pitch Error threshold)
        ffe = self._compute_ffe_loss(vuv_prob, f0_pred, vuv_label, f0_label)

        # === Total Loss ===
        total_loss = (
            self.config.vuv_loss_weight * vuv_loss
            + self.config.f0_loss_weight * f0_loss
            + self.config.ffe_loss_weight * ffe
        )

        return {
            "loss": total_loss,
            "vuv_loss": vuv_loss,
            "f0_loss": f0_loss,
            "ffe": ffe,
        }

    def _compute_ffe_loss(
        self,
        vuv_prob: torch.Tensor,
        f0_pred: torch.Tensor,
        vuv_label: torch.Tensor,
        f0_label: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute differentiable F0 Frame Error (FFE) as auxiliary loss.

        FFE considers a frame as erroneous if:
        1. V/UV detection is wrong (false positive or false negative), OR
        2. V/UV is correct but F0 error > 20% (Gross Pitch Error)

        We use a smooth differentiable approximation for training.
        """
        # V/UV error: probability of wrong V/UV decision
        vuv_error = torch.abs(vuv_prob - vuv_label)  # (batch, time, 1)

        # F0 error: relative error on voiced frames
        voiced_mask = (vuv_label > 0.5).float()
        f0_relative_error = torch.zeros_like(f0_pred)

        if voiced_mask.sum() > 0:
            f0_relative_error = (
                (f0_pred - f0_label).abs() / f0_label.clamp(min=1.0)
            ) * voiced_mask

        # GPE threshold: 20% relative error
        # Smooth approximation using sigmoid
        gpe_prob = torch.sigmoid(50.0 * (f0_relative_error - 0.2)) * voiced_mask

        # FFE: frame is erroneous if V/UV is wrong OR (V/UV correct AND GPE)
        # Use probabilistic OR: P(A or B) = P(A) + P(B) - P(A)*P(B)
        ffe_prob = vuv_error + gpe_prob - vuv_error * gpe_prob

        return ffe_prob.mean()

    def predict(self, mel: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inference: predict V/UV and F0 from mel spectrogram.

        Args:
            mel: (batch, n_mels, time) mel spectrogram

        Returns:
            vuv: (batch, time) boolean voiced/unvoiced decision
            f0: (batch, time) predicted F0 in Hz (0 for unvoiced frames)
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(mel)
            vuv_prob = outputs["vuv_prob"].squeeze(-1)  # (batch, time)
            f0_pred = outputs["f0_pred"].squeeze(-1)  # (batch, time)

            vuv = vuv_prob > 0.5
            f0 = f0_pred * vuv.float()  # Zero F0 for unvoiced

        return vuv, f0

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
