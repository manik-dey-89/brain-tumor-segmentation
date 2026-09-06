"""
Shared building blocks used across all segmentation architectures.

Blocks
------
- ConvBnAct        : Conv2d -> Norm -> Activation (configurable)
- DoubleConvBlock  : Two ConvBnAct units
- ResidualBlock    : Two ConvBnAct + skip-connection projection
- AttentionGate    : Spatial attention gate (additive formulation)
- DownBlock        : MaxPool2d -> DoubleConvBlock / ResidualBlock
- UpBlock          : Upsample (bilinear or ConvTranspose2d) -> concat -> DoubleConvBlock
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Literal


# Normalisation helper

def _norm_layer(
    norm_type: str,
    num_channels: int,
    num_groups: int = 8,
) -> nn.Module:
    """Return the requested normalisation layer."""
    if norm_type == "batch":
        return nn.BatchNorm2d(num_channels)
    if norm_type == "group":
        # num_groups must divide num_channels
        ng = min(num_groups, num_channels)
        while num_channels % ng != 0:
            ng -= 1
        return nn.GroupNorm(ng, num_channels)
    if norm_type == "instance":
        return nn.InstanceNorm2d(num_channels, affine=True)
    raise ValueError(f"Unknown norm_type: {norm_type!r}")


# ConvBnAct

class ConvBnAct(nn.Module):
    """Conv2d -> Norm -> ReLU (or no activation when act=False)."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        padding: int = 1,
        norm_type: str = "batch",
        num_groups: int = 8,
        act: bool = True,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, bias=False),
            _norm_layer(norm_type, out_ch, num_groups),
        ]
        if act:
            layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# DoubleConvBlock

class DoubleConvBlock(nn.Module):
    """Two consecutive ConvBnAct blocks."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        mid_ch: Optional[int] = None,
        norm_type: str = "batch",
        num_groups: int = 8,
        dropout_rate: float = 0.0,
    ) -> None:
        super().__init__()
        mid_ch = mid_ch or out_ch
        self.conv1 = ConvBnAct(in_ch, mid_ch, norm_type=norm_type, num_groups=num_groups)
        self.drop = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.conv2 = ConvBnAct(mid_ch, out_ch, norm_type=norm_type, num_groups=num_groups)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2(self.drop(self.conv1(x)))


# ResidualBlock

class ResidualBlock(nn.Module):
    """
    Residual block with optional projection shortcut.
    Residual: x -> Conv1 -> BN -> ReLU -> Conv2 -> BN -> (+x) -> ReLU
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        norm_type: str = "batch",
        num_groups: int = 8,
        dropout_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.conv1 = ConvBnAct(in_ch, out_ch, norm_type=norm_type, num_groups=num_groups)
        self.drop = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.conv2 = ConvBnAct(out_ch, out_ch, norm_type=norm_type, num_groups=num_groups, act=False)
        self.relu = nn.ReLU(inplace=True)

        # Projection shortcut if channel dimensions differ
        if in_ch != out_ch:
            self.shortcut: nn.Module = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
                _norm_layer(norm_type, out_ch, num_groups),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.drop(out)
        out = self.conv2(out)
        return self.relu(out + residual)


# AttentionGate

class AttentionGate(nn.Module):
    """
    Additive attention gate as in Oktay et al. (2018).

    g  : gating signal from decoder (coarser scale)
    x  : skip-connection feature map from encoder (finer scale)
    """

    def __init__(self, g_ch: int, x_ch: int, inter_ch: int) -> None:
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(g_ch, inter_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_ch),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(x_ch, inter_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_ch),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter_ch, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # Align spatial resolution: upsample g to match x if needed
        if g.shape[2:] != x.shape[2:]:
            g = F.interpolate(g, size=x.shape[2:], mode="bilinear", align_corners=False)
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


# DownBlock  (Encoder step)

class DownBlock(nn.Module):
    """MaxPool2d followed by a DoubleConvBlock or ResidualBlock."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        block_type: Literal["double", "residual"] = "double",
        norm_type: str = "batch",
        num_groups: int = 8,
        dropout_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        BlockCls = DoubleConvBlock if block_type == "double" else ResidualBlock
        self.conv = BlockCls(
            in_ch, out_ch, norm_type=norm_type, num_groups=num_groups, dropout_rate=dropout_rate
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


# UpBlock  (Decoder step)

class UpBlock(nn.Module):
    """
    Upsample -> concatenate skip -> DoubleConvBlock.
    Upsampling: bilinear interpolation OR transposed convolution.
    """

    def __init__(
        self,
        in_ch: int,
        skip_ch: int,
        out_ch: int,
        bilinear: bool = False,
        norm_type: str = "batch",
        num_groups: int = 8,
        dropout_rate: float = 0.0,
    ) -> None:
        super().__init__()
        if bilinear:
            self.up: nn.Module = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
            conv_in_ch = in_ch + skip_ch
        else:
            self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
            conv_in_ch = in_ch // 2 + skip_ch

        self.conv = DoubleConvBlock(
            conv_in_ch, out_ch, norm_type=norm_type, num_groups=num_groups, dropout_rate=dropout_rate
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Pad if spatial dims differ (odd input sizes)
        if x.shape != skip.shape:
            x = F.pad(x, [0, skip.shape[3] - x.shape[3], 0, skip.shape[2] - x.shape[2]])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)
