"""
Classic U-Net (Ronneberger et al., 2015)
========================================
Symmetric encoder–decoder with skip connections.

Architecture (depth=4, base_filters=64):
  Encoder: 1->64 -> 64->128 -> 128->256 -> 256->512 -> 512->1024 (bottleneck)
  Decoder: 1024+512->512 -> 512+256->256 -> 256+128->128 -> 128+64->64
  Head:    64 -> num_classes (1×1 conv)
"""

import torch
import torch.nn as nn
from models.blocks import DoubleConvBlock, DownBlock, UpBlock


class UNet(nn.Module):
    """
    Configurable U-Net segmentation model.

    Parameters
    ----------
    in_channels  : number of input MRI channels (1 for grayscale, 3 for RGB)
    num_classes  : number of output classes (1 for binary, >1 for multi-class)
    base_filters : feature-map count in first encoder block (doubles each level)
    depth        : number of encoder / decoder levels (≥ 2)
    bilinear     : use bilinear upsampling instead of transposed convolutions
    norm_type    : "batch" | "group" | "instance"
    num_groups   : GroupNorm group count (used when norm_type="group")
    dropout_rate : spatial dropout probability (applied inside conv blocks)
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 1,
        base_filters: int = 64,
        depth: int = 4,
        bilinear: bool = False,
        norm_type: str = "batch",
        num_groups: int = 8,
        dropout_rate: float = 0.0,
    ) -> None:
        super().__init__()
        assert depth >= 2, "depth must be at least 2"

        self.depth = depth
        self.num_classes = num_classes

        # ── Encoder ──────────────────────────────────────────────────────────
        # stem: initial double-conv (no pooling)
        self.stem = DoubleConvBlock(
            in_channels, base_filters,
            norm_type=norm_type, num_groups=num_groups, dropout_rate=dropout_rate,
        )
        self.encoder_blocks = nn.ModuleList()
        ch = base_filters
        self.encoder_channels: list[int] = [ch]   # channels after each encoder level

        for _ in range(depth - 1):
            out_ch = ch * 2
            self.encoder_blocks.append(
                DownBlock(ch, out_ch, block_type="double",
                          norm_type=norm_type, num_groups=num_groups,
                          dropout_rate=dropout_rate)
            )
            ch = out_ch
            self.encoder_channels.append(ch)

        # bottleneck
        self.bottleneck = DownBlock(
            ch, ch * 2, block_type="double",
            norm_type=norm_type, num_groups=num_groups, dropout_rate=dropout_rate,
        )
        ch = ch * 2

        # ── Decoder ──────────────────────────────────────────────────────────
        self.decoder_blocks = nn.ModuleList()
        for skip_ch in reversed(self.encoder_channels):
            out_ch = ch // 2
            self.decoder_blocks.append(
                UpBlock(ch, skip_ch, out_ch, bilinear=bilinear,
                        norm_type=norm_type, num_groups=num_groups,
                        dropout_rate=dropout_rate)
            )
            ch = out_ch

        # ── Segmentation head ─────────────────────────────────────────────
        self.head = nn.Conv2d(ch, num_classes, kernel_size=1)

        self._init_weights()

    # ------------------------------------------------------------------
    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.InstanceNorm2d)):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        skip_maps: list[torch.Tensor] = []
        x = self.stem(x)
        skip_maps.append(x)

        for enc in self.encoder_blocks:
            x = enc(x)
            skip_maps.append(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        for dec, skip in zip(self.decoder_blocks, reversed(skip_maps)):
            x = dec(x, skip)

        return self.head(x)

    # ------------------------------------------------------------------
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return (
            f"UNet(depth={self.depth}, "
            f"params={self.count_parameters():,})"
        )
