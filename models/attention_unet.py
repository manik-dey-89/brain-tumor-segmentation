"""
Attention U-Net (Oktay et al., 2018 – "Attention U-Net: Learning Where to
Look for the Pancreas")
=======================================================================
Extends standard U-Net by inserting Attention Gates between each encoder
skip connection and the corresponding decoder input.  The gates suppress
irrelevant feature activations and highlight salient regions (tumour areas).

Reference: https://arxiv.org/abs/1804.03999
"""

import torch
import torch.nn as nn
from models.blocks import DoubleConvBlock, DownBlock, UpBlock, AttentionGate


class AttentionUNet(nn.Module):
    """
    Attention U-Net with configurable depth, normalization and dropout.

    Parameters
    ----------
    in_channels  : number of input MRI channels
    num_classes  : 1 for binary tumour segmentation, >1 for multi-class
    base_filters : feature maps in the first encoder block (doubled per level)
    depth        : encoder / decoder levels (≥ 2)
    bilinear     : bilinear upsampling (True) vs. transposed convolution (False)
    norm_type    : "batch" | "group" | "instance"
    num_groups   : groups for GroupNorm
    dropout_rate : spatial dropout (Dropout2d) probability
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
        dropout_rate: float = 0.3,
    ) -> None:
        super().__init__()
        assert depth >= 2, "depth must be at least 2"

        self.depth = depth
        self.num_classes = num_classes

        # ── Encoder ──────────────────────────────────────────────────────────
        self.stem = DoubleConvBlock(
            in_channels, base_filters,
            norm_type=norm_type, num_groups=num_groups, dropout_rate=dropout_rate,
        )
        self.encoder_blocks = nn.ModuleList()
        ch = base_filters
        self.encoder_channels: list[int] = [ch]

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

        # ── Decoder + Attention Gates ─────────────────────────────────────
        self.decoder_blocks = nn.ModuleList()
        self.attention_gates = nn.ModuleList()

        for skip_ch in reversed(self.encoder_channels):
            # inter_ch for attention = skip_ch // 2 (at least 1)
            inter_ch = max(skip_ch // 2, 1)
            self.attention_gates.append(
                AttentionGate(g_ch=ch, x_ch=skip_ch, inter_ch=inter_ch)
            )
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
        # ── Encoder ──────────────────────────────────────────────────
        skip_maps: list[torch.Tensor] = []
        x = self.stem(x)
        skip_maps.append(x)

        for enc in self.encoder_blocks:
            x = enc(x)
            skip_maps.append(x)

        # ── Bottleneck ───────────────────────────────────────────────
        x = self.bottleneck(x)

        # ── Decoder with attention ───────────────────────────────────
        for attn, dec, skip in zip(
            self.attention_gates,
            self.decoder_blocks,
            reversed(skip_maps),
        ):
            attended_skip = attn(g=x, x=skip)   # gate: suppress irrelevant features
            x = dec(x, attended_skip)

        return self.head(x)

    # ------------------------------------------------------------------
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_attention_maps(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Forward pass that also returns intermediate attention maps for
        visualisation / interpretability.

        Returns
        -------
        logits       : (B, num_classes, H, W)
        attn_maps    : list of attention weight tensors (one per decoder level)
        """
        skip_maps: list[torch.Tensor] = []
        x = self.stem(x)
        skip_maps.append(x)

        for enc in self.encoder_blocks:
            x = enc(x)
            skip_maps.append(x)

        x = self.bottleneck(x)

        attn_maps: list[torch.Tensor] = []
        for attn, dec, skip in zip(
            self.attention_gates,
            self.decoder_blocks,
            reversed(skip_maps),
        ):
            # Compute psi (attention coefficient) manually
            import torch.nn.functional as F
            g1 = attn.W_g(
                F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
                if x.shape[2:] != skip.shape[2:] else x
            )
            x1 = attn.W_x(skip)
            psi_val = attn.psi(attn.relu(g1 + x1))
            attn_maps.append(psi_val.detach())

            attended_skip = attn(g=x, x=skip)
            x = dec(x, attended_skip)

        return self.head(x), attn_maps

    def __repr__(self) -> str:
        return (
            f"AttentionUNet(depth={self.depth}, "
            f"params={self.count_parameters():,})"
        )
