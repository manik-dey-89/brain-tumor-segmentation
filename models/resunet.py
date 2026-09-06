"""
ResUNet – Residual U-Net
========================
Replaces every double-conv block in the classic U-Net with a Residual Block
(Zhang et al., 2018 – "Road Extraction by Deep Residual U-Net").

Optionally uses a pretrained ResNet encoder (resnet18 / resnet34 / resnet50)
from torchvision when `use_pretrained_encoder=True`.

Reference: https://arxiv.org/abs/1711.10684
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from models.blocks import ResidualBlock, ConvBnAct, UpBlock, _norm_layer


# Scratch-trained ResUNet (no external backbone dependency)

class ResUNet(nn.Module):
    """
    Residual U-Net with configurable depth, normalization and dropout.

    Parameters
    ----------
    in_channels          : input MRI channels
    num_classes          : output segmentation classes
    base_filters         : channels in first encoder residual block
    depth                : encoder / decoder levels (= 2)
    bilinear             : bilinear vs transposed-conv upsampling
    norm_type            : "batch" | "group" | "instance"
    num_groups           : GroupNorm groups
    dropout_rate         : spatial dropout probability
    use_pretrained_encoder: use torchvision ResNet as encoder backbone
    pretrained_encoder   : "resnet18" | "resnet34" | "resnet50"
    freeze_encoder       : freeze pretrained encoder weights
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
        use_pretrained_encoder: bool = False,
        pretrained_encoder: str = "resnet34",
        freeze_encoder: bool = False,
    ) -> None:
        super().__init__()
        assert depth >= 2

        self.depth = depth
        self.num_classes = num_classes
        self.use_pretrained_encoder = use_pretrained_encoder

        if use_pretrained_encoder:
            self._build_pretrained_encoder(
                in_channels, pretrained_encoder, freeze_encoder, num_classes,
                bilinear, norm_type, num_groups, dropout_rate,
            )
        else:
            self._build_scratch_encoder(
                in_channels, base_filters, depth,
                bilinear, norm_type, num_groups, dropout_rate, num_classes,
            )

        self._init_weights()

    # ------------------------------------------------------------------
    # Scratch encoder/decoder
    # ------------------------------------------------------------------

    def _build_scratch_encoder(
        self,
        in_ch: int,
        base_filters: int,
        depth: int,
        bilinear: bool,
        norm_type: str,
        num_groups: int,
        dropout_rate: float,
        num_classes: int,
    ) -> None:
        # stem
        self.stem = ResidualBlock(
            in_ch, base_filters, norm_type=norm_type, num_groups=num_groups,
            dropout_rate=dropout_rate,
        )
        ch = base_filters
        self.encoder_channels: list[int] = [ch]

        self.encoder_blocks = nn.ModuleList()
        for _ in range(depth - 1):
            out_ch = ch * 2
            self.encoder_blocks.append(
                nn.Sequential(
                    nn.MaxPool2d(2),
                    ResidualBlock(ch, out_ch, norm_type=norm_type,
                                  num_groups=num_groups, dropout_rate=dropout_rate),
                )
            )
            ch = out_ch
            self.encoder_channels.append(ch)

        # bottleneck
        self.bottleneck = nn.Sequential(
            nn.MaxPool2d(2),
            ResidualBlock(ch, ch * 2, norm_type=norm_type,
                          num_groups=num_groups, dropout_rate=dropout_rate),
        )
        ch = ch * 2

        # decoder
        self.decoder_blocks = nn.ModuleList()
        for skip_ch in reversed(self.encoder_channels):
            out_ch = ch // 2
            self.decoder_blocks.append(
                UpBlock(ch, skip_ch, out_ch, bilinear=bilinear,
                        norm_type=norm_type, num_groups=num_groups,
                        dropout_rate=dropout_rate)
            )
            ch = out_ch

        self.head = nn.Conv2d(ch, num_classes, kernel_size=1)

    # ------------------------------------------------------------------
    # Pretrained ResNet encoder
    # ------------------------------------------------------------------

    def _build_pretrained_encoder(
        self,
        in_ch: int,
        backbone_name: str,
        freeze: bool,
        num_classes: int,
        bilinear: bool,
        norm_type: str,
        num_groups: int,
        dropout_rate: float,
    ) -> None:
        try:
            import torchvision.models as tvm
        except ImportError as exc:
            raise ImportError("torchvision is required for pretrained encoders.") from exc

        backbone_fn = getattr(tvm, backbone_name, None)
        if backbone_fn is None:
            raise ValueError(f"Unknown backbone: {backbone_name}")

        backbone = backbone_fn(weights="DEFAULT")

        # Patch first conv if input channels ? 3
        if in_ch != 3:
            orig = backbone.layer0[0] if hasattr(backbone, "layer0") else backbone.conv1
            new_conv = nn.Conv2d(
                in_ch, orig.out_channels, orig.kernel_size,  # type: ignore[arg-type]
                orig.stride, orig.padding, bias=False,
            )
            if in_ch == 1:
                new_conv.weight.data = orig.weight.data.mean(dim=1, keepdim=True)
            if hasattr(backbone, "layer0"):
                backbone.layer0[0] = new_conv
            else:
                backbone.conv1 = new_conv

        if freeze:
            for p in backbone.parameters():
                p.requires_grad = False

        # Extract encoder feature stages
        if backbone_name in ("resnet18", "resnet34"):
            enc_channels = [64, 64, 128, 256, 512]
        else:  # resnet50 / resnet101
            enc_channels = [64, 256, 512, 1024, 2048]

        self.enc0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.pool0 = backbone.maxpool
        self.enc1 = backbone.layer1
        self.enc2 = backbone.layer2
        self.enc3 = backbone.layer3
        self.enc4 = backbone.layer4  # bottleneck

        self.encoder_channels = enc_channels[:-1]

        # decoder
        ch = enc_channels[-1]
        self.decoder_blocks = nn.ModuleList()
        for skip_ch in reversed(enc_channels[:-1]):
            out_ch = ch // 2
            self.decoder_blocks.append(
                UpBlock(ch, skip_ch, out_ch, bilinear=bilinear,
                        norm_type=norm_type, num_groups=num_groups,
                        dropout_rate=dropout_rate)
            )
            ch = out_ch

        self.head = nn.Conv2d(ch, num_classes, kernel_size=1)

    # ------------------------------------------------------------------
    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d) and not self.use_pretrained_encoder:
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_pretrained_encoder:
            return self._forward_pretrained(x)
        return self._forward_scratch(x)

    def _forward_scratch(self, x: torch.Tensor) -> torch.Tensor:
        skip_maps: list[torch.Tensor] = []
        x = self.stem(x)
        skip_maps.append(x)

        for enc in self.encoder_blocks:
            x = enc(x)
            skip_maps.append(x)

        x = self.bottleneck(x)

        for dec, skip in zip(self.decoder_blocks, reversed(skip_maps)):
            x = dec(x, skip)

        return self.head(x)

    def _forward_pretrained(self, x: torch.Tensor) -> torch.Tensor:
        e0 = self.enc0(x)            # stride 2
        e0p = self.pool0(e0)         # stride 4
        e1 = self.enc1(e0p)          # stride 4
        e2 = self.enc2(e1)           # stride 8
        e3 = self.enc3(e2)           # stride 16
        e4 = self.enc4(e3)           # stride 32  <- bottleneck

        skips = [e0, e1, e2, e3]

        x = e4
        for dec, skip in zip(self.decoder_blocks, reversed(skips)):
            x = dec(x, skip)

        # Upsample to original resolution if needed
        x = F.interpolate(x, size=(x.shape[2] * 2, x.shape[3] * 2),
                          mode="bilinear", align_corners=False)
        return self.head(x)

    # ------------------------------------------------------------------
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return (
            f"ResUNet(depth={self.depth}, pretrained={self.use_pretrained_encoder}, "
            f"params={self.count_parameters():,})"
        )
