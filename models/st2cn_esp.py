# ST2CN with ESP (Efficient Spatial Pyramid) Bottleneck
# Based on Lin & Zheng (2023), extended with ESP-style bottleneck
#
# Motivation:
#   PSPNet's adaptive pooling collapses at the 1×3 bottleneck —
#   pooling scales (1,1), (1,2), (1,3) produce near-identical branches,
#   giving no meaningful multi-scale context.
#
#   The ESP module (Mehta et al., ECCV 2018) achieves multi-scale context
#   via parallel dilated convolutions rather than pooling, making it
#   resolution-agnostic and effective even at 1×3.
#
# ESP module design (adapted for height=1 bottleneck):
#   - Pointwise 1×1 conv reduces channels: 2048 → d per branch
#   - K=4 parallel dilated convs, kernel (1×3), rates 1, 2, 4, 8
#     → effective receptive field widths: 3, 5, 9, 17
#   - Hierarchical Feature Fusion (HFF) removes gridding artifacts
#   - Concat branches → out_channels, then residual add
#
# Channel math (K=4, d=out_channels//K):
#   in:      2048
#   reduced: 512  (per branch, 4 branches)
#   concat:  4096 (= baseline bottleneck output, decoder unchanged)
#
# References:
#   Mehta et al. (2018) ESPNet, ECCV 2018
#   Lin & Zheng (2023) ST2CN, IEEE ITSC 2023, DOI: 10.1109/ITSC57777.2023.10421903

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.st2cn import (
    INTERACTION_CLASSES, NUM_CLASSES,
    EncoderBlock, DecoderBlock,
)


class ESPModule(nn.Module):
    """
    Efficient Spatial Pyramid (ESP) module adapted for asymmetric feature maps.

    Replaces PSPNet's adaptive average pooling with K parallel dilated
    convolutions, each seeing a different effective receptive field.
    Works at any spatial resolution — including height=1.

    Key differences from the paper's original ESP:
      - Kernel size (1, 3) instead of (3, 3): handles height=1 bottleneck.
      - Dilation applied to width dimension only: (1, 2^k) for k=0..K-1.
      - padding=(0, 2^k) preserves spatial size at each dilation rate.

    HFF (Hierarchical Feature Fusion):
      Each branch accumulates from the previous before concatenation,
      suppressing the checkerboard/gridding artifact that arises from
      parallel dilated convolutions.

    Args:
        in_channels  (int): input channels (2048 at bottleneck).
        out_channels (int): output channels (4096 to match baseline decoder).
        K            (int): number of parallel branches (default 4).
    """

    def __init__(self, in_channels: int, out_channels: int, K: int = 4):
        super().__init__()
        assert out_channels % K == 0, "out_channels must be divisible by K"
        d = out_channels // K   # channels per branch after reduce

        # ── Step 1: pointwise reduction ────────────────────────────────────
        # Projects M-dim feature maps to d-dim before spatial convolutions.
        # Reduces computation while preserving channel information.
        self.reduce = nn.Sequential(
            nn.Conv2d(in_channels, d, kernel_size=1, bias=False),
            nn.BatchNorm2d(d),
            nn.PReLU(d),
        )

        # ── Step 2: K parallel dilated convolutions ─────────────────────────
        # kernel (1,3): height-1 safe. padding=(0, rate) maintains spatial size.
        # Dilation rates: 1, 2, 4, 8 → ERF widths: 3, 5, 9, 17
        self.dilated_convs = nn.ModuleList([
            nn.Conv2d(
                d, d,
                kernel_size=(1, 3),
                dilation=(1, 2 ** k),
                padding=(0, 2 ** k),
                bias=False,
            )
            for k in range(K)
        ])

        # ── Step 3: BN + activation after HFF concat ───────────────────────
        self.bn_act = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.PReLU(out_channels),
        )

        # ── Skip connection ─────────────────────────────────────────────────
        # Projects input to out_channels for the residual add.
        if in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, in_channels, H, W]  (at bottleneck: [B, 2048, 1, 3])

        reduced = self.reduce(x)   # [B, d, H, W]

        # Parallel dilated convs with HFF accumulation
        branches = []
        for i, dconv in enumerate(self.dilated_convs):
            y = dconv(reduced)
            if i > 0:
                y = y + branches[i - 1]   # HFF: accumulate to de-grid
            branches.append(y)

        # Concat all K branches → [B, K*d, H, W] = [B, out_channels, H, W]
        out = torch.cat(branches, dim=1)
        out = self.bn_act(out)

        # Residual add (element-wise sum with skip-projected input)
        out = out + self.skip(x)
        return out


class ST2CN_ESP(nn.Module):
    """
    ST2CN with ESP bottleneck.

    Replaces the degenerate PSPNet pooling module at the 1×3 bottleneck
    with an Efficient Spatial Pyramid (ESP) module that uses parallel
    dilated convolutions instead of adaptive pooling.

    Interface identical to ST2CN and ST2CN_PSP:
        input  [B, in_channels, 256, 768]
        output [B, num_classes,   1, 768]
    """

    def __init__(self, in_channels: int = 4,
                 num_classes: int = NUM_CLASSES,
                 base_filters: int = 16):
        super().__init__()
        self.num_classes = num_classes

        f = base_filters
        enc_channels = [in_channels] + [f * (2 ** i) for i in range(8)]
        # [4, 16, 32, 64, 128, 256, 512, 1024, 2048]

        # ── Encoder (unchanged from baseline) ──────────────────────────────
        self.encoders = nn.ModuleList([
            EncoderBlock(enc_channels[i], enc_channels[i + 1])
            for i in range(8)
        ])

        # ── ESP Bottleneck ──────────────────────────────────────────────────
        # Input:  [B, 2048, 1, 3]
        # Output: [B, 4096, 1, 3]  (same shape as baseline / PSP bottleneck)
        bottleneck_in_ch  = enc_channels[-1]       # 2048
        bottleneck_out_ch = enc_channels[-1] * 2   # 4096

        self.bottleneck = ESPModule(
            in_channels=bottleneck_in_ch,
            out_channels=bottleneck_out_ch,
            K=4,
        )

        # ── Decoder (unchanged from baseline) ──────────────────────────────
        dec_blocks = []
        x_ch = bottleneck_out_ch
        for i in range(8):
            skip_ch = enc_channels[8 - i]
            out_ch  = enc_channels[8 - i]
            dec_blocks.append(DecoderBlock(x_ch, skip_ch, out_ch))
            x_ch = out_ch
        self.decoders = nn.ModuleList(dec_blocks)

        # ── Segmentation head (unchanged) ───────────────────────────────────
        self.head = nn.Conv2d(x_ch, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, in_channels, 256, 768]
        skips = []
        for enc in self.encoders:
            x, skip = enc(x)
            skips.append(skip)
        # x: [B, 2048, 1, 3]

        x = self.bottleneck(x)   # [B, 4096, 1, 3]

        for i, dec in enumerate(self.decoders):
            x = dec(x, skips[-(i + 1)])

        return self.head(x)      # [B, NUM_CLASSES, 1, 768]


class ST2CNLoss(nn.Module):
    def __init__(self, class_weights=None):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=class_weights, ignore_index=255)

    def forward(self, logits, targets):
        return self.ce(logits.squeeze(2), targets)


# ── Sanity check ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = ST2CN_ESP(in_channels=4, num_classes=NUM_CLASSES).to(device)
    dummy  = torch.randn(2, 4, 256, 768).to(device)

    with torch.no_grad():
        out = model(dummy)

    assert out.shape == (2, NUM_CLASSES, 1, 768), f"Bad output shape: {out.shape}"

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Output shape : {out.shape}")
    print(f"ESP params   : {n_params:,}")

    from models.st2cn import ST2CN
    n_base = sum(p.numel() for p in ST2CN(4, NUM_CLASSES).parameters()
                 if p.requires_grad)
    print(f"Baseline params: {n_base:,}")

    from models.st2cn_psp import ST2CN_PSP
    n_psp = sum(p.numel() for p in ST2CN_PSP(4, NUM_CLASSES).parameters()
                if p.requires_grad)
    print(f"PSP params   : {n_psp:,}")

    print("Sanity check passed")