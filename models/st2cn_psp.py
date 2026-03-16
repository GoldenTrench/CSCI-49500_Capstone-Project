# ST2CN with PSPNet Pyramid Pooling Module
# Based on Lin & Zheng (2023), extended with PSPNet-style bottleneck
# (Zhao et al., 2017 - Pyramid Scene Parsing Network)
#
# Changes from st2cn.py:
#   - Bottleneck: 1x1 conv -> PyramidPoolingModule
#
# PyramidPoolingModule pools the bottleneck feature map at 4 scales,
# upsamples each back to the original size, concatenates with the original
# features, then reduces channels back with a 1x1 conv.
#
# At the bottleneck the spatial size is [B, 2048, 1, 3].
# Pool scales: (1,1), (1,2), (1,3) -- limited by the 1x3 spatial size.
# (A 4th scale of (1,1) duplicates itself so we use 3 meaningful scales
# plus the identity, giving 4 total branches.)
#
# Channel math:
#   Original:       2048
#   3 pool branches: 3 * 512 = 1536  (each branch reduces to 512 via 1x1)
#   Concat:         2048 + 1536 = 3584
#   Output conv:    3584 -> 4096
#
# Everything downstream (decoder, head) is unchanged from st2cn.py.

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


class PyramidPoolingModule(nn.Module):
    # Pools input at multiple scales, upsamples back, concats, reduces channels.
    # bin_sizes: list of (H, W) output sizes for each pooling branch.
    # At bottleneck spatial size 1x3, meaningful sizes are (1,1), (1,2), (1,3).
    def __init__(self, in_channels, bin_sizes=((1, 1), (1, 2), (1, 3))):
        super().__init__()
        # Each branch reduces channels by 4x before upsampling to save memory
        branch_ch = in_channels // 4

        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=size),
                nn.Conv2d(in_channels, branch_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(branch_ch),
                nn.ReLU(inplace=True),
            )
            for size in bin_sizes
        ])

        # Fuse: original features + all branches -> out_channels
        fuse_in  = in_channels + branch_ch * len(bin_sizes)
        out_channels = in_channels * 2  # match baseline bottleneck output (4096)
        self.fuse = nn.Sequential(
            nn.Conv2d(fuse_in, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        h, w = x.shape[2], x.shape[3]
        parts = [x]
        for branch in self.branches:
            pooled = branch(x)
            # Upsample back to original spatial size
            pooled = F.interpolate(pooled, size=(h, w),
                                   mode='bilinear', align_corners=False)
            parts.append(pooled)
        return self.fuse(torch.cat(parts, dim=1))


class ST2CN_PSP(nn.Module):
    """
    ST2CN with PSPNet pyramid pooling at the bottleneck.
    Same interface as ST2CN: input [B, 4, 256, 768], output [B, NUM_CLASSES, 1, 768].
    """

    def __init__(self, in_channels=4, num_classes=NUM_CLASSES, base_filters=16):
        super().__init__()
        self.num_classes = num_classes

        f = base_filters
        enc_channels = [in_channels] + [f * (2 ** i) for i in range(8)]
        # [4, 16, 32, 64, 128, 256, 512, 1024, 2048]

        self.encoders = nn.ModuleList([
            EncoderBlock(enc_channels[i], enc_channels[i + 1])
            for i in range(8)
        ])

        # PSP replaces the baseline 1x1 conv bottleneck
        # Input:  [B, 2048, 1, 3]
        # Output: [B, 4096, 1, 3]  (same shape as baseline)
        self.bottleneck = PyramidPoolingModule(
            in_channels=enc_channels[-1],   # 2048
            bin_sizes=((1, 1), (1, 2), (1, 3)),
        )

        bottleneck_out_ch = enc_channels[-1] * 2  # 4096

        dec_blocks = []
        x_ch = bottleneck_out_ch
        for i in range(8):
            skip_ch = enc_channels[8 - i]
            out_ch  = enc_channels[8 - i]
            dec_blocks.append(DecoderBlock(x_ch, skip_ch, out_ch))
            x_ch = out_ch
        self.decoders = nn.ModuleList(dec_blocks)

        self.head = nn.Conv2d(x_ch, num_classes, kernel_size=1)

    def forward(self, x):
        # x: [B, 4, 256, 768]
        skips = []
        for enc in self.encoders:
            x, skip = enc(x)
            skips.append(skip)

        x = self.bottleneck(x)  # [B, 4096, 1, 3]

        for i, dec in enumerate(self.decoders):
            x = dec(x, skips[-(i + 1)])

        return self.head(x)     # [B, NUM_CLASSES, 1, 768]


class ST2CNLoss(nn.Module):
    def __init__(self, class_weights=None):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=class_weights, ignore_index=255)

    def forward(self, logits, targets):
        return self.ce(logits.squeeze(2), targets)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = ST2CN_PSP(in_channels=4, num_classes=NUM_CLASSES).to(device)
    dummy  = torch.randn(2, 4, 256, 768).to(device)

    with torch.no_grad():
        out = model(dummy)

    assert out.shape == (2, NUM_CLASSES, 1, 768), f"Bad output shape: {out.shape}"
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Output shape: {out.shape}  |  Params: {n_params:,}")

    from models.st2cn import ST2CN
    baseline = ST2CN(in_channels=4, num_classes=NUM_CLASSES)
    n_base   = sum(p.numel() for p in baseline.parameters() if p.requires_grad)
    print(f"Baseline params: {n_base:,}  |  PSP params: {n_params:,}")
    print("Sanity check passed")




























