# ST2CN: Spatial-Temporal 2D CNN for Vehicle Interaction Recognition
# Based on Lin & Zheng (2023), IEEE ITSC 2023
#
# Input:  [B, 4, 256, 768] — RGB motion profile (3ch) + vehicle width trace (1ch)
# Output: [B, NUM_CLASSES, 1, 768] — class prediction at the last temporal row only
#
# Architecture: U-Net encoder-decoder, 8+8 blocks, skip connections
# Using 3x3 conv with pad=1 (preserves spatial dims — paper uses 2x2 but
# that causes dimension drift without careful padding)

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# 15 interaction classes from VMVI 2025 Table 1
# (paper had 13 — new classes are off_road and stopping split from ego stop/turning)
INTERACTION_CLASSES = [
    "cut_in",           #  0  CI   (255, 140,   0)
    "lane_change",      #  1  CL   (255, 165,   0)
    "front_approach",   #  2  FA   (255,   0,   0)
    "front_leaving",    #  3  FL   (240, 128, 128)
    "front_following",  #  4  FF   (255,  99,  71)
    "parallel_next",    #  5  PN   (255, 255,   0)
    "passing",          #  6  PS   (255, 215,   0)
    "being_passed",     #  7  PD   (218, 165,  32)
    "merging",          #  8  M    ( 60, 179, 113)
    "opposite",         #  9  O    (128,   0, 128)
    "crossing",         # 10  C    (  0,   0, 255)
    "turning_away",     # 11  TW   (  0, 255, 255)
    "off_road",         # 12  OR   (255, 255, 255)
    "stopping",         # 13  ST   (128, 128, 128)
    "background",       # 14  BG   (  0,   0,   0)
]

NUM_CLASSES = len(INTERACTION_CLASSES)  # 15


class EncoderBlock(nn.Module):
    # Conv -> BN -> ReLU -> MaxPool
    # Returns (pooled output, skip connection before pooling)
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.bn   = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x    = self.relu(self.bn(self.conv(x)))
        skip = x
        x    = self.pool(x)
        return x, skip


class DecoderBlock(nn.Module):
    # Upsample -> concat skip -> Conv -> BN -> ReLU
    # Takes explicit in_ch + skip_ch because bottleneck doubles channels,
    # so skip_ch != in_ch at the first decoder block
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch + skip_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.bn   = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, skip):
        # Upsample to match skip width; height stays at 1 in the decoder
        x    = F.interpolate(x, size=(1, skip.shape[3]), mode='bilinear', align_corners=False)
        skip = skip[:, :, -1:, :]          # take last (most recent) temporal row
        x    = torch.cat([x, skip], dim=1)
        x    = self.relu(self.bn(self.conv(x)))
        return x


class ST2CN(nn.Module):
    """
    U-Net style encoder-decoder for vehicle interaction recognition.

    Spatial dimension trace (base_filters=16, input 256x768):
        enc[0]: 256x768 -> pool -> 128x384
        enc[1]: 128x384 -> pool ->  64x192
        enc[2]:  64x192 -> pool ->  32x96
        enc[3]:  32x96  -> pool ->  16x48
        enc[4]:  16x48  -> pool ->   8x24
        enc[5]:   8x24  -> pool ->   4x12
        enc[6]:   4x12  -> pool ->   2x6
        enc[7]:   2x6   -> pool ->   1x3
        bottleneck:                  1x3
        dec[0..7]: upsample+cat back to 1x768
        head:                        1x768 -> [B, 15, 1, 768]
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

        bottleneck_ch = enc_channels[-1] * 2  # 2048 -> 4096
        self.bottleneck = nn.Sequential(
            nn.Conv2d(enc_channels[-1], bottleneck_ch, kernel_size=1),
            nn.BatchNorm2d(bottleneck_ch),
            nn.ReLU(inplace=True),
        )

        dec_blocks = []
        x_ch = bottleneck_ch
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
    # Cross-entropy on the last temporal line only
    def __init__(self, class_weights=None):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=class_weights, ignore_index=255)

    def forward(self, logits, targets):
        # logits: [B, C, 1, W] -> [B, C, W]; targets: [B, W]
        return self.ce(logits.squeeze(2), targets)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = ST2CN(in_channels=4, num_classes=NUM_CLASSES).to(device)
    dummy  = torch.randn(2, 4, 256, 768).to(device)

    with torch.no_grad():
        out = model(dummy)

    assert out.shape == (2, NUM_CLASSES, 1, 768), f"Bad output shape: {out.shape}"
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Output shape: {out.shape}  |  Params: {n_params:,}")
    print("Sanity check passed")





