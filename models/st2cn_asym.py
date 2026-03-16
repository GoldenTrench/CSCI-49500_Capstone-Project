# ST2CN with Asymmetric Pooling
# Variant of ST2CN that only pools the spatial (width) axis at each encoder block,
# preserving the full temporal dimension throughout the network.
#
# Motivation: the professor pointed out that pooling both axes equally discards
# temporal information aggressively. Since the temporal axis encodes *when*
# an interaction is happening, preserving it should help the network learn
# temporal patterns more accurately.
#
# Changes from st2cn.py:
#   - EncoderBlock: MaxPool(2,2) -> MaxPool(1,2) — spatial only
#   - DecoderBlock: upsample to full skip shape (not just width)
#   - Forward: take last temporal row at the head, not in decoder
#
# Dimension trace (base_filters=16, input 256x768):
#   enc[0]: 256x768 -> pool -> 256x384
#   enc[1]: 256x384 -> pool -> 256x192
#   enc[2]: 256x192 -> pool -> 256x96
#   enc[3]: 256x96  -> pool -> 256x48
#   enc[4]: 256x48  -> pool -> 256x24
#   enc[5]: 256x24  -> pool -> 256x12
#   enc[6]: 256x12  -> pool -> 256x6
#   enc[7]: 256x6   -> pool -> 256x3
#   bottleneck:                256x3
#   dec[0..7]: upsample spatially back to 256x768
#   head: take last row -> [B, 15, 1, 768]
#
# Memory note: feature maps are 256x larger in temporal dim vs symmetric version.
# Should fit in 32GB but watch AMP usage.

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from models.st2cn import INTERACTION_CLASSES, NUM_CLASSES


class EncoderBlock(nn.Module):
    # Conv -> BN -> ReLU -> MaxPool(1x2)
    # Pools spatial axis only — temporal stays intact
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.bn   = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))

    def forward(self, x):
        x    = self.relu(self.bn(self.conv(x)))
        skip = x
        x    = self.pool(x)
        return x, skip


class DecoderBlock(nn.Module):
    # Upsample spatially -> concat full skip -> Conv -> BN -> ReLU
    # Skip is the full [B, ch, T, W] feature map (temporal not collapsed)
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch + skip_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.bn   = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, skip):
        # Upsample to match full skip shape (T stays the same, W doubles)
        x = F.interpolate(x, size=(skip.shape[2], skip.shape[3]),
                          mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.relu(self.bn(self.conv(x)))
        return x


class ST2CN_Asym(nn.Module):
    """
    ST2CN with asymmetric pooling — temporal axis preserved throughout.
    Same interface as ST2CN: input [B, 4, 256, 768], output [B, NUM_CLASSES, 1, 768].
    """

    def __init__(self, in_channels=4, num_classes=NUM_CLASSES, base_filters=16):
        super().__init__()
        self.num_classes = num_classes

        f = base_filters
        enc_channels = [in_channels] + [f * (2 ** i) for i in range(8)]

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

        x = self.bottleneck(x)  # [B, 4096, 256, 3]

        for i, dec in enumerate(self.decoders):
            x = dec(x, skips[-(i + 1)])
        # x: [B, 16, 256, 768]

        logits = self.head(x)           # [B, NUM_CLASSES, 256, 768]
        return logits[:, :, -1:, :]     # [B, NUM_CLASSES, 1, 768] — last temporal row


class ST2CNLoss(nn.Module):
    def __init__(self, class_weights=None):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=class_weights, ignore_index=255)

    def forward(self, logits, targets):
        return self.ce(logits.squeeze(2), targets)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = ST2CN_Asym(in_channels=4, num_classes=NUM_CLASSES).to(device)
    dummy  = torch.randn(2, 4, 256, 768).to(device)

    with torch.no_grad():
        out = model(dummy)

    assert out.shape == (2, NUM_CLASSES, 1, 768), f"Bad output shape: {out.shape}"
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Output shape: {out.shape}  |  Params: {n_params:,}")

    # Compare param count to baseline
    from models.st2cn import ST2CN
    baseline = ST2CN(in_channels=4, num_classes=NUM_CLASSES)
    n_base   = sum(p.numel() for p in baseline.parameters() if p.requires_grad)
    print(f"Baseline params: {n_base:,}  |  Asym params: {n_params:,}")
    print("Sanity check passed")
























