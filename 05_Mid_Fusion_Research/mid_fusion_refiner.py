import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1):
        super().__init__()
        p = k // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, s, p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class TinyBackbone(nn.Module):
    """
    Lightweight feature extractor for one modality.
    Input patch: [B, 3, H, W]
    Output feat: [B, C, H/8, W/8]
    """
    def __init__(self, in_ch: int = 3, base_ch: int = 32):
        super().__init__()
        self.stem = ConvBlock(in_ch, base_ch, 3, 2)
        self.stage1 = ConvBlock(base_ch, base_ch * 2, 3, 2)
        self.stage2 = ConvBlock(base_ch * 2, base_ch * 4, 3, 2)
        self.out_ch = base_ch * 4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        return x


class MidFusionRefiner(nn.Module):
    """
    Mid-level fusion refiner:
    - Extract visible and lwir features separately
    - Fuse at feature level (concat + 1x1 + 3x3)
    - Predict refined objectness score in [0,1]
    """
    def __init__(self, base_ch: int = 32):
        super().__init__()
        self.vis_backbone = TinyBackbone(3, base_ch)
        self.ir_backbone = TinyBackbone(3, base_ch)
        c = self.vis_backbone.out_ch

        self.fusion = nn.Sequential(
            ConvBlock(c * 2, c, k=1, s=1),
            ConvBlock(c, c, k=3, s=1),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c, c // 2),
            nn.SiLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(c // 2, 1),
        )

    def forward(self, vis_patch: torch.Tensor, ir_patch: torch.Tensor) -> torch.Tensor:
        fv = self.vis_backbone(vis_patch)
        fi = self.ir_backbone(ir_patch)
        f = self.fusion(torch.cat([fv, fi], dim=1))
        z = self.pool(f)
        logits = self.head(z)
        return logits

