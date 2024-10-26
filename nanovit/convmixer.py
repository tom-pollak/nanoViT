# %%
from dataclasses import dataclass
from torch import nn
import torch.nn.functional as F


@dataclass
class ConvMixerConfig:
    n_layers: int
    d_model: int
    d_proj: int
    kernel_size: int
    patch_size: int


class PatchEmbed(nn.Module):
    def __init__(self, cfg: ConvMixerConfig):
        # alternatively do it with a linear layer & einops like in vit.py
        self.patch_conv = nn.Conv2d(
            3, cfg.d_model, kernel_size=cfg.patch_size, stride=cfg.patch_size
        )
        self.bn = nn.BatchNorm2d(cfg.d_model)

    def forward(self, pixel_values):
        embed = F.gelu(self.patch_conv(pixel_values))
        embed = self.bn(embed)
        return embed


class MixerBlock(nn.Module):
    def __init__(self, cfg: ConvMixerConfig):
        self.cfg = cfg
        self.depth_path = nn.Sequential(
            nn.Conv2d(
                cfg.d_model,
                cfg.d_model,
                kernel_size=cfg.kernel_size,
                groups=cfg.d_model,
                padding="same",
            ),
            nn.GELU(),
            nn.BatchNorm2d(cfg.d_model),
        )
        self.point_path = nn.Sequential(
            nn.Conv2d(cfg.d_model, cfg.d_model, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm2d(cfg.d_model),
        )

    def forward(self, x):
        # resid(depth_path) -> point_path
        x = x + self.depth_path(x)
        x = self.point_path(x)
        return x


class ConvMixer(nn.Module):
    def __init__(self, cfg: ConvMixerConfig):
        self.cfg = cfg
        self.embed = PatchEmbed(cfg)
        self.blocks = nn.Sequential(*[MixerBlock(cfg) for _ in range(cfg.n_layers)])
        self.out_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_proj)

    def forward(self, pixel_values):
        x = self.embed(pixel_values)
        x = self.blocks(x)
        x = self.out_pool(x).flatten()
        x = self.out_proj(x)
        return x
