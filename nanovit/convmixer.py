# %%
from dataclasses import dataclass
from torch import nn
import torch as t
import torch.nn.functional as F
from torch import Tensor
import einops
from jaxtyping import Float


@dataclass
class ConvMixerConfig:
    n_layers: int
    d_model: int
    d_proj: int
    kernel_size: int
    patch_size: int


class PatchEmbedding(nn.Module):
    def __init__(self, cfg: ConvMixerConfig):
        self.patch_embedding = nn.Parameter(t.empty(cfg.d_model, 3 * cfg.patch_size**2))
        # alternatively:
        # self.patch_conv = nn.Conv2d(
        #     3, cfg.d_model, kernel_size=cfg.patch_size, stride=cfg.patch_size
        # )

        self.bn = nn.BatchNorm2d(cfg.d_model)

    def forward(
        self, pixel_values: Float[Tensor, "batch channels height width"]
    ) -> Float[Tensor, "batch num_patch_h num_patch_w d_model"]:
        # Only difference with vit:
        # height, width patches are 2D rather than 1D with positional embedding
        # aka num_patch_h num_patch_w rather than (num_patch_h num_patch_w)
        patched_pixels = einops.rearrange(
            pixel_values,
            """ \
            batch channel (num_patch_h patch_size_h) (num_patch_w patch_size_w) \
         -> batch num_patch_h num_patch_w (channel patch_size_h patch_size_w) \
            """,
            num_patch_h=self.cfg.num_patches[0],
            num_patch_w=self.cfg.num_patches[1],
            patch_size_h=self.cfg.patch_size,
            patch_size_w=self.cfg.patch_size,
        )

        patch_embeds = einops.einsum(
            patched_pixels,
            self.patch_embedding,
            "batch num_patch_h num_patch_w patch, d_model patch -> batch num_patch_h num_patch_w d_model",
        )
        # we also have a non-linearity after embedding
        patch_embeds = F.gelu(patch_embeds)
        patch_embeds = self.bn(patch_embeds)
        return patch_embeds


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
        self.embed = PatchEmbedding(cfg)
        self.blocks = nn.Sequential(*[MixerBlock(cfg) for _ in range(cfg.n_layers)])
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_proj)

    def forward(self, pixel_values):
        x = self.embed(pixel_values)
        x = self.blocks(x)
        x = x.mean(dim=(-2, -1)).flatten()
        x = self.out_proj(x)
        return x
