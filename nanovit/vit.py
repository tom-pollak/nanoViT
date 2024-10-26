# fmt: off
import math
from dataclasses import dataclass
from functools import partial
import einops
import torch as t
import torch.nn as nn
from torchvision import transforms


@dataclass
class ViTConfig:
    n_layers: int
    d_model: int
    d_proj: int
    image_res: tuple[int, int]
    patch_size: int
    n_heads: int
    norm_data: tuple[tuple[float, float, float], tuple[float, float, float]] # (mean, std) to norm image

    mlp_mult: int = 4
    causal_attn: bool = False

    # Calculated in __post_init__
    d_head: int = None # type: ignore
    num_patches: tuple[int, int] = None # type: ignore
    seq_length: int = None # type: ignore

    def __post_init__(self):
        im_h, im_w = self.image_res
        assert im_h % self.patch_size == 0 and im_w % self.patch_size == 0
        self.num_patches = (im_h // self.patch_size, im_w // self.patch_size)
        self.seq_length = self.num_patches[0] * self.num_patches[1] + 1

        assert self.d_model % self.n_heads == 0
        self.d_head = self.d_model // self.n_heads


def build_preprocessor(cfg: ViTConfig) -> transforms.Compose:
    h, w = cfg.image_res
    mean, std = cfg.norm_data
    return transforms.Compose([
        transforms.Lambda(lambda im: im.convert("RGB")),
        transforms.Resize((h, w), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def quick_gelu(x):
    return x * t.sigmoid(1.702 * x)


# ███████████████████████████████████  Model  ████████████████████████████████████


class PatchEmbedding(nn.Module):
    """
    Fixed size image ViT

    [CLS] patch_1 patch_2
    """

    def __init__(self, cfg: ViTConfig):
        super().__init__()
        self.cfg = cfg
        self.class_embedding = nn.Parameter(t.empty(cfg.d_model))
        self.patch_embedding = nn.Parameter(t.empty(cfg.d_model, 3 * cfg.patch_size**2))
        self.position_embedding = nn.Parameter(t.empty(cfg.seq_length, cfg.d_model))

    def forward(self, pixel_values):
        patched_pixels = einops.rearrange(
            pixel_values,
            """ \
            batch channel (patch_h patch_size_h) (patch_w patch_size_w) \
         -> batch (patch_h patch_w) (channel patch_size_h patch_size_w) \
            """,
            patch_h=self.cfg.num_patches[0], patch_w=self.cfg.num_patches[1],
            patch_size_h=self.cfg.patch_size, patch_size_w=self.cfg.patch_size,
        )

        # batch, seq_length-1, d_model
        patch_embeds = einops.einsum(
            patched_pixels, self.patch_embedding,
            "batch seq_no_cls patch, d_model patch -> batch seq_no_cls d_model"
        )

        # [CLS] patch_1 patch_2... (batch, seq_length, d_model)
        class_embeds = einops.repeat(
            self.class_embedding,
            "d_model -> batch 1 d_model",
            batch=patch_embeds.shape[0],
        )
        embeddings = t.cat([class_embeds, patch_embeds], dim=1)

        embeddings = embeddings + self.position_embedding

        return embeddings


class Attention(nn.Module):
    def __init__(self, cfg: ViTConfig):
        super().__init__()
        self.cfg = cfg
        self.q_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=True)
        self.k_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=True)
        self.v_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=True)
        self.c_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=True)

    def forward(self, x):
        # batch seq d_model
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        head_tfm = partial(
            einops.rearrange,
            pattern="batch seq (head d_head) -> batch seq head d_head",
            head=self.cfg.n_heads,
            d_head=self.cfg.d_head,
        )
        # batch n_heads seq d_model
        q, k, v = head_tfm(q), head_tfm(k), head_tfm(v)
        attn = einops.einsum(
            q, k,
            """ \
            batch seq_q head d_head, \
            batch seq_k head d_head \
         -> batch head seq_q seq_k \
            """,
        ) / self.cfg.d_head**0.5

        # full attention -- no causal masking
        scores = t.nn.functional.softmax(attn, dim=-1)  # seq_k
        z = einops.einsum(
            scores, v,
            """ \
            batch head seq_q seq_k, \
            batch seq_k head d_head \
         -> batch seq_q head d_head \
            """,
        )
        z = einops.rearrange(z, "batch seq head d_head -> batch seq (head d_head)")
        out = self.c_proj(z)
        return out


class MLP(nn.Module):
    def __init__(self, cfg: ViTConfig):
        super().__init__()
        self.cfg = cfg
        self.c_fc = nn.Linear(cfg.d_model, cfg.d_model * cfg.mlp_mult, bias=True)
        self.c_proj = nn.Linear(cfg.d_model * cfg.mlp_mult, cfg.d_model, bias=True)

    def forward(self, x):
        h = self.c_fc(x)
        acts = quick_gelu(h)
        out = self.c_proj(acts)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ViTConfig):
        super().__init__()
        self.cfg = cfg
        self.attn = Attention(cfg)
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.mlp = MLP(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class ViT(nn.Module):
    def __init__(self, cfg: ViTConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = PatchEmbedding(cfg)
        self.pre_ln = nn.LayerNorm(cfg.d_model)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.post_ln = nn.LayerNorm(cfg.d_model)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_proj, bias=False)

    def forward(self, pixel_values):
        x = self.embed(pixel_values)
        x = self.pre_ln(x)
        for block in self.blocks:
            x = block(x)

        x = x[:, 0, :] # select [CLS] residual (index 0)
        x = self.post_ln(x)
        x = self.out_proj(x)
        return x

    def init_weights_(self):
        for name, param in self.named_parameters(recurse=True):
            if name.endswith("c_proj.weight"): # residual stream projection, GPT2 init
                t.nn.init.normal_(param, mean=0.0, std=0.02 / math.sqrt(2 * self.cfg.n_layers))
            elif isinstance(param, nn.Linear):
                t.nn.init.normal_(param.weight, mean=0.0, std=0.02)
                if param.bias is not None:
                    t.nn.init.zeros_(param.bias)
            elif isinstance(param, nn.Embedding) or name.endswith("embedding"):
                t.nn.init.normal_(param.weight if hasattr(param, "weight") else param, mean=0.0, std=0.02) # type: ignore
