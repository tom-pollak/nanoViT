# fmt: off
# %%
import einops
from dataclasses import dataclass
from functools import partial
import torch as t
from torch import nn, Tensor
import torch.nn.functional as F
from jaxtyping import Float
from huggingface_hub import hf_hub_download
from fastcore.all import patch

# %%

from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('clip-ViT-B-16')

# %%
sd_path = hf_hub_download(
    "sentence-transformers/clip-ViT-B-16",
    "0_CLIPModel/pytorch_model.bin",
    local_files_only=True,
)
sd = t.load(sd_path, map_location="cpu", weights_only=False)
for name in sd.keys():
    if (".layers" in name and ".0" not in name):# or "mlp" not in name:
        continue
    print(name)

# %%
size = (224, 224)
mean, std =

# %%


@dataclass
class ViTConfig:
    n_layers: int
    d_model: int
    d_proj: int
    image_res: tuple[int, int]
    patch_size: tuple[int, int]
    n_heads: int
    d_head: int
    norm_data: tuple[tuple[int, int, int], tuple[int, int, int]] # mean std to norm image

    mlp_mult: int = 4
    causal_attn: bool = False

    # Calculated in __post_init__
    num_patches: int = None
    num_positions: int = None

    def __post_init__(self):
        im_h, im_w = self.image_res
        npatches_h, npatches_w = self.patch_size
        assert im_h % npatches_h == 0 and im_w % npatches_w == 0

        self.num_patches = (im_h // npatches_h) * (im_w // npatches_w)
        self.num_positions = self.num_patches + 1


cfg_vit_b_16 = ViTConfig(
    n_layers=12,
    d_model=768,
    d_proj=512,
    image_res=(224, 224),
    patch_size=(16, 16),
    n_heads=24,
    d_head=32,
    norm_data=([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]),
    mlp_mult=4,
)

pixel_values = t.randn(16, 3, *cfg_vit_b_16.image_res)
residual = t.randn(16, cfg_vit_b_16.num_positions, cfg_vit_b_16.d_model)
# %%


class PatchEmbeddings(nn.Module):
    """
    Fixed size image ViT

    [CLS] patch_1 patch_2
    """

    def __init__(self, cfg: ViTConfig):
        super().__init__()
        self.cfg = cfg
        im_h, im_w = cfg.image_res

        self.class_embedding = nn.Parameter(t.empty(cfg.d_model))
        self.patch_embedding = nn.Linear(
            3 * cfg.patch_size[0] * cfg.patch_size[1], cfg.d_model, bias=False
        )
        self.position_embedding = nn.Parameter(t.empty(cfg.num_positions, cfg.d_model))

    def forward(self, pixel_values):
        "B C H W"
        patched_pixels = einops.rearrange(
            pixel_values,
            "b c (h p1) (w p2) -> b (h w) (c p1 p2)",
            p1=self.cfg.patch_size[0],
            p2=self.cfg.patch_size[1],
        )

        # batch, num_patches-1, d_model
        patch_embeds = self.patch_embedding(patched_pixels)
        # [CLS] patch_1 patch_2... (batch, num_patches, d_model)
        class_embeds = self.class_embedding.expand(pixel_values.shape[0], 1, -1)
        embeddings = t.cat([class_embeds, patch_embeds], dim=1)

        embeddings = embeddings + self.position_embedding

        return embeddings


@patch
def load_clip_weights(self: PatchEmbeddings, sd: dict):  # type: ignore
    root_key = "vision_model.embeddings"
    self.class_embedding.data = sd[f"{root_key}.class_embedding"].data

    # (d_model, channels, patch_size, patch_size => channels * patch_size**2, d_model)
    self.patch_embedding.weight.data = (
        sd[f"{root_key}.patch_embedding.weight"].reshape(self.cfg.d_model, -1).T.data
    )
    self.position_embedding.data = sd[f"{root_key}.position_embedding.weight"].data


embed = PatchEmbeddings(cfg_vit_b_16)
embed.load_clip_weights(sd)
embed(pixel_values).shape

# %%


class MultiHeadAttention(nn.Module):
    def __init__(self, cfg: ViTConfig):
        super().__init__()
        self.cfg = cfg
        self.q_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=True)
        self.k_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=True)
        self.v_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=True)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=True)

    def forward(self, x):
        # batch seq d_model
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        head_tfm = partial(
            einops.rearrange,
            pattern="batch seq (n_heads d_head) -> batch seq n_heads d_head",
            n_heads=self.cfg.n_heads,
            d_head=self.cfg.d_head,
        )
        # batch seq n_heads d_model
        q, k, v = head_tfm(q), head_tfm(k), head_tfm(v)
        attn = einops.einsum(
            q, k,
            """ \
            batch seq_q n_heads d_head, \
            batch seq_k n_heads d_head \
            -> batch n_heads seq_q seq_k \
            """,
        ) / self.cfg.d_head**0.5

        # full attention -- no causal masking
        scores = F.softmax(attn, dim=-1)  # seq_k
        z = einops.einsum(
            scores,
            v,
            """ \
            batch n_heads seq_q seq_k, \
            batch seq_k n_heads d_head \
            -> batch seq_q n_heads d_head \
            """,
        )
        z = einops.rearrange(z, "batch seq n_heads d_head -> batch seq (n_heads d_head)")
        up_proj = self.out_proj(z)
        return up_proj


@patch
def load_clip_weights(  # noqa: F811
    self: MultiHeadAttention,
    sd: dict,
    layer: int,
):
    root_key = f"vision_model.encoder.layers.{layer}.self_attn"
    self.q_proj.weight.data = sd[f"{root_key}.q_proj.weight"]
    self.q_proj.bias.data = sd[f"{root_key}.q_proj.bias"]
    self.k_proj.weight.data = sd[f"{root_key}.k_proj.weight"]
    self.k_proj.bias.data = sd[f"{root_key}.k_proj.bias"]
    self.v_proj.weight.data = sd[f"{root_key}.v_proj.weight"]
    self.v_proj.bias.data = sd[f"{root_key}.v_proj.bias"]
    self.out_proj.weight.data = sd[f"{root_key}.out_proj.weight"]
    self.out_proj.bias.data = sd[f"{root_key}.out_proj.bias"]


attn = MultiHeadAttention(cfg_vit_b_16)
attn.load_clip_weights(sd, layer=0)
attn(residual).shape

# %%


class MLP(nn.Module):
    def __init__(self, cfg: ViTConfig):
        super().__init__()
        self.cfg = cfg
        self.up_proj = nn.Linear(cfg.d_model, cfg.d_model * cfg.mlp_mult, bias=True)
        self.down_proj = nn.Linear(cfg.d_model * cfg.mlp_mult, cfg.d_model, bias=True)

    def forward(self, x):
        h = self.up_proj(x)
        acts = F.relu(h)
        x_p = self.down_proj(acts)
        return x_p


@patch
def load_clip_weights(  # noqa: F811
    self: MLP,
    sd: dict,
    layer: int,
):
    root_key = f"vision_model.encoder.layers.{layer}.mlp"
    self.up_proj.weight.data = sd[f"{root_key}.fc1.weight"]
    self.up_proj.bias.data = sd[f"{root_key}.fc1.bias"]
    self.down_proj.weight.data = sd[f"{root_key}.fc2.weight"]
    self.down_proj.bias.data = sd[f"{root_key}.fc2.bias"]

mlp = MLP(cfg_vit_b_16)
mlp.load_clip_weights(sd, layer=0)
mlp(residual).shape

# %%


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ViTConfig):
        super().__init__()
        self.attn = MultiHeadAttention(cfg)
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.mlp = MLP(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)

    def forward(self, x):
        x = x + self.ln1(self.attn(x))
        x = x + self.ln2(self.mlp(x))
        return x


@patch
def load_clip_weights(  # noqa: F811
    self: TransformerBlock,
    sd: dict,
    layer: int,
):
    root_key = f"vision_model.encoder.layers.{layer}"
    self.ln1.weight.data = sd[f"{root_key}.layer_norm1.weight"]
    self.ln1.bias.data = sd[f"{root_key}.layer_norm1.bias"]
    self.ln2.weight.data = sd[f"{root_key}.layer_norm2.weight"]
    self.ln2.bias.data = sd[f"{root_key}.layer_norm2.bias"]
    self.attn.load_clip_weights(sd, layer)
    self.mlp.load_clip_weights(sd, layer)

block = TransformerBlock(cfg_vit_b_16)
block.load_clip_weights(sd, 0)
block(residual).shape

# %%

class ViT(nn.Module):
    def __init__(self, cfg: ViTConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = PatchEmbeddings(cfg)
        self.pre_ln = nn.LayerNorm(cfg.d_model)
        self.blocks = nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg.n_layers)]
        )
        self.post_ln = nn.LayerNorm(cfg.d_model)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_proj, bias=False)

    def forward(self, pixel_values):
        x = self.embed(pixel_values)
        x = self.pre_ln(x)
        for block in self.blocks:
            x = block(x)
        x = self.post_ln(x)
        x = self.out_proj(x)
        return x

@patch
def load_clip_weights(  # noqa: F811
    self: ViT,
    sd: dict,
):
    root_key = "vision_model"
    self.pre_ln.weight.data = sd[f"{root_key}.pre_layrnorm.weight"] # yes that is a spelling mistake
    self.pre_ln.bias.data = sd[f"{root_key}.pre_layrnorm.bias"]
    self.post_ln.weight.data = sd[f"{root_key}.post_layernorm.weight"]
    self.post_ln.bias.data = sd[f"{root_key}.post_layernorm.bias"]
    self.out_proj.weight.data = sd["visual_projection.weight"]

    self.embed.load_clip_weights(sd)
    for i, block in enumerate(self.blocks):
        block.load_clip_weights(sd, layer=i)


vit = ViT(cfg_vit_b_16)
vit.load_clip_weights(sd)
vit(pixel_values).shape

# %%




# %%

