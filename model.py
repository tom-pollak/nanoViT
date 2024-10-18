# %%
# fmt: off

# %%
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
raw_model = model[0]

# %%

from PIL import Image
from urllib.request import urlopen

img = Image.open(urlopen(
    'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png'
))


# %%
# sd_path = hf_hub_download(
#     "sentence-transformers/clip-ViT-B-16",
#     "0_CLIPModel/pytorch_model.bin",
#     local_files_only=True,
# )
sd_path = hf_hub_download(
    "openai/clip-vit-base-patch16",
    "pytorch_model.bin"
)

sd = t.load(sd_path, map_location="cpu", weights_only=False)
for name in sd.keys():
    # if "vision_model.encoder.layers.0.self_attn" in name:
    # if (".layers" in name and ".0" not in name):# or "mlp" not in name:
    #     continue
    if "vision" in name:
        print(name, sd[name].shape)

# %%
# import clip
# model, preprocess = clip.load("ViT-B/16", device="cpu")

modules = dict(model.named_modules())
for name, mod in modules.items():
    # if "model.vision_model.encoder.layers.0" in name:
    print(name)

# %%


@dataclass
class ViTConfig:
    n_layers: int
    d_model: int
    d_proj: int
    image_res: tuple[int, int]
    patch_size: tuple[int, int]
    n_heads: int
    norm_data: tuple[tuple[int, int, int], tuple[int, int, int]] # mean std to norm image

    mlp_mult: int = 4
    causal_attn: bool = False

    # Calculated in __post_init__
    d_head: int = None
    num_patches: int = None
    num_positions: int = None

    def __post_init__(self):
        im_h, im_w = self.image_res
        npatches_h, npatches_w = self.patch_size
        assert im_h % npatches_h == 0 and im_w % npatches_w == 0
        self.num_patches = (im_h // npatches_h) * (im_w // npatches_w)
        self.num_positions = self.num_patches + 1

        assert self.d_model % self.n_heads == 0
        self.d_head = self.d_model // self.n_heads


cfg_vit_b_16 = ViTConfig(
    n_layers=12,
    d_model=768,
    d_proj=512,
    image_res=(224, 224),
    patch_size=(16, 16),
    n_heads=12,
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
        sd[f"{root_key}.patch_embedding.weight"].reshape(self.cfg.d_model, -1).data
    )
    self.position_embedding.data = sd[f"{root_key}.position_embedding.weight"].data


embed = PatchEmbeddings(cfg_vit_b_16)
embed.load_clip_weights(sd)
patch_emb_out = embed(pixel_values)

gt_embed = modules['model.vision_model.embeddings'].cpu()
patch_emb_out_gt = gt_embed(pixel_values)

print(patch_emb_out.shape)
print(patch_emb_out_gt.shape)
t.allclose(patch_emb_out, patch_emb_out_gt, atol=1e-5)

# %%


class Attention(nn.Module):
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
        # batch n_heads seq d_model
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
    self: Attention,
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


attn = Attention(cfg_vit_b_16)
attn.load_clip_weights(sd, layer=0)
attn_out = attn(residual)
print(attn_out.shape)

gt_attn_out = modules['model.vision_model.encoder.layers.0.self_attn'].cpu()(residual)[0]
print(gt_attn_out.shape)
print(t.allclose(attn_out, gt_attn_out, atol=1e-5))

# %%

def quick_gelu(x):
    return x * t.sigmoid(1.702 * x)

class MLP(nn.Module):
    def __init__(self, cfg: ViTConfig):
        super().__init__()
        self.cfg = cfg
        self.up_proj = nn.Linear(cfg.d_model, cfg.d_model * cfg.mlp_mult, bias=True)
        self.down_proj = nn.Linear(cfg.d_model * cfg.mlp_mult, cfg.d_model, bias=True)

    def forward(self, x):
        h = self.up_proj(x)
        acts = quick_gelu(h)
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
mlp_out = mlp(residual)

gt_mlp_out = modules['model.vision_model.encoder.layers.0.mlp'].cpu()(residual)
print(t.allclose(mlp_out, gt_mlp_out, atol=1e-8))

# %%


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ViTConfig):
        super().__init__()
        self.attn = Attention(cfg)
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.mlp = MLP(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
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
block_out = block(residual)


gt_block = modules["model.vision_model.encoder.layers.0"].cpu()
attn_mask = t.full((197, 197,), 1.)
gt_block_out = gt_block(residual, attn_mask, attn_mask)[0]
print(t.allclose(block_out, gt_block_out, atol=1e-6))


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

        x = x[:, 0, :] # select [CLS] residual (index 0)
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


vit = ViT(cfg_vit_b_16).eval()
vit.load_clip_weights(sd)

# %%


# %%
from torchvision import transforms

def build_preproc(cfg: ViTConfig):
    h, w = cfg.image_res
    mean, std = cfg.norm_data
    return transforms.Compose([
        transforms.Resize((h, w), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

preproc = build_preproc(cfg_vit_b_16)
pixel_values = preproc(img).unsqueeze(0)

pixel_values_gt = preprocess(img).unsqueeze(0)
t.allclose(pixel_values, pixel_values_gt)

# %%

with t.no_grad():
    emb = vit(pixel_values).squeeze(0)
    gt_emb = model.encode_image(pixel_values).squeeze(0)
print(pixel_values.shape)
print(emb.shape)

# gt_emb = t.tensor(model.encode(img))
# print(gt_emb.shape)

print(t.allclose(emb, gt_emb, atol=1e-5))
# %%

# %%

