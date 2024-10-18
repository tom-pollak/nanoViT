# %%
# fmt: off
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

img
# %%
sd_path = hf_hub_download(
    "openai/clip-vit-base-patch16",
    "pytorch_model.bin"
)

sd = t.load(sd_path, map_location="cpu", weights_only=False)
for name in sd.keys():
    if "vision" in name:
        print(name, tuple(sd[name].shape))

# %%
# import clip
# model, preprocess = clip.load("ViT-B/16", device="cpu")

modules = dict(raw_model.named_modules())
for name, mod in modules.items():
    # if "model.vision_model.encoder.layers.0" in name:
    print(name)

# %%


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
residual = t.randn(16, cfg_vit_b_16.seq_length, cfg_vit_b_16.d_model)
# %%


@patch
def load_clip_weights(self: PatchEmbeddings, sd: dict):  # type: ignore
    root_key = "vision_model.embeddings"
    self.class_embedding.data = sd[f"{root_key}.class_embedding"].data
    self.position_embedding.data = sd[f"{root_key}.position_embedding.weight"].data
    # (d_model, channels, patch_size, patch_size => d_model, channels * patch_size**2)
    self.patch_embedding.data = sd[f"{root_key}.patch_embedding.weight"].reshape(self.cfg.d_model, -1).data



embed = PatchEmbeddings(cfg_vit_b_16)
embed.load_clip_weights(sd)
patch_emb_out = embed(pixel_values)

gt_embed = modules['model.vision_model.embeddings'].cpu()
patch_emb_out_gt = gt_embed(pixel_values)

print(patch_emb_out.shape)
print(patch_emb_out_gt.shape)
t.allclose(patch_emb_out, patch_emb_out_gt, atol=1e-5)

# %%


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
