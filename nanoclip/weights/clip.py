from huggingface_hub import hf_hub_download
from fastcore.all import patch
import torch as t
from nanoclip.vit import (
    ViTConfig,
    PatchEmbeddings,
    Attention,
    MLP,
    TransformerBlock,
    ViT,
)

__all__ = ["CFG_VIT_B_16", "clip_vit_b_16"]

CFG_VIT_B_16 = ViTConfig(
    n_layers=12,
    d_model=768,
    d_proj=512,
    image_res=(224, 224),
    patch_size=(16, 16),
    n_heads=12,
    norm_data=(
        (0.48145466, 0.4578275, 0.40821073),
        (0.26862954, 0.26130258, 0.27577711),
    ),
    mlp_mult=4,
)


def clip_vit_b_16():
    sd_path = hf_hub_download("openai/clip-vit-base-patch16", "pytorch_model.bin")
    sd = t.load(sd_path, map_location="cpu", weights_only=False)
    vit = ViT(CFG_VIT_B_16)
    vit.load_clip_weights_(sd)
    return vit.eval()


@patch
def load_clip_weights_(self: PatchEmbeddings, sd: dict):  # type: ignore # noqa: F811
    root_key = "vision_model.embeddings"
    self.class_embedding.data = sd[f"{root_key}.class_embedding"].data
    self.position_embedding.data = sd[f"{root_key}.position_embedding.weight"].data
    # (d_model, channels, patch_size, patch_size => d_model, channels * patch_size**2)
    self.patch_embedding.data = (
        sd[f"{root_key}.patch_embedding.weight"].reshape(self.cfg.d_model, -1).data
    )


@patch
def load_clip_weights_(  # type: ignore # noqa: F811
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


@patch
def load_clip_weights_(  # type: ignore # noqa: F811
    self: MLP,
    sd: dict,
    layer: int,
):
    root_key = f"vision_model.encoder.layers.{layer}.mlp"
    self.up_proj.weight.data = sd[f"{root_key}.fc1.weight"]
    self.up_proj.bias.data = sd[f"{root_key}.fc1.bias"]
    self.down_proj.weight.data = sd[f"{root_key}.fc2.weight"]
    self.down_proj.bias.data = sd[f"{root_key}.fc2.bias"]


@patch
def load_clip_weights_(  # type: ignore # noqa: F811
    self: TransformerBlock,
    sd: dict,
    layer: int,
):
    root_key = f"vision_model.encoder.layers.{layer}"
    self.ln1.weight.data = sd[f"{root_key}.layer_norm1.weight"]
    self.ln1.bias.data = sd[f"{root_key}.layer_norm1.bias"]
    self.ln2.weight.data = sd[f"{root_key}.layer_norm2.weight"]
    self.ln2.bias.data = sd[f"{root_key}.layer_norm2.bias"]
    self.attn.load_clip_weights_(sd, layer)
    self.mlp.load_clip_weights_(sd, layer)


@patch
def load_clip_weights_(  # type: ignore # noqa: F811
    self: ViT,
    sd: dict,
):
    root_key = "vision_model"
    self.pre_ln.weight.data = sd[
        f"{root_key}.pre_layrnorm.weight"
    ]  # yes that is a spelling mistake
    self.pre_ln.bias.data = sd[f"{root_key}.pre_layrnorm.bias"]
    self.post_ln.weight.data = sd[f"{root_key}.post_layernorm.weight"]
    self.post_ln.bias.data = sd[f"{root_key}.post_layernorm.bias"]
    self.out_proj.weight.data = sd["visual_projection.weight"]

    self.embed.load_clip_weights_(sd)
    for i, block in enumerate(self.blocks):
        block.load_clip_weights_(sd, layer=i)
