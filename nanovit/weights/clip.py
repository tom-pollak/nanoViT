from huggingface_hub import hf_hub_download

import torch as t
from nanovit.vit import (
    ViTConfig,
    PatchEmbedding,
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
    patch_size=16,
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
    load_vit_weights(vit, sd)
    return vit.eval()


def load_patch_embedding_weights(patch_embedding: PatchEmbedding, sd: dict):
    root_key = "vision_model.embeddings"
    patch_embedding.class_embedding.data = sd[f"{root_key}.class_embedding"].data
    patch_embedding.position_embedding.data = sd[
        f"{root_key}.position_embedding.weight"
    ].data
    patch_embedding.patch_embedding.data = (
        sd[f"{root_key}.patch_embedding.weight"]
        .reshape(patch_embedding.cfg.d_model, -1)
        .data
    )


def load_attention_weights(attention: Attention, sd: dict, layer: int):
    root_key = f"vision_model.encoder.layers.{layer}.self_attn"
    attention.q_proj.weight.data = sd[f"{root_key}.q_proj.weight"]
    attention.q_proj.bias.data = sd[f"{root_key}.q_proj.bias"]
    attention.k_proj.weight.data = sd[f"{root_key}.k_proj.weight"]
    attention.k_proj.bias.data = sd[f"{root_key}.k_proj.bias"]
    attention.v_proj.weight.data = sd[f"{root_key}.v_proj.weight"]
    attention.v_proj.bias.data = sd[f"{root_key}.v_proj.bias"]
    attention.c_proj.weight.data = sd[f"{root_key}.out_proj.weight"]
    attention.c_proj.bias.data = sd[f"{root_key}.out_proj.bias"]


def load_mlp_weights(mlp: MLP, sd: dict, layer: int):
    root_key = f"vision_model.encoder.layers.{layer}.mlp"
    mlp.c_fc.weight.data = sd[f"{root_key}.fc1.weight"]
    mlp.c_fc.bias.data = sd[f"{root_key}.fc1.bias"]
    mlp.c_proj.weight.data = sd[f"{root_key}.fc2.weight"]
    mlp.c_proj.bias.data = sd[f"{root_key}.fc2.bias"]


def load_transformer_block_weights(block: TransformerBlock, sd: dict, layer: int):
    root_key = f"vision_model.encoder.layers.{layer}"
    block.ln1.weight.data = sd[f"{root_key}.layer_norm1.weight"]
    block.ln1.bias.data = sd[f"{root_key}.layer_norm1.bias"]
    block.ln2.weight.data = sd[f"{root_key}.layer_norm2.weight"]
    block.ln2.bias.data = sd[f"{root_key}.layer_norm2.bias"]
    load_attention_weights(block.attn, sd, layer)
    load_mlp_weights(block.mlp, sd, layer)


def load_vit_weights(vit: ViT, sd: dict):
    root_key = "vision_model"
    vit.pre_ln.weight.data = sd[
        f"{root_key}.pre_layrnorm.weight"
    ]  # yes that is a spelling mistake
    vit.pre_ln.bias.data = sd[f"{root_key}.pre_layrnorm.bias"]
    vit.post_ln.weight.data = sd[f"{root_key}.post_layernorm.weight"]
    vit.post_ln.bias.data = sd[f"{root_key}.post_layernorm.bias"]
    vit.out_proj.weight.data = sd["visual_projection.weight"]

    load_patch_embedding_weights(vit.embed, sd)
    for i, block in enumerate(vit.blocks):
        load_transformer_block_weights(block, sd, layer=i)
