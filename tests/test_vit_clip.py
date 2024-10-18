import pytest
from PIL import Image
from urllib.request import urlopen
import torch as t
from huggingface_hub import hf_hub_download
from sentence_transformers import SentenceTransformer

from nanoclip.vit import (
    ViT,
    build_preprocessor,
    PatchEmbeddings,
    Attention,
    MLP,
    TransformerBlock,
)
from nanoclip.weights.clip import clip_vit_b_16, CFG_VIT_B_16


@pytest.fixture(scope="module")
def sd():
    sd_path = hf_hub_download("openai/clip-vit-base-patch16", "pytorch_model.bin")
    sd = t.load(sd_path, map_location="cpu", weights_only=False)
    return sd


@pytest.fixture(scope="module")
def modules():
    model = SentenceTransformer("clip-ViT-B-16", device="cpu")
    return dict(model[0].named_modules())


@pytest.fixture(scope="module")
def img():
    img = Image.open(
        urlopen(
            "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png"
        )
    )
    return img


@pytest.fixture(scope="module")
def pixel_values():
    t.manual_seed(42)
    return t.randn(16, 3, *CFG_VIT_B_16.image_res)


@pytest.fixture(scope="module")
def residual():
    t.manual_seed(42)
    return t.randn(16, CFG_VIT_B_16.seq_length, CFG_VIT_B_16.d_model)


@t.no_grad()
def test_e2e(img):
    gt_clip = SentenceTransformer("clip-ViT-B-16")  # Uses hf impl

    vit: ViT = clip_vit_b_16()
    preproc = build_preprocessor(vit.cfg)
    emb = vit(preproc(img).unsqueeze(0)).squeeze(0)  # type: ignore
    gt_emb = t.tensor(gt_clip.encode(img))  # type: ignore
    assert t.allclose(emb, gt_emb, atol=1e-5)


@t.no_grad()
def test_patch_embedding(sd, modules, pixel_values):
    embed = PatchEmbeddings(CFG_VIT_B_16)
    embed.load_clip_weights_(sd)
    patch_emb_out = embed(pixel_values)

    gt_embed = modules["model.vision_model.embeddings"]
    patch_emb_out_gt = gt_embed(pixel_values)

    assert t.allclose(patch_emb_out, patch_emb_out_gt, atol=1e-5)


@t.no_grad()
def test_attn(sd, modules, residual):
    attn = Attention(CFG_VIT_B_16)
    attn.load_clip_weights_(sd, layer=0)
    attn_out = attn(residual)

    gt_attn = modules["model.vision_model.encoder.layers.0.self_attn"]
    gt_attn_out = gt_attn(residual)[0]
    assert t.allclose(attn_out, gt_attn_out, atol=1e-5)


@t.no_grad()
def test_mlp(sd, modules, residual):
    mlp = MLP(CFG_VIT_B_16)
    mlp.load_clip_weights_(sd, layer=0)
    mlp_out = mlp(residual)

    gt_mlp = modules["model.vision_model.encoder.layers.0.mlp"]
    gt_mlp_out = gt_mlp(residual)

    assert t.allclose(mlp_out, gt_mlp_out, atol=1e-8)


@t.no_grad()
def test_xfmer_block(sd, modules, residual):
    block = TransformerBlock(CFG_VIT_B_16)
    block.load_clip_weights_(sd, 0)
    block_out = block(residual)

    gt_block = modules["model.vision_model.encoder.layers.0"]
    attn_mask = t.full((197, 197), 1.0)
    gt_block_out = gt_block(residual, attn_mask, attn_mask)[0]

    assert t.allclose(block_out, gt_block_out, atol=1e-5)
