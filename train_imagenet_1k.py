# %%
import torch as t
from torch import nn
from tqdm import tqdm
from nanovit import ViT, ViTConfig, build_preprocessor
from datasets import load_dataset, DatasetDict

dd: DatasetDict = load_dataset(
    "microsoft/cats_vs_dogs", split="train"
).train_test_split(0.2)  # type: ignore
lbl_feat = dd["train"].features["labels"]

# %%


# ███████████████████████████████████  config  ███████████████████████████████████

vit_cfg = ViTConfig(
    n_layers=8,
    d_model=192,
    d_proj=2,  # 2 classes
    image_res=(224, 224),
    patch_size=16,
    n_heads=6,
    norm_data=(
        (0.48145466, 0.4578275, 0.40821073),
        (0.26862954, 0.26130258, 0.27577711),
    ),
    mlp_mult=4,
)

nepochs = 5
bs = 64
lr = 4e-4
wd = 1e-2

val_bs = bs * 2

device = (
    "cuda"
    if t.cuda.is_available()
    else "mps"
    if t.backends.mps.is_available()
    else "cpu"
)
