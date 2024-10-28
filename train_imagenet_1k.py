# %%
import torch as t
from torch import nn
from tqdm import tqdm
from datasets import load_dataset, DatasetDict

from nanovit import ViT, ViTConfig, build_preprocessor
from nanovit.schedule import cosine_schedule

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
# pool_type: gap
# posemb: sincos2d

nclasses = 1000
nepochs = 90
bs = 1024
lr = 1e-3
wd = 1e-4
grad_clip = 1.0
warmup_steps = 10_000
sched = cosine_schedule
# bfloat16
# mixup: 0.2
# 99% train test split

loss = "softmax_xent"

# inception crop(224) flip_lr randaug(2, 10)
# resize_small(256) central_crop(224)


val_bs = bs * 2

device = (
    "cuda"
    if t.cuda.is_available()
    else "mps"
    if t.backends.mps.is_available()
    else "cpu"
)

#  %%


# %%


