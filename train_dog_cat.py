# %%
import torch as t
from torch import nn
from tqdm import tqdm
from nanoclip import ViT, ViTConfig, build_preprocessor
from datasets import load_dataset, DatasetDict

dd: DatasetDict = load_dataset(
    "microsoft/cats_vs_dogs", split="train"
).train_test_split(0.2)  # type: ignore
lbl_feat = dd["train"].features["labels"]

# ███████████████████████████████████  params  ███████████████████████████████████

vit_cfg = ViTConfig(
    n_layers=8,
    d_model=192,
    d_proj=2,  # 2 classes
    image_res=(224, 224),
    patch_size=(32, 32),
    n_heads=12,
    norm_data=(
        (0.48145466, 0.4578275, 0.40821073),
        (0.26862954, 0.26130258, 0.27577711),
    ),
    mlp_mult=4,
)

nepochs = 5
bs = 64
lr = 1e-3
wd = 1e-2

val_bs = bs * 2

device = (
    "cuda"
    if t.cuda.is_available()
    else "mps"
    if t.backends.mps.is_available()
    else "cpu"
)


# ███████████████████████████████████  setup  ████████████████████████████████████

from nanoclip.lsuv import LSUV_

LSUV_()

# def init_weights_(vit: ViT):
#     for name, param in vit.named_parameters(recurse=True):
#         if param.requires_grad:
#             if "bias" in name:
#                 init_func = nn.init.zeros_
#             elif "class_embedding" in name:
#                 init_func = nn.init.normal_
#             elif "ln" in name:
#                 init_func = None
#             else:
#                 init_func = nn.init.kaiming_normal_

#             if init_func is not None:
#                 init_func(param)


vit = ViT(vit_cfg).to(device)
init_weights_(vit)
preproc = build_preprocessor(vit_cfg)

opt = t.optim.AdamW(vit.parameters(), lr=lr, weight_decay=wd)


# ███████████████████████████████████  train  ████████████████████████████████████

for epoch in range(nepochs):
    # Train
    train_dl = dd["train"].shuffle(epoch).iter(bs)
    vit.train()
    pbar = tqdm(enumerate(train_dl), total=len(dd["train"]) // bs, leave=False)
    for step, batch in pbar:
        images, labels = batch["image"], batch["labels"]  # type: ignore
        pixel_values = t.stack([preproc(im) for im in images]).to(device)
        logits = vit(pixel_values)
        loss = t.nn.functional.cross_entropy(logits, t.tensor(labels, device=device))

        opt.zero_grad()
        loss.backward()
        opt.step()
        pbar.set_postfix({"epoch": epoch, "loss": f"{loss:.4f}"})
        pbar.update(1)

    # Test
    valid_dl = dd["test"].iter(val_bs)
    val_pbar = tqdm(enumerate(valid_dl), total=len(dd["test"]) // val_bs, leave=False)
    vit.eval()
    with t.no_grad():
        losses = []
        accuracies = []
        for step, batch in val_pbar:
            images, labels = batch["image"], batch["labels"]  # type: ignore
            pixel_values = t.stack([preproc(im) for im in images]).to(device)
            logits = vit(pixel_values)
            loss = t.nn.functional.cross_entropy(
                logits, t.tensor(labels, device=device)
            )
            preds = t.argmax(logits, dim=1)
            accuracy = (preds == t.tensor(labels, device=device)).float().mean().item()
            losses.append(loss)
            accuracies.append(accuracy)
            pbar.set_postfix(
                {"epoch": epoch, "loss": f"{loss:.4f}", "accuracy": f"{accuracy:.4f}"}
            )
            val_pbar.update(1)
        print(
            f"Epoch: {epoch} | Loss: {t.tensor(losses).mean():.4f} | Accuracy: {t.tensor(accuracies).mean():.4f}"
        )
