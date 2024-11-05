# %%
import wandb
import torch as t
from torch import nn
from tqdm import tqdm
from dataclasses import dataclass, asdict, field
from typing import Literal
from datasets import load_dataset, DatasetDict

from nanovit import ViT, ViTConfig, build_preprocessor
from nanovit.schedule import cosine_schedule

print = tqdm.external_write_mode()(print)  # tqdm friendly print

# %%

# ███████████████████████████████████  config  ███████████████████████████████████

vit_cfg = ViTConfig(
    n_layers=8,
    d_model=192,
    d_proj=100,  # 100 classes
    image_res=(32, 32),
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


@dataclass
class TrainConfig:
    nclasses: int = 1000
    nepochs: int = 90
    bs: int = 1024
    val_bs: int = 2048
    lr: float = 1e-3
    wd: float = 1e-4
    grad_clip: float = 1.0
    warmup_steps: int = 10_000
    sched: Literal["cosine_schedule", "linear_schedule"] = "cosine_schedule"


train_cfg = TrainConfig()

# %%
# bfloat16
# mixup: 0.2
# 99% train test split

# loss = "softmax_xent"

# inception crop(224) flip_lr randaug(2, 10)
# resize_small(256) central_crop(224)


device = (
    "cuda"
    if t.cuda.is_available()
    else "mps"
    if t.backends.mps.is_available()
    else "cpu"
)

vit = ViT(vit_cfg).to(device)

preproc = build_preprocessor(vit_cfg)

opt = t.optim.AdamW(vit.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.wd)

# %%

dd: DatasetDict = load_dataset("uoft-cs/cifar100")  # type: ignore
feats = dd["train"].features


# %%

wandb.init(
    project="nanovit-cifar100",
    config={"vit_cfg": asdict(vit_cfg), "train_cfg": asdict(train_cfg)},
)


def single_step(batch: dict):
    images, labels = batch["image"], batch["fine_label"]  # type: ignore
    pixel_values = t.stack([preproc(im) for im in images]).to(device)
    logits = vit(pixel_values)
    loss = t.nn.functional.cross_entropy(logits, t.tensor(labels, device=device))
    accuracy = (logits.argmax(dim=-1) == labels).float().mean()
    return loss, accuracy


for epoch in range(train_cfg.nepochs):
    train_dl = dd["train"].shuffle(seed=epoch).iter(train_cfg.bs)
    vit.train()
    pbar = tqdm(
        enumerate(train_dl), total=len(dd["train"]) // train_cfg.bs, leave=False
    )
    for step, batch in pbar:
        loss, accuracy = single_step(batch)  # type: ignore
        wandb.log({"train_loss": loss, "train_acc": accuracy})

        opt.zero_grad()
        loss.backward()
        opt.step()
        pbar.set_postfix(
            {"epoch": epoch, "loss": f"{loss:.4f}", "acc": f"{accuracy:.4f}"}
        )
        pbar.update(1)

    valid_dl = dd["test"].iter(train_cfg.val_bs)
    val_pbar = tqdm(
        enumerate(valid_dl), total=len(dd["test"]) // train_cfg.val_bs, leave=False
    )
    vit.eval()
    losses, accuracies = [], []
    with t.no_grad():
        for step, batch in val_pbar:
            loss, accuracy = single_step(batch)  # type: ignore
            val_pbar.set_postfix(
                {"epoch": epoch, "loss": f"{loss:.4f}", "acc": f"{accuracy:.4f}"}
            )
            val_pbar.update(1)
            losses.append(loss.item())
            accuracies.append(accuracy.item())

    val_loss = t.tensor(losses).mean()
    val_acc = t.tensor(accuracies).mean()
    wandb.log({"val_loss": val_loss, "val_acc": val_acc})
    print(f"Epoch {epoch} loss: {val_loss:.4f}, acc: {val_acc:.4f}")
