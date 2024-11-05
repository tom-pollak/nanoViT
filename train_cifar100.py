# %%
from typing import Literal
from dataclasses import dataclass, asdict
from tqdm import tqdm
import wandb

import torch as t
from torchvision import transforms
from datasets import load_dataset, DatasetDict

from nanovit import ViT, ViTConfig, build_preprocessor
from nanovit.schedule import cosine_schedule, linear_schedule

print = tqdm.external_write_mode()(print)  # tqdm friendly print

# %% ███████████████████████████████████  config  ███████████████████████████████████

vit_cfg = ViTConfig(
    n_layers=8,
    d_model=512,
    d_proj=100,  # 100 classes
    image_res=(32, 32),
    patch_size=4,
    n_heads=8,
    norm_data=(
        (0.48145466, 0.4578275, 0.40821073),
        (0.26862954, 0.26130258, 0.27577711),
    ),
    mlp_mult=4,
)
# pool_type: gap
# posemb: sincos2d

# bfloat16
# mixup: 0.2
# 99% train test split

# loss = "softmax_xent"

# inception crop(224) flip_lr randaug(2, 10)
# resize_small(256) central_crop(224)


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

# %% ███████████████████████████████████  model  ████████████████████████████████████

device = (
    "cuda"
    if t.cuda.is_available()
    else "mps"
    if t.backends.mps.is_available()
    else "cpu"
)

vit = ViT(vit_cfg).to(device)
vit.init_weights_()


opt = t.optim.AdamW(vit.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.wd)


# %% ███████████████████████████████  dataset & xfms  ███████████████████████████████

dd: DatasetDict = load_dataset("uoft-cs/cifar100")  # type: ignore
feats = dd["train"].features

img2tensor: transforms.Compose = build_preprocessor(vit_cfg)

train_xfms = transforms.Compose(
    [
        transforms.RandomResizedCrop(
            size=vit_cfg.image_res,
            scale=(0.9, 1.0),
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=2, magnitude=10),
        img2tensor,
    ]
)

valid_xfms = transforms.Compose([img2tensor])


# %% ██████████████████████████████████  training  ██████████████████████████████████

wandb.init(
    project="nanovit-cifar100",
    config={"vit_cfg": asdict(vit_cfg), "train_cfg": asdict(train_cfg)},
)


def single_step(batch: dict, preproc: transforms.Compose):
    images, labels = batch["img"], t.tensor(batch["fine_label"], device=device)  # type: ignore
    pixel_values = t.stack([preproc(im) for im in images]).to(device)
    with t.autocast(device_type=device):
        logits = vit(pixel_values)
        loss = t.nn.functional.cross_entropy(logits, labels)
    accuracy = (logits.argmax(dim=-1) == labels).float().mean()
    return loss, accuracy


sched = cosine_schedule if train_cfg.sched == "cosine_schedule" else linear_schedule
steps_per_epoch = len(dd["train"]) // train_cfg.bs
max_steps = train_cfg.nepochs * steps_per_epoch

for epoch in range(train_cfg.nepochs):
    train_dl = dd["train"].shuffle(seed=epoch).iter(train_cfg.bs, drop_last_batch=True)
    vit.train()
    pbar = tqdm(enumerate(train_dl), total=steps_per_epoch, leave=False)
    for step, batch in pbar:
        global_step = epoch * steps_per_epoch + step
        lr = sched(
            step=global_step,
            max_lr=train_cfg.lr,
            max_steps=max_steps,
            warmup_steps=train_cfg.warmup_steps,
        )
        opt.param_groups[0]["lr"] = lr
        wandb.log({"lr": lr}, step=global_step)

        loss, accuracy = single_step(batch, train_xfms)  # type: ignore
        wandb.log({"train_loss": loss, "train_acc": accuracy}, step=global_step)

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
            loss, accuracy = single_step(batch, valid_xfms)  # type: ignore
            val_pbar.set_postfix(
                {"epoch": epoch, "loss": f"{loss:.4f}", "acc": f"{accuracy:.4f}"}
            )
            val_pbar.update(1)
            losses.append(loss.item())
            accuracies.append(accuracy.item())

    val_loss = t.tensor(losses).mean()
    val_acc = t.tensor(accuracies).mean()
    wandb.log({"val_loss": val_loss, "val_acc": val_acc}, step=global_step)
    print(f"Epoch {epoch} loss: {val_loss:.4f}, acc: {val_acc:.4f}")
