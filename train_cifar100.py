# %%
from typing import Literal
from dataclasses import dataclass, asdict
from tqdm import tqdm
import wandb

import torch as t
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import v2  # type: ignore
from datasets import load_dataset, DatasetDict

from nanovit import ViT, ViTConfig, build_preprocessor
from nanovit.schedule import cosine_schedule, linear_schedule

print = tqdm.external_write_mode()(print)  # tqdm friendly print

num_workers = 10

device = (
    "cuda"
    if t.cuda.is_available()
    else "mps"
    if t.backends.mps.is_available()
    else "cpu"
)


# %% ███████████████████████████████████  config  ███████████████████████████████████


@dataclass
class TrainConfig:
    nepochs: int = 200
    nclasses: int = 100
    bs: int = 256
    val_bs: int = 512
    lr: float = 1e-3
    wd: float = 3e-2
    grad_clip: float = 1.0
    warmup_steps: int = 5_000
    sched: Literal["cosine_schedule", "linear_schedule"] = "cosine_schedule"


train_cfg = TrainConfig()


vit_cfg = ViTConfig(
    n_layers=24,
    d_model=512,
    d_proj=train_cfg.nclasses,
    image_res=(32, 32),
    patch_size=4,
    n_heads=8,
    dropout=0.1,
    norm_data=(
        (0.48145466, 0.4578275, 0.40821073),
        (0.26862954, 0.26130258, 0.27577711),
    ),
    mlp_mult=4,
)


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
        # transforms.RandAugment(num_ops=2, magnitude=10),
        transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10),
        img2tensor,
    ]
)


valid_xfms = transforms.Compose([img2tensor])

mixup_or_cutmix = transforms.RandomChoice(
    [
        transforms.v2.MixUp(num_classes=train_cfg.nclasses),  # type: ignore
        transforms.v2.CutMix(num_classes=train_cfg.nclasses),  # type: ignore
    ]
)


def train_collate_fn(batch: list[dict]):
    pixel_values = t.stack([train_xfms(x["img"]) for x in batch])
    labels = t.tensor([x["fine_label"] for x in batch])
    pixel_values, labels = mixup_or_cutmix(pixel_values, labels)
    return pixel_values, labels


def valid_collate_fn(batch: list[dict]):
    pixel_values = t.stack([valid_xfms(x["img"]) for x in batch])
    labels = t.tensor([x["fine_label"] for x in batch])
    return pixel_values, labels


train_dl = DataLoader(
    dd["train"],  # type: ignore
    batch_size=train_cfg.bs,
    shuffle=True,
    drop_last=True,
    num_workers=num_workers,
    collate_fn=train_collate_fn,
)

valid_dl = DataLoader(
    dd["test"],  # type: ignore
    batch_size=train_cfg.val_bs,
    shuffle=False,
    num_workers=num_workers,
    collate_fn=valid_collate_fn,
)


# %% ████████████████████████████████  init & train  ████████████████████████████████

wandb.init(
    project="nanovit-cifar100",
    config={"vit_cfg": asdict(vit_cfg), "train_cfg": asdict(train_cfg)},
)

vit = ViT(vit_cfg).to(device)
vit.init_weights_()
opt = t.optim.AdamW(vit.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.wd)


def single_step(batch):
    pixel_values, labels = batch
    pixel_values, labels = pixel_values.to(device), labels.to(device)
    with t.autocast(device_type=device):
        logits = vit(pixel_values)
        loss = t.nn.functional.cross_entropy(logits, labels)

    if labels.ndim != 1:  # mixup/cutmix
        labels = labels.argmax(dim=-1)

    accuracy = (logits.argmax(dim=-1) == labels).float().mean()
    return loss, accuracy


sched = cosine_schedule if train_cfg.sched == "cosine_schedule" else linear_schedule
steps_per_epoch = len(dd["train"]) // train_cfg.bs
max_steps = train_cfg.nepochs * steps_per_epoch

for epoch in tqdm(range(train_cfg.nepochs)):
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

        loss, accuracy = single_step(batch)  # type: ignore
        wandb.log({"train_loss": loss, "train_acc": accuracy}, step=global_step)

        opt.zero_grad()
        loss.backward()
        t.nn.utils.clip_grad_norm_(vit.parameters(), train_cfg.grad_clip)
        opt.step()
        pbar.set_postfix(
            {"epoch": epoch, "loss": f"{loss:.4f}", "acc": f"{accuracy:.4f}"}
        )

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
            losses.append(loss.item())
            accuracies.append(accuracy.item())

    val_loss = t.tensor(losses).mean()
    val_acc = t.tensor(accuracies).mean()
    wandb.log({"val_loss": val_loss, "val_acc": val_acc}, step=global_step)
    print(f"Epoch {epoch} loss: {val_loss:.4f}, acc: {val_acc:.4f}")
