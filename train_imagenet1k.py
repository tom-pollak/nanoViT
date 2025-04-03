# %%
import os
import random
import inspect
from dataclasses import dataclass, asdict
from typing import Literal
from tqdm import tqdm
from contextlib import nullcontext

import torch as t
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import v2  # type: ignore

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

import wandb
from datasets import load_dataset, DatasetDict

from nanovit import ViT, ViTConfig, build_preprocessor
from nanovit.schedule import cosine_schedule, linear_schedule

# %%

ddp = False

if ddp:
    init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])

    device = f"cuda:{ddp_local_rank}"
    t.cuda.set_device(device)
    master_process = ddp_rank == 0
    seed_offset = ddp_rank  # each gpu gets different seed

    t.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    t.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn

else:
    device = (
        "cuda"
        if t.cuda.is_available()
        else "mps"
        if t.backends.mps.is_available()
        else "cpu"
    )

    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1

    master_process = True
    seed_offset = 0


num_workers = 10
t.manual_seed(1337 + seed_offset)

dtype = "bfloat16"
ptdtype = t.bfloat16


# %%


@dataclass
class TrainConfig:
    n_classes: int = 100
    n_epochs: int = 90
    bs: int = 32
    val_bs: int = 64
    lr: float = 1e-3
    wd: float = 1e-4
    grad_clip: float = 1.0
    warmup_steps: int = 10_000
    sched: Literal["cosine_schedule", "linear_schedule"] = "cosine_schedule"
    grad_accum_steps: int = 32
    # augmentation
    rand_crop_scale: tuple[float, float] = (0.9, 1.0)
    flip_p: float = 0.5
    mixup_p: float = 0.2
    autoaugment_policy: str = "imagenet"

    def __post_init__(self):
        assert self.grad_accum_steps % ddp_world_size == 0
        self.grad_accum_steps //= ddp_world_size
        self.warmup_steps

    @property
    def full_batch_size(self):
        return self.bs * self.grad_accum_steps * ddp_world_size


train_cfg = TrainConfig()


vit_cfg = ViTConfig(
    n_layers=12,
    d_model=512,
    d_proj=train_cfg.n_classes,
    image_res=(224, 224),
    patch_size=16,
    n_heads=16,
    dropout=0.1,
    norm_data=(
        (0.48145466, 0.4578275, 0.40821073),
        (0.26862954, 0.26130258, 0.27577711),
    ),
    mlp_mult=4,
)


# %% ████████████████████████████████████  xfms  ████████████████████████████████████

img2tensor: transforms.Compose = build_preprocessor(vit_cfg)

train_xfms = transforms.Compose(
    [
        transforms.RandomResizedCrop(
            size=vit_cfg.image_res,
            scale=train_cfg.rand_crop_scale,
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        transforms.RandomHorizontalFlip(p=train_cfg.flip_p),
        # Like ViT Imagenet1k
        # transforms.RandAugment(
        #     num_ops=2,
        #     magnitude=10,
        #     interpolation=transforms.InterpolationMode.BICUBIC,
        # ),
        transforms.AutoAugment(
            policy=transforms.autoaugment.AutoAugmentPolicy(
                train_cfg.autoaugment_policy
            )
        ),
        img2tensor,
    ]
)


valid_xfms = transforms.Compose([img2tensor])

# batch xfm
mixup = transforms.v2.MixUp(alpha=1.0, num_classes=train_cfg.n_classes)  # type: ignore


# %% ████████████████████████████████  dataset & dl  ████████████████████████████████

dd: DatasetDict = load_dataset(
    "ILSVRC/imagenet-1k", trust_remote_code=True, streaming=True
)  # type: ignore

### DEBUG
feats = dd["train"].features
i2s = feats["label"].int2str
s2i = feats["label"].str2int
row = next(dd["train"].iter(1))
###


def train_collate_fn(batch: list[dict]):
    pixel_values = t.stack([train_xfms(x["image"]) for x in batch])
    labels = t.tensor([x["label"] for x in batch])
    if random.random() < train_cfg.mixup_p:
        pixel_values, labels = mixup(pixel_values, labels)
    return pixel_values, labels


def valid_collate_fn(batch: list[dict]):
    pixel_values = t.stack([valid_xfms(x["image"]) for x in batch])
    labels = t.tensor([x["label"] for x in batch])
    return pixel_values, labels


train_dl = DataLoader(
    dd["train"],  # type: ignore
    batch_size=train_cfg.bs,
    shuffle=False,  # DEBUG: should shuffle
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


# %% ████████████████████████████████████  init  ████████████████████████████████████

vit = ViT(vit_cfg).to(device)
vit.init_weights_()
vit = t.compile(vit)

if ddp:
    vit = DDP(vit, device_ids=[ddp_local_rank])

n_params = sum(p.numel() for p in vit.parameters())
print(f"Number of parameters: {n_params:,}")


def configure_optimizers(model, weight_decay, learning_rate, betas, device_type):
    """
    From NanoGPT
    """
    # start with all of the candidate parameters
    param_dict = {pn: p for pn, p in model.named_parameters()}
    # filter out those that do not require grad
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    decay_params = [p for p in param_dict.values() if p.dim() >= 2]
    nodecay_params = [p for p in param_dict.values() if p.dim() < 2]
    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    print(
        f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
    )
    print(
        f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
    )
    # Create AdamW optimizer and use the fused version if it is available
    fused_available = "fused" in inspect.signature(t.optim.AdamW).parameters
    use_fused = fused_available and device_type == "cuda"
    extra_args = dict(fused=True) if use_fused else dict()
    optimizer = t.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
    print(f"using fused AdamW: {use_fused}")
    return optimizer


# configure optimizers nanogpt
opt = configure_optimizers(
    vit, train_cfg.wd, train_cfg.lr, betas=(0.9, 0.95), device_type=device
)
scaler = t.amp.GradScaler(device=device, enabled=False)

# %% ███████████████████████████████████  train  ████████████████████████████████████


if master_process:
    wandb.init(
        project="nanovit-cifar100",
        config={"vit_cfg": asdict(vit_cfg), "train_cfg": asdict(train_cfg)},
        settings=wandb.Settings(code_dir="."),
    )


def single_step(batch):
    pixel_values, labels = batch
    pixel_values, labels = pixel_values.to(device), labels.to(device)
    with t.autocast(device_type=device, dtype=ptdtype):
        logits = vit(pixel_values)
        loss = t.nn.functional.cross_entropy(logits, labels)

    if labels.ndim != 1:  # mixup/cutmix
        labels = labels.argmax(dim=-1)

    accuracy = (logits.argmax(dim=-1) == labels).float().mean()
    return loss, accuracy


sched = cosine_schedule if train_cfg.sched == "cosine_schedule" else linear_schedule
steps_per_epoch = len(train_dl)
max_steps = train_cfg.n_epochs * steps_per_epoch

for epoch in tqdm(range(train_cfg.n_epochs)):
    vit.train()
    if master_process:
        pbar = tqdm(enumerate(train_dl), total=steps_per_epoch, leave=False)
    else:
        pbar = enumerate(train_dl)

    grad_accum_step = 0
    for step, batch in pbar:
        grad_accum_step += 1
        global_step = epoch * steps_per_epoch + step
        lr = sched(
            step=global_step,
            max_lr=train_cfg.lr,
            max_steps=max_steps,
            warmup_steps=train_cfg.warmup_steps,
        )
        for param_group in opt.param_groups:
            param_group["lr"] = lr

        should_update = grad_accum_step == train_cfg.grad_accum_steps
        if ddp:
            vit.require_backward_grad_sync = should_update

        loss, accuracy = single_step(batch)  # type: ignore
        scaler.scale(loss).backward()
        if master_process:
            wandb.log(
                {"train_loss": loss, "train_acc": accuracy, "lr": lr}, step=global_step
            )
            assert isinstance(pbar, tqdm)
            pbar.set_postfix({"loss": f"{loss:.4f}", "acc": f"{accuracy:.4f}"})

        if should_update:
            grad_accum_step = 0

            scaler.unscale_(opt)
            t.nn.utils.clip_grad_norm_(vit.parameters(), train_cfg.grad_clip)

            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)

    vit.eval()
    losses, accuracies = [], []
    if master_process:
        pbar = tqdm(valid_dl, leave=False)
    else:
        pbar = valid_dl
    for step, batch in enumerate(pbar):
        with t.no_grad():
            loss, accuracy = single_step(batch)  # type: ignore
        losses.append(loss.item())
        accuracies.append(accuracy.item())

    val_loss = t.tensor(losses).mean()
    val_acc = t.tensor(accuracies).mean()
    if master_process:
        wandb.log({"val_loss": val_loss, "val_acc": val_acc}, step=global_step)
        print(f"Epoch {epoch} loss: {val_loss:.4f}, acc: {val_acc:.4f}")
