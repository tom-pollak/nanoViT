# ViTs From Scratch

> Minimal ViT implementation from scratch in PyTorch. Inspired by [nanogpt](https://github.com/karpathy/nanoGPT).

Model code: [nanovit/vit.py](nanovit/vit.py).
- Verified by loading CLIP ViT-B/32 model weights.
- Accompanying exercise: implement each module from scratch yourself in [Colab](https://colab.research.google.com/github/tom-pollak/nanoViT/blob/main/tutorials/vit_from_scratch.ipynb)

Training script for CIFAR-100: [train_cifar100.py](train_cifar100.py).
- 2.7M params ViT runs in ~15mins on a 3090, achieves 63%

Also a ConvMixer implementation ([nanovit/conv_mixer.py](nanovit/conv_mixer.py)) as a baseline.

