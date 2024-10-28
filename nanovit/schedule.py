import math


def linear_schedule(step: int, max_lr: float, max_steps: int, eps: float = 1e-8):
    frac = 1 - step / max_steps
    return max_lr * frac + eps


def cosine_schedule(
    step: int, max_lr: float, max_steps: int, warmup_steps: int, eps: float = 1e-8
):
    if step < warmup_steps:
        frac = step / warmup_steps
        return max_lr * frac + eps

    anneal_step = step - warmup_steps
    anneal_max_steps = max_steps - warmup_steps
    frac = (math.cos(math.pi * anneal_step / anneal_max_steps) + 1) / 2
    return max_lr * frac + eps


if __name__ == "__main__":
    import matplotlib.pyplot as plt  # noqa: E402
    import numpy as np

    lr = 1e-3
    X = np.arange(10_000)
    y_cos = [cosine_schedule(x, lr, 10_000, 1000) for x in X]
    y_lin = [linear_schedule(x, lr, 10_000) for x in X]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3), sharey=True)  # type: ignore
    ax1.plot(X, y_cos)
    ax2.plot(X, y_lin)
    ax1.set_title("cosine")
    ax2.set_title("linear")
    plt.show()
