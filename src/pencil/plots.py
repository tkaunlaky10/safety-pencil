"""Small matplotlib helpers shared by the scripts' figures (headless, 160 dpi)."""

from __future__ import annotations

import math


def subplots(*args, **kw):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt.subplots(*args, **kw)


def save(fig, path) -> None:
    import matplotlib.pyplot as plt

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def log10_or_nan(x: float) -> float:
    return math.log10(x) if x > 0 else float("nan")


def heatmap(
    rows: list[list[float]],
    xticklabels,
    path,
    *,
    title: str,
    xlabel: str = "",
    ylabel: str = "layer",
    colorbar: str = "",
    width: float = 4.8,
) -> None:
    """One heatmap (rows × columns) saved to `path`; NaN cells are left blank."""
    fig, ax = subplots(figsize=(width, 0.18 * len(rows) + 1.4))
    im = ax.imshow(rows, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(xticklabels)), xticklabels)
    ax.set(xlabel=xlabel, ylabel=ylabel)
    ax.set_title(title, fontsize=8)
    fig.colorbar(im, ax=ax, label=colorbar)
    save(fig, path)
