"""Small matplotlib helpers shared by the scripts' figures.

Figures are drawn headless (the Agg backend, so no display is needed) and saved at 160 dpi.
"""

from __future__ import annotations

import math

DPI = 160
# Heatmap height: this many inches per row, plus room for the title and axis labels.
HEATMAP_ROW_HEIGHT = 0.18
HEATMAP_MARGIN_HEIGHT = 1.4


def subplots(*args, **kwargs):
    """matplotlib.pyplot.subplots on the headless Agg backend; returns (figure, axes)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt.subplots(*args, **kwargs)


def save(fig, path) -> None:
    """Lay out, save to `path` at DPI and close the figure."""
    import matplotlib.pyplot as plt

    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def log10_or_nan(x: float) -> float:
    """log10(x) for positive x, NaN otherwise (NaN cells are drawn blank)."""
    if x > 0:
        return math.log10(x)
    return float("nan")


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
    height = HEATMAP_ROW_HEIGHT * len(rows) + HEATMAP_MARGIN_HEIGHT
    figure, axes = subplots(figsize=(width, height))
    image = axes.imshow(rows, aspect="auto", cmap="viridis")
    axes.set_xticks(range(len(xticklabels)), xticklabels)
    axes.set(xlabel=xlabel, ylabel=ylabel)
    axes.set_title(title, fontsize=8)
    figure.colorbar(image, ax=axes, label=colorbar)
    save(figure, path)
