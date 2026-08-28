"""
Shared matplotlib styling for the analysis figures.

Collects the rcParams and the colour palette used by more than one plotting
script, so the figures keep a single consistent look and the styling is
defined once.

The palette assigns a fixed colour to a role rather than to a quantity:
stiff for data-visible quantities, soft for data-invisible ones, accent for a
third series, and grey for reference lines and baselines.
"""

import matplotlib as mpl

PALETTE = {
    "stiff": "#2166ac",
    "soft": "#d6604d",
    "accent": "#5aae61",
    "grey": "#888888",
}


def apply_style() -> None:
    """Set the shared rcParams. Call once near the top of a plotting script."""
    mpl.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 130,
            "font.family": "DejaVu Sans",
        }
    )


def save(fig, outdir, name: str) -> None:
    """Write a figure as both PDF (vector, for the writeup) and PNG."""
    from pathlib import Path

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(outdir / f"{name}.png", dpi=180, bbox_inches="tight")
