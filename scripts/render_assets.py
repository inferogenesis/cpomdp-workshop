"""Regenerate every figure in ``assets/`` from the package.

    uv run --no-sync python scripts/render_assets.py

Each figure has one generating function in ``cpomdp_workshop.plot``. Nothing here
draws anything itself.
"""

import os
from pathlib import Path

# Before pyplot is imported: the figures are written, never shown.
os.environ.setdefault("MPLBACKEND", "Agg")

from cpomdp_workshop import plot

ASSETS = Path(__file__).resolve().parents[1] / "assets"

FIGURES = {
    "fig_world_rock.png": plot.preview,
    "tile_task1.png": plot.tile_task1,
    "tile_task2.png": plot.tile_task2,
    "tile_task3.png": plot.tile_task3,
}


def main() -> None:
    """Render each figure and write it beside the others."""
    ASSETS.mkdir(exist_ok=True)
    for name, draw in FIGURES.items():
        figure = draw()
        figure.savefig(
            ASSETS / name,
            dpi=200,
            facecolor=figure.get_facecolor(),
            bbox_inches="tight",
        )
        print(f"wrote assets/{name}")


if __name__ == "__main__":
    main()
