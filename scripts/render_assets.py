"""Regenerate every figure in ``assets/`` from the package.

    uv run --no-sync python scripts/render_assets.py

Each figure has one generating function in ``cpomdp_workshop.plot``. Nothing here
draws anything itself. The route frames are the slides' and take the light palette.
The rest show what the notebook draws, on the dark one.

The timer slides' clock clip and its poster come from ``timer_clip``. They need the
deck extra and the deck's Calibri Bold.
"""

import os
from functools import partial
from pathlib import Path

# Before pyplot is imported: the figures are written, never shown.
os.environ.setdefault("MPLBACKEND", "Agg")

import timer_clip
from build_deck import TASK_MINUTES
from cpomdp_workshop import plot
from deck_layout import metrics

ASSETS = Path(__file__).resolve().parents[1] / "assets"

ROUTE_FILES = {
    "straight": "route_1_straight.png",
    "beacon": "route_2_beacon.png",
    "too_curious": "route_2_too_curious.png",
    "blackout": "route_3_blackout.png",
    "rock": "route_4_rock.png",
}
"""Each route frame's file, numbered by the task whose slides show it."""

FIGURES = {
    "fig_world_rock.png": plot.preview,
    "tile_task1.png": plot.tile_task1,
    "tile_task2.png": plot.tile_task2,
    "tile_task3.png": plot.tile_task3,
    "route_2_balance.png": partial(plot.plot_balance, palette=plot.LIGHT),
    **{
        name: partial(plot.plot_route, stage, palette=plot.LIGHT)
        for stage, name in ROUTE_FILES.items()
    },
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
    fonts = metrics()
    if fonts is None:
        raise SystemExit("no folder holds Calibri Bold, so the clock clip is not drawn")
    timer_clip.write(TASK_MINUTES, fonts.directory)
    print(f"wrote assets/{timer_clip.CLIP.name} and assets/{timer_clip.POSTER.name}")


if __name__ == "__main__":
    main()
