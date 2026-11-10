"""Regenerate every figure in ``assets/`` from the package.

    uv run --no-sync python scripts/render_assets.py

Each figure has one generating function in ``cpomdp_workshop.plot``. Nothing here
draws anything itself. The route frames are the slides' and take the light palette.
The rest show what the notebook draws, on the dark one.

The timer slides' clock clips and their posters come from ``timer_clip``, one pair
for each length a timer runs. They need the deck extra and the deck's Calibri Bold.
"""

import os
from functools import partial
from pathlib import Path

# Before pyplot is imported: the figures are written, never shown.
os.environ.setdefault("MPLBACKEND", "Agg")

from matplotlib.figure import Figure

import timer_clip
from build_deck import TIMER_MINUTES
from cpomdp_workshop import plot
from deck_layout import metrics

ASSETS = Path(__file__).resolve().parents[1] / "assets"

ROUTE_FILES = {
    "straight": "route_1_straight.png",
    "beacon": "route_2_beacon.png",
    "too_curious": "route_2_too_curious.png",
}
"""Each route frame the slides show, numbered by the task whose slides show it."""


def hidden_pad() -> Figure:
    """Task 3's reference pair on the hidden pad, on the slides' palette."""
    return plot.plot_pad(plot.pad_pair(), palette=plot.LIGHT)


FIGURES = {
    "tile_task1.png": plot.tile_task1,
    "tile_task2.png": plot.tile_task2,
    "route_2_balance.png": partial(plot.plot_balance, palette=plot.LIGHT),
    "route_3_pad.png": hidden_pad,
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
    for minutes in TIMER_MINUTES:
        timer_clip.write(minutes, fonts.directory)
        clip, poster = timer_clip.clip(minutes), timer_clip.poster(minutes)
        print(f"wrote assets/{clip.name} and assets/{poster.name}")


if __name__ == "__main__":
    main()
