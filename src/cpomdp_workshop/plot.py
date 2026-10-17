"""Pictures of the world and the flights through it.

The sensor field is drawn as ``log10 R(x)`` in one muted green ramp, brighter where
the reading is noisier, so the blackout glows around the rock and the beacon is a dark
well. The 1960 rocket is a dashed ink line. The curious rockets take amber, then
blue, in the order given. Text is ink or muted, never a series colour, and a figure
with more than one rocket carries a legend.

Every function returns a figure without showing it. A notebook cell that ends with
one displays it once.
"""

from collections.abc import Mapping, Sequence

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.figure import Figure
from matplotlib.patches import Circle, Ellipse

from cpomdp_workshop.build import curious_rocket, the_1960_rocket
from cpomdp_workshop.sim import DEFAULT_SEED, ROCK, Flight, Hazard, fly
from cpomdp_workshop.world import (
    BEACON,
    BLACKOUT_GAIN,
    LAUNCH,
    NOISE_FLOOR,
    PAD,
    NoiseFn,
    SensorParams,
    beacon_sensor,
    default_params,
    reference_sensor,
)

__all__ = [
    "IMPACT_MARKER",
    "plot_beliefs",
    "plot_flights",
    "plot_path",
    "plot_world",
    "preview",
    "tile_task1",
    "tile_task2",
    "tile_task3",
    "world_flights",
]

BACKGROUND = "#0C1916"
PANEL = "#14261F"
RULE = "#2A3D35"
INK = "#F3F0E6"
MUTED = "#8C968F"
AMBER = "#C48200"
BLUE = "#3D8ED4"
ROCK_FILL = "#5A625D"
FIELD_RAMP = LinearSegmentedColormap.from_list("reading_noise", [PANEL, "#4F7A63"])
# One scale for every figure, from the floor to the floor plus the whole blackout, so
# the same shade means the same noise from tile to tile.
FIELD_SCALE = Normalize(np.log10(NOISE_FLOOR), np.log10(NOISE_FLOOR + BLACKOUT_GAIN))

IMPACT_MARKER = "*"
CURIOUS_COLOURS = (AMBER, BLUE)

_EXTENT = ((-0.6, 5.6), (-1.1, 2.5))
_FIELD_SAMPLES = (125, 73)


def covariance_ellipse(cov: np.ndarray, *, sigmas: float = 2.0) -> tuple[float, ...]:
    """The ``(width, height, angle_degrees)`` of a covariance ellipse.

    Adapted from ``examples/gallery.py`` in cpomdp v0.5.0 (MIT).
    """
    values, vectors = np.linalg.eigh(cov)
    values = np.clip(values, 1e-9, None)
    angle = float(np.degrees(np.arctan2(vectors[1, 0], vectors[0, 0])))
    width, height = 2 * sigmas * np.sqrt(values)
    return float(width), float(height), angle


def _canvas(title: str, size: tuple[float, float] = (9.0, 5.0)) -> tuple[Figure, Axes]:
    """An empty map of the world in the workshop's palette."""
    figure, axes = plt.subplots(figsize=size, facecolor=BACKGROUND)
    plt.close(figure)  # returned, never shown by pyplot itself
    axes.set_facecolor(PANEL)
    axes.set_xlim(*_EXTENT[0])
    axes.set_ylim(*_EXTENT[1])
    axes.set_aspect("equal")
    axes.set_title(title, color=INK, loc="left", fontsize=12)
    axes.tick_params(colors=MUTED, labelsize=9)
    for spine in axes.spines.values():
        spine.set_color(RULE)
        spine.set_linewidth(0.8)
    return figure, axes


def _noise_levels(
    sensor: NoiseFn, params: SensorParams, points: np.ndarray
) -> np.ndarray:
    """``log10`` of each point's reading noise, as one standard deviation squared."""

    def level(point):
        return jnp.sqrt(jnp.linalg.det(jnp.asarray(sensor(point, params))))

    try:
        noise = np.asarray(jax.vmap(level)(jnp.asarray(points)))
    except Exception:  # a NumPy sensor cannot be traced, so point by point
        noise = np.array(
            [np.sqrt(np.linalg.det(np.asarray(sensor(p, params)))) for p in points]
        )
    return np.log10(noise)


def _draw_field(
    figure: Figure, axes: Axes, sensor: NoiseFn, params: SensorParams
) -> None:
    """The sensor field under everything else, with its scale."""
    xs = np.linspace(*_EXTENT[0], _FIELD_SAMPLES[0])
    ys = np.linspace(*_EXTENT[1], _FIELD_SAMPLES[1])
    grid_x, grid_y = np.meshgrid(xs, ys)
    points = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    levels = _noise_levels(sensor, params, points).reshape(grid_x.shape)
    mesh = axes.pcolormesh(
        grid_x, grid_y, levels, cmap=FIELD_RAMP, norm=FIELD_SCALE, shading="auto",
        zorder=0,
    )  # fmt: skip
    scale = figure.colorbar(mesh, ax=axes, shrink=0.8, pad=0.02)
    scale.set_label("reading noise, log10 R", color=MUTED, fontsize=9)
    scale.ax.tick_params(colors=MUTED, labelsize=8)
    scale.outline.set_edgecolor(RULE)


def _draw_landmarks(axes: Axes, hazard: Hazard | None) -> None:
    """The launch site, the pad, the beacon and, when given, the rock."""
    if hazard is not None:
        centre, radius = hazard
        axes.add_patch(Circle(centre, radius, facecolor=ROCK_FILL, zorder=2))
        axes.annotate(
            "rock", (centre[0], centre[1] - radius), xytext=(0, -4),
            textcoords="offset points", ha="center", va="top", color=INK, fontsize=9,
            zorder=6,
        )  # fmt: skip
    # Each name sits on the side no flight path crosses.
    for point, name, offset, align in (
        (LAUNCH, "launch", (-9, 0), "right"),
        (PAD, "pad", (0, -15), "center"),
        (BEACON, "beacon", (9, 0), "left"),
    ):
        axes.plot(
            *point, marker="o", ms=7, color=INK if name != "beacon" else MUTED,
            markeredgecolor=PANEL, markeredgewidth=2, zorder=5, linestyle="none",
        )  # fmt: skip
        axes.annotate(
            name, point, xytext=offset, textcoords="offset points", ha=align,
            va="center", color=INK, fontsize=9, zorder=6,
        )  # fmt: skip


def _draw_flight(
    axes: Axes, flight: Flight, colour: str, label: str, *, dashed: bool, inline: bool
) -> None:
    """One path, ending in an impact star if it crashed.

    The impact step goes in the legend entry, or beside the star when ``inline``.
    """
    if flight.crashed and not inline:
        label = f"{label}, hit on step {flight.impact_step}"
    axes.plot(
        flight.path[:, 0], flight.path[:, 1], color=colour, lw=2,
        linestyle=(0, (5, 3)) if dashed else "solid", solid_capstyle="round",
        label=label, zorder=4,
    )  # fmt: skip
    if not flight.crashed:
        return
    x, y = flight.final
    axes.plot(
        x, y, marker=IMPACT_MARKER, ms=16, color=colour, markeredgecolor=PANEL,
        markeredgewidth=2, linestyle="none", zorder=7,
    )  # fmt: skip
    if inline:
        axes.annotate(
            f"impact, step {flight.impact_step}", (x, y), xytext=(-8, 12),
            textcoords="offset points", ha="right", color=INK, fontsize=9, zorder=7,
        )  # fmt: skip


def _legend(axes: Axes) -> None:
    legend = axes.legend(loc="upper left", frameon=True, fontsize=9)
    legend.get_frame().set_facecolor(PANEL)
    legend.get_frame().set_edgecolor(RULE)
    for text in legend.get_texts():
        text.set_color(INK)


def world_flights(
    sensor: NoiseFn,
    beacon_gains: Sequence[float],
    *,
    horizon: int = 1,
    hazard: Hazard | None = ROCK,
    seed: int = DEFAULT_SEED,
) -> dict[float, Flight]:
    """One rocket per beacon gain, flown from the same seed past the same hazard.

    A gain of zero is the 1960 rocket. It models a reading that is equally noisy
    everywhere and flies in that world, so nothing but the rock is hidden from it.
    Every other gain is a curious rocket that models ``sensor`` and flies in it. A
    gain given twice is flown once.

    Args:
        sensor: the noise field the curious rockets model and fly in.
        beacon_gains: k for each rocket.
        horizon: how many moves ahead the curious rockets look, H.
        hazard: ``(centre, radius)``, or ``None`` for open sky.
        seed: the reading noise, shared by every rocket.

    Returns:
        Each gain's flight, in the order given.
    """
    flights = {}
    for beacon_gain in dict.fromkeys(beacon_gains):
        if beacon_gain == 0:
            rocket = the_1960_rocket()
        else:
            rocket = curious_rocket(beacon_gain, sensor, horizon=horizon)
        flights[beacon_gain] = fly(rocket, seed=seed, hazard=hazard)
    return flights


def plot_flights(
    flights: Mapping[str, Flight],
    sensor: NoiseFn | None = None,
    params: SensorParams | None = None,
    *,
    baseline: str | None = None,
    hazard: Hazard | None = None,
    title: str = "",
) -> Figure:
    """Named flights over the field they flew in.

    Args:
        flights: each flight under the name its legend entry gives it.
        sensor: the noise field to draw under the paths, or ``None`` for none.
        params: that field's parameters.
        baseline: the name of the flight to draw as the 1960 rocket, dashed ink.
            The others take amber, then blue, in the order given.
        hazard: ``(centre, radius)`` to draw, or ``None``.
        title: what the figure is called. With one flight it names that flight,
            and there is no legend box.

    Returns:
        The figure.

    Raises:
        ValueError: for more flights than there are series colours. A third hue
            would not stay distinct, so split the flights across figures.
    """
    curious = [name for name in flights if name != baseline]
    if len(curious) > len(CURIOUS_COLOURS):
        raise ValueError(
            f"{len(curious)} curious flights. Draw at most {len(CURIOUS_COLOURS)} "
            "per figure."
        )
    figure, axes = _canvas(title)
    if sensor is not None:
        _draw_field(figure, axes, sensor, params if params is not None else {})
    _draw_landmarks(axes, hazard)
    inline = len(flights) == 1
    colours = dict(zip(curious, CURIOUS_COLOURS, strict=False))
    for name, flight in flights.items():
        if name == baseline:
            _draw_flight(axes, flight, INK, name, dashed=True, inline=inline)
        else:
            _draw_flight(axes, flight, colours[name], name, dashed=False, inline=inline)
    if not inline:
        _legend(axes)
    return figure


def plot_path(
    flight: Flight,
    sensor: NoiseFn | None = None,
    params: SensorParams | None = None,
    *,
    hazard: Hazard | None = None,
    title: str = "your rocket",
) -> Figure:
    """One flight over the field it flew in.

    Args:
        flight: from [`fly`][cpomdp_workshop.fly].
        sensor: the noise field to draw under the path, or ``None`` for none.
        params: that field's parameters.
        hazard: ``(centre, radius)`` to draw, or ``None``.
        title: what the figure is called. It names the one flight.

    Returns:
        The figure.
    """
    return plot_flights({title: flight}, sensor, params, hazard=hazard, title=title)


def plot_world(
    sensor: NoiseFn = reference_sensor,
    beacon_gains: Sequence[float] = (0.0, 0.08, 0.20),
    *,
    horizon: int = 1,
    hazard: Hazard | None = ROCK,
    title: str = "three rockets, one rock",
) -> Figure:
    """Fly one rocket per beacon gain and draw them all over the field.

    A gain of zero is the 1960 rocket, drawn dashed. The field drawn is ``sensor`` at
    the first gain above zero, the one the room's slider sets. A gain given twice is
    drawn once.

    Args:
        sensor: the noise field the curious rockets model and fly in.
        beacon_gains: k for each rocket.
        horizon: how many moves ahead the curious rockets look, H.
        hazard: ``(centre, radius)``, or ``None`` for open sky.
        title: what the figure is called.

    Returns:
        The figure.
    """
    flights = world_flights(sensor, tuple(beacon_gains), horizon=horizon, hazard=hazard)
    named = {
        ("1960 rocket" if k == 0 else f"k = {k:.2f}"): flight
        for k, flight in flights.items()
    }
    curious = [k for k in flights if k != 0]
    params = default_params(curious[0] if curious else 0.0)
    return plot_flights(
        named, sensor, params, baseline="1960 rocket", hazard=hazard, title=title
    )


def plot_beliefs(
    flight: Flight,
    steps: Sequence[int] = (1, 2, 4, 8),
    *,
    title: str = "what the 1960 rocket believes",
) -> Figure:
    """A flight with the rocket's two-sigma belief after each of ``steps`` readings.

    Args:
        flight: from [`fly`][cpomdp_workshop.fly].
        steps: which readings to draw the belief after, counting from one.
        title: what the figure is called. It names the one flight.

    Returns:
        The figure.
    """
    figure, axes = _canvas(title, size=(9.0, 4.2))
    _draw_landmarks(axes, None)
    _draw_flight(axes, flight, INK, title, dashed=True, inline=True)
    for step in steps:
        width, height, angle = covariance_ellipse(flight.covs[step - 1])
        axes.add_patch(
            Ellipse(
                flight.means[step - 1], width, height, angle=angle, facecolor=AMBER,
                alpha=0.18, edgecolor=AMBER, lw=1.2, zorder=3,
            )
        )  # fmt: skip
        axes.annotate(
            f"after {step}", flight.means[step - 1], xytext=(0, -16),
            textcoords="offset points", ha="center", color=MUTED, fontsize=8,
        )  # fmt: skip
    return figure


def preview() -> Figure:
    """The finished picture: what the room will have built by the end of session 3."""
    return plot_world(title="what you will build today")


def tile_task1() -> Figure:
    """Task 1's tile: the 1960 rocket and its shrinking belief."""
    return plot_beliefs(fly(the_1960_rocket()))


def tile_task2() -> Figure:
    """Task 2's tile: the k = 0.4 arc up to the beacon, over the beacon field."""
    flight = fly(curious_rocket(0.4, beacon_sensor))
    return plot_path(
        flight, beacon_sensor, default_params(0.4), title="k = 0.40, beacon only"
    )


def tile_task3() -> Figure:
    """Task 3's tile: the blackout world, before the rock is put in it."""
    return plot_world(hazard=None, title="the blackout, before the rock")
