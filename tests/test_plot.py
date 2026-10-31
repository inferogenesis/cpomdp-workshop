import json
import time
from pathlib import Path

import numpy as np
import pytest
from matplotlib.colors import to_hex
from matplotlib.figure import Figure
from matplotlib.patches import Circle

from cpomdp_workshop.build import curious_rocket, the_1960_rocket
from cpomdp_workshop.plot import (
    CURIOUS_COLOURS,
    DARK,
    IMPACT_MARKER,
    LIGHT,
    ROUTE_STAGES,
    plot_beliefs,
    plot_flights,
    plot_path,
    plot_route,
    plot_world,
    preview,
    tile_task1,
    tile_task2,
    tile_task3,
)
from cpomdp_workshop.sim import fly
from cpomdp_workshop.world import beacon_sensor, default_params, reference_sensor

TIMINGS = Path(__file__).resolve().parents[1] / "artifacts" / "timings.json"


def impact_markers(figure):
    return [
        line
        for axes in figure.axes
        for line in axes.get_lines()
        if line.get_marker() == IMPACT_MARKER
    ]


def legend_texts(figure):
    legend = figure.axes[0].get_legend()
    return [] if legend is None else [text.get_text() for text in legend.get_texts()]


def record_timing(name, seconds):
    TIMINGS.parent.mkdir(exist_ok=True)
    timings = json.loads(TIMINGS.read_text()) if TIMINGS.exists() else {}
    timings[name] = round(seconds, 3)
    TIMINGS.write_text(json.dumps(timings, indent=2, sort_keys=True) + "\n")


def test_a_single_flight_renders_without_a_legend_box():
    figure = plot_path(fly(the_1960_rocket(), steps=6))
    assert isinstance(figure, Figure)
    assert figure.axes[0].get_legend() is None


def test_a_flight_renders_over_the_field_it_flew_in():
    params = default_params(0.4)
    figure = plot_path(fly(curious_rocket(0.4, beacon_sensor)), beacon_sensor, params)
    assert figure.axes[0].collections, "the field is drawn as a mesh"


def test_a_sensor_written_in_numpy_still_draws_its_field():
    def numpy_sensor(x, params):
        return np.eye(2) * (0.05 + float(np.sum(np.asarray(x) ** 2)))

    figure = plot_path(fly(the_1960_rocket(), steps=3), numpy_sensor, {})
    assert figure.axes[0].collections


def test_the_world_shows_three_rockets_and_the_crash():
    started = time.perf_counter()
    figure = plot_world(reference_sensor, (0.0, 0.08, 0.20))
    record_timing("plot_world", time.perf_counter() - started)
    assert legend_texts(figure) == [
        "1960 rocket, hit on step 6",
        "k = 0.08",
        "k = 0.20",
    ]
    assert len(impact_markers(figure)) == 1


def test_the_world_without_the_1960_rocket_shows_no_crash():
    figure = plot_world(reference_sensor, (0.08, 0.20))
    assert impact_markers(figure) == []


def test_the_belief_tile_draws_one_ellipse_per_step_shown():
    figure = plot_beliefs(fly(the_1960_rocket(), steps=8), steps=(1, 2, 4, 8))
    ellipses = [p for p in figure.axes[0].patches if type(p).__name__ == "Ellipse"]
    assert len(ellipses) == 4


def test_more_curious_flights_than_colours_is_refused():
    flight = fly(the_1960_rocket(), steps=2)
    with pytest.raises(ValueError, match="at most 2"):
        plot_flights({"a": flight, "b": flight, "c": flight})


def test_a_named_baseline_is_drawn_dashed():
    flight = fly(the_1960_rocket(), steps=2)
    figure = plot_flights({"base": flight, "mine": flight}, baseline="base")
    lines = figure.axes[0].get_lines()
    paths = [line for line in lines if not str(line.get_label()).startswith("_")]
    styles = {line.get_label(): line.get_linestyle() for line in paths}
    assert styles == {"base": "--", "mine": "-"}


@pytest.mark.parametrize("tile", [tile_task1, tile_task2, tile_task3])
def test_every_tile_renders(tile):
    assert isinstance(tile(), Figure)


def test_the_preview_renders_inside_a_minute():
    started = time.perf_counter()
    figure = preview()
    elapsed = time.perf_counter() - started
    record_timing("preview", elapsed)
    assert len(impact_markers(figure)) == 1
    assert elapsed < 60


@pytest.fixture(scope="module")
def route():
    return {stage: plot_route(stage) for stage in ROUTE_STAGES}


def map_of(figure):
    axes = figure.axes[0]
    return (
        tuple(figure.get_size_inches()),
        axes.get_xlim(),
        axes.get_ylim(),
        tuple(axes.get_position().bounds),
    )


def test_the_route_is_built_up_in_four_frames():
    assert ROUTE_STAGES == ("straight", "beacon", "blackout", "rock")


def test_every_route_frame_draws_the_same_map(route):
    assert len({map_of(figure) for figure in route.values()}) == 1


@pytest.mark.parametrize(
    ("stage", "legend"),
    [
        ("straight", ["1960 rocket"]),
        ("beacon", ["1960 rocket", "k = 0.40"]),
        ("blackout", ["1960 rocket", "k = 0.08", "k = 0.20"]),
        ("rock", ["1960 rocket, hit on step 6", "k = 0.08", "k = 0.20"]),
    ],
)
def test_each_route_frame_names_its_rockets(route, stage, legend):
    assert legend_texts(route[stage]) == legend


@pytest.mark.parametrize("stage", ROUTE_STAGES)
def test_only_the_last_route_frame_holds_the_rock_and_the_crash(route, stage):
    patches = route[stage].axes[0].patches
    expected = 1 if stage == "rock" else 0
    assert len([p for p in patches if isinstance(p, Circle)]) == expected
    assert len(impact_markers(route[stage])) == expected


def test_the_first_route_frame_is_one_noise_level_everywhere(route):
    field = route["straight"].axes[0].collections[0].get_array()
    assert np.ptp(field) == 0


def test_the_1960_route_is_the_same_line_until_the_rock(route):
    def dashed(figure):
        line = next(
            line
            for line in figure.axes[0].get_lines()
            if str(line.get_label()).startswith("1960 rocket")
        )
        return np.asarray(line.get_xydata())

    first = dashed(route["straight"])
    for stage in ("beacon", "blackout"):
        np.testing.assert_array_equal(dashed(route[stage]), first)
    np.testing.assert_array_equal(dashed(route["rock"]), first[:7])


def test_an_unknown_route_frame_is_refused():
    with pytest.raises(ValueError, match="straight, beacon, blackout, rock"):
        plot_route("moon")


def test_figures_are_dark_unless_a_palette_is_given(route):
    assert to_hex(route["rock"].get_facecolor()) == DARK.background.lower()
    assert to_hex(preview().get_facecolor()) == DARK.background.lower()


def test_a_light_route_frame_draws_the_same_map_on_the_light_palette(route):
    light = plot_route("rock", palette=LIGHT)
    axes = light.axes[0]
    assert to_hex(light.get_facecolor()) == LIGHT.background.lower()
    assert to_hex(axes.get_facecolor()) == LIGHT.panel.lower()
    assert map_of(light) == map_of(route["rock"])
    assert legend_texts(light) == legend_texts(route["rock"])


def test_the_rockets_keep_their_colours_on_either_palette(route):
    def colours(figure):
        return [to_hex(line.get_color()) for line in figure.axes[0].get_lines()]

    dark, light = colours(route["rock"]), colours(plot_route("rock", palette=LIGHT))
    curious = [to_hex(c) for c in CURIOUS_COLOURS]
    assert [c for c in dark if c in curious] == [c for c in light if c in curious]
    assert [c for c in light if c in curious]
