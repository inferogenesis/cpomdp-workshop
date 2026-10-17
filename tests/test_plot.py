import json
import time
from pathlib import Path

import numpy as np
import pytest
from matplotlib.figure import Figure

from cpomdp_workshop.build import curious_rocket, the_1960_rocket
from cpomdp_workshop.plot import (
    IMPACT_MARKER,
    plot_beliefs,
    plot_flights,
    plot_path,
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
