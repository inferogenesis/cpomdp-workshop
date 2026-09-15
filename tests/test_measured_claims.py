"""One test per value the deck or the notebook quotes.

A number in the prose has a test here, and the test is what the prose is patched to
when the two disagree. Every flight runs under the pinned cpomdp release; a re-pin
re-runs this file first.
"""

from importlib import metadata

import numpy as np
import pytest
from cpomdp import Belief, CallableSensor, LinearGaussianModel, probe_model

from cpomdp_workshop import WORKSHOP_CPOMDP_VERSION
from cpomdp_workshop.build import (
    beacon_rocket,
    curious_rocket,
    model_with,
    the_1960_rocket,
)
from cpomdp_workshop.planner import compass_actions, compass_planner
from cpomdp_workshop.sim import ROCK, fly
from cpomdp_workshop.world import (
    BEACON,
    PAD,
    beacon_sensor,
    default_params,
    reference_sensor,
)

MOVES = list(np.asarray(compass_actions().actions))
I = np.eye(2)


def test_the_numbers_were_measured_on_the_pinned_release():
    assert metadata.version("cpomdp") == WORKSHOP_CPOMDP_VERSION


def test_no_move_of_the_1960_rocket_is_worth_more_than_any_other():
    report = probe_model(model_with(), model_with().prior, MOVES)
    lo, hi = report.epistemic_range
    assert report.flattens is True
    assert report.epistemic_varies is False
    assert f"{hi - lo:.6f} nats" == "0.000000 nats"


def test_the_curious_rocket_separates_its_moves_by_more_than_a_hundredth_of_a_nat():
    model = model_with(reference_sensor, default_params(0.08))
    report = probe_model(model, model.prior, MOVES)
    lo, hi = report.epistemic_range
    assert report.flattens is False
    assert hi - lo > 0.01


def test_the_search_is_decided_rather_than_sampled():
    model = model_with(reference_sensor, default_params(0.08))
    certificate = compass_planner(model).certificate
    assert str(certificate) == "PROVED (finite set, |A|^H = 9, visited 9)"


@pytest.mark.parametrize("seed", range(10))
def test_the_1960_rocket_hits_the_rock_on_step_five_whatever_the_noise_does(seed):
    flight = fly(the_1960_rocket(), seed=seed, hazard=ROCK)
    assert flight.crashed is True
    assert flight.impact_step == 5


@pytest.mark.parametrize("beacon_gain", [0.08, 0.20])
def test_a_curious_rocket_clears_the_rock(beacon_gain):
    params = default_params(beacon_gain)
    flight = fly(curious_rocket(beacon_gain), reference_sensor, params, hazard=ROCK)
    assert flight.crashed is False


def test_the_default_slider_setting_lands_the_rocket():
    params = default_params(0.08)
    flight = fly(curious_rocket(0.08), reference_sensor, params, hazard=ROCK)
    assert flight.distance_to(PAD) < 0.5
    assert flight.path[:, 1].max() >= BEACON[1] - 0.5


def test_session_two_bends_the_flight_path_toward_the_beacon():
    params = default_params(0.4)
    flight = fly(beacon_rocket(0.4), beacon_sensor, params)
    assert flight.path[:, 1].max() >= 1.0
    assert flight.distance_to(PAD) < 0.35


def test_session_two_at_the_slider_default_does_not_bend_at_all():
    params = default_params(0.08)
    flight = fly(beacon_rocket(0.08), beacon_sensor, params)
    assert flight.path[:, 1].max() == pytest.approx(0.0, abs=1e-9)


def test_papercut_one_still_bites_on_this_release():
    sensor = CallableSensor(I, reference_sensor, default_params(0.08))
    with pytest.raises(TypeError, match="not 'CallableSensor'"):
        LinearGaussianModel(
            dynamics=I,
            sensor_model=I,
            dynamics_noise=1e-3 * I,
            sensor_noise=sensor,
            prior=Belief(mean=[0.0, 0.0], cov=4 * I),
            control=I,
        )
