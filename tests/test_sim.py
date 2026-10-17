import numpy as np
import pytest

from cpomdp_workshop.build import curious_rocket, the_1960_rocket
from cpomdp_workshop.sim import DEFAULT_STEPS, ROCK, _hits, fly
from cpomdp_workshop.world import LAUNCH, NOISE_FLOOR, default_params, reference_sensor


def test_a_flight_records_the_launch_site_and_one_row_per_move():
    flight = fly(the_1960_rocket(), steps=8)
    assert flight.path.shape == (9, 2)
    assert np.allclose(flight.path[0], LAUNCH)
    assert flight.means.shape == (8, 2)
    assert flight.covs.shape == (8, 2, 2)
    assert flight.crashed is False
    assert flight.impact_step is None


def test_a_rocket_that_has_flown_is_refused_a_second_flight():
    rocket = the_1960_rocket()
    fly(rocket, steps=2)
    with pytest.raises(ValueError, match="has flown already"):
        fly(rocket)


def test_a_flight_uses_every_step_it_is_given():
    assert len(fly(the_1960_rocket()).path) == DEFAULT_STEPS + 1


def test_the_same_seed_flies_the_same_path():
    first = fly(the_1960_rocket(), steps=6)
    again = fly(the_1960_rocket(), steps=6)
    assert np.allclose(first.path, again.path)
    assert not np.allclose(fly(the_1960_rocket(), steps=6, seed=11).path, first.path)


def test_the_first_reading_is_the_legacy_draw_for_the_seed():
    flight = fly(the_1960_rocket(), steps=1, seed=3)
    noise = np.sqrt(NOISE_FLOOR) * np.random.RandomState(3).randn(2)
    # One reading against a prior of 4: the posterior mean is the reading, shrunk.
    shrink = 4.001 / (4.001 + NOISE_FLOOR)
    assert np.allclose(flight.means[0], shrink * noise)


def test_a_flight_leaves_the_global_random_state_alone():
    # The legacy global calls are the point: this is the state a flight must not touch.
    np.random.seed(0)  # noqa: NPY002
    expected = np.random.rand()  # noqa: NPY002
    np.random.seed(0)  # noqa: NPY002
    fly(the_1960_rocket(), steps=2)
    assert np.random.rand() == expected  # noqa: NPY002


def test_a_rocket_flies_its_own_world_unless_told_otherwise():
    params = default_params(0.08)
    own = fly(curious_rocket(0.08), steps=6)
    told = fly(curious_rocket(0.08), reference_sensor, params, steps=6)
    assert np.allclose(own.path, told.path)


def test_the_world_can_blind_a_rocket_that_never_modelled_the_blackout():
    params = default_params(0.08)
    blind = fly(the_1960_rocket(), reference_sensor, params, steps=8)
    clear = fly(the_1960_rocket(), steps=8)
    assert not np.allclose(blind.path, clear.path)


def test_a_crash_truncates_the_path_at_the_point_of_impact():
    flight = fly(the_1960_rocket(), hazard=ROCK)
    assert flight.crashed is True
    assert flight.impact_step is not None
    assert len(flight.path) == flight.impact_step + 1
    assert np.linalg.norm(flight.path[-1] - np.asarray(ROCK[0])) < 1.0


def test_a_move_that_crosses_the_rock_and_comes_out_the_far_side_still_hits_it():
    centre, radius = ROCK
    assert _hits(np.array([0.0, 0.0]), np.array([5.0, 0.0]), centre, radius)
    assert not _hits(np.array([0.0, 2.0]), np.array([5.0, 2.0]), centre, radius)
    assert not _hits(np.array([0.0, 0.0]), np.array([1.0, 0.0]), centre, radius)


def test_the_belief_sharpens_once_the_rocket_starts_reading():
    flight = fly(the_1960_rocket(), steps=4)
    traces = [float(np.trace(cov)) for cov in flight.covs]
    assert traces[0] < 2 * 4.0
    assert traces == sorted(traces, reverse=True)
