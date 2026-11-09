import numpy as np

from cpomdp_workshop.build import (
    curious_rocket,
    model_with,
    pad_preference,
    the_1960_rocket,
)
from cpomdp_workshop.sim import fly
from cpomdp_workshop.solutions import CLEAR_SKY_ROCK, curiosity_decided, efe_split
from cpomdp_workshop.world import BEACON, BLACKOUT, beacon_sensor


def test_the_clear_sky_rock_is_clear_of_the_blackout_and_the_beacon():
    centre, radius = CLEAR_SKY_ROCK
    assert np.linalg.norm(np.asarray(centre) - np.asarray(BLACKOUT)) > 4 * radius
    assert np.linalg.norm(np.asarray(centre) - np.asarray(BEACON)) > 2 * radius


def test_one_split_row_per_move_and_the_epistemic_term_is_positive():
    rocket = curious_rocket(0.08)
    flight = fly(rocket, steps=5)
    split = efe_split(flight, rocket.model, pad_preference())
    assert split.shape == (5, 2)
    assert (split[:, 1] > 0).all()


def test_the_1960_rocket_learns_the_same_from_every_move_it_could_make():
    rocket = the_1960_rocket()
    flight = fly(rocket, steps=3)
    model = model_with()
    first = efe_split(flight, model, pad_preference())[0, 1]
    moved = flight.path.copy()
    moved[1] = moved[0] + np.array([0.0, 0.5])  # a different first move
    other = efe_split(
        type(flight)(moved, flight.means, flight.covs, False, None),
        model,
        pad_preference(),
    )[0, 1]
    assert np.isclose(first, other)


def test_curiosity_decides_no_move_of_a_rocket_that_prefers_the_pad_tightly():
    rocket = curious_rocket(0.08, beacon_sensor, precision=1.0)
    flight = fly(rocket, steps=8)
    decided = curiosity_decided(flight, rocket.model, pad_preference(1.0))
    assert decided.shape == (8,)
    assert not decided.any()


def test_curiosity_decides_some_moves_of_a_loose_rocket():
    rocket = curious_rocket(0.08, beacon_sensor)
    flight = fly(rocket, steps=8)
    assert curiosity_decided(flight, rocket.model, pad_preference()).any()
