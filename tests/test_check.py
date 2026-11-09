import re

import jax.numpy as jnp
import numpy as np
import pytest
from cpomdp import Agent, Belief, KalmanBackend, LinearGaussianModel

from cpomdp_workshop.build import model_with, pad_goal, pad_preference, the_1960_rocket
from cpomdp_workshop.check import (
    LANDED,
    STRAIGHT_BEND,
    check_agent,
    check_backends,
    check_my_sensor,
    grade_sensor,
    route_bend,
    show_landings,
)
from cpomdp_workshop.planner import compass_planner
from cpomdp_workshop.plot import curiosity_pair
from cpomdp_workshop.sim import fly
from cpomdp_workshop.world import (
    LAUNCH,
    PAD,
    beacon_sensor,
    default_params,
    reference_sensor,
)


def numpy_beacon(x, params):
    distance_squared = jnp.sum((x - params["beacon"]) ** 2)
    capped = np.minimum(distance_squared, params["beacon_cap"])
    return jnp.eye(2) * (params["noise_floor"] + params["beacon_gain"] * capped)


def flat_route_params(k):
    # Every point of the straight route is past a cap of 1, so the noise is flat
    # along it and no move there teaches more than another.
    return {**default_params(k), "beacon_cap": jnp.asarray(1.0)}


def test_a_route_with_flat_noise_stays_straight():
    assert grade_sensor(beacon_sensor, flat_route_params(0.08)).grade == "straight"


def test_the_starting_beacon_bends_the_path_and_lands():
    grade = grade_sensor(beacon_sensor, default_params(0.08))
    assert grade.grade == "bends"
    assert grade.matches == "beacon_sensor"


def test_a_beacon_too_strong_stops_the_rocket_short():
    grade = grade_sensor(beacon_sensor, default_params(0.4))
    assert grade.grade == "stops short"
    assert grade.distance is not None
    assert grade.distance > 1.0


def test_the_full_field_is_recognised():
    assert grade_sensor(reference_sensor, default_params(0.08)).matches == (
        "reference_sensor"
    )


def test_a_sensor_jax_cannot_trace_is_caught_before_it_flies():
    grade = grade_sensor(numpy_beacon, default_params(0.4))
    assert grade.grade == "broken"
    assert "jnp.minimum" in grade.message


def test_a_flat_sensor_is_the_1960_rockets():
    def flat(x, params):
        return 0.05 * jnp.eye(2)

    assert grade_sensor(flat, default_params(0.4)).grade == "flat"


def test_a_sensor_of_the_wrong_shape_is_told_the_shape_it_needs():
    def vector(x, params):
        return jnp.ones(2)

    grade = grade_sensor(vector, default_params(0.4))
    assert grade.grade == "broken"
    assert "2x2" in grade.message


def uncapped_beacon(x, params):
    distance_squared = jnp.sum((x - params["beacon"]) ** 2)
    return jnp.eye(2) * (
        params["noise_floor"] + params["beacon_gain"] * distance_squared
    )


def unwritten_beacon(x, params):
    return jnp.eye(2) * (params["noise_floor"] + ...)


def test_a_landing_wobble_is_no_bend():
    grade = grade_sensor(beacon_sensor, flat_route_params(0.4))
    assert grade.grade == "straight"
    assert grade.bend is not None
    assert grade.bend < STRAIGHT_BEND


def test_the_bend_is_measured_only_while_the_rocket_is_away_from_the_pad():
    model = model_with(beacon_sensor, flat_route_params(0.4))
    flight = fly(Agent(model, pad_preference(), selector=compass_planner(model)))
    away = np.linalg.norm(flight.path - np.asarray(PAD), axis=1) > LANDED
    assert route_bend(flight) == float(np.abs(flight.path[away, 1]).max())
    assert route_bend(flight) < float(np.abs(flight.path[:, 1]).max())


def test_a_broken_sensor_stops_the_cell_with_the_fix_last():
    with pytest.raises(ValueError, match=re.escape("jnp.minimum")) as raised:
        check_my_sensor(numpy_beacon, default_params(0.4))
    assert str(raised.value).startswith("numpy_beacon is not a sensor yet")
    assert raised.value.__cause__ is None


def test_a_broken_sensors_message_fits_a_narrow_cell():
    with pytest.raises(ValueError, match="not a sensor yet") as raised:
        check_my_sensor(numpy_beacon, default_params(0.4))
    assert max(len(line) for line in str(raised.value).splitlines()) <= 80


def test_a_sensor_that_flies_but_differs_from_the_beacon_says_so():
    grade = grade_sensor(uncapped_beacon, default_params(0.4))
    assert grade.matches is None
    assert "does not match the workshop's beacon_sensor" in grade.message


def test_a_matching_sensor_says_so_whatever_it_does():
    for beacon_gain in (0.08, 0.4, 0.6):
        grade = grade_sensor(beacon_sensor, default_params(beacon_gain))
        assert "It matches the workshop's beacon_sensor." in grade.message


def test_a_setting_given_as_a_tuple_is_named_with_its_fix():
    params = {**default_params(0.4), "beacon": (2.5, 0.0)}
    grade = grade_sensor(beacon_sensor, params)  # ty: ignore[invalid-argument-type]
    assert grade.grade == "broken"
    assert 'params["beacon"] is a tuple' in grade.message
    assert "jnp.array" in grade.message


def test_a_line_still_holding_the_ellipsis_is_named():
    grade = grade_sensor(unwritten_beacon, default_params(0.4))
    assert grade.grade == "broken"
    assert "still holds ..." in grade.message


def test_the_reference_rocket_is_told_it_matches(capsys):
    check_agent(the_1960_rocket())
    assert capsys.readouterr().out.splitlines()[-1] == (
        "It matches the workshop's 1960 rocket."
    )


@pytest.mark.parametrize(
    ("observation_matrix", "observation_noise", "named"),
    [
        (jnp.ones((2, 2)), 0.05 * jnp.eye(2), "observation_matrix"),
        (jnp.eye(2), jnp.sqrt(0.05) * jnp.eye(2), "observation_noise"),
    ],
)
def test_a_rocket_that_flies_but_differs_is_told_which_part(
    capsys, observation_matrix, observation_noise, named
):
    model = LinearGaussianModel(
        jnp.eye(2),
        control_matrix=jnp.eye(2),
        observation_matrix=observation_matrix,
        observation_noise=observation_noise,
        dynamics_noise=1e-3 * jnp.eye(2),
        prior=Belief(mean=LAUNCH, cov=4 * jnp.eye(2)),
    )
    check_agent(Agent(model, pad_goal()))
    assert capsys.readouterr().out.splitlines()[-1] == (
        f"It differs from the workshop's 1960 rocket in {named}."
    )


def test_a_backend_left_unwritten_is_named():
    with pytest.raises(TypeError, match=re.escape("backend is still ...")):
        check_backends(KalmanBackend(model_with()), ...)  # ty: ignore[invalid-argument-type]


def test_every_flight_gets_a_line_saying_whether_it_landed(capsys):
    flights = curiosity_pair(0.4)
    show_landings(flights)
    assert capsys.readouterr().out.splitlines() == [
        "curiosity on   ended 2.79 from the pad, did not land",
        "curiosity off  ended 0.24 from the pad, landed",
    ]


def test_a_straight_route_is_told_to_raise_k():
    grade = grade_sensor(beacon_sensor, flat_route_params(0.08))
    assert "Try raising it" in grade.message
