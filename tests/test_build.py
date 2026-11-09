import numpy as np
import pytest
from cpomdp import Agent, CouplingGraphBackend

from cpomdp_workshop import _seams
from cpomdp_workshop.build import (
    CURIOUS_PRECISION,
    THRUSTER_COST,
    curious_rocket,
    model_with,
    one_node,
    pad_goal,
    pad_model,
    pad_preference,
    radar_preference,
    the_1960_rocket,
)
from cpomdp_workshop.world import (
    LAUNCH,
    NOISE_FLOOR,
    PAD,
    PAD_GUESS,
    PAD_READINGS,
    beacon_sensor,
    broadcast_noise,
    default_params,
    pad_params,
    reference_sensor,
)


def test_the_fixed_model_reads_the_position_with_the_floor_noise():
    model = model_with()
    assert model.observation_model is None
    assert np.allclose(model.observation_matrix, np.eye(2))
    assert np.allclose(model.observation_noise, NOISE_FLOOR * np.eye(2))


def test_the_sensed_model_reads_its_noise_from_the_sensor():
    params = default_params(0.08)
    model = model_with(reference_sensor, params)
    assert model.observation_model is not None
    assert np.allclose(
        model.observation_noise_at(np.asarray(PAD)), reference_sensor(PAD, params)
    )


def test_a_sensor_without_params_is_refused():
    with pytest.raises(ValueError, match="needs its params"):
        model_with(reference_sensor)


def test_the_pad_goal_charges_the_thruster_cost():
    goal = pad_goal()
    assert np.allclose(goal.target, PAD)
    assert np.allclose(goal.effort, THRUSTER_COST * np.eye(2))
    assert np.allclose(pad_preference().target, PAD)


def test_the_1960_rocket_steers_by_lqr():
    assert "LQRSelector" in repr(the_1960_rocket())


def test_the_curious_rocket_searches_the_compass_set():
    rocket = curious_rocket(0.08)
    assert isinstance(rocket.selector, _seams.RecedingHorizonSelector)
    assert rocket.selector.horizon == 1


def test_the_curious_rocket_takes_any_sensor_and_horizon():
    rocket = curious_rocket(0.4, beacon_sensor, horizon=2)
    assert isinstance(rocket.selector, _seams.RecedingHorizonSelector)
    assert rocket.selector.horizon == 2
    sensor = rocket.model.observation_model
    assert getattr(sensor, "noise_fn", None) is beacon_sensor


def test_both_rockets_choose_a_move():
    for rocket in (the_1960_rocket(), curious_rocket(0.08)):
        rocket.infer_states(np.asarray([0.1, 0.0]))
        assert np.asarray(rocket.sample_action()).shape == (2,)


def test_one_node_refuses_a_model_it_cannot_reproduce():
    with pytest.raises(ValueError, match="fixed-sensor model") as raised:
        one_node(model_with(reference_sensor, default_params(0.08)))
    assert "the 1960 rocket's model" in str(raised.value)


def test_one_node_refuses_a_rocket_and_names_the_model_instead():
    with pytest.raises(TypeError, match="takes a model, not a rocket"):
        one_node(the_1960_rocket())  # ty: ignore[invalid-argument-type]


def test_one_node_carries_the_same_plant():
    backend = one_node(model_with())
    assert isinstance(backend, CouplingGraphBackend)
    assert backend.graph.dims == (2,)
    flat = backend.to_flat_model()
    reference = model_with()
    for name in (
        "dynamics_matrix",
        "control_matrix",
        "dynamics_noise",
        "observation_noise",
    ):
        assert np.allclose(getattr(flat, name), getattr(reference, name))


def test_the_1960_rocket_flies_on_the_graph_backend():
    agent = Agent.from_backend(one_node(model_with()), pad_goal())
    assert "backend=CouplingGraphBackend" in repr(agent)


def test_a_move_shifts_the_rocket_and_leaves_the_pad_where_it_is():
    model = pad_model(broadcast_noise, pad_params(), 1.5)
    assert np.allclose(model.dynamics_matrix, np.eye(4))
    assert np.allclose(model.control_matrix, np.vstack([np.eye(2), np.zeros((2, 2))]))
    assert np.allclose(model.observation_matrix, PAD_READINGS)


def test_the_rocket_starts_at_launch_and_guesses_the_pad_with_its_spread():
    prior = pad_model(broadcast_noise, pad_params(), 1.5).prior
    assert np.allclose(prior.mean, [*LAUNCH, *PAD_GUESS])
    assert np.allclose(np.diag(prior.cov), [0.25, 0.25, 2.25, 2.25])


def test_the_radar_carries_the_goal_and_the_other_readings_barely_count():
    preference = radar_preference()
    assert np.allclose(preference.target, np.zeros(6))
    weights = np.diag(np.asarray(preference.precision))
    assert np.allclose(weights[2:4], CURIOUS_PRECISION)
    assert (weights[[0, 1, 4, 5]] < 1e-6).all()
