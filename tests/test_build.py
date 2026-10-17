import numpy as np
import pytest
from cpomdp import Agent, CouplingGraphBackend

from cpomdp_workshop import _seams
from cpomdp_workshop.build import (
    THRUSTER_COST,
    curious_rocket,
    model_with,
    one_node,
    pad_goal,
    pad_preference,
    the_1960_rocket,
)
from cpomdp_workshop.world import (
    NOISE_FLOOR,
    PAD,
    beacon_sensor,
    default_params,
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
    with pytest.raises(ValueError, match="fixed-sensor model"):
        one_node(model_with(reference_sensor, default_params(0.08)))


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
