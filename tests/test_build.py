import numpy as np
import pytest
from cpomdp import Agent, ObservationGoal, StateGoal, probe_model

from cpomdp_workshop.build import (
    curious_rocket,
    goal_for,
    model_with,
    one_node,
    the_1960_rocket,
)
from cpomdp_workshop.planner import compass_actions
from cpomdp_workshop.world import default_params, reference_sensor

MOVES = list(np.asarray(compass_actions().actions))


def probe(model):
    return probe_model(model, model.prior, MOVES)


def test_the_1960_rocket_has_nothing_to_be_curious_about():
    report = probe(model_with())
    lo, hi = report.epistemic_range
    assert report.flattens is True
    assert report.epistemic_varies is False
    assert f"{hi - lo:.6f}" == "0.000000"


def test_the_curious_rocket_can_tell_its_moves_apart():
    report = probe(model_with(reference_sensor, default_params(0.08)))
    lo, hi = report.epistemic_range
    assert report.flattens is False
    assert hi - lo > 0.01


def test_the_goal_follows_the_sensor():
    assert isinstance(goal_for(model_with()), StateGoal)
    assert isinstance(
        goal_for(model_with(reference_sensor, default_params(0.08))), ObservationGoal
    )


def test_the_library_refuses_a_state_goal_on_a_varying_sensor():
    model = model_with(reference_sensor, default_params(0.08))
    with pytest.raises(ValueError, match="fixed sensor"):
        Agent(model, StateGoal(np.asarray([5.0, 0.0])))


def test_the_library_refuses_an_observation_goal_on_a_fixed_sensor():
    model = model_with()
    with pytest.raises(ValueError, match="output regulation"):
        Agent(model, ObservationGoal(np.asarray([5.0, 0.0]), (-0.5, 0.5)))


def test_both_rockets_build_and_choose_a_move():
    for agent in (the_1960_rocket(), curious_rocket(0.08)):
        agent.infer_states(np.asarray([0.1, 0.0]))
        assert np.asarray(agent.sample_action()).shape == (2,)


def test_a_sensor_without_params_is_refused():
    with pytest.raises(ValueError, match="needs its params"):
        model_with(reference_sensor)


def test_one_node_refuses_a_model_it_cannot_reproduce():
    with pytest.raises(ValueError, match="fixed-sensor model"):
        one_node(model_with(reference_sensor, default_params(0.08)))


def test_one_node_carries_the_same_plant():
    backend = one_node(model_with())
    assert np.allclose(np.asarray(backend.model.dynamics), np.eye(2))
    assert np.allclose(np.asarray(backend.model.sensor_noise), 0.05 * np.eye(2))
