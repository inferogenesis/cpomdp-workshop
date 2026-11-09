"""The cpomdp forms this workshop is written against, on the pinned release.

Each test names one form the package or the notebook relies on. A cpomdp release that
renames a field, a class or a refusal fails here first, before any flight is flown.
"""

import dataclasses

import jax.numpy as jnp
import numpy as np
import pytest
from cpomdp import (
    Agent,
    Belief,
    CallableSensor,
    EFESelector,
    KalmanBackend,
    LinearGaussianModel,
    ObservationGoal,
    Preference,
    ProbeReport,
    StateGoal,
)

from cpomdp_workshop import _seams
from cpomdp_workshop.planner import compass_planner
from cpomdp_workshop.world import PAD, default_params, reference_sensor

I = np.eye(2)
PRIOR = Belief(mean=[0.0, 0.0], cov=4 * I)


def a_sensor():
    return CallableSensor(I, reference_sensor, default_params(0.08))


def a_fixed_model():
    return LinearGaussianModel(
        I,
        control_matrix=I,
        observation_matrix=I,
        observation_noise=0.05 * I,
        dynamics_noise=1e-3 * I,
        prior=PRIOR,
    )


def a_sensed_model():
    return LinearGaussianModel(
        I,
        control_matrix=I,
        dynamics_noise=1e-3 * I,
        prior=PRIOR,
        observation_model=a_sensor(),
    )


def a_selector(model):
    actions = _seams.FiniteActionSet(np.vstack([np.zeros(2), 0.5 * I]), version="t")
    return _seams.RecedingHorizonSelector(
        _seams.EnumeratedEFESearch(model, actions, horizon=1)
    )


def test_the_model_takes_role_named_fields():
    model = a_fixed_model()
    assert np.allclose(model.dynamics_matrix, I)
    assert np.allclose(model.control_matrix, I)
    assert np.allclose(model.observation_noise, 0.05 * I)


@pytest.mark.parametrize(
    "old_name", ["dynamics", "control", "sensor_model", "sensor_noise", "observation"]
)
def test_the_0_4_field_names_are_refused(old_name):
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        LinearGaussianModel(I, prior=PRIOR, **{old_name: I})  # ty: ignore[invalid-argument-type]


def test_a_sensor_supplies_the_matrix_and_the_noise_at_the_prior_mean():
    model = a_sensed_model()
    assert np.allclose(model.observation_matrix, I)
    expected = reference_sensor(np.zeros(2), default_params(0.08))
    assert np.allclose(model.observation_noise, expected)


def test_a_noise_beside_the_sensor_must_be_the_sensors_own():
    with pytest.raises(ValueError, match="not the observation_model's R"):
        LinearGaussianModel(
            I,
            control_matrix=I,
            dynamics_noise=1e-3 * I,
            prior=PRIOR,
            observation_model=a_sensor(),
            observation_noise=0.06 * I,
        )


def test_a_state_goal_carries_the_thruster_cost():
    goal = StateGoal(PAD, precision=I, effort=100 * I)
    assert np.allclose(goal.effort, 100 * I)


def test_a_state_goal_under_a_state_dependent_sensor_is_refused():
    # Why the curious rocket prefers a reading, and so pays for its noise.
    with pytest.raises(ValueError, match="a StateGoal needs a fixed sensor"):
        Agent(a_sensed_model(), StateGoal(PAD))


@pytest.mark.parametrize(
    ("cov", "refusal"),
    [
        (2.25, "must be a square 2-D matrix"),
        (2.25 * np.eye(2), "must be 4x4 to match a 4-D mean"),
    ],
)
def test_a_prior_covariance_of_the_wrong_shape_is_refused_by_name(cov, refusal):
    # Task 3's second line is the prior's covariance, and cpomdp's refusal is the fix.
    with pytest.raises(ValueError, match=refusal):
        Belief(mean=np.zeros(4), cov=cov)


def test_a_goal_beside_a_selector_is_refused():
    model = a_sensed_model()
    with pytest.raises(TypeError, match="selector= takes a Preference"):
        Agent(model, StateGoal(PAD), selector=a_selector(model))


def test_a_preference_without_a_selector_is_refused():
    with pytest.raises(TypeError, match="a Preference carries no search"):
        Agent(a_sensed_model(), Preference(PAD))


def test_an_observation_goal_cannot_search_a_two_dimensional_move():
    with pytest.raises(ValueError, match="For p>1 use EnumeratedEFESearch"):
        Agent(a_sensed_model(), ObservationGoal(PAD, action_bounds=(-0.5, 0.5)))


def test_the_efe_grid_names_the_search_to_use_instead():
    with pytest.raises(
        ValueError, match=r"searches a 1-D action grid \(p=1\); got p=2"
    ):
        EFESelector(a_sensed_model(), action_bounds=(-0.5, 0.5))


def test_the_search_drivers_read_the_preference_as_an_observation():
    model = a_sensed_model()
    with pytest.raises(ValueError, match="as an observation, a vector of length 2"):
        Agent(model, Preference([5.0, 0.0, 0.0]), selector=a_selector(model))


def test_an_agent_swaps_its_backend_through_from_backend():
    agent = Agent.from_backend(KalmanBackend(a_fixed_model()), StateGoal(PAD))
    assert "backend=KalmanBackend" in repr(agent)


def test_the_model_on_an_agent_is_read_only():
    agent = Agent(a_fixed_model(), StateGoal(PAD))
    with pytest.raises(AttributeError, match="has no setter"):
        agent.model = a_fixed_model()  # ty: ignore[invalid-assignment]


def test_the_probe_report_carries_the_fields_the_notebook_reads():
    names = {field.name for field in dataclasses.fields(ProbeReport)}
    assert {
        "observation_noise_varies",
        "observation_noise_spread",
        "dynamics_noise_varies",
        "dynamics_noise_spread",
        "epistemic_varies",
        "epistemic_range",
    } <= names
    assert isinstance(ProbeReport.flattens, property)
    assert callable(ProbeReport.summary)


def test_the_efe_trace_splits_g_into_its_two_terms_exactly():
    # The curiosity switch reweights these two terms, and at weight one it must
    # rank plans exactly as the search ranks them by g.
    plan = jnp.full((2, 2), 0.5)
    trace = _seams.policy_efe_trace(a_sensed_model(), PRIOR, plan, Preference(PAD))
    assert np.array_equal(
        np.asarray(trace.g), np.asarray(trace.pragmatic - trace.epistemic)
    )


def test_two_agents_share_one_model_with_a_custom_selector_beside_a_search():
    model = a_sensed_model()
    preference = Preference(PAD, precision=0.1 * I)
    curious = Agent(model, preference, selector=compass_planner(model))
    incurious = Agent(
        model, preference, selector=compass_planner(model, curiosity=False)
    )
    for agent in (curious, incurious):
        agent.infer_states(np.zeros(2))
        assert np.asarray(agent.sample_action()).shape == (2,)
    assert curious.belief is not incurious.belief
