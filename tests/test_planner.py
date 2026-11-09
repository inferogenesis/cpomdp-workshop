import jax.numpy as jnp
import numpy as np
import pytest
from cpomdp import Agent, Belief, Preference

from cpomdp_workshop import _seams
from cpomdp_workshop.build import model_with
from cpomdp_workshop.planner import (
    AXIS_VERSION,
    COMPASS_VERSION,
    STEP,
    WeightedEFESelector,
    axis_actions,
    compass_actions,
    compass_moves,
    compass_planner,
)
from cpomdp_workshop.sim import fly
from cpomdp_workshop.world import PAD, beacon_sensor, default_params, reference_sensor

I = np.eye(2)


def a_curious_model(noise_fn=reference_sensor):
    return model_with(noise_fn, default_params(0.08))


def test_the_compass_set_is_stay_plus_eight_moves_of_one_step():
    actions = compass_actions()
    assert actions.size == 9
    assert actions.version == COMPASS_VERSION == "compass-9"
    moves = np.asarray(actions.actions)
    assert np.allclose(moves[0], [0.0, 0.0])
    assert np.allclose(np.linalg.norm(moves[1:], axis=1), STEP)
    assert len({tuple(np.round(m, 12)) for m in moves}) == 9


def test_the_compass_moves_are_the_set_as_a_list():
    moves = compass_moves()
    assert len(moves) == 9
    assert np.allclose(np.asarray(moves), np.asarray(compass_actions().actions))


def test_the_axis_set_is_stay_plus_four_moves():
    actions = axis_actions()
    assert actions.size == 5
    assert actions.version == AXIS_VERSION == "axis-5"
    moves = np.asarray(actions.actions)
    assert np.allclose(np.abs(moves).sum(axis=1), [0.0] + [STEP] * 4)


def test_the_planner_holds_no_certificate_until_it_has_planned():
    planner = compass_planner(a_curious_model())
    assert planner.certificate is None
    planner.select(a_curious_model().prior, Preference(PAD))
    assert planner.certificate is not None
    assert planner.warrant.value == "PROVED"


def test_the_chunked_search_plans_the_same_move():
    model = a_curious_model()
    belief, preference = model.prior, Preference(PAD)
    front_loaded = compass_planner(model, horizon=2)
    chunked = compass_planner(model, horizon=2, search=_seams.ChunkedEFESearch)
    assert np.allclose(
        front_loaded.select(belief, preference), chunked.select(belief, preference)
    )


traces = []


def counted_sensor(x, params):
    traces.append(None)
    return reference_sensor(x, params)


def test_a_second_plan_reuses_the_first_plans_program():
    model = a_curious_model(counted_sensor)
    planner = compass_planner(model)
    preference = Preference(PAD)
    built = len(traces)  # the model and the sensor each call it once to check it
    planner.select(model.prior, preference)
    traced = len(traces)
    planner.select(Belief(mean=jnp.asarray([1.0, 0.5]), cov=0.3 * I), preference)
    assert traced > built
    assert len(traces) == traced


LOOSE = Preference(np.asarray(PAD), precision=0.1 * I)


def a_beacon_model(k):
    return model_with(beacon_sensor, default_params(k))


def summed_terms(model, belief, plan, preference):
    trace = _seams.policy_efe_trace(model, belief, plan, preference)
    return float(jnp.sum(trace.pragmatic)), float(jnp.sum(trace.epistemic))


# Seed 3's first decision at k = 0.4: the belief is wide and north-east of the
# straight line, so north-east teaches more and east costs less.
FIRST_DECISION = Belief(mean=jnp.asarray([1.63, 0.40]), cov=1.08**2 * I)


def test_with_curiosity_off_the_planner_takes_the_lowest_pragmatic_move():
    model = a_beacon_model(0.4)
    moves = compass_moves()
    pragmatic = [
        summed_terms(model, FIRST_DECISION, move[None, :], LOOSE)[0] for move in moves
    ]
    chosen = compass_planner(model, curiosity=False).select(FIRST_DECISION, LOOSE)
    assert np.allclose(chosen, moves[int(np.argmin(pragmatic))])


def test_switching_curiosity_off_changes_a_loose_rockets_first_decision():
    model = a_beacon_model(0.4)
    on = compass_planner(model).select(FIRST_DECISION, LOOSE)
    off = compass_planner(model, curiosity=False).select(FIRST_DECISION, LOOSE)
    assert np.allclose(on, [STEP / np.sqrt(2), STEP / np.sqrt(2)])
    assert np.allclose(off, [STEP, 0.0])


def test_curiosity_off_scores_every_plan_two_moves_ahead():
    model = a_beacon_model(0.08)
    selector = compass_planner(model, horizon=2, curiosity=False)
    assert isinstance(selector, WeightedEFESelector)
    assert selector.horizon == 2
    assert selector.n_plans == 81
    assert selector.epistemic_weight == 0.0


def fly_both(k, seed, horizon):
    model = a_beacon_model(k)
    searched = Agent(model, LOOSE, selector=compass_planner(model, horizon=horizon))
    weighted = Agent(
        model,
        LOOSE,
        selector=WeightedEFESelector(
            model, compass_actions(), horizon=horizon, epistemic_weight=1.0
        ),
    )
    return fly(searched, seed=seed), fly(weighted, seed=seed)


@pytest.mark.slow
@pytest.mark.parametrize("horizon", [1, 2])
def test_at_weight_one_the_selector_flies_the_compass_planners_route(horizon):
    for k in (0.08, 0.4):
        for seed in range(10):
            searched, weighted = fly_both(k, seed, horizon)
            assert np.array_equal(searched.path, weighted.path), (k, seed)
