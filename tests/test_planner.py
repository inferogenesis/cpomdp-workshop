import jax.numpy as jnp
import numpy as np
from cpomdp import Belief, Preference

from cpomdp_workshop import _seams
from cpomdp_workshop.build import model_with
from cpomdp_workshop.planner import (
    AXIS_VERSION,
    COMPASS_VERSION,
    STEP,
    axis_actions,
    compass_actions,
    compass_moves,
    compass_planner,
)
from cpomdp_workshop.world import PAD, default_params, reference_sensor

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
