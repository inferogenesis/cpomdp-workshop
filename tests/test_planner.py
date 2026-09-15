from typing import cast

import jax.numpy as jnp
import numpy as np
from cpomdp import Belief, CallableSensor, LinearGaussianModel, ObservationModel

from cpomdp_workshop.planner import STEP, compass_actions, compass_planner
from cpomdp_workshop.world import default_params, reference_sensor

I = np.eye(2)


def a_curious_model():
    params = default_params(0.08)
    sensor = cast(ObservationModel, CallableSensor(I, reference_sensor, params))
    return LinearGaussianModel(
        dynamics=I,
        sensor_model=I,
        dynamics_noise=1e-3 * I,
        sensor_noise=np.asarray(reference_sensor(jnp.zeros(2), params)),
        prior=Belief(mean=[0.0, 0.0], cov=4 * I),
        control=I,
        observation=sensor,
    )


def test_the_action_set_is_stay_plus_eight_compass_moves():
    actions = compass_actions()
    assert actions.size == 9
    assert actions.version == "c"
    moves = np.asarray(actions.actions)
    assert np.allclose(moves[0], [0.0, 0.0])
    assert np.allclose(np.linalg.norm(moves[1:], axis=1), STEP)
    assert len({tuple(np.round(m, 12)) for m in moves}) == 9


def test_the_planner_decides_rather_than_samples():
    planner = compass_planner(a_curious_model())
    assert planner.warrant.value == "PROVED"
    assert str(planner.certificate) == "PROVED (finite set, |A|^H = 9, visited 9)"


def test_the_horizon_is_visible_in_the_certificate_and_the_cost():
    planner = compass_planner(a_curious_model(), horizon=2)
    assert str(planner.certificate) == "PROVED (finite set, |A|^H = 81, visited 81)"
    assert planner.cost_per_cycle == 81 * 2
