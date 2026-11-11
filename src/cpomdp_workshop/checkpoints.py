"""The names each session's reference cell leaves behind, built fresh.

A checkpoint cell runs one of these and puts the result in the notebook's namespace,
so anyone whose own cell did not run can carry on from the reference. Each checkpoint
holds everything the sessions before it defined, since a later session reads names
from an earlier one. Nothing here reuses an object a learner built: a rocket's model
cannot be swapped once built, so every object is made new.
"""

from cpomdp import Agent

from cpomdp_workshop.build import (
    model_with,
    pad_goal,
    pad_model,
    pad_preference,
    radar_preference,
)
from cpomdp_workshop.check import probe
from cpomdp_workshop.planner import compass_planner
from cpomdp_workshop.sim import fly, fly_pad
from cpomdp_workshop.world import (
    PAD_SPREAD_WORLD,
    beacon_sensor,
    broadcast_noise,
    default_params,
    pad_params,
)

__all__ = ["after_session_1", "after_session_2", "after_session_3"]

SESSION_2_GAIN = 0.4


def after_session_1() -> dict[str, object]:
    """The 1960 rocket, its model and the probe report on it."""
    model = model_with()
    return {"model": model, "rocket": Agent(model, pad_goal()), "report": probe(model)}


def after_session_2() -> dict[str, object]:
    """Session 1's names, plus the beacon sensor and one rocket flown both ways.

    The rocket flies once with curiosity on and once with it off, as the attempt cell
    flies it.
    """
    params = default_params(SESSION_2_GAIN)
    curious_model = model_with(beacon_sensor, params)
    preference = pad_preference()
    planner = compass_planner(curious_model)
    curious = Agent(curious_model, preference, selector=planner)
    incurious = Agent(
        curious_model,
        preference,
        selector=compass_planner(curious_model, curiosity=False),
    )
    return {
        **after_session_1(),
        "my_sensor": beacon_sensor,
        "beacon_gain": SESSION_2_GAIN,
        "params": params,
        "curious_model": curious_model,
        "preference": preference,
        "planner": planner,
        "curious": curious,
        "incurious": incurious,
        "flights": {"curiosity on": fly(curious), "curiosity off": fly(incurious)},
    }


def after_session_3() -> dict[str, object]:
    """Session 2's names, with the hidden pad's put over them.

    Task 3 reuses ``params``, ``model``, ``preference``, ``curious``, ``incurious``
    and ``flights`` for the hidden-pad rocket, as its attempt cell does.
    """
    params = pad_params()
    model = pad_model(broadcast_noise, params, PAD_SPREAD_WORLD)
    preference = radar_preference()
    curious = Agent(model, preference, selector=compass_planner(model))
    incurious = Agent(
        model, preference, selector=compass_planner(model, curiosity=False)
    )
    return {
        **after_session_2(),
        "params": params,
        "pad_spread": PAD_SPREAD_WORLD,
        "prior_mean": model.prior.mean,
        "prior_cov": model.prior.cov,
        "model": model,
        "preference": preference,
        "curious": curious,
        "incurious": incurious,
        "flights": {
            "curiosity on": fly_pad(curious),
            "curiosity off": fly_pad(incurious),
        },
    }
