"""The names each session's reference cell leaves behind, built fresh.

A checkpoint cell runs one of these and puts the result in the notebook's namespace,
so anyone whose own cell did not run can carry on from the reference. Each checkpoint
holds everything the sessions before it defined, since a later session reads names
from an earlier one. Nothing here reuses an object a learner built: a rocket's model
cannot be swapped once built, so every object is made new.
"""

from cpomdp import Agent

from cpomdp_workshop.build import model_with, one_node, pad_goal, pad_preference
from cpomdp_workshop.check import probe
from cpomdp_workshop.planner import compass_planner
from cpomdp_workshop.sim import fly
from cpomdp_workshop.world import beacon_sensor, default_params, reference_sensor

__all__ = ["after_session_1", "after_session_2", "after_session_3"]

SESSION_2_GAIN = 0.4
SESSION_3_GAIN = 0.08


def after_session_1() -> dict[str, object]:
    """The 1960 rocket, its model and the probe report on it."""
    model = model_with()
    return {"model": model, "rocket": Agent(model, pad_goal()), "report": probe(model)}


def after_session_2() -> dict[str, object]:
    """Session 1's names, plus the beacon sensor and the rocket flown under it."""
    params = default_params(SESSION_2_GAIN)
    curious_model = model_with(beacon_sensor, params)
    planner = compass_planner(curious_model)
    curious = Agent(curious_model, pad_preference(), selector=planner)
    return {
        **after_session_1(),
        "my_sensor": beacon_sensor,
        "beacon_gain": SESSION_2_GAIN,
        "params": params,
        "curious_model": curious_model,
        "planner": planner,
        "curious": curious,
        "flight": fly(curious),
    }


def after_session_3() -> dict[str, object]:
    """Session 2's names, plus the blackout world and the 1960 model as a graph."""
    return {
        **after_session_2(),
        "my_world": reference_sensor,
        "beacon_gain": SESSION_3_GAIN,
        "backend": one_node(model_with()),
    }
