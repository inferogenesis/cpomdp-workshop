"""Assembling the two rockets, and the same model expressed as a one-node graph.

Both rockets fly the same plant: position in the plane, a move added straight to it,
a reading of where they are. They differ in one thing. The 1960 rocket's reading is
equally noisy everywhere, so it is handed a state to reach and steers by LQR. The
curious rocket's reading depends on where it is, so it is handed a reading to prefer
and searches the move set for the sequence that best trades getting there against
seeing clearly.

Which goal goes with which sensor is not a free choice: cpomdp refuses a state goal on
a state-dependent sensor and refuses an observation goal on a fixed one. The routing
here follows the sensor so a caller never meets that refusal by accident.
"""

from typing import cast

import numpy as np
from cpomdp import (
    Agent,
    Belief,
    CallableSensor,
    CouplingGraph,
    CouplingGraphBackend,
    GaussianObservation,
    GaussianTransition,
    LinearGaussianModel,
    LQRSelector,
    ObservationGoal,
    ObservationModel,
    StateGoal,
)
from cpomdp.control import LQRController

from cpomdp_workshop.planner import STEP, ThrustLimited, compass_planner
from cpomdp_workshop.world import (
    DYNAMICS_NOISE,
    LAUNCH,
    NOISE_FLOOR,
    PAD,
    PRIOR_COV,
    NoiseFn,
    SensorParams,
    beacon_sensor,
    default_params,
    reference_sensor,
)

__all__ = [
    "beacon_rocket",
    "curious_rocket",
    "goal_for",
    "model_with",
    "one_node",
    "rocket",
    "the_1960_rocket",
]

_I = np.eye(2)


def model_with(
    sensor: NoiseFn | None = None, params: SensorParams | None = None
) -> LinearGaussianModel:
    """The rocket's generative model, with or without a state-dependent sensor.

    Args:
        sensor: a noise function ``R(x, params)``, or ``None`` for a reading that is
            equally noisy everywhere.
        params: the sensor's parameters. Required when ``sensor`` is given.

    Returns:
        A two-dimensional model: the rocket stays where it is unless moved, a move
        adds straight to the position, and the reading is the position plus noise.

    Raises:
        ValueError: if a sensor is given without parameters.
    """
    prior = Belief(mean=np.asarray(LAUNCH), cov=PRIOR_COV * _I)
    if sensor is None:
        observation, sensor_noise = None, NOISE_FLOOR * _I
    elif params is None:
        raise ValueError("a sensor needs its params; pass default_params(k).")
    else:
        # CallableSensor sets `is_fixed` as a class attribute, which ty reads as
        # incompatible with the protocol's writable `is_fixed: bool` (cpomdp 0.4.4).
        observation = cast(ObservationModel, CallableSensor(_I, sensor, params))
        sensor_noise = np.asarray(sensor(prior.mean, params))
    return LinearGaussianModel(
        dynamics=_I,
        sensor_model=_I,
        dynamics_noise=DYNAMICS_NOISE * _I,
        sensor_noise=sensor_noise,
        prior=prior,
        control=_I,
        observation=observation,
    )


def goal_for(model: LinearGaussianModel) -> StateGoal | ObservationGoal:
    """The goal cpomdp will accept for this model's sensor — the pad, either way.

    Args:
        model: from [`model_with`][cpomdp_workshop.model_with].

    Returns:
        A state goal for a fixed sensor, an observation goal for a varying one.
    """
    if model.observation is None:
        return StateGoal(np.asarray(PAD), precision=_I)
    return ObservationGoal(np.asarray(PAD), (-STEP, STEP), precision=_I)


def rocket(model: LinearGaussianModel) -> Agent:
    """An agent that flies ``model`` to the pad.

    Args:
        model: from [`model_with`][cpomdp_workshop.model_with].

    Returns:
        An agent carrying the goal its sensor allows, and, where the sensor varies,
        the compass search that plans under it.
    """
    goal = goal_for(model)
    if isinstance(goal, StateGoal):
        controller = LQRController(
            model, goal_precision=goal.precision, effort_penalty=_I
        )
        return Agent(model, goal, selector=ThrustLimited(LQRSelector(controller)))
    return Agent(model, goal, selector=compass_planner(model))


def the_1960_rocket() -> Agent:
    """The Kalman filter with a goal bolted on: one fixed reading, steered by LQR."""
    return rocket(model_with())


def curious_rocket(beacon_gain: float) -> Agent:
    """A rocket whose reading sharpens near the beacon and fails in the blackout.

    Args:
        beacon_gain: k, how hard the beacon pulls. Zero leaves the beacon flat and
            the blackout still blinding.

    Returns:
        An agent searching the compass set under the full sensor field.
    """
    return rocket(model_with(reference_sensor, default_params(beacon_gain)))


def beacon_rocket(beacon_gain: float) -> Agent:
    """A rocket under the beacon alone — the world of session 2, with no blackout.

    Args:
        beacon_gain: k, how hard the beacon pulls.

    Returns:
        An agent searching the compass set under a field whose only feature is the
        beacon.
    """
    return rocket(model_with(beacon_sensor, default_params(beacon_gain)))


def one_node(model: LinearGaussianModel) -> CouplingGraphBackend:
    """The same model as a graph of one node, to run it on the other backend.

    A coupling graph is for branching models, where separate variables evolve at
    their own rates and only some of them are seen. A single node with no edges is
    the degenerate case: the same filtering problem the Kalman backend solves, routed
    through the message-passing code instead. Two engines on one model is the point —
    they should agree to the last few digits.

    Args:
        model: a fixed-sensor model from [`model_with`][cpomdp_workshop.model_with].

    Returns:
        A backend to hand to ``Agent(model, goal, backend=...)``.

    Raises:
        ValueError: if the model's sensor varies with the state. That model cannot be
            flattened back, so the two backends would not be running the same thing.
    """
    if model.observation is not None:
        raise ValueError(
            "one_node takes a fixed-sensor model; a state-dependent R(x) on a graph "
            "node is not the same model as the flat one."
        )
    graph = CouplingGraph(
        root=0,
        dims=[model.n_states],
        couplings=[],
        observations={0: GaussianObservation(model.sensor_model, model.sensor_noise)},
    )
    transition = GaussianTransition(model.dynamics, model.dynamics_noise)
    return CouplingGraphBackend(graph, [transition], control=model.control)
