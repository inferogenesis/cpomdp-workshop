"""The two rockets, and the 1960 one again as a graph of one node.

Both rockets fly the same plant: a position in the plane, a move added straight to it,
a reading of where they are. They differ in the reading. The 1960 rocket's reading is
equally noisy everywhere, so no move can teach it more than another. It is handed a
state to reach and steers by LQR. The curious rocket's reading depends on where it is.
It is handed a reading to prefer and searches the compass moves for the plan that
best trades getting there against seeing clearly.
"""

import numpy as np
from cpomdp import (
    Agent,
    Belief,
    CallableSensor,
    CouplingGraph,
    CouplingGraphBackend,
    FixedSensor,
    GaussianTransition,
    LinearGaussianModel,
    Preference,
    StateGoal,
)

from cpomdp_workshop._seams import as_observation_model
from cpomdp_workshop.planner import compass_planner
from cpomdp_workshop.world import (
    DYNAMICS_NOISE,
    LAUNCH,
    NOISE_FLOOR,
    PAD,
    PRIOR_COV,
    NoiseFn,
    SensorParams,
    default_params,
    reference_sensor,
)

__all__ = [
    "THRUSTER_COST",
    "curious_rocket",
    "model_with",
    "one_node",
    "pad_goal",
    "pad_preference",
    "the_1960_rocket",
]

THRUSTER_COST = 100.0  # effort

_IDENTITY = np.eye(2)


def model_with(
    sensor: NoiseFn | None = None, params: SensorParams | None = None
) -> LinearGaussianModel:
    """The rocket's generative model, with or without a state-dependent sensor.

    Args:
        sensor: a noise function ``R(x, params)``, or ``None`` for a reading that is
            equally noisy everywhere.
        params: the sensor's parameters. Required when ``sensor`` is given.

    Returns:
        A two-dimensional model. The rocket stays where it is unless moved, a move
        adds straight to the position, and the reading is the position plus noise.

    Raises:
        ValueError: if a sensor is given without parameters.
    """
    if sensor is None:
        observation_model = None
        observation_matrix, observation_noise = (
            _IDENTITY,
            NOISE_FLOOR * _IDENTITY,
        )  # C, R
    elif params is None:
        raise ValueError("a sensor needs its params. Pass default_params(k).")
    else:
        # The model reads C, and R at the prior mean, from the sensor itself.
        observation_model = as_observation_model(
            CallableSensor(_IDENTITY, sensor, params)
        )
        observation_matrix = observation_noise = None
    return LinearGaussianModel(
        _IDENTITY,
        control_matrix=_IDENTITY,
        observation_matrix=observation_matrix,
        observation_noise=observation_noise,
        dynamics_noise=DYNAMICS_NOISE * _IDENTITY,
        prior=Belief(mean=np.asarray(LAUNCH), cov=PRIOR_COV * _IDENTITY),
        observation_model=observation_model,
    )


def pad_goal() -> StateGoal:
    """The pad as a state to reach, with every unit of thrust charged for.

    The effort weight is the thruster cost. Left at its default of one, LQR would
    cover more than half the distance to the pad in its first move.
    """
    return StateGoal(
        np.asarray(PAD), precision=_IDENTITY, effort=THRUSTER_COST * _IDENTITY
    )


def pad_preference() -> Preference:
    """The pad as a reading to prefer, which is what a compass search scores."""
    return Preference(np.asarray(PAD), precision=_IDENTITY)


def the_1960_rocket() -> Agent:
    """The Kalman filter with a goal bolted on: one fixed reading, steered by LQR."""
    return Agent(model_with(), pad_goal())


def curious_rocket(
    beacon_gain: float, sensor: NoiseFn = reference_sensor, *, horizon: int = 1
) -> Agent:
    """A rocket whose reading depends on where it is, searching the compass moves.

    Args:
        beacon_gain: k, how fast the reading degrades away from the beacon.
        sensor: the noise field the rocket models. The default has the beacon and
            the blackout. ``beacon_sensor`` is the beacon alone.
        horizon: how many moves ahead the search looks, H.

    Returns:
        An agent that prefers to read the pad and plans every step over every
        compass plan of length ``horizon``.
    """
    model = model_with(sensor, default_params(beacon_gain))
    planner = compass_planner(model, horizon=horizon)
    return Agent(model, pad_preference(), selector=planner)


def one_node(model: LinearGaussianModel) -> CouplingGraphBackend:
    """The same model as a graph of one node, to run it on the message-passing engine.

    A coupling graph is for branching models, where separate variables evolve at
    their own rates and only some of them are read. A single node with no edges is
    the degenerate case: the filtering problem the Kalman backend solves, routed
    through the graph code instead. The two engines should agree to the last few
    digits.

    Args:
        model: a fixed-sensor model from [`model_with`][cpomdp_workshop.model_with].

    Returns:
        A backend for ``Agent.from_backend``. Its graph is ``backend.graph``.

    Raises:
        ValueError: if the model's sensor depends on the state. The backends would
            then not be running the same model.
    """
    if model.observation_model is not None:
        raise ValueError(
            "one_node takes a fixed-sensor model. Build it with model_with()."
        )
    sensor = as_observation_model(
        FixedSensor(model.observation_matrix, observation_noise=model.observation_noise)
    )
    graph = CouplingGraph(
        dims=[model.n_states], couplings=[], observation_models={0: sensor}
    )
    transition = GaussianTransition(
        model.dynamics_matrix, dynamics_noise=model.dynamics_noise
    )
    return CouplingGraphBackend(
        graph, [transition], prior=model.prior, control_matrix=model.control_matrix
    )
