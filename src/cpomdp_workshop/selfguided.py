"""The self-guided tail: models past what one flat filter can hold.

A rocket towing a probe is two nodes. The engine moves the rocket. The probe trails
halfway back toward the rocket on every step. The full field, beacon and blackout,
reads the probe and not the rocket. The coupling shifts the probe's predicted mean,
and the field's noise is read at the predicted mean. No single fixed linear-Gaussian
model reproduces that, and cpomdp refuses to pretend otherwise.
"""

import numpy as np
from cpomdp import (
    Belief,
    CallableSensor,
    Coupling,
    CouplingGraph,
    CouplingGraphBackend,
    GaussianTransition,
)

from cpomdp_workshop._seams import as_observation_model
from cpomdp_workshop.world import (
    DYNAMICS_NOISE,
    PRIOR_COV,
    default_params,
    reference_sensor,
)

__all__ = ["TETHER_PULL", "two_node_backend"]

TETHER_PULL = 0.5
"""How far the probe closes on the rocket each step, as a fraction of the gap."""

_IDENTITY = np.eye(2)


def two_node_backend(beacon_gain: float = 0.08) -> CouplingGraphBackend:
    """A rocket and the probe it tows, with the probe read under the full field.

    Args:
        beacon_gain: k for the field the probe is read under.

    Returns:
        A graph backend over the four-dimensional joint state: the rocket's position
        first, the probe's second. The engine moves the rocket alone.
    """
    graph = CouplingGraph(
        dims=[2, 2],
        couplings=[
            Coupling(
                0, 1, TETHER_PULL * _IDENTITY, coupling_noise=DYNAMICS_NOISE * _IDENTITY
            )
        ],
        observation_models={
            1: as_observation_model(
                CallableSensor(_IDENTITY, reference_sensor, default_params(beacon_gain))
            )
        },
    )
    rocket = GaussianTransition(_IDENTITY, dynamics_noise=DYNAMICS_NOISE * _IDENTITY)
    probe = GaussianTransition(
        (1 - TETHER_PULL) * _IDENTITY, dynamics_noise=DYNAMICS_NOISE * _IDENTITY
    )
    return CouplingGraphBackend(
        graph,
        [rocket, probe],
        prior=Belief(mean=np.zeros(4), cov=PRIOR_COV * np.eye(4)),
        control_matrix=np.vstack([_IDENTITY, np.zeros((2, 2))]),
    )
