"""What the bonus and coda cells call beyond the core package.

The reference cells are generated from the attempt cells and the hint lines, so they
cannot drift from either. What lives here is the code a cell would be too long to
hold.
"""

import numpy as np
from cpomdp import LinearGaussianModel, Preference

from cpomdp_workshop._seams import (
    CompletenessCertificate,
    EnumeratedEFESearch,
    policy_efe_trace,
)
from cpomdp_workshop.build import model_with, pad_preference
from cpomdp_workshop.planner import axis_actions
from cpomdp_workshop.sim import Flight
from cpomdp_workshop.world import ROCK_RADIUS, default_params, reference_sensor

__all__ = [
    "CLEAR_SKY_ROCK",
    "CODA_HORIZON",
    "axis_certificate",
    "efe_split",
]

CODA_HORIZON = 7
CLEAR_SKY_ROCK = ((4.2, -0.7), ROCK_RADIUS)
"""A rock the size of the first, in clear sky where most curious routes pass."""


def axis_certificate(horizon: int = CODA_HORIZON) -> CompletenessCertificate:
    """Plan once over every axis-move sequence of length ``horizon`` and certify it.

    At seven moves the five-move set gives 78125 plans. They are all held at once,
    which takes about half a gigabyte and a second.

    Args:
        horizon: how many moves each plan makes, H.

    Returns:
        The certificate the search earned, which counts every plan it scored.
    """
    model = model_with(reference_sensor, default_params(0.08))
    search = EnumeratedEFESearch(model, axis_actions(), horizon=horizon)
    return search.evaluate(model.prior, pad_preference()).certificate


def efe_split(
    flight: Flight, model: LinearGaussianModel, preference: Preference
) -> np.ndarray:
    """The two terms of the expected free energy of each move the rocket made.

    Each move is scored from the belief the rocket held when it chose it, one step
    ahead, as the compass search scored it. The expected free energy is the first
    term minus the second.

    Args:
        flight: from [`fly`][cpomdp_workshop.fly].
        model: the rocket's model.
        preference: the reading it prefers.

    Returns:
        One row per move: the pragmatic term (how far the reading is expected to land
        from the preferred one) and the epistemic term (how much the reading is
        expected to teach), both in nats.
    """
    rows = []
    for step, move in enumerate(np.diff(flight.path, axis=0)):
        belief = type(model.prior)(flight.means[step], flight.covs[step])
        trace = policy_efe_trace(model, belief, move[None, :], preference)
        rows.append([float(trace.pragmatic[0]), float(trace.epistemic[0])])
    return np.asarray(rows)
