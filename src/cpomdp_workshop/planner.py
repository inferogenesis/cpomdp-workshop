"""The move set the curious rocket searches over, the search itself, and the engine.

Nine moves: stay put, or travel one step along one of the eight compass directions.
Diagonals are scaled to the same length as the axis moves, so every move costs the
rocket the same distance. The set is declared and versioned because cpomdp will only
call a search decisive over a set it was told about in advance.

Both rockets fly with the same engine, and ``STEP`` is what it can do in one cycle.
The move set is already built to that limit; LQR is not, and would otherwise cross
half the map on its first move, so it flies behind
[`ThrustLimited`][cpomdp_workshop.ThrustLimited]. The limit sits on the *selector*
rather than on the flight loop, so the action the rocket commits to is the action the
world applies, and the filter predicts with the move that was really made.
"""

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from cpomdp import ActionSelector, Belief, LinearGaussianModel, Preference
from cpomdp.enumeration import (
    EnumeratedEfeSearch,
    FiniteActionSet,
    RecedingHorizonSelector,
)
from jax import Array

__all__ = [
    "ACTION_SET_VERSION",
    "STEP",
    "ThrustLimited",
    "compass_actions",
    "compass_planner",
]

STEP = 0.5
ACTION_SET_VERSION = "c"

_BEARINGS = np.arange(8) * (np.pi / 4)


def compass_actions(step: float = STEP) -> FiniteActionSet:
    """Stay, plus the eight compass moves of length ``step``.

    Args:
        step: how far one move travels.

    Returns:
        The declared, versioned set the search enumerates.
    """
    compass = step * np.column_stack([np.cos(_BEARINGS), np.sin(_BEARINGS)])
    moves = np.vstack([np.zeros((1, 2)), compass])
    return FiniteActionSet(moves, version=ACTION_SET_VERSION)


def compass_planner(
    model: LinearGaussianModel, *, step: float = STEP, horizon: int = 1
) -> RecedingHorizonSelector:
    """Plan by scoring every move sequence the rocket could fly, then take the first.

    Args:
        model: the rocket's generative model. It needs a control matrix; a rocket
            that cannot act has nothing to plan.
        step: how far one move travels.
        horizon: how many moves ahead to look. The search scores ``9 ** horizon``
            sequences on every step, so this is where the compute goes.

    Returns:
        A selector that re-plans on the current belief at every step, carrying the
        certificate that says the search was exhaustive rather than sampled.
    """
    search = EnumeratedEfeSearch(model, compass_actions(step), horizon=horizon)
    return RecedingHorizonSelector(search)


@dataclass(frozen=True)
class ThrustLimited:
    """A selector that shortens any move longer than the engine can manage.

    Attributes:
        selector: the selector being limited.
        limit: the longest move the engine can make in one cycle.
    """

    selector: ActionSelector
    limit: float = STEP

    def select(self, belief: Belief, preference: Preference) -> Array:
        """Ask the wrapped selector, then rescale anything past the limit."""
        action = jnp.asarray(self.selector.select(belief, preference), dtype=float)
        length = jnp.linalg.norm(action)
        return jnp.where(length > self.limit, action * self.limit / length, action)
