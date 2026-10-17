"""The moves the curious rocket chooses between, and the search that chooses.

Nine moves: stay put, or travel one step along one of the eight compass directions.
Diagonals are scaled to the same length as the axis moves, so every move covers the
same distance. The set is declared and versioned because cpomdp only calls a search
decided over a set it was told about in advance. The certificate then names the set
by its version.

A smaller set of five, stay plus the four axis moves, is what the coda searches seven
steps deep.
"""

from collections.abc import Callable

import numpy as np
from cpomdp import LinearGaussianModel

from cpomdp_workshop._seams import (
    EnumeratedEFESearch,
    ExhaustiveEFESearch,
    FiniteActionSet,
    RecedingHorizonSelector,
)

__all__ = [
    "AXIS_VERSION",
    "COMPASS_VERSION",
    "STEP",
    "axis_actions",
    "compass_actions",
    "compass_moves",
    "compass_planner",
]

STEP = 0.5
COMPASS_VERSION = "compass-9"
AXIS_VERSION = "axis-5"

_COMPASS_BEARINGS = np.arange(8) * (np.pi / 4)
_AXIS_BEARINGS = np.arange(4) * (np.pi / 2)


def _stay_and_move(bearings: np.ndarray, step: float) -> np.ndarray:
    """A zero move followed by one move of length ``step`` along each bearing."""
    moves = step * np.column_stack([np.cos(bearings), np.sin(bearings)])
    return np.vstack([np.zeros((1, 2)), moves])


def compass_actions(step: float = STEP) -> FiniteActionSet:
    """Stay, plus the eight compass moves of length ``step``.

    Args:
        step: how far one move travels.

    Returns:
        The declared set the search enumerates, versioned ``compass-9``.
    """
    return FiniteActionSet(
        _stay_and_move(_COMPASS_BEARINGS, step), version=COMPASS_VERSION
    )


def axis_actions(step: float = STEP) -> FiniteActionSet:
    """Stay, plus the four axis moves of length ``step``.

    Args:
        step: how far one move travels.

    Returns:
        The declared set, versioned ``axis-5``.
    """
    return FiniteActionSet(_stay_and_move(_AXIS_BEARINGS, step), version=AXIS_VERSION)


def compass_moves(step: float = STEP) -> list[np.ndarray]:
    """The compass set as a list of moves, the form ``probe_model`` takes.

    Args:
        step: how far one move travels.

    Returns:
        Nine moves, stay first.
    """
    return list(_stay_and_move(_COMPASS_BEARINGS, step))


def compass_planner(
    model: LinearGaussianModel,
    *,
    step: float = STEP,
    horizon: int = 1,
    search: Callable[..., ExhaustiveEFESearch] = EnumeratedEFESearch,
) -> RecedingHorizonSelector:
    """Score every compass plan the rocket could fly, then take the first move.

    Args:
        model: the rocket's generative model. It needs a control matrix.
        step: how far one move travels.
        horizon: how many moves ahead to look, H. The search scores ``9 ** horizon``
            plans on every step, so this is where the compute goes.
        search: the exhaustive search to plan with. ``ChunkedEFESearch`` scores the
            plans a block at a time, for horizons whose plans do not fit in memory at
            once. Either gives the same move.

    Returns:
        A selector that re-plans on the current belief at every step. After its
        first plan it carries the certificate that says the search covered every
        plan rather than a sample of them.
    """
    return RecedingHorizonSelector(
        search(model, compass_actions(step), horizon=horizon)
    )
