"""The moves the curious rocket chooses between, and the search that chooses.

Nine moves: stay put, or travel one step along one of the eight compass directions.
Diagonals are scaled to the same length as the axis moves, so every move covers the
same distance. The set is declared and versioned because cpomdp only calls a search
decided over a set it was told about in advance. The certificate then names the set
by its version.

A smaller set of five, stay plus the four axis moves, is what the coda searches seven
steps deep.

The curiosity switch is a workshop helper, not a cpomdp feature. With curiosity off,
the planner ranks the same plans by the pragmatic term alone, so a flight with it off
is the same rocket with the epistemic term taken out of every decision.
"""

import itertools
from collections.abc import Callable
from typing import Literal, overload

import jax
import jax.numpy as jnp
import numpy as np
from cpomdp import Belief, LinearGaussianModel, Preference
from jax import Array

from cpomdp_workshop._seams import (
    EnumeratedEFESearch,
    ExhaustiveEFESearch,
    FiniteActionSet,
    PolicyEFETrace,
    RecedingHorizonSelector,
    policy_efe_trace,
)

__all__ = [
    "AXIS_VERSION",
    "COMPASS_VERSION",
    "STEP",
    "WeightedEFESelector",
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


class WeightedEFESelector:
    """Ranks every plan by its pragmatic term minus a weight on its epistemic term.

    Expected free energy is the pragmatic term minus the epistemic term, so a weight
    of one ranks plans as cpomdp's search does and a weight of zero ranks them by the
    pragmatic term alone. It re-plans on the current belief at every step and takes
    the first move of the best plan. Plans are enumerated in the order cpomdp's
    search enumerates them, and a tie goes to the earlier plan, as it does there.

    Args:
        model: the rocket's generative model. It needs a control matrix.
        actions: the declared moves each step of a plan chooses from.
        horizon: how many moves each plan makes, H.
        epistemic_weight: w, the weight on the epistemic term.
    """

    def __init__(
        self,
        model: LinearGaussianModel,
        actions: FiniteActionSet,
        *,
        horizon: int,
        epistemic_weight: float,
    ) -> None:
        self._model = model
        self._horizon = horizon
        self._epistemic_weight = epistemic_weight
        moves = np.asarray(actions.actions)
        self._plans = jnp.asarray(
            [list(plan) for plan in itertools.product(moves, repeat=horizon)]
        )  # (|A|^H, H, p)
        self._scores = jax.jit(self._score_every_plan)

    def _weighted(self, trace: PolicyEFETrace) -> Array:
        """One plan's pragmatic term minus w times its epistemic term, summed."""
        return jnp.sum(trace.pragmatic - self._epistemic_weight * trace.epistemic)

    def _score_every_plan(self, belief: Belief, preference: Preference) -> Array:
        return jax.vmap(
            lambda plan: self._weighted(
                policy_efe_trace(self._model, belief, plan, preference)
            )
        )(self._plans)

    def select(self, belief: Belief, preference: Preference) -> Array:
        """Score every plan from ``belief`` and return the best plan's first move.

        Raises:
            ValueError: if no plan scores below infinity, so there is nothing to pick.
        """
        scores = self._scores(belief, preference)
        masked = jnp.where(jnp.isnan(scores), jnp.inf, scores)
        best = int(jnp.argmin(masked))
        if not bool(masked[best] < jnp.inf):
            raise ValueError("no plan scored below infinity, so no move can be picked.")
        return self._plans[best, 0]

    @property
    def model(self) -> LinearGaussianModel:
        """The model the plans are scored on."""
        return self._model

    @property
    def horizon(self) -> int:
        """How many moves each plan makes, H."""
        return self._horizon

    @property
    def n_plans(self) -> int:
        """How many plans every step scores, ``|A|^H``."""
        return int(self._plans.shape[0])

    @property
    def epistemic_weight(self) -> float:
        """w, the weight on the epistemic term."""
        return self._epistemic_weight


@overload
def compass_planner(
    model: LinearGaussianModel,
    *,
    step: float = ...,
    horizon: int = ...,
    search: Callable[..., ExhaustiveEFESearch] = ...,
    curiosity: Literal[True] = ...,
) -> RecedingHorizonSelector: ...


@overload
def compass_planner(
    model: LinearGaussianModel,
    *,
    step: float = ...,
    horizon: int = ...,
    search: Callable[..., ExhaustiveEFESearch] = ...,
    curiosity: Literal[False],
) -> WeightedEFESelector: ...


@overload
def compass_planner(
    model: LinearGaussianModel,
    *,
    step: float = ...,
    horizon: int = ...,
    search: Callable[..., ExhaustiveEFESearch] = ...,
    curiosity: bool,
) -> RecedingHorizonSelector | WeightedEFESelector: ...


def compass_planner(
    model: LinearGaussianModel,
    *,
    step: float = STEP,
    horizon: int = 1,
    search: Callable[..., ExhaustiveEFESearch] = EnumeratedEFESearch,
    curiosity: bool = True,
) -> RecedingHorizonSelector | WeightedEFESelector:
    """Score every compass plan the rocket could fly, then take the first move.

    Args:
        model: the rocket's generative model. It needs a control matrix.
        step: how far one move travels.
        horizon: how many moves ahead to look, H. The search scores ``9 ** horizon``
            plans on every step, so this is where the compute goes.
        search: the exhaustive search to plan with when curiosity is on.
            ``ChunkedEFESearch`` scores the plans a block at a time, for horizons
            whose plans do not fit in memory at once. Either gives the same move.
        curiosity: whether the epistemic term counts. Off, the same plans are ranked
            by the pragmatic term alone.

    Returns:
        With curiosity on, cpomdp's selector, which re-plans on the current belief at
        every step. After its first plan it carries the certificate that says the
        search covered every plan rather than a sample of them. With curiosity off, a
        [`WeightedEFESelector`][cpomdp_workshop.planner.WeightedEFESelector] with no
        weight on the epistemic term.
    """
    actions = compass_actions(step)
    if not curiosity:
        return WeightedEFESelector(
            model, actions, horizon=horizon, epistemic_weight=0.0
        )
    return RecedingHorizonSelector(search(model, actions, horizon=horizon))
