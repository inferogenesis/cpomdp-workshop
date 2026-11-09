"""The five errors the room is most likely to hit, each with its cause and fix.

The list comes from a walk-through of the notebook on Colab. It names what people
actually see, which is mostly a ``...`` left in a cell, a runtime that restarted, or
NumPy inside a sensor. The fifth is the mistake the hidden-pad cell invites: a
broadcast that returns a matrix, as task 2's sensor did. A symptom that raises carries
the line that raises it, and a test checks that the words quoted under "You see" are
in that error. The printout fits a narrow Colab cell.
"""

from dataclasses import dataclass

import jax.numpy as jnp
from cpomdp import Belief, KalmanBackend, LinearGaussianModel

from cpomdp_workshop.check import check_my_broadcast, check_my_sensor
from cpomdp_workshop.world import LAUNCH, beacon_sensor, default_params, pad_params

__all__ = ["SYMPTOMS", "Symptom", "symptoms"]


@dataclass(frozen=True)
class Symptom:
    """One mistake, what it looks like, and its fix.

    Attributes:
        seen: what the learner sees, in a few words.
        cause: what went wrong.
        fix: what to do about it.
        wrong: a line that raises the error, or ``None`` for a symptom that raises
            nothing.
        quoted: the words of that error that ``seen`` quotes, word for word.
    """

    seen: str
    cause: str
    fix: str
    wrong: str | None = None
    quoted: str | None = None


SYMPTOMS = (
    Symptom(
        "an error naming 'ellipsis', or saying something is still ...",
        "a line marked ... is still unwritten",
        "write both lines marked ..., then run the cell again",
        "LinearGaussianModel(jnp.eye(2), control_matrix=jnp.eye(2), "
        "observation_matrix=..., observation_noise=..., "
        "dynamics_noise=1e-3 * jnp.eye(2), prior=prior)",
        "ellipsis",
    ),
    Symptom(
        "NameError: name 'model' is not defined (or my_sensor, my_broadcast)",
        "the runtime restarted or disconnected, which clears every name",
        "run the checkpoint above the task, then the task's cell",
        "KalmanBackend(model)",
        "name 'model' is not defined",
    ),
    Symptom(
        "JAX cannot trace it, or an error naming a Tracer or Concretization",
        "Python's min, or math or np, inside my_sensor or my_broadcast",
        "use jnp.minimum, jnp.exp and jnp.sum instead",
        "check_my_sensor(lambda x, params: jnp.eye(2) * "
        "(0.05 + min(jnp.sum(x**2), 1.0)), default_params(0.4))",
        "JAX cannot trace it",
    ),
    Symptom(
        'params["beacon"] is a tuple',
        "a setting in params given as a tuple, such as (2.5, 0.0)",
        'params["beacon"] = jnp.array([2.5, 0.0])',
        'check_my_sensor(beacon_sensor, {**default_params(0.4), "beacon": (2.5, 0.0)})',
        'params["beacon"] is a tuple',
    ),
    Symptom(
        "my_broadcast is not a broadcast yet: it returned shape (2, 2)",
        "my_broadcast multiplies by jnp.eye(2), as my_sensor did in task 2",
        'return one number, params["noise_floor"] + fade',
        "check_my_broadcast(lambda x, params: jnp.eye(2) * "
        '(params["noise_floor"] + jnp.sum(x**2)), pad_params())',
        "it returned shape (2, 2)",
    ),
)


def _namespace() -> dict[str, object]:
    """The names the symptom lines use, as a notebook cell would have them.

    ``model`` is left out on purpose. A restarted runtime has lost it.
    """
    return {
        "KalmanBackend": KalmanBackend,
        "LinearGaussianModel": LinearGaussianModel,
        "beacon_sensor": beacon_sensor,
        "check_my_broadcast": check_my_broadcast,
        "check_my_sensor": check_my_sensor,
        "default_params": default_params,
        "jnp": jnp,
        "pad_params": pad_params,
        "prior": Belief(mean=jnp.asarray(LAUNCH), cov=4 * jnp.eye(2)),
    }


def run(line: str) -> None:
    """Run one symptom line in a fresh copy of the names it uses."""
    exec(line, _namespace())  # the lines are this module's own constants


def symptoms() -> None:
    """Print each symptom: what you see, its cause and its fix."""
    for number, symptom in enumerate(SYMPTOMS, start=1):
        print(f"{number}. You see  {symptom.seen}")
        print(f"   Cause    {symptom.cause}")
        print(f"   Fix      {symptom.fix}")
        print()
