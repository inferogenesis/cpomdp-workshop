"""The five errors the room is most likely to hit, each with the line that fixes it.

Each symptom is two lines of code: one that raises, one that runs. ``symptoms()`` runs
the first and prints the message cpomdp gives, so the text in the notebook is always
the pinned release's own words, never a paraphrase of them.
"""

from dataclasses import dataclass

import jax.numpy as jnp
from cpomdp import (
    Agent,
    Belief,
    CallableSensor,
    LinearGaussianModel,
    Preference,
    StateGoal,
)

from cpomdp_workshop.build import (
    model_with,
    pad_goal,
    the_1960_rocket,
)
from cpomdp_workshop.planner import compass_planner
from cpomdp_workshop.world import LAUNCH, PAD, default_params, reference_sensor

__all__ = ["SYMPTOMS", "Symptom", "symptoms"]


@dataclass(frozen=True)
class Symptom:
    """One mistake and its fix.

    Attributes:
        name: what the person was trying to do.
        wrong: the line that raises.
        right: the line that does what was meant.
        error: the exception ``wrong`` raises.
    """

    name: str
    wrong: str
    right: str
    error: type[Exception]


SYMPTOMS = (
    Symptom(
        "a state-dependent sensor in the noise slot",
        "LinearGaussianModel(I, observation_noise=sensor, control_matrix=I, "
        "dynamics_noise=1e-3 * I, prior=prior)",
        "LinearGaussianModel(I, observation_model=sensor, control_matrix=I, "
        "dynamics_noise=1e-3 * I, prior=prior)",
        TypeError,
    ),
    Symptom(
        "a goal beside a planner",
        "Agent(curious_model, StateGoal(PAD), selector=planner)",
        "Agent(curious_model, Preference(PAD), selector=planner)",
        TypeError,
    ),
    Symptom(
        "a state goal under a sensor that depends on the state",
        "Agent(curious_model, StateGoal(PAD))",
        "Agent(curious_model, Preference(PAD), selector=planner)",
        ValueError,
    ),
    Symptom(
        "a preference of the wrong length",
        "Agent(curious_model, Preference([5.0, 0.0, 0.0]), selector=planner)",
        "Agent(curious_model, Preference([5.0, 0.0]), selector=planner)",
        ValueError,
    ),
    Symptom(
        "swapping the model under a built rocket",
        "rocket.model = model_with()",
        "rocket = Agent(model_with(), pad_goal())",
        AttributeError,
    ),
)


def _namespace() -> dict[str, object]:
    """The names the symptom lines use, as a notebook cell would have them."""
    identity = jnp.eye(2)
    curious_model = model_with(reference_sensor, default_params(0.08))
    return {
        "Agent": Agent,
        "I": identity,
        "LinearGaussianModel": LinearGaussianModel,
        "PAD": PAD,
        "Preference": Preference,
        "StateGoal": StateGoal,
        "curious_model": curious_model,
        "model_with": model_with,
        "pad_goal": pad_goal,
        "planner": compass_planner(curious_model),
        "prior": Belief(mean=LAUNCH, cov=4 * identity),
        "rocket": the_1960_rocket(),
        "sensor": CallableSensor(identity, reference_sensor, default_params(0.08)),
    }


def run(line: str) -> None:
    """Run one symptom line in a fresh copy of the names it uses."""
    exec(line, _namespace())  # the lines are this module's own constants


def symptoms() -> None:
    """Print each mistake, the error cpomdp gives for it, and the fix."""
    for number, symptom in enumerate(SYMPTOMS, start=1):
        try:
            run(symptom.wrong)
        except symptom.error as error:
            message = f"{type(error).__name__}: {error}"
        else:  # a release that stops refusing this shows up here, not as a crash
            message = "(this release no longer raises here)"
        print(f"{number}. {symptom.name}")
        print(f"   you wrote  {symptom.wrong}")
        print(f"   cpomdp     {message}")
        print(f"   write      {symptom.right}")
        print()
