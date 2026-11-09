"""Build a curious rocket: the workshop package.

Everything the notebook imports lives here. The world and its sensor field are in
``world``, the compass moves and the search over them in ``planner``, the rockets
in ``build``, the flight loop in ``sim`` and the checks the room runs in ``check``.

The cpomdp release the workshop was measured against is pinned in ``pyproject.toml``
and read back here as ``WORKSHOP_CPOMDP_VERSION``, so the notebook, the tests and the
install cannot disagree about which library produced a number.
"""

import sys
from importlib import metadata

from cpomdp_workshop.build import (
    CURIOUS_PRECISION,
    THRUSTER_COST,
    curious_rocket,
    model_with,
    one_node,
    pad_goal,
    pad_model,
    pad_preference,
    radar_preference,
    the_1960_rocket,
)
from cpomdp_workshop.check import (
    check_agent,
    check_backends,
    check_my_broadcast,
    check_my_sensor,
    epistemic_spread,
    grade_sensor,
    probe,
    show_landings,
    show_misses,
    show_probe,
)
from cpomdp_workshop.checkpoints import (
    after_session_1,
    after_session_2,
    after_session_3,
)
from cpomdp_workshop.planner import (
    STEP,
    compass_actions,
    compass_moves,
    compass_planner,
)
from cpomdp_workshop.plot import (
    curiosity_pair,
    pad_pair,
    plot_beliefs,
    plot_flights,
    plot_pad,
    plot_path,
    plot_world,
    preview,
    world_flights,
)
from cpomdp_workshop.selfguided import two_node_backend
from cpomdp_workshop.sim import ROCK, Flight, PadFlight, fly, fly_pad
from cpomdp_workshop.solutions import (
    CLEAR_SKY_ROCK,
    axis_certificate,
    curiosity_decided,
    efe_split,
)
from cpomdp_workshop.symptoms import symptoms
from cpomdp_workshop.world import (
    BEACON,
    BLACKOUT,
    LAUNCH,
    PAD,
    PAD_GUESS,
    ROCK_RADIUS,
    beacon_sensor,
    broadcast_noise,
    default_params,
    pad_params,
    pad_sensor,
    reference_sensor,
)

__all__ = [
    "BEACON",
    "BLACKOUT",
    "CLEAR_SKY_ROCK",
    "CURIOUS_PRECISION",
    "LAUNCH",
    "MINIMUM_PYTHON",
    "PAD",
    "PAD_GUESS",
    "ROCK",
    "ROCK_RADIUS",
    "STEP",
    "THRUSTER_COST",
    "WORKSHOP_CPOMDP_VERSION",
    "Flight",
    "PadFlight",
    "after_session_1",
    "after_session_2",
    "after_session_3",
    "axis_certificate",
    "beacon_sensor",
    "broadcast_noise",
    "check_agent",
    "check_backends",
    "check_my_broadcast",
    "check_my_sensor",
    "compass_actions",
    "compass_moves",
    "compass_planner",
    "curiosity_decided",
    "curiosity_pair",
    "curious_rocket",
    "default_params",
    "efe_split",
    "epistemic_spread",
    "fly",
    "fly_pad",
    "grade_sensor",
    "hello",
    "model_with",
    "one_node",
    "pad_goal",
    "pad_model",
    "pad_pair",
    "pad_params",
    "pad_preference",
    "pad_sensor",
    "plot_beliefs",
    "plot_flights",
    "plot_pad",
    "plot_path",
    "plot_world",
    "preview",
    "probe",
    "radar_preference",
    "reference_sensor",
    "show_landings",
    "show_misses",
    "show_probe",
    "symptoms",
    "the_1960_rocket",
    "two_node_backend",
    "world_flights",
]

MINIMUM_PYTHON = (3, 11)


def _pinned_cpomdp_version() -> str:
    """The cpomdp version this distribution pins, read from its own metadata."""
    for requirement in metadata.requires("cpomdp-workshop") or ():
        name, pinned, rest = requirement.partition("==")
        if name.strip() == "cpomdp" and pinned:
            return rest.split(";")[0].strip()
    raise RuntimeError(
        "cpomdp-workshop must pin cpomdp with '==' in pyproject.toml. The notebook "
        "and the tests read the pin from there."
    )


WORKSHOP_CPOMDP_VERSION = _pinned_cpomdp_version()


def hello() -> None:
    """Print the versions in use and the world, after checking the install.

    Raises:
        RuntimeError: if Python is older than 3.11 or the installed cpomdp is not
            the release the workshop was measured on.
    """
    python = ".".join(str(part) for part in sys.version_info[:3])
    if sys.version_info[:2] < MINIMUM_PYTHON:
        raise RuntimeError(f"Python {python} is too old. The workshop needs 3.11+.")
    installed = metadata.version("cpomdp")
    if installed != WORKSHOP_CPOMDP_VERSION:
        raise RuntimeError(
            f"cpomdp {installed} is installed and the workshop was measured on "
            f"{WORKSHOP_CPOMDP_VERSION}. Run: pip install cpomdp=="
            f"{WORKSHOP_CPOMDP_VERSION}"
        )
    print(f"cpomdp {installed}, jax {metadata.version('jax')}, Python {python}. Ready.")
    print(
        f"launch {LAUNCH} -> pad {PAD}, navigation beacon at {BEACON}. "
        f"In task 3 the pad is hidden, somewhere around {PAD_GUESS}."
    )
