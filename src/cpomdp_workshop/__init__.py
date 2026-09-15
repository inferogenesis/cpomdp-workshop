"""Build a curious rocket — the workshop package.

Everything the notebook imports lives here. The world and its sensor field are in
``world``, the action set the curious rocket searches over in ``planner``, the
assembled agents in ``build``, and the flight loop in ``sim``.

The cpomdp release the workshop was measured against is pinned in ``pyproject.toml``
and read back here as ``WORKSHOP_CPOMDP_VERSION``, so the notebook, the tests and the
install cannot disagree about which library produced a number.
"""

from importlib import metadata

from cpomdp_workshop.world import (
    BEACON,
    BLACKOUT,
    LAUNCH,
    PAD,
    ROCK_RADIUS,
    beacon_sensor,
    default_params,
    reference_sensor,
)

__all__ = [
    "BEACON",
    "BLACKOUT",
    "LAUNCH",
    "PAD",
    "ROCK_RADIUS",
    "WORKSHOP_CPOMDP_VERSION",
    "beacon_sensor",
    "default_params",
    "hello",
    "reference_sensor",
]


def _pinned_cpomdp_version() -> str:
    """The cpomdp version this distribution pins, read from its own metadata."""
    for requirement in metadata.requires("cpomdp-workshop") or ():
        name, pinned, rest = requirement.partition("==")
        if name.strip() == "cpomdp" and pinned:
            return rest.split(";")[0].strip()
    raise RuntimeError(
        "cpomdp-workshop must pin cpomdp with '==' in pyproject.toml; the notebook "
        "and the tests read the pin from there."
    )


WORKSHOP_CPOMDP_VERSION = _pinned_cpomdp_version()


def hello() -> None:
    """Print the cpomdp version in use and a one-line summary of the world."""
    installed = metadata.version("cpomdp")
    match = "as pinned" if installed == WORKSHOP_CPOMDP_VERSION else "NOT the pin"
    print(f"cpomdp {installed} ({match}: {WORKSHOP_CPOMDP_VERSION})")
    print(
        f"launch {LAUNCH} -> pad {PAD}, nav beacon at {BEACON}, "
        f"blackout at {BLACKOUT} with a rock of radius {ROCK_RADIUS}"
    )
