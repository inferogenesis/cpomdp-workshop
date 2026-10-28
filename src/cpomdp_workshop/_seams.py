"""Where this package meets cpomdp past its top-level exports.

The finite-action search and the per-step EFE trace live in submodules the library
does not re-export at its top level. They are imported here and nowhere else, so a
rename in one of them is a fix to this file alone. The notebook reaches them through
the package and never imports a submodule itself.

The one typing workaround the pinned release needs lives here too.
"""

from typing import cast

from cpomdp import CallableSensor, FixedSensor, ObservationModel
from cpomdp.efe import PolicyEFETrace, policy_efe_trace
from cpomdp.enumeration import (
    ChunkedEFESearch,
    CompletenessCertificate,
    EnumeratedEFESearch,
    ExhaustiveEFESearch,
    FiniteActionSet,
    RecedingHorizonSelector,
)

__all__ = [
    "ChunkedEFESearch",
    "CompletenessCertificate",
    "EnumeratedEFESearch",
    "ExhaustiveEFESearch",
    "FiniteActionSet",
    "PolicyEFETrace",
    "RecedingHorizonSelector",
    "as_observation_model",
    "policy_efe_trace",
]


def as_observation_model(sensor: FixedSensor | CallableSensor) -> ObservationModel:
    """``sensor``, typed as the protocol it implements.

    cpomdp 0.5.0 declares the protocol's ``is_fixed`` writable and sets it as a class
    attribute on both sensors, so a type checker refuses either sensor where the
    protocol is expected. The fix belongs in the library.
    """
    return cast(ObservationModel, sensor)
