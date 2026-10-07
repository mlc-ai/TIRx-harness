"""Native split defaults with per-process overrides used by emission and caches."""

from .native_frontend import default_split_thresholds

_DEFAULTS = default_split_thresholds()
SPLIT_THRESHOLD_NAMES = tuple(sorted(_DEFAULTS))
globals().update(_DEFAULTS)


def thresholds() -> dict[str, int]:
    return {name: globals()[name] for name in SPLIT_THRESHOLD_NAMES}
