"""Look up a table operation by name.

Callers that pick an operation from a config file or a command line go through
here rather than importing the kernel modules directly.
"""

from typing import Callable, Dict, Tuple

from .config import DEFAULT_CONFIG, GatherConfig
from .kernels.gather import gather_rows
from .kernels.rowsum import row_sums
from .kernels.scatter import scatter_rows_

_REGISTRY: Dict[str, Callable] = {
    "gather_rows": gather_rows,
    "row_sums": row_sums,
    "scatter_rows_": scatter_rows_,
}


def available_ops() -> Tuple[str, ...]:
    """Names accepted by :func:`run_op`, in a stable order."""
    return tuple(sorted(_REGISTRY))


def get_op(name: str) -> Callable:
    """Return the callable registered under ``name``."""
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"unknown op {name!r}; available: {', '.join(available_ops())}"
        ) from None


def run_op(name: str, *args, config: GatherConfig = DEFAULT_CONFIG, **kwargs):
    """Run a registered operation with ``config`` threaded through to it."""
    return get_op(name)(*args, config=config, **kwargs)
