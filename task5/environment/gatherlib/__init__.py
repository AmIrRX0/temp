"""Triton table-access primitives for large quantised row tables.

A *table* here is any 2-D CUDA tensor. It does not have to own its storage and
it does not have to be contiguous, and it may be as wide as the device can
hold. Every operation walks it by flat element offset.
"""

from .addressing import addressable_span, element_strides, row_offsets
from .config import DEFAULT_CONFIG, GatherConfig
from .dispatch import available_ops, get_op, run_op
from .kernels.gather import gather_rows
from .kernels.rowsum import row_sums
from .kernels.scatter import scatter_rows_

__all__ = [
    "DEFAULT_CONFIG",
    "GatherConfig",
    "addressable_span",
    "available_ops",
    "element_strides",
    "gather_rows",
    "get_op",
    "row_offsets",
    "row_sums",
    "run_op",
    "scatter_rows_",
]
