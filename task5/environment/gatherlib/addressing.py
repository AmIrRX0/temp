"""Turning tensors into the scalars the kernels address memory with.

Every kernel in this package walks a table by flat element offset rather than
by (row, column) pair, so something has to turn a tensor into a pitch per axis
and a bound the kernel can clamp against.
"""

from typing import Tuple

import torch


def element_strides(table: torch.Tensor) -> Tuple[int, int]:
    """Per-axis element strides of a 2-D table, as the tensor reports them."""
    return int(table.stride(-2)), int(table.stride(-1))


def addressable_span(table: torch.Tensor) -> int:
    """Elements spanned between the first and last element of ``table``.

    A kernel that clamps its offsets against this value cannot touch memory
    outside the region the tensor covers, whatever layout it happens to have.
    This is the *span*, not the element count: a strided view reaches further
    than ``numel()`` because of the gaps it steps over.
    """
    if table.numel() == 0:
        return 0
    span = 1
    for size, stride in zip(table.shape, table.stride()):
        span += (int(size) - 1) * int(stride)
    return span


def row_offsets(table: torch.Tensor, row_ids: torch.Tensor) -> torch.Tensor:
    """Flat offset of the first element of each requested row.

    Worked out host side in 64-bit so a kernel can load a ready-made offset
    instead of repeating the multiply in every program.
    """
    row_pitch, _ = element_strides(table)
    return row_ids.to(torch.int64) * row_pitch
