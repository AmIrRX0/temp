"""Everything a launch needs to know about one table, worked out once."""

from dataclasses import dataclass

import torch

from .addressing import addressable_span, element_strides
from .config import GatherConfig
from .utils import pick_block_size


@dataclass(frozen=True)
class AccessPlan:
    """Launch scalars for a table.

    Deriving these means reading the tensor's geometry and consulting the
    config, which is cheap but not free, and a caller that gathers from the
    same table in a loop should not pay for it on every call. See
    :mod:`gatherlib.cache`.
    """

    row_pitch: int
    col_pitch: int
    span: int
    row_len: int
    block: int
    num_warps: int

    @property
    def col_blocks(self) -> int:
        return -(-self.row_len // self.block)

    def grid(self, n_gathered: int):
        """One program per (gathered row, block of columns)."""
        return (n_gathered, self.col_blocks)


def build_plan(table: torch.Tensor, config: GatherConfig) -> AccessPlan:
    """Derive the launch scalars for ``table`` under ``config``."""
    row_pitch, col_pitch = element_strides(table)
    row_len = int(table.shape[-1])
    return AccessPlan(
        row_pitch=row_pitch,
        col_pitch=col_pitch,
        span=addressable_span(table),
        row_len=row_len,
        block=pick_block_size(row_len, config),
        num_warps=config.num_warps,
    )
