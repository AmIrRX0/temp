"""Write whole rows back into a table at selected row ids."""

import torch
import triton
import triton.language as tl

from ..addressing import addressable_span, row_offsets
from ..config import DEFAULT_CONFIG, GatherConfig
from ..utils import ceil_div
from ..validate import (
    check_table,
    check_values,
    normalize_row_ids,
    require_packed_rows,
)


@triton.jit
def _scatter_rows_kernel(
    table_ptr,
    dst_offsets_ptr,
    values_ptr,
    row_len,
    val_row_pitch,
    table_span,
    BLOCK: tl.constexpr,
):
    pid_row = tl.program_id(0)
    pid_col = tl.program_id(1)

    # As in row_sums: the destination offset is already a 64-bit flat offset and
    # the row is packed, so the walk along it is a plain add.
    dst_base = tl.load(dst_offsets_ptr + pid_row)
    lane = pid_col * BLOCK + tl.arange(0, BLOCK)
    active = lane < row_len

    src_off = pid_row.to(tl.int64) * val_row_pitch + lane
    vals = tl.load(values_ptr + src_off, mask=active, other=0)

    dst_off = dst_base + lane
    inside = (dst_off >= 0) & (dst_off < table_span)
    tl.store(table_ptr + dst_off, vals, mask=active & inside)


def scatter_rows_(
    table: torch.Tensor,
    row_ids: torch.Tensor,
    values: torch.Tensor,
    config: GatherConfig = DEFAULT_CONFIG,
) -> torch.Tensor:
    """Copy ``values[i]`` over ``table[row_ids[i]]`` in place, returning the table.

    Row ids are expected to be distinct; overlapping ids race with each other
    and the surviving write is unspecified. Requires a packed trailing axis.
    """
    check_table(table)
    require_packed_rows(table, "scatter_rows_")
    rows = normalize_row_ids(row_ids, int(table.shape[0]), config)
    n_gathered = int(rows.numel())
    check_values(table, values, n_gathered)

    row_len = int(table.shape[1])
    dst_offsets = row_offsets(table, rows)
    span = addressable_span(table)

    values = values.contiguous()
    block = config.resolved_block(row_len)
    _scatter_rows_kernel[(n_gathered, ceil_div(row_len, block))](
        table,
        dst_offsets,
        values,
        row_len,
        values.stride(0),
        span,
        BLOCK=block,
        num_warps=config.num_warps,
    )
    return table
