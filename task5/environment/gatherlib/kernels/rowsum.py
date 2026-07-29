"""Sum of every element of a selected table row."""

import torch
import triton
import triton.language as tl

from ..addressing import addressable_span, element_strides
from ..config import DEFAULT_CONFIG, GatherConfig
from ..utils import next_power_of_2
from ..validate import check_table, normalize_row_ids


@triton.jit
def _row_sums_kernel(
    table_ptr,
    row_ids_ptr,
    out_ptr,
    row_len,
    row_pitch,
    col_pitch,
    table_span,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)

    src_row = tl.load(row_ids_ptr + pid).to(tl.int64)
    lane = tl.arange(0, BLOCK)
    active = lane < row_len

    src_off = src_row * row_pitch + lane.to(tl.int64) * col_pitch
    inside = (src_off >= 0) & (src_off < table_span)

    vals = tl.load(table_ptr + src_off, mask=active & inside, other=0)
    tl.store(out_ptr + pid, tl.sum(vals.to(tl.float32), axis=0))


def row_sums(
    table: torch.Tensor,
    row_ids: torch.Tensor,
    config: GatherConfig = DEFAULT_CONFIG,
) -> torch.Tensor:
    """Return a float32 vector holding the sum of each requested table row.

    One program per requested row, covering the whole row in a single block, so
    the reduction never has to round-trip through global memory.
    """
    check_table(table)
    rows = normalize_row_ids(row_ids, int(table.shape[0]), config)

    row_len = int(table.shape[1])
    row_pitch, col_pitch = element_strides(table)
    span = addressable_span(table)

    n_gathered = int(rows.numel())
    out = torch.empty((n_gathered,), dtype=torch.float32, device=table.device)

    block = next_power_of_2(row_len)
    _row_sums_kernel[(n_gathered,)](
        table,
        rows,
        out,
        row_len,
        row_pitch,
        col_pitch,
        span,
        BLOCK=block,
        num_warps=config.num_warps,
    )
    return out
