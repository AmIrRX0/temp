"""Sum of every element of a selected table row."""

import torch
import triton
import triton.language as tl

from ..addressing import addressable_span, row_offsets
from ..config import DEFAULT_CONFIG, GatherConfig
from ..utils import next_power_of_2
from ..validate import check_table, normalize_row_ids, require_packed_rows


@triton.jit
def _row_sums_kernel(
    table_ptr,
    src_offsets_ptr,
    out_ptr,
    row_len,
    table_span,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)

    # The row's flat offset arrives ready made, and the row is one contiguous
    # run, so walking it is a single add.
    src_base = tl.load(src_offsets_ptr + pid)
    lane = tl.arange(0, BLOCK)
    active = lane < row_len

    src_off = src_base + lane
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
    the reduction never round-trips through global memory. Requires a packed
    trailing axis.
    """
    check_table(table)
    require_packed_rows(table, "row_sums")
    rows = normalize_row_ids(row_ids, int(table.shape[0]), config)

    row_len = int(table.shape[1])
    src_offsets = row_offsets(table, rows)
    span = addressable_span(table)

    n_gathered = int(rows.numel())
    out = torch.empty((n_gathered,), dtype=torch.float32, device=table.device)

    _row_sums_kernel[(n_gathered,)](
        table,
        src_offsets,
        out,
        row_len,
        span,
        BLOCK=next_power_of_2(row_len),
        num_warps=config.num_warps,
    )
    return out
