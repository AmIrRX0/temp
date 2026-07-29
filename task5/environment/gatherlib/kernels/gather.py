"""Copy selected rows of a table into a freshly packed output block."""

import torch
import triton
import triton.language as tl

from ..addressing import addressable_span, packed_row_strides
from ..config import DEFAULT_CONFIG, GatherConfig
from ..utils import as_launch_grid
from ..validate import check_table, normalize_row_ids


@triton.jit
def _gather_rows_kernel(
    table_ptr,
    row_ids_ptr,
    out_ptr,
    row_len,
    row_pitch,
    col_pitch,
    out_row_pitch,
    table_span,
    BLOCK: tl.constexpr,
):
    pid_row = tl.program_id(0)
    pid_col = tl.program_id(1)

    src_row = tl.load(row_ids_ptr + pid_row)
    lane = pid_col * BLOCK + tl.arange(0, BLOCK)
    active = lane < row_len

    row_base = src_row * row_pitch
    src_off = row_base + lane.to(tl.int64) * col_pitch
    inside = (src_off >= 0) & (src_off < table_span)

    vals = tl.load(table_ptr + src_off, mask=active & inside, other=0)

    dst_off = pid_row.to(tl.int64) * out_row_pitch + lane.to(tl.int64)
    tl.store(out_ptr + dst_off, vals, mask=active)


def gather_rows(
    table: torch.Tensor,
    row_ids: torch.Tensor,
    config: GatherConfig = DEFAULT_CONFIG,
) -> torch.Tensor:
    """Return a packed ``(len(row_ids), table.shape[1])`` copy of the named rows.

    ``table`` may be any 2-D view of a larger allocation; the output is always
    freshly allocated and contiguous, and carries the table's dtype and device.
    """
    check_table(table)
    rows = normalize_row_ids(row_ids, int(table.shape[0]), config)

    row_len = int(table.shape[1])
    row_pitch, col_pitch = packed_row_strides(table)
    span = addressable_span(table)

    n_gathered = int(rows.numel())
    out = torch.empty(
        (n_gathered, row_len), dtype=table.dtype, device=table.device
    )

    block = config.resolved_block(row_len)
    grid = as_launch_grid(n_gathered, row_len, block)
    _gather_rows_kernel[grid](
        table,
        rows,
        out,
        row_len,
        row_pitch,
        col_pitch,
        out.stride(0),
        span,
        BLOCK=block,
        num_warps=config.num_warps,
    )
    return out
