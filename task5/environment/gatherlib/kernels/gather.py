"""Copy selected rows of a table into a freshly packed output block."""

import torch
import triton
import triton.language as tl

from ..cache import plan_for
from ..config import DEFAULT_CONFIG, GatherConfig
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

    # Both halves of the address are widened before they meet, so a table that
    # spans more than a 32-bit offset range is still addressed correctly.
    row_base = src_row.to(tl.int64) * row_pitch
    in_row = (lane * col_pitch).to(tl.int64)
    src_off = row_base + in_row

    inside = (src_off >= 0) & (src_off < table_span)
    vals = tl.load(table_ptr + src_off, mask=active & inside, other=0)

    dst_off = pid_row.to(tl.int64) * out_row_pitch + lane
    tl.store(out_ptr + dst_off, vals, mask=active)


def gather_rows(
    table: torch.Tensor,
    row_ids: torch.Tensor,
    config: GatherConfig = DEFAULT_CONFIG,
) -> torch.Tensor:
    """Return a packed ``(len(row_ids), table.shape[1])`` copy of the named rows.

    ``table`` may be any 2-D view of a larger allocation, whatever its layout;
    the output is always freshly allocated and contiguous and carries the
    table's dtype and device.
    """
    check_table(table)
    rows = normalize_row_ids(row_ids, int(table.shape[0]), config)
    plan = plan_for(table, config)

    n_gathered = int(rows.numel())
    out = torch.empty(
        (n_gathered, plan.row_len), dtype=table.dtype, device=table.device
    )

    _gather_rows_kernel[plan.grid(n_gathered)](
        table,
        rows,
        out,
        plan.row_len,
        plan.row_pitch,
        plan.col_pitch,
        out.stride(0),
        plan.span,
        BLOCK=plan.block,
        num_warps=plan.num_warps,
    )
    return out
