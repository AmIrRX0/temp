"""Input checks run before any kernel launch."""

import torch

from .config import GatherConfig

SUPPORTED_DTYPES = (
    torch.int8,
    torch.uint8,
    torch.int16,
    torch.float16,
    torch.bfloat16,
    torch.float32,
)

_INTEGER_DTYPES = (torch.int16, torch.int32, torch.int64, torch.uint8)


def check_table(table: torch.Tensor) -> None:
    """Reject tables the kernels cannot address."""
    if not isinstance(table, torch.Tensor):
        raise TypeError("table must be a torch.Tensor")
    if table.dim() != 2:
        raise ValueError(f"table must be 2-D, got {table.dim()}-D")
    if not table.is_cuda:
        raise ValueError("table must live on a CUDA device")
    if table.dtype not in SUPPORTED_DTYPES:
        raise TypeError(f"unsupported table dtype {table.dtype}")
    if table.shape[0] == 0 or table.shape[1] == 0:
        raise ValueError("table must not be empty")
    if min(int(s) for s in table.stride()) < 1:
        raise ValueError("table must have positive strides on every axis")


def require_packed_rows(table: torch.Tensor, op: str) -> None:
    """Reject a table whose trailing axis is not packed.

    Only :func:`gatherlib.gather_rows` walks an arbitrary layout. The in-place
    and reducing kernels address a row as one contiguous run, which is what
    lets them skip a per-element pitch multiply, so they need the trailing axis
    packed and say so rather than reading the wrong memory.
    """
    if int(table.stride(-1)) != 1:
        raise ValueError(
            f"{op} needs a table whose trailing axis is packed; "
            f"got strides {tuple(int(s) for s in table.stride())}"
        )


def check_values(table: torch.Tensor, values: torch.Tensor, n_rows: int) -> None:
    """Reject a value block that does not line up with the table it writes to."""
    if values.dim() != 2:
        raise ValueError(f"values must be 2-D, got {values.dim()}-D")
    if values.dtype != table.dtype:
        raise TypeError(f"values dtype {values.dtype} != table dtype {table.dtype}")
    if values.device != table.device:
        raise ValueError("values must be on the same device as the table")
    if values.shape != (n_rows, table.shape[1]):
        raise ValueError(
            f"values must be {(n_rows, table.shape[1])}, got {tuple(values.shape)}"
        )


def normalize_row_ids(
    row_ids: torch.Tensor,
    n_rows: int,
    config: GatherConfig,
) -> torch.Tensor:
    """Return the requested row ids in the form the kernels read them.

    The kernels take a contiguous 32-bit index vector, so anything wider is
    narrowed here after the range check has had a chance to reject it.
    """
    if not isinstance(row_ids, torch.Tensor):
        raise TypeError("row_ids must be a torch.Tensor")
    if row_ids.dim() != 1:
        raise ValueError(f"row_ids must be 1-D, got {row_ids.dim()}-D")
    if row_ids.dtype not in _INTEGER_DTYPES:
        raise TypeError(f"row_ids must be an integer tensor, got {row_ids.dtype}")
    if row_ids.numel() == 0:
        raise ValueError("row_ids must not be empty")

    if config.check_row_ids:
        out_of_range = ((row_ids < 0) | (row_ids >= n_rows)).any()
        if bool(out_of_range.item()):
            raise IndexError(f"row_ids contains an id outside [0, {n_rows})")

    return row_ids.contiguous().to(torch.int32)
