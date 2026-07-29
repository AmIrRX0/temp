"""Tunable knobs for the table-access kernels."""

from dataclasses import dataclass
from typing import Optional

# Elements one program handles along a row unless a config says otherwise.
DEFAULT_BLOCK = 1024

# Largest block we are willing to ask Triton for. Bigger blocks cost registers
# without buying bandwidth for the narrow dtypes these tables use.
MAX_BLOCK = 4096


@dataclass(frozen=True)
class GatherConfig:
    """Settings shared by every operation in :mod:`gatherlib`.

    Attributes
    ----------
    block_size:
        Elements per program along the row axis. ``None`` means "derive it from
        the row length", see :func:`gatherlib.utils.pick_block_size`.
    num_warps:
        Warps per program, forwarded straight to Triton.
    check_row_ids:
        Reject row ids that fall outside the table before launching anything.
        Costs one device-to-host synchronisation per call, so callers that have
        already validated their indices can turn it off.
    """

    block_size: Optional[int] = None
    num_warps: int = 4
    check_row_ids: bool = True

    def resolved_block(self, row_len: int) -> int:
        """Block size this config wants for ``row_len``-wide rows."""
        from .utils import pick_block_size

        return pick_block_size(row_len, self)


DEFAULT_CONFIG = GatherConfig()
