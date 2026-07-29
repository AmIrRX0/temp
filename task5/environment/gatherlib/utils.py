"""Small helpers shared by the kernel wrappers."""

from typing import TYPE_CHECKING

from .config import DEFAULT_BLOCK, MAX_BLOCK

if TYPE_CHECKING:  # pragma: no cover
    from .config import GatherConfig


def ceil_div(numerator: int, denominator: int) -> int:
    """Integer ceiling division, used to size launch grids."""
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    return -(-numerator // denominator)


def next_power_of_2(value: int) -> int:
    """Smallest power of two that is >= ``value`` (and at least 1)."""
    if value <= 1:
        return 1
    return 1 << (value - 1).bit_length()


def pick_block_size(row_len: int, config: "GatherConfig") -> int:
    """Block size for rows of ``row_len`` elements.

    An explicit ``config.block_size`` wins, rounded up to a power of two
    because Triton block shapes have to be powers of two. Otherwise we take the
    smaller of the row width and :data:`gatherlib.config.DEFAULT_BLOCK`, so a
    narrow table does not pay for a block it cannot fill.
    """
    if config.block_size is not None:
        if config.block_size <= 0:
            raise ValueError("block_size must be positive")
        return min(next_power_of_2(config.block_size), MAX_BLOCK)
    return min(next_power_of_2(row_len), DEFAULT_BLOCK)


def as_launch_grid(n_rows: int, row_len: int, block: int):
    """Two dimensional grid: one program per (gathered row, block of columns)."""
    return (n_rows, ceil_div(row_len, block))
