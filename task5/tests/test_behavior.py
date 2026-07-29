"""Behavioural tests for gatherlib.

Every expectation is built without going through gatherlib: either from plain
Python lists taken off the tensor, or from torch's own indexing operators.
"""

import sys

sys.path.insert(0, "/workspace")

import pytest  # noqa: E402
import torch  # noqa: E402
import triton  # noqa: E402

import gatherlib  # noqa: E402
from gatherlib import GatherConfig, gather_rows, row_sums, scatter_rows_  # noqa: E402

DEVICE = "cuda"

ROW_LEN = 2560
TABLE_ROWS = 840_000
PROBE_ROWS = [5, 4099, 300_000, 700_000, 838_000, 839_101, 839_999]

SMALL_ROWS = 96
SMALL_COLS = 128
SMALL_PICK = [0, 3, 11, 17, 4, 3]


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def full_table():
    """A table wide enough to exercise the whole address range of a device.

    Values vary along both axes, so a row read from the wrong place does not
    look like the row that was asked for.
    """
    table = torch.empty((TABLE_ROWS, ROW_LEN), dtype=torch.int8, device=DEVICE)
    col = torch.arange(ROW_LEN, device=DEVICE, dtype=torch.int32)
    chunk = 4096
    for start in range(0, TABLE_ROWS, chunk):
        stop = min(start + chunk, TABLE_ROWS)
        rows = torch.arange(start, stop, device=DEVICE, dtype=torch.int32).unsqueeze(1)
        block = (rows * 37 + col * 11 + (rows >> 3)) % 251 - 125
        table[start:stop].copy_(block)
    yield table
    del table
    torch.cuda.empty_cache()


@pytest.fixture(scope="module")
def small_base():
    """A modest contiguous allocation the view fixtures carve pieces out of."""
    generator = torch.Generator().manual_seed(20250729)
    host = torch.randint(
        -120, 120, (SMALL_ROWS, SMALL_COLS), dtype=torch.int8, generator=generator
    )
    return host.to(DEVICE)


# --------------------------------------------------------------------------
# references
# --------------------------------------------------------------------------


def python_gather(table, row_ids):
    """Gather done in plain Python over the values the tensor reports."""
    values = table.cpu().tolist()
    return [list(values[int(r)]) for r in row_ids]


def rows_one_at_a_time(table, row_ids):
    """Each requested row fetched on its own and read back as a Python list."""
    return [table[int(r)].cpu().tolist() for r in row_ids]


def ids(row_ids, dtype=torch.int64):
    return torch.tensor(row_ids, dtype=dtype, device=DEVICE)


# --------------------------------------------------------------------------
# fail_to_pass
# --------------------------------------------------------------------------


def test_gather_matches_direct_indexing_on_a_full_size_table(full_table):
    picked = ids(PROBE_ROWS)
    got = gather_rows(full_table, picked)
    want = full_table[picked]
    assert torch.equal(got, want)


def test_gather_matches_index_select_on_a_full_size_table(full_table):
    picked = ids(PROBE_ROWS)
    got = gather_rows(full_table, picked)
    want = torch.index_select(full_table, 0, picked)
    assert torch.equal(got, want)


def test_gather_matches_per_row_reference_on_a_full_size_table(full_table):
    got = gather_rows(full_table, ids(PROBE_ROWS))
    assert got.cpu().tolist() == rows_one_at_a_time(full_table, PROBE_ROWS)


def test_gather_of_one_row_matches_that_row_on_a_full_size_table(full_table):
    for row in (838_000, 839_101, 839_999):
        got = gather_rows(full_table, ids([row]))
        assert got.shape == (1, ROW_LEN)
        assert got[0].cpu().tolist() == full_table[row].cpu().tolist()


def test_gather_matches_reference_for_a_row_step_sliced_table(small_base):
    table = small_base[::3]
    got = gather_rows(table, ids(SMALL_PICK))
    assert got.cpu().tolist() == python_gather(table, SMALL_PICK)


def test_gather_matches_reference_for_a_column_narrowed_table(small_base):
    table = small_base[:, : SMALL_COLS // 2]
    got = gather_rows(table, ids(SMALL_PICK))
    assert got.cpu().tolist() == python_gather(table, SMALL_PICK)


def test_gather_matches_reference_for_a_column_windowed_table(small_base):
    table = small_base[:, 16:80]
    got = gather_rows(table, ids(SMALL_PICK))
    assert got.cpu().tolist() == python_gather(table, SMALL_PICK)


def test_gather_agrees_across_tables_holding_the_same_values(small_base):
    """A view and a packed copy of it hold equal data, so they must gather alike."""
    view = small_base[::3]
    packed = view.contiguous()
    want = python_gather(view, SMALL_PICK)

    from_view = gather_rows(view, ids(SMALL_PICK))
    from_packed = gather_rows(packed, ids(SMALL_PICK))

    assert from_packed.cpu().tolist() == want
    assert from_view.cpu().tolist() == want
    assert torch.equal(from_view, from_packed)


# --------------------------------------------------------------------------
# pass_to_pass
# --------------------------------------------------------------------------


def test_gather_matches_reference_for_a_contiguous_table(small_base):
    got = gather_rows(small_base, ids(SMALL_PICK))
    assert got.cpu().tolist() == python_gather(small_base, SMALL_PICK)


def test_gather_matches_reference_for_a_transposed_table(small_base):
    table = small_base.t()
    got = gather_rows(table, ids(SMALL_PICK))
    assert got.cpu().tolist() == python_gather(table, SMALL_PICK)


def test_gather_matches_reference_for_a_column_step_sliced_table(small_base):
    table = small_base[:, ::2]
    got = gather_rows(table, ids(SMALL_PICK))
    assert got.cpu().tolist() == python_gather(table, SMALL_PICK)


def test_gather_matches_reference_for_a_row_offset_table(small_base):
    table = small_base[8:]
    got = gather_rows(table, ids(SMALL_PICK))
    assert got.cpu().tolist() == python_gather(table, SMALL_PICK)


def test_gather_output_has_expected_shape_dtype_and_device(small_base):
    got = gather_rows(small_base, ids(SMALL_PICK))
    assert got.shape == (len(SMALL_PICK), SMALL_COLS)
    assert got.dtype == small_base.dtype
    assert got.device == small_base.device
    assert got.is_contiguous()


def test_gather_accepts_int32_and_int64_row_ids(small_base):
    wide = gather_rows(small_base, ids(SMALL_PICK, dtype=torch.int64))
    narrow = gather_rows(small_base, ids(SMALL_PICK, dtype=torch.int32))
    assert torch.equal(wide, narrow)


def test_gather_is_invariant_to_block_size(small_base):
    baseline = gather_rows(small_base, ids(SMALL_PICK))
    for block in (64, 256, 2048):
        other = gather_rows(
            small_base, ids(SMALL_PICK), GatherConfig(block_size=block)
        )
        assert torch.equal(baseline, other)


def test_gather_runs_a_triton_kernel(small_base):
    """A torch-level reimplementation would satisfy the value tests; this pins
    the gather to the compiled kernel the package is built around."""
    from gatherlib.kernels import gather as gather_module

    launched = []

    class Tripwire:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                launched.append(grid)
                raise RuntimeError("tripwire")

            return launch

    kernels = {
        name: value
        for name, value in vars(gather_module).items()
        if isinstance(value, triton.JITFunction)
    }
    assert kernels, "gather module defines no Triton kernel"

    for name in kernels:
        setattr(gather_module, name, Tripwire())
    try:
        with pytest.raises(RuntimeError, match="tripwire"):
            gather_rows(small_base, ids([0, 1]))
    finally:
        for name, value in kernels.items():
            setattr(gather_module, name, value)
    assert launched


def test_row_sums_match_reference_on_a_full_size_table(full_table):
    picked = ids(PROBE_ROWS)
    got = row_sums(full_table, picked)
    want = torch.index_select(full_table, 0, picked).to(torch.int64).sum(dim=1)
    assert got.dtype == torch.float32
    assert got.cpu().tolist() == want.to(torch.float32).cpu().tolist()


def test_row_sums_match_reference_for_a_column_narrowed_table(small_base):
    table = small_base[:, : SMALL_COLS // 2]
    got = row_sums(table, ids(SMALL_PICK))
    want = [float(sum(row)) for row in python_gather(table, SMALL_PICK)]
    assert got.cpu().tolist() == want


def test_scatter_writes_expected_rows_of_a_column_narrowed_table(small_base):
    base = small_base.clone()
    table = base[:, : SMALL_COLS // 2]
    before = base.cpu().tolist()

    targets = [2, 9, 40]
    payload = torch.full(
        (len(targets), table.shape[1]), 61, dtype=table.dtype, device=DEVICE
    )
    scatter_rows_(table, ids(targets), payload)

    after = base.cpu().tolist()
    for r in range(SMALL_ROWS):
        for c in range(SMALL_COLS):
            if r in targets and c < SMALL_COLS // 2:
                assert after[r][c] == 61, (r, c)
            else:
                assert after[r][c] == before[r][c], (r, c)


def test_out_of_range_row_id_is_rejected(small_base):
    with pytest.raises(IndexError):
        gather_rows(small_base, ids([0, SMALL_ROWS]))
    with pytest.raises(IndexError):
        gather_rows(small_base, ids([-1]))


def test_registry_exposes_every_operation(small_base):
    assert gatherlib.available_ops() == ("gather_rows", "row_sums", "scatter_rows_")
    through_registry = gatherlib.run_op("gather_rows", small_base, ids(SMALL_PICK))
    assert torch.equal(through_registry, gather_rows(small_base, ids(SMALL_PICK)))
