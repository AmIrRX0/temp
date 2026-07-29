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
from gatherlib import (  # noqa: E402
    GatherConfig,
    clear_plans,
    gather_rows,
    row_sums,
    scatter_rows_,
)

DEVICE = "cuda"

# One allocation backs every full size case; its transpose is a second layout
# over the same bytes.
STORE_ROWS = 840_000
STORE_COLS = 2_560

SMALL_ROWS = 96
SMALL_COLS = 128
SMALL_PICK = [0, 3, 11, 17, 4, 3]


@pytest.fixture(autouse=True)
def fresh_plan_cache():
    """Each test starts with nothing cached, so tests cannot leak into one another."""
    clear_plans()
    yield
    clear_plans()


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def store():
    """A table wide enough to exercise a device's whole address range.

    Values vary along both axes, so a value read from the wrong place does not
    look like the value that was asked for.
    """
    table = torch.empty((STORE_ROWS, STORE_COLS), dtype=torch.int8, device=DEVICE)
    col = torch.arange(STORE_COLS, device=DEVICE, dtype=torch.int32)
    chunk = 4096
    for start in range(0, STORE_ROWS, chunk):
        stop = min(start + chunk, STORE_ROWS)
        rows = torch.arange(start, stop, device=DEVICE, dtype=torch.int32).unsqueeze(1)
        block = (rows * 37 + col * 11 + (rows >> 3)) % 251 - 125
        table[start:stop].copy_(block)
    yield table
    del table
    torch.cuda.empty_cache()


def _host_block(rows, cols, seed):
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(
        -120, 120, (rows, cols), dtype=torch.int8, generator=generator
    ).to(DEVICE)


@pytest.fixture(scope="module")
def small_base():
    """A modest contiguous allocation the view fixtures carve pieces out of."""
    return _host_block(SMALL_ROWS, SMALL_COLS, 20250729)


@pytest.fixture(scope="module")
def narrow_views():
    """Same shape, same dtype, none of them packed, all with different pitches."""
    return {
        pitch: _host_block(SMALL_ROWS, pitch, 900 + pitch)[:, :SMALL_COLS]
        for pitch in (256, 384, 512)
    }


@pytest.fixture(scope="module")
def shared_base_views():
    """Same shape and pitchless-in-common, both carved out of one allocation."""
    base = _host_block(2 * SMALL_ROWS, 2 * SMALL_COLS, 4242)
    return base, base[:SMALL_ROWS, :SMALL_COLS], base[::2, :SMALL_COLS]


# --------------------------------------------------------------------------
# references
# --------------------------------------------------------------------------


def python_gather(table, row_ids):
    """Gather done in plain Python over the values the tensor reports."""
    values = table.cpu().tolist()
    return [list(values[int(r)]) for r in row_ids]


def rows_one_at_a_time(table, row_ids):
    """Each requested row fetched on its own, as a tensor."""
    return [table[int(r)].clone() for r in row_ids]


def ids(row_ids, dtype=torch.int64):
    return torch.tensor(row_ids, dtype=dtype, device=DEVICE)


def triton_kernels(module):
    """Every compiled kernel object a module defines, JIT or interpreted."""
    kinds = [triton.JITFunction]
    try:  # TRITON_INTERPRET swaps the class out from under us
        from triton.runtime.interpreter import InterpretedFunction

        kinds.append(InterpretedFunction)
    except Exception:  # pragma: no cover
        pass
    return {
        name: value
        for name, value in vars(module).items()
        if isinstance(value, tuple(kinds))
    }


# --------------------------------------------------------------------------
# fail_to_pass
# --------------------------------------------------------------------------


def test_gather_matches_direct_indexing_on_a_transposed_full_size_table(store):
    table = store.t()
    picked = ids([0, 700, 2559])
    assert torch.equal(gather_rows(table, picked), table[picked])


def test_gather_matches_index_select_on_a_transposed_full_size_table(store):
    table = store.t()
    picked = ids([1, 1279, 2558])
    got = gather_rows(table, picked)
    assert torch.equal(got, torch.index_select(table, 0, picked))


def test_gather_matches_per_row_reference_on_a_transposed_full_size_table(store):
    table = store.t()
    picks = [5, 2047, 2559]
    got = gather_rows(table, ids(picks))
    for i, want in enumerate(rows_one_at_a_time(table, picks)):
        assert torch.equal(got[i], want), picks[i]


def test_gather_matches_direct_indexing_on_a_strided_full_size_table(store):
    table = store.t()[:, ::2]
    picked = ids([3, 1500, 2559])
    assert torch.equal(gather_rows(table, picked), table[picked])


def test_gather_is_unaffected_by_an_earlier_gather_from_a_wider_table(narrow_views):
    first, second = narrow_views[256], narrow_views[384]
    gather_rows(first, ids(SMALL_PICK))
    got = gather_rows(second, ids(SMALL_PICK))
    assert got.cpu().tolist() == python_gather(second, SMALL_PICK)


def test_gather_is_unaffected_by_an_earlier_gather_from_a_narrower_table(narrow_views):
    first, second = narrow_views[512], narrow_views[256]
    gather_rows(first, ids(SMALL_PICK))
    got = gather_rows(second, ids(SMALL_PICK))
    assert got.cpu().tolist() == python_gather(second, SMALL_PICK)


def test_gather_is_correct_for_every_table_in_a_mixed_sequence(narrow_views):
    order = [256, 384, 512, 384, 256, 512]
    for pitch in order:
        table = narrow_views[pitch]
        got = gather_rows(table, ids(SMALL_PICK))
        assert got.cpu().tolist() == python_gather(table, SMALL_PICK), pitch


def test_gather_is_correct_for_two_views_of_one_allocation(shared_base_views):
    _, dense, spaced = shared_base_views
    gather_rows(dense, ids(SMALL_PICK))
    got = gather_rows(spaced, ids(SMALL_PICK))
    assert got.cpu().tolist() == python_gather(spaced, SMALL_PICK)


# --------------------------------------------------------------------------
# pass_to_pass
# --------------------------------------------------------------------------


def test_gather_matches_direct_indexing_on_a_full_size_table(store):
    picked = ids([5, 4099, 300_000, 700_000, 839_999])
    assert torch.equal(gather_rows(store, picked), store[picked])


def test_gather_matches_per_row_reference_on_a_full_size_table(store):
    picks = [7, 512_000, 839_998]
    got = gather_rows(store, ids(picks))
    for i, want in enumerate(rows_one_at_a_time(store, picks)):
        assert torch.equal(got[i], want), picks[i]


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


def test_gather_matches_reference_for_a_row_step_sliced_table(small_base):
    table = small_base[::3]
    got = gather_rows(table, ids([0, 3, 11, 17, 4]))
    assert got.cpu().tolist() == python_gather(table, [0, 3, 11, 17, 4])


def test_gather_matches_reference_for_each_narrow_view_on_its_own(narrow_views):
    for pitch, table in sorted(narrow_views.items()):
        clear_plans()
        got = gather_rows(table, ids(SMALL_PICK))
        assert got.cpu().tolist() == python_gather(table, SMALL_PICK), pitch


def test_gather_is_correct_after_a_gather_from_a_packed_table_of_equal_shape(
    small_base, narrow_views
):
    table = narrow_views[384]
    gather_rows(small_base, ids(SMALL_PICK))
    got = gather_rows(table, ids(SMALL_PICK))
    assert got.cpu().tolist() == python_gather(table, SMALL_PICK)


def test_gather_is_correct_after_a_gather_from_a_view_of_another_shape(narrow_views):
    other = narrow_views[256][: SMALL_ROWS // 2, : SMALL_COLS // 2]
    table = narrow_views[384]
    gather_rows(other, ids([0, 1, 2]))
    got = gather_rows(table, ids(SMALL_PICK))
    assert got.cpu().tolist() == python_gather(table, SMALL_PICK)


def test_repeated_gathers_from_one_table_agree_with_the_reference(narrow_views):
    table = narrow_views[512]
    want = python_gather(table, SMALL_PICK)
    for _ in range(3):
        assert gather_rows(table, ids(SMALL_PICK)).cpu().tolist() == want


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
    want = python_gather(small_base, SMALL_PICK)
    for block in (64, 256, 2048):
        clear_plans()
        got = gather_rows(small_base, ids(SMALL_PICK), GatherConfig(block_size=block))
        assert got.cpu().tolist() == want, block


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

    kernels = triton_kernels(gather_module)
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


def test_row_sums_match_reference_on_a_full_size_table(store):
    picked = ids([5, 4099, 300_000, 839_999])
    got = row_sums(store, picked)
    want = torch.index_select(store, 0, picked).to(torch.int64).sum(dim=1)
    assert got.dtype == torch.float32
    assert got.cpu().tolist() == want.to(torch.float32).cpu().tolist()


def test_row_sums_match_reference_for_a_narrow_view(narrow_views):
    table = narrow_views[384]
    got = row_sums(table, ids(SMALL_PICK))
    want = [float(sum(row)) for row in python_gather(table, SMALL_PICK)]
    assert got.cpu().tolist() == want


def test_scatter_writes_expected_rows_of_a_narrow_view():
    # Its own allocation: this one gets written to, so it must not be shared.
    base = _host_block(SMALL_ROWS, 2 * SMALL_COLS, 777)
    table = base[:, :SMALL_COLS]
    before = base.cpu().tolist()

    targets = [2, 9, 40]
    payload = torch.full(
        (len(targets), SMALL_COLS), 61, dtype=table.dtype, device=DEVICE
    )
    scatter_rows_(table, ids(targets), payload)

    after = base.cpu().tolist()
    for r in range(SMALL_ROWS):
        for c in range(2 * SMALL_COLS):
            if r in targets and c < SMALL_COLS:
                assert after[r][c] == 61, (r, c)
            else:
                assert after[r][c] == before[r][c], (r, c)


def test_reducing_and_writing_kernels_reject_an_unpacked_trailing_axis(small_base):
    table = small_base.t()
    with pytest.raises(ValueError, match="packed"):
        row_sums(table, ids([0, 1]))
    payload = torch.zeros(
        (2, table.shape[1]), dtype=table.dtype, device=DEVICE
    )
    with pytest.raises(ValueError, match="packed"):
        scatter_rows_(table, ids([0, 1]), payload)


def test_out_of_range_row_id_is_rejected(small_base):
    with pytest.raises(IndexError):
        gather_rows(small_base, ids([0, SMALL_ROWS]))
    with pytest.raises(IndexError):
        gather_rows(small_base, ids([-1]))


def test_plan_cache_can_be_inspected_and_emptied(small_base):
    assert gatherlib.cached_plan_count() == 0
    gather_rows(small_base, ids(SMALL_PICK))
    assert gatherlib.cached_plan_count() >= 1
    clear_plans()
    assert gatherlib.cached_plan_count() == 0


def test_registry_exposes_every_operation(small_base):
    assert gatherlib.available_ops() == ("gather_rows", "row_sums", "scatter_rows_")
    through_registry = gatherlib.run_op("gather_rows", small_base, ids(SMALL_PICK))
    assert through_registry.cpu().tolist() == python_gather(small_base, SMALL_PICK)
