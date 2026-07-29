#!/usr/bin/env python3
"""Verify-first harness for the gatherlib task. Run this on a GPU box.

It answers the four questions that decide whether the task is viable, and that
cannot be answered without a CUDA device:

  1. Does the int32 row base actually wrap in generated Triton code?
  2. Is the wrap *silent* -- zeros through the bounds clamp, not an illegal
     memory access that poisons the CUDA context?
  3. Which layouts does the pitch helper get wrong, and which does it get right
     (the pass_to_pass set depends on the right ones)?
  4. How much device memory does the full size fixture actually need?

Usage
-----
    python tools/verify_first.py                # run everything, print verdict
    python tools/verify_first.py --probe DIR    # internal single-variant probe

Exit code is 0 only if every expectation holds.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

REPO = pathlib.Path(__file__).resolve().parent.parent
ENVIRONMENT = REPO / "task5" / "environment"
SOLVE = REPO / "task5" / "solution" / "solve.sh"

ROW_LEN = 2560
TABLE_ROWS = 840_000
PROBE_ROWS = [5, 4099, 300_000, 700_000, 838_000, 839_101, 839_999]

# Layout name -> expected verdict for the code as shipped.
# "ok" layouts back the pass_to_pass set; "wrong" ones back fail_to_pass.
EXPECTED_LAYOUTS = {
    "contiguous": "ok",
    "transposed": "ok",
    "column_step": "ok",
    "row_offset": "ok",
    "row_step": "wrong",
    "column_narrow": "wrong",
    "column_window": "wrong",
}


# ---------------------------------------------------------------------------
# probe: runs inside a subprocess against one copy of the library
# ---------------------------------------------------------------------------


def build_full_table(torch, device="cuda"):
    table = torch.empty((TABLE_ROWS, ROW_LEN), dtype=torch.int8, device=device)
    col = torch.arange(ROW_LEN, device=device, dtype=torch.int32)
    chunk = 4096
    for start in range(0, TABLE_ROWS, chunk):
        stop = min(start + chunk, TABLE_ROWS)
        rows = torch.arange(start, stop, device=device, dtype=torch.int32).unsqueeze(1)
        block = (rows * 37 + col * 11 + (rows >> 3)) % 251 - 125
        table[start:stop].copy_(block)
    return table


def layout_cases(torch, device="cuda"):
    generator = torch.Generator().manual_seed(20250729)
    base = torch.randint(
        -120, 120, (96, 128), dtype=torch.int8, generator=generator
    ).to(device)
    return {
        "contiguous": base,
        "transposed": base.t(),
        "column_step": base[:, ::2],
        "row_offset": base[8:],
        "row_step": base[::3],
        "column_narrow": base[:, :64],
        "column_window": base[:, 16:80],
    }


def probe(source_dir: str) -> dict:
    os.environ.setdefault("TRITON_DEBUG", "0")
    sys.path.insert(0, source_dir)

    import torch
    import triton

    report: dict = {
        "torch": torch.__version__,
        "triton": triton.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
    }
    if not report["cuda_available"]:
        report["fatal"] = "no CUDA device"
        return report

    report["device"] = torch.cuda.get_device_name(0)
    free, total = torch.cuda.mem_get_info()
    report["device_free_gib"] = round(free / 2**30, 3)
    report["device_total_gib"] = round(total / 2**30, 3)

    from gatherlib import gather_rows, row_sums

    # --- layouts -----------------------------------------------------------
    picks = [0, 3, 11, 17, 4, 3]
    ids = torch.tensor(picks, dtype=torch.int64, device="cuda")
    layouts = {}
    for name, table in layout_cases(torch).items():
        values = table.cpu().tolist()
        want = [list(values[r]) for r in picks]
        got = gather_rows(table, ids).cpu().tolist()
        layouts[name] = "ok" if got == want else "wrong"
    report["layouts"] = layouts

    # --- full size table ---------------------------------------------------
    torch.cuda.reset_peak_memory_stats()
    table = build_full_table(torch)
    report["table_gib"] = round(table.numel() / 2**30, 4)

    picked = torch.tensor(PROBE_ROWS, dtype=torch.int64, device="cuda")
    got = gather_rows(table, picked)
    want = table[picked]
    torch.cuda.synchronize()

    mismatched, all_zero = [], []
    for i, row in enumerate(PROBE_ROWS):
        same = bool(torch.equal(got[i], want[i]))
        if not same:
            mismatched.append(row)
            if not bool(got[i].any()):
                all_zero.append(row)
    report["mismatched_rows"] = mismatched
    report["mismatched_rows_that_are_all_zero"] = all_zero

    # Exact boundary: first row whose flat base crosses 2**31.
    boundary = 2**31 // ROW_LEN + 1
    window = [boundary - 2, boundary - 1, boundary, boundary + 1]
    w_ids = torch.tensor(window, dtype=torch.int64, device="cuda")
    w_got = gather_rows(table, w_ids)
    w_want = table[w_ids]
    report["predicted_first_failing_row"] = boundary
    report["boundary_window"] = {
        str(r): bool(torch.equal(w_got[i], w_want[i])) for i, r in enumerate(window)
    }

    # The correct sibling kernel must be unaffected on the same table.
    sums = row_sums(table, picked)
    ref = torch.index_select(table, 0, picked).to(torch.int64).sum(dim=1)
    report["sibling_kernel_ok"] = sums.cpu().tolist() == ref.to(torch.float32).cpu().tolist()

    # A live context after the gather proves nothing faulted.
    torch.cuda.synchronize()
    canary = (torch.ones(1024, device="cuda") * 3).sum().item()
    report["context_alive_after_gather"] = canary == 3072.0
    report["peak_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 3)
    return report


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def run_probe(source_dir: pathlib.Path, label: str) -> dict:
    out = subprocess.run(
        [sys.executable, str(pathlib.Path(__file__).resolve()), "--probe", str(source_dir)],
        capture_output=True,
        text=True,
    )
    sys.stderr.write(out.stderr)
    match = re.search(r"^REPORT (\{.*\})\s*$", out.stdout, re.M)
    if not match:
        print(f"--- {label} probe produced no report ---")
        print(out.stdout[-4000:])
        raise SystemExit(f"{label} probe failed (exit {out.returncode})")
    return json.loads(match.group(1))


def patched_copy(workdir: pathlib.Path) -> pathlib.Path:
    """Copy the library and apply the oracle patch out of solve.sh."""
    target = workdir / "fixed"
    shutil.copytree(ENVIRONMENT, target)
    body = SOLVE.read_text()
    inner = body.split("python3 - <<'PY'", 1)[1].split("\nPY\n", 1)[0]
    script = workdir / "patch.py"
    script.write_text(inner)
    subprocess.run([sys.executable, str(script)], cwd=target, check=True)
    return target


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", metavar="DIR")
    args = parser.parse_args()

    if args.probe:
        print("REPORT " + json.dumps(probe(args.probe)))
        return 0

    problems: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        workdir = pathlib.Path(tmp)
        broken = run_probe(ENVIRONMENT, "as-shipped")
        fixed = run_probe(patched_copy(workdir), "patched")

    print("=" * 74)
    print("environment")
    print("=" * 74)
    for key in ("torch", "triton", "device", "device_total_gib", "device_free_gib"):
        print(f"  {key:<20} {broken.get(key)}")
    print(f"  {'table size':<20} {broken.get('table_gib')} GiB")
    print(f"  {'peak allocated':<20} {broken.get('peak_gib')} GiB")

    if broken.get("fatal") or fixed.get("fatal"):
        print(f"\nFATAL: {broken.get('fatal') or fixed.get('fatal')}")
        return 1

    print()
    print("=" * 74)
    print("1/2. int32 row base: does it wrap, and is the wrap silent?")
    print("=" * 74)
    boundary = broken["predicted_first_failing_row"]
    print(f"  predicted first failing row      {boundary}")
    print(f"  rows that came back wrong        {broken['mismatched_rows']}")
    print(f"  ...of which entirely zero        {broken['mismatched_rows_that_are_all_zero']}")
    print(f"  boundary window (row -> equal)   {broken['boundary_window']}")
    print(f"  CUDA context alive afterwards    {broken['context_alive_after_gather']}")
    print(f"  correct sibling kernel unaffected {broken['sibling_kernel_ok']}")

    if not broken["mismatched_rows"]:
        problems.append(
            "the int32 row base did NOT wrap: no row came back wrong on the full "
            "size table. Triton may have promoted the multiply; inspect the "
            "generated TTIR before continuing."
        )
    if not broken["context_alive_after_gather"]:
        problems.append(
            "the CUDA context did not survive the gather, so the wrap is faulting "
            "instead of being clamped. The bounds guard in the kernel is not doing "
            "its job and the failure is not silent."
        )
    if broken["mismatched_rows"] and set(broken["mismatched_rows"]) != set(
        broken["mismatched_rows_that_are_all_zero"]
    ):
        problems.append(
            "some wrong rows were not all zero, so the wrapped offset is landing "
            "inside the table rather than being clamped. Still silent, but "
            "re-check that no read escapes the allocation."
        )
    expected_window = {
        str(boundary - 2): True,
        str(boundary - 1): True,
        str(boundary): False,
        str(boundary + 1): False,
    }
    if broken["boundary_window"] != expected_window:
        problems.append(
            f"boundary is not where predicted: expected {expected_window}, got "
            f"{broken['boundary_window']}"
        )
    if not broken["sibling_kernel_ok"]:
        problems.append("row_sums is wrong on the full size table; it must be correct")

    print()
    print("=" * 74)
    print("3. pitch helper: which layouts are wrong as shipped?")
    print("=" * 74)
    for name, expected in EXPECTED_LAYOUTS.items():
        actual = broken["layouts"].get(name)
        flag = "OK " if actual == expected else "!! "
        print(f"  {flag}{name:<16} expected {expected:<6} got {actual}")
        if actual != expected:
            problems.append(f"layout {name}: expected {expected}, got {actual}")

    print()
    print("=" * 74)
    print("4. patched library must be correct everywhere")
    print("=" * 74)
    bad_layouts = [n for n, v in fixed["layouts"].items() if v != "ok"]
    print(f"  layouts still wrong after fix    {bad_layouts or 'none'}")
    print(f"  rows still wrong after fix       {fixed['mismatched_rows'] or 'none'}")
    print(f"  boundary window after fix        {fixed['boundary_window']}")
    if bad_layouts:
        problems.append(f"patched library still wrong on layouts: {bad_layouts}")
    if fixed["mismatched_rows"]:
        problems.append(
            f"patched library still wrong on rows: {fixed['mismatched_rows']}"
        )
    if not all(fixed["boundary_window"].values()):
        problems.append("patched library still wrong at the boundary")

    print()
    print("=" * 74)
    if problems:
        print(f"VERDICT: {len(problems)} problem(s) -- do NOT package yet")
        for item in problems:
            print(f"  - {item}")
        return 1
    print("VERDICT: all expectations hold. Both defects are silent and independent.")
    print("Next: harbor run -p . -a nop   (separate invocation)")
    print("      harbor run -p . -a oracle")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
