#!/usr/bin/env python3
"""Verify-first harness for the gatherlib task. Run this on a GPU.

Answers the questions that decide whether the task is viable and that cannot be
answered without a CUDA device:

  1. Does the 32-bit lane product actually wrap in generated Triton code, and at
     exactly the predicted lane?
  2. Is the wrap silent -- clamped to zeros rather than an illegal access that
     poisons the CUDA context -- and confined to the tail of a row?
  3. Do the cases that must stay correct stay correct: a large contiguous table,
     a small transposed table?
  4. Does the plan cache key really collide for two unpacked views of one shape,
     while still separating a packed table from a view?
  5. How much device memory does the shared full size allocation need?

Usage
-----
    python tools/verify_first.py                # run everything, print verdict
    python tools/verify_first.py --probe DIR    # internal single-variant probe

On a Windows host use tools/verify_in_docker.ps1, which runs this inside the
task's own image; Triton has no official Windows support.

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

STORE_ROWS = 840_000
STORE_COLS = 2_560

# Layout name -> expected verdict for the code as shipped.
EXPECTED_LAYOUTS = {
    "large_contiguous": "ok",
    "small_transposed": "ok",
    "small_contiguous": "ok",
    "large_transposed": "wrong",
    "large_transposed_strided": "wrong",
}


def build_store(torch, device="cuda"):
    table = torch.empty((STORE_ROWS, STORE_COLS), dtype=torch.int8, device=device)
    col = torch.arange(STORE_COLS, device=device, dtype=torch.int32)
    chunk = 4096
    for start in range(0, STORE_ROWS, chunk):
        stop = min(start + chunk, STORE_ROWS)
        rows = torch.arange(start, stop, device=device, dtype=torch.int32).unsqueeze(1)
        block = (rows * 37 + col * 11 + (rows >> 3)) % 251 - 125
        table[start:stop].copy_(block)
    return table


def probe(source_dir: str) -> dict:
    os.environ.setdefault("TRITON_DEBUG", "0")
    sys.path.insert(0, source_dir)

    try:
        import torch
        import triton  # noqa: F401
    except ImportError as exc:
        return {
            "fatal": f"{exc}. This probe needs torch and Triton on a CUDA device. "
            f"Triton has no official Windows support, so run this inside the "
            f"task's own image instead -- see tools/verify_in_docker.ps1.",
            "interpreter": sys.executable,
            "platform": sys.platform,
        }

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

    from gatherlib import clear_plans, gather_rows, plan_for
    from gatherlib.cache import plan_key
    from gatherlib.config import DEFAULT_CONFIG

    # ---- small layouts ----------------------------------------------------
    generator = torch.Generator().manual_seed(20250729)
    small = torch.randint(
        -120, 120, (96, 128), dtype=torch.int8, generator=generator
    ).cuda()
    picks = [0, 3, 11, 17, 4, 3]
    ids = torch.tensor(picks, dtype=torch.int64, device="cuda")

    def check_layout(table):
        clear_plans()
        values = table.cpu().tolist()
        want = [list(values[r]) for r in picks]
        return "ok" if gather_rows(table, ids).cpu().tolist() == want else "wrong"

    layouts = {
        "small_contiguous": check_layout(small),
        "small_transposed": check_layout(small.t()),
    }

    # ---- the shared full size allocation ---------------------------------
    torch.cuda.reset_peak_memory_stats()
    store = build_store(torch)
    report["store_gib"] = round(store.numel() / 2**30, 4)

    def check_big(table, rows):
        clear_plans()
        picked = torch.tensor(rows, dtype=torch.int64, device="cuda")
        got = gather_rows(table, picked)
        want = table[picked]
        if bool(torch.equal(got, want)):
            return "ok", None
        differing = (got != want).nonzero()
        return "wrong", int(differing[0][1])

    layouts["large_contiguous"], _ = check_big(store, [5, 300_000, 839_999])
    layouts["large_transposed"], first_bad = check_big(store.t(), [0, 700, 2559])
    layouts["large_transposed_strided"], _ = check_big(store.t()[:, ::2], [3, 2559])
    report["layouts"] = layouts

    # The lane where a 32-bit product first crosses 2**31, for col_pitch = 2560.
    col_pitch = STORE_COLS
    report["predicted_first_bad_lane"] = 2**31 // col_pitch + (
        0 if 2**31 % col_pitch == 0 else 1
    )
    report["observed_first_bad_lane"] = first_bad

    # Wrong values must be zeros (clamped) and confined to the tail.
    clear_plans()
    row = torch.tensor([1234], dtype=torch.int64, device="cuda")
    transposed = store.t()
    got = gather_rows(transposed, row)[0]
    want = transposed[1234]
    mismatch = (got != want)
    report["n_wrong_in_row"] = int(mismatch.sum())
    report["row_len"] = int(want.numel())
    report["all_wrong_values_are_zero"] = bool((got[mismatch] == 0).all()) if int(
        mismatch.sum()
    ) else True
    report["wrong_values_are_a_tail"] = (
        bool(mismatch[-1]) and bool(~mismatch[0]) if int(mismatch.sum()) else True
    )

    # ---- the plan cache key ----------------------------------------------
    def unpacked(pitch, seed):
        gen = torch.Generator().manual_seed(seed)
        return torch.randint(
            -120, 120, (96, pitch), dtype=torch.int8, generator=gen
        ).cuda()[:, :128]

    p, q = unpacked(256, 901), unpacked(384, 902)
    report["key_collides_for_two_unpacked_views"] = plan_key(
        p, DEFAULT_CONFIG
    ) == plan_key(q, DEFAULT_CONFIG)
    report["key_separates_packed_from_unpacked"] = plan_key(
        small, DEFAULT_CONFIG
    ) != plan_key(p, DEFAULT_CONFIG)

    def sequence_result(first, second):
        clear_plans()
        gather_rows(first, ids)
        values = second.cpu().tolist()
        want = [list(values[r]) for r in picks]
        return "ok" if gather_rows(second, ids).cpu().tolist() == want else "wrong"

    report["sequences"] = {
        "unpacked_then_other_unpacked": sequence_result(p, q),
        "other_unpacked_then_unpacked": sequence_result(q, p),
        "packed_then_unpacked": sequence_result(small, q),
        "same_table_twice": sequence_result(q, q),
    }
    # A stale plan must read real neighbouring data, not masked zeros.
    clear_plans()
    gather_rows(p, ids)
    stale = gather_rows(q, ids)
    truth = q[ids]
    wrong = stale != truth
    n_wrong = int(wrong.sum())
    report["stale_wrong_values"] = n_wrong
    report["stale_wrong_values_that_are_zero"] = (
        int((stale[wrong] == 0).sum()) if n_wrong else 0
    )

    # ---- the context must still be alive --------------------------------
    torch.cuda.synchronize()
    canary = (torch.ones(1024, device="cuda") * 3).sum().item()
    report["context_alive"] = canary == 3072.0
    report["peak_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 3)
    del store
    torch.cuda.empty_cache()
    return report


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
        shipped = run_probe(ENVIRONMENT, "as-shipped")
        fixed = run_probe(patched_copy(workdir), "patched")

    fatal = shipped.get("fatal") or fixed.get("fatal")
    if fatal:
        print("=" * 74)
        print(f"FATAL: {fatal}")
        print("=" * 74)
        print(f"  interpreter  {shipped.get('interpreter', sys.executable)}")
        print(f"  platform     {shipped.get('platform', sys.platform)}")
        return 1

    print("=" * 74)
    print("environment")
    print("=" * 74)
    for key in ("torch", "triton", "device", "device_total_gib", "device_free_gib"):
        print(f"  {key:<24} {shipped.get(key)}")
    print(f"  {'shared allocation':<24} {shipped.get('store_gib')} GiB")
    print(f"  {'peak allocated':<24} {shipped.get('peak_gib')} GiB")

    print()
    print("=" * 74)
    print("1. does the 32-bit lane product wrap, and where?")
    print("=" * 74)
    predicted = shipped["predicted_first_bad_lane"]
    observed = shipped["observed_first_bad_lane"]
    print(f"  predicted first wrong lane   {predicted}")
    print(f"  observed first wrong lane    {observed}")
    if observed != predicted:
        problems.append(
            f"first wrong lane is {observed}, predicted {predicted}. If it is None "
            f"the product did not wrap at all: inspect the generated TTIR before "
            f"continuing."
        )

    print()
    print("=" * 74)
    print("2. is the wrap silent, and confined to a tail?")
    print("=" * 74)
    print(f"  wrong values in one row      {shipped['n_wrong_in_row']} "
          f"of {shipped['row_len']}")
    print(f"  every wrong value is zero    {shipped['all_wrong_values_are_zero']}")
    print(f"  wrong values form a tail     {shipped['wrong_values_are_a_tail']}")
    print(f"  CUDA context alive after     {shipped['context_alive']}")
    if not shipped["context_alive"]:
        problems.append(
            "the CUDA context did not survive, so the wrap is faulting instead of "
            "being clamped; the failure is not silent"
        )
    if not shipped["all_wrong_values_are_zero"]:
        problems.append(
            "some wrong values are not zero, so a wrapped offset landed inside the "
            "table; re-check that no read escapes the allocation"
        )
    if not shipped["wrong_values_are_a_tail"]:
        problems.append("the corruption is not a tail slice of the row")
    if not 0 < shipped["n_wrong_in_row"] < shipped["row_len"]:
        problems.append(
            f"expected part of the row wrong, got {shipped['n_wrong_in_row']} "
            f"of {shipped['row_len']}"
        )

    print()
    print("=" * 74)
    print("3. which layouts are wrong as shipped?")
    print("=" * 74)
    for name, expected in EXPECTED_LAYOUTS.items():
        actual = shipped["layouts"].get(name)
        flag = "OK " if actual == expected else "!! "
        print(f"  {flag}{name:<26} expected {expected:<6} got {actual}")
        if actual != expected:
            problems.append(f"layout {name}: expected {expected}, got {actual}")

    print()
    print("=" * 74)
    print("4. the plan cache key")
    print("=" * 74)
    print(f"  collides for two unpacked views  "
          f"{shipped['key_collides_for_two_unpacked_views']}")
    print(f"  separates packed from unpacked   "
          f"{shipped['key_separates_packed_from_unpacked']}")
    if not shipped["key_collides_for_two_unpacked_views"]:
        problems.append("the key does not collide, so the cache defect cannot fire")
    if not shipped["key_separates_packed_from_unpacked"]:
        problems.append(
            "the key fails to separate a packed table from a view, which would "
            "break the pass_to_pass mislead"
        )
    expected_sequences = {
        "unpacked_then_other_unpacked": "wrong",
        "other_unpacked_then_unpacked": "wrong",
        "packed_then_unpacked": "ok",
        "same_table_twice": "ok",
    }
    for name, expected in expected_sequences.items():
        actual = shipped["sequences"].get(name)
        flag = "OK " if actual == expected else "!! "
        print(f"  {flag}{name:<32} expected {expected:<6} got {actual}")
        if actual != expected:
            problems.append(f"sequence {name}: expected {expected}, got {actual}")
    n_wrong = shipped["stale_wrong_values"]
    n_zero = shipped["stale_wrong_values_that_are_zero"]
    print(f"  stale plan -> {n_wrong} wrong values, {n_zero} of them zero")
    if n_wrong and n_zero > n_wrong // 2:
        problems.append(
            "most wrong values from a stale plan are zeros, so the symptom points "
            "at the mask rather than looking like misread data"
        )

    print()
    print("=" * 74)
    print("5. the patched library must be correct everywhere")
    print("=" * 74)
    bad_layouts = [n for n, v in fixed["layouts"].items() if v != "ok"]
    bad_sequences = [n for n, v in fixed["sequences"].items() if v != "ok"]
    print(f"  layouts still wrong    {bad_layouts or 'none'}")
    print(f"  sequences still wrong  {bad_sequences or 'none'}")
    print(f"  wrong values in a row  {fixed['n_wrong_in_row']}")
    if bad_layouts:
        problems.append(f"patched library still wrong on layouts: {bad_layouts}")
    if bad_sequences:
        problems.append(f"patched library still wrong on sequences: {bad_sequences}")
    if fixed["n_wrong_in_row"]:
        problems.append("patched library still corrupts part of a row")

    print()
    print("=" * 74)
    if problems:
        print(f"VERDICT: {len(problems)} problem(s) -- do NOT package yet")
        for item in problems:
            print(f"  - {item}")
        return 1
    print("VERDICT: all expectations hold. Both defects are silent, narrowly")
    print("         triggered, and repaired by solve.sh.")
    print("Next: python tools/verify_harness.py")
    print("      python tools/verify_harness.py --independence")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
