# Task 5 — Triton gather: 32-bit lane product + an under-keyed plan cache

Implements **Idea 5** from `task_ideas.md` (Kernel Structural Correctness, GPU).

`v1` is tagged `v1-verified` (commit `5e2d013`). It passed every mechanical
check on GPU — `nop` 0, `oracle` 1, defects independent — and was then **solved
on the first attempt** by a solver agent. This is v2, rebuilt to close the three
shortcuts that made v1 easy.

```
task5/                      # zip root
├── environment/
│   ├── Dockerfile
│   └── gatherlib/          # 12 files
├── tests/                  # test.sh + run_script.sh + parser.py + config.json
├── solution/solve.sh
├── instruction.md          # agent facing
├── task_reasoning.md       # reviewer facing
└── task.toml
tools/
├── verify_first.py         # probes the two mechanisms with actionable diagnostics
├── verify_harness.py       # runs the real verifier on a GPU without Harbor
├── make_solver_zip.py      # answer-free package for difficulty measurement
├── solver_env.Dockerfile   # neutral GPU image for the solver experiment
└── make_zip.py             # forward-slash, LF-normalising archive builder
```

## Why v1 was easy

| Shortcut | What it gave away |
|---|---|
| `rowsum.py` contained `.to(tl.int64)` before the *same* multiply | The solver said so explicitly: "this cast is exactly what is missing in gather". A one-line diff, no understanding needed. |
| `element_strides` sat ten lines above `packed_row_strides` in one file | `stride(-2)` vs `shape[-1]` side by side is an answer key, not a decoy. |
| Broken rows came back **entirely zero** | Points at the load mask; the only mask term that can be false for a valid row is `src_off >= 0`, which names the mechanism. |

## The v2 defects

| | File | Defect | Trigger | Symptom |
|---|---|---|---|---|
| **A** | `kernels/gather.py` | Offset built from two halves, `row_base = src_row.to(tl.int64) * row_pitch` and `in_row = (lane * col_pitch).to(tl.int64)` — both are i64 when they meet, and a comment says so. The cast on the second lands **one step too late**: `lane` is i32 from `tl.arange`, `col_pitch` is an i32 arg, so the product already wrapped | Needs a table **both** large **and** addressed down a column. Large contiguous is fine (`col_pitch == 1`); small transposed is fine (pitch tiny) | The last **0.136%** of every row zeroed |
| **B** | `cache.py` | Plan cache keyed on `shape, dtype, device, is_contiguous(), stride(-1), config` — a stride **is** in the key, so "it ignores layout" is false. It misses the *leading* pitch | Needs **two unpacked views of the same shape** that agree on the trailing pitch and differ on the leading one, gathered in sequence | Values read from real neighbouring rows, no masking |

**A is reflex-proof**: an agent applying the standard Triton rule ("cast offset
arithmetic to int64") finds a cast on every term already and changes nothing; the
fix is moving one cast inside a parenthesis.

**B is not.** A solver enumerated the key's elements against what `build_plan`
consumes and spotted the missing leading pitch on its first attempt — see the
solver-result section below. B stays because the suite is graded whole, so both
must be fixed to score 1, but the difficulty rests on A.

Nothing in the package multiplies a lane index by a column pitch anywhere else:
`row_sums` and `scatter_rows_` declare a packed-trailing-axis requirement
(validated, and a `pass_to_pass` test asserts it raises), take a ready-made
64-bit row offset from the host, and walk the row with a plain add. So there is
no exemplar to diff and no correct twin beside the wrong value.

## pass_to_pass is built to mislead

Large contiguous works, small transposed works, every unpacked view works **on
its own**, and gathering from a packed table then an unpacked one of the same
shape works — so the obvious probe for state leaking between calls comes back
clean. A decoy repair is available too: `validate.py` narrows row ids to int32,
which looks like the cause of an overflow but is not, because the row term is
already widened.

## Verified without a GPU

- **Triton 3.3.1 integer semantics, read from source.** `tl.arange` → `int32`
  (`language/semantic.py:arange`); `mul(i32,i32)` → `i32` with no promotion from
  the consuming context (`integer_promote_impl`); a Python `int` kernel argument
  specialises to `"i32"` when it fits, and to `constexpr` when it equals 1
  (`runtime/jit.py:299-308`) — which is why `col_pitch == 1` is safe.
- **The overflow sanitiser is compiled out.** Triton defaults to
  `sanitize_overflow=True` and emits
  `device_assert("int32 overflow detected for operation mul")`, but
  `semantic.device_assert` returns early unless `options.debug`, which defaults
  to `False` (`runtime/jit.py:526`). **`run_script.sh` pins `TRITON_DEBUG=0`**;
  with it set to 1 the task would raise instead of failing silently.
- **Both defects simulated on CPU** with those type rules: `col_pitch=1` correct,
  `col_pitch=2560` wrong from lane **838861** (exactly `2**31 // 2560 + 1`),
  1139 of 840000 lanes, masked so no out-of-bounds read; small transposed
  correct. And `key(p) == key(q)` collides while `key(p) == key(packed)` does
  not, with a stale plan giving 384/512 values wrong of which only 3 are zero.
- **29 tests collect**, `config.json` matches exactly (8 `fail_to_pass`,
  21 `pass_to_pass`).
- **`solve.sh` applies cleanly**, refuses to double-apply, and
  `verify_harness.py` can stage either half alone for the independence check.
- **Leakage audit** — nothing in `environment/`, `tests/` or `solution/` names
  overflow, wrapping, staleness, or a bug.

## Confirmed on hardware (RTX 2050, 4 GiB, inside the task's own image)

`tools/verify_first.py`:

- **The wrap starts at exactly the predicted lane** — 838861 predicted, 838861
  observed. Not approximate; the arithmetic boundary.
- **Silent and tail-confined** — 1134 of 840000 values in a row wrong, every one
  of them zero, all at the end, CUDA context alive afterwards. (1139 lanes are
  masked; about five of them happened to hold a genuine zero, so they do not
  register as wrong.)
- **All five layouts as designed** — large contiguous, small transposed and small
  contiguous correct; large transposed and large-transposed-strided wrong.
- **The key collides for the failing pair and separates packed from unpacked**,
  and all four call sequences behave as designed.
- **A stale plan gives 640 wrong values, only 3 of them zero** — misread real
  data, not masking.
- **Patched library correct everywhere**: no layout, sequence or value wrong.
- **Peak 2.12 GiB** against 3.23 GiB free.

`tools/verify_harness.py`, through the real verifier:

- `nop` -> reward **0**, 29/29 outcomes as expected (exactly the 8 `fail_to_pass`
  `FAILED`, all 21 `pass_to_pass` `PASSED`, nothing ERRORed).
- `oracle` -> reward **1**, 29/29.
- `--independence` -> repairing A alone leaves B's four failing and vice versa,
  29/29 each way.

## Solver result (first legitimate run, `make_solver_zip` package, no tests)

The solver **did not fix the task**. It found defect B (the cache key) on attempt
one and stopped, declaring success on its own 23-case suite. Defect A survived,
so on real grading this submission scores **reward 0** — A's four `fail_to_pass`
tests still fail.

Two honest caveats:

- **The run was blind to A.** It used `TRITON_INTERPRET=1` on CPU, and A needs a
  >2 GiB CUDA allocation, so its baseline was 21 pass / 2 fail: its suite never
  had a case that could trigger A. We learned nothing about A's resistance. The
  next run must be on a GPU.
- **Hardening B by adding `stride(-1)` to the key backfired.** The solver's
  reasoning was "builds the key from shape, dtype, device, is_contiguous(),
  stride(-1) and config but does not include stride(-2)". The prediction that a
  `grep stride` hit would look reassuring was wrong: including one of two strides
  puts the reader's eye on exactly the right line. Partial inclusion is a
  stronger hint than total omission. B is the weaker defect and the reviewer docs
  now say so.

B being findable matters less than it looks, because the suite is graded whole —
an agent must fix **both** to score 1.

**Graded result** (`tools/verify_harness.py --grade`, real suite, RTX 2050):

    reward Harbor would give: 0
      fail_to_pass fixed     4/8      <- defect B's four
      pass_to_pass intact    21/21    <- clean fix, no collateral damage
      still failing: the four transposed / strided full-size gather tests

A second solver run, given the same package, reached the same place: `cache.py`
found by reading, `gather.py` untouched, reward 0. Full record in
`factory/records/task5-adversarial.md`.

## Environment lessons carried over from v1

- **The `-runtime` base image has no C compiler.** Triton builds a CPython
  extension for its CUDA driver shim on the first kernel launch, so every test
  errors with `Failed to find C compiler`. The Dockerfile installs `gcc` and
  asserts at build time that the compiler and Python headers work together.
- **`task.toml` needs `gpus` under `[environment]`** or Harbor brings the
  container up with no device.
- **Harbor's local `docker` environment cannot run a GPU task at all** —
  `DockerEnvironment.capabilities` never sets `gpus`
  (`environments/docker/docker.py:291`), so `_validate_gpu_support` raises. Its
  GPU-capable `--env` backends need a cloud account. `verify_harness.py` covers
  the same ground with plain `docker run --gpus all`.
- **CRLF.** `make_zip.py` normalises text files to LF, marks `.sh` `0o755`, and
  re-opens the archive to assert none survived — a Windows checkout would
  otherwise have shipped CRLF into QA's Linux run.

## Run order

```powershell
.\tools\verify_in_docker.ps1          # builds the task image, runs verify_first inside it
python tools\verify_harness.py         # nop -> 0 with exactly the 8 f2p failing; oracle -> 1
python tools\verify_harness.py --independence
python tools\make_zip.py
```

## Measuring difficulty with a solver agent

Do **not** hand a solver the submission archive: `solution/solve.sh`,
`task_reasoning.md` and `task.toml` each state the answer outright.

```powershell
python tools\make_solver_zip.py        # instruction.md + environment/ only
```

The solver does **not** need a GPU of its own. What decides the experiment is
whether its final patch passes the real suite, and you can score that yourself:

```powershell
python tools\verify_harness.py --grade C:\path\to\solver_test\environment
```

That runs the actual `test.sh` against the solver's patched library on your GPU
and prints the reward Harbor would give, plus exactly which `fail_to_pass` tests
are still failing. A sandboxed solver with no CUDA can still reason its way to
both fixes from the source; if it cannot, `--grade` says so with reward 0.

Note the asymmetry: a CPU-only solver failing is **not** proof the task is hard,
only that the source does not give the defects away by reading. A real Harbor
agent has a device and can probe. A GPU-backed solver run is still the
conclusive test.

`tests/` is excluded on purpose. Harbor never shows the agent the suite, and
this task's fail_to_pass names describe both triggers — one says "an earlier
gather from a wider table", the other says "a transposed full size table". A
solver given those is being measured on reading test names, not on the defects.
`--with-tests` exists for a second, weaker run.

The packager refuses to build from a tree where `solve.sh` has already been
applied, so a leftover repair cannot silently produce a package with no bug in
it.

## Device memory

One allocation of `840000 × 2560` int8 = **2.003 GiB** backs every full-size
case, and its transpose is the second layout over the same bytes. With the CUDA
context the suite needs ~2.5 GiB free. This cannot be reduced: a pitch times an
index has to be able to cross 2³¹, and int8 is already the narrowest dtype.
Set the compute profile to **GPU** on the submit form.

## Before you submit

`instruction.md` and `task_reasoning.md` are drafts. The project rules say
instructions and task reasoning must be written by you — rewrite both in your
own words. In `instruction.md`, keep the contracts and the symptom class, and do
not add any sentence saying *when* the bug fires.
