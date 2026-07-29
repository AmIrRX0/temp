# Task 5 — Triton gather kernel: int32 address overflow + wrong row pitch

Implements **Idea 5** from `task_ideas.md` (Kernel Structural Correctness, GPU).

```
task5/                      # this is the zip root
├── environment/
│   ├── Dockerfile
│   └── gatherlib/          # 10 files, ~470 lines
├── tests/                  # full harness shape: test.sh + run_script.sh + parser.py + config.json
├── solution/solve.sh
├── instruction.md          # agent facing
├── task_reasoning.md       # reviewer facing
└── task.toml
tools/
├── verify_first.py         # RUN THIS ON A GPU BOX FIRST
├── verify_harness.py       # runs the real verifier on a GPU without Harbor
└── make_zip.py             # forward-slash, LF-normalising archive builder (S8)
```

## The two defects

| | File | Defect | Symptom |
|---|---|---|---|
| **A** | `kernels/gather.py` | `row_base = src_row * row_pitch` — both operands are i32 in generated code, so the product wraps past element 2³¹. The kernel *does* widen to int64, but on the column term and the output offset instead of the row base. | Rows beyond the wrap point come back **all zero** (the kernel's bounds clamp turns the negative offset into a masked load). |
| **B** | `addressing.py` | `packed_row_strides` returns `shape[-1]` as the row pitch instead of `stride(-2)` when the trailing axis is packed. | Rows read from the **wrong position**. |

Different symptom signatures, different files, independently fatal (S2).

Defect B is correct for contiguous, transposed, column-strided and row-offset
tables, and wrong only for row-step-sliced, column-narrowed and column-windowed
ones — so "views are broken" is a wrong hypothesis, and the `pass_to_pass` set
rewards it.

## What is verified, and what is not

Verified in this container (no GPU here):

- **Triton 3.3.1 integer semantics, read from source.** `tl.arange` → `int32`
  (`language/semantic.py:arange`); `mul(i32,i32)` → `i32` with no promotion from
  the consuming context (`integer_promote_impl`); a Python `int` kernel argument
  specialises to `"i32"` when it fits in i32, and to `constexpr` when it equals 1
  (`runtime/jit.py:299-308`).
- **The overflow sanitiser is compiled out.** Triton 3.3.1 defaults to
  `sanitize_overflow=True` and `binary_op_sanitize_overflow_impl` emits
  `device_assert("int32 overflow detected for operation mul")` — but
  `semantic.device_assert` returns early unless `options.debug`, and `debug`
  defaults to `False` (`runtime/jit.py:526`). So the wrap is silent. **This is
  why `run_script.sh` exports `TRITON_DEBUG=0`**: with `TRITON_DEBUG=1` the task
  would raise instead of failing silently.
- **Both defects, simulated on CPU** with those exact type rules: which layouts
  Defect B breaks, and that Defect A's wrapped offset is masked (no
  out-of-bounds read) rather than faulting.
- **Test suite collects** — 21 tests, and `config.json` matches the collected
  names exactly (8 `fail_to_pass`, 13 `pass_to_pass`, no extras, no dupes).
- **`solve.sh` applies cleanly** and refuses to double-apply; the patched
  library is correct on all seven layouts in simulation.
- **Leakage audit (S7)** — no `TODO`/`FIXME`/`HACK`, and no occurrence of
  `int32`, `int64`, `overflow`, `wrap`, `bug`, `wrong` anywhere in
  `environment/`, `tests/` or `solution/`.

Found on real hardware and fixed:

- **The `-runtime` base image has no C compiler.** Triton builds a CPython
  extension for its CUDA driver shim on the first kernel launch
  (`runtime/build.py:_build`), so every test errored with
  `Failed to find C compiler` before any kernel ran. The Dockerfile now installs
  `gcc` and asserts at build time that the compiler and the Python headers work
  together, resolving the include directory the way Triton's builder does.
  Without this the `oracle` run could never have reached 1.0.

**Confirmed on hardware** (RTX 2050, 4 GiB, inside the task's own image):

- **The int32 wrap happens in generated code, at exactly the predicted row.**
  `2**31 // 2560 + 1` = 838861. Row 838860 gathers correctly, row 838861 does
  not. The boundary is not approximate -- it is the arithmetic boundary.
- **The wrap is silent.** Every wrong row comes back entirely zero, i.e. the
  bounds clamp masked the negative offset, and the CUDA context is still alive
  afterwards. No illegal memory access. This is what the clamp was added for:
  Idea 5 as written would have faulted instead, which would have taken down the
  `pass_to_pass` set too.
- **All seven layouts behave as designed** -- contiguous, transposed,
  column-strided and row-offset correct; row-step-sliced, column-narrowed and
  column-windowed wrong.
- **The patched library is correct everywhere**, including at the boundary.
- **The correct sibling kernel is unaffected** on the same full size table.
- **Memory: 2.12 GiB peak** against 3.23 GiB free on a 4 GiB card.

Also found by running Harbor:

- **`task.toml` needs `gpus` under `[environment]`.** Without it Harbor brings
  the container up with no device and all 21 tests error with
  `Found no NVIDIA driver`, which makes the `nop` run report 0.0 for entirely
  the wrong reason. The key name comes from Harbor's own trial config, which
  carries an `override_gpus` field alongside `override_cpus`,
  `override_memory_mb` and `override_storage_mb`.

- **Harbor's local `docker` environment cannot run a GPU task at all.**
  `DockerEnvironment.capabilities` (`environments/docker/docker.py:291`) never
  sets `gpus`, so it defaults to False and `_validate_gpu_support` raises before
  a container is ever created. It is a hard no, not a flag. `harbor run --env`
  offers GPU-capable backends (modal, novita, beam, blaxmith, ...) but they all
  need a cloud account.

  `tools/verify_harness.py` closes that gap without one: it runs the task's real
  verifier -- the same `test.sh`, `run_script.sh`, `parser.py` and `config.json`
  -- under plain `docker run --gpus all`, once as shipped and once with
  `solve.sh` applied, and checks the two outcomes Harbor would check.

  The failed CPU-only Harbor run did still validate the plumbing end to end: the
  image builds, `test.sh` runs, `parser.py` reported all 21 test names correctly
  into `report.json` (so `config.json` matches what pytest prints), both reward
  paths were written, and the oracle agent log shows `solution applied`.

- **A Windows checkout leaves CRLF in the working tree**, and `bash` in the
  container then dies with `$'\r': command not found` on every line of
  `test.sh`. `.gitattributes` only helps a checkout made *after* it exists, so a
  tree cloned earlier stays broken. The dangerous part was that `make_zip.py`
  read those same bytes, so the submitted archive would have carried CRLF into
  QA's Linux run. `make_zip.py` now normalises text files to LF on the way in,
  sets `0o755` on `.sh`, uses a fixed timestamp, and re-opens the archive to
  assert no CRLF survived. `verify_harness.py` stages `tests/` and the library
  the same way, so it exercises what ships rather than what is on disk.

Still outstanding:

- `python tools/verify_harness.py` -- the pass/fail outcomes on a GPU.
- Optionally a real `harbor run` on a GPU-capable `--env` backend.

## Run order on the GPU box

Triton has no official Windows support, so on a Windows host do **not** try to
install it natively — run the probe inside the task's own image, which is the
environment the task is graded in anyway:

```powershell
.\tools\verify_in_docker.ps1
```

On Linux, or in any environment that already has torch + Triton + CUDA:

```bash
python tools/verify_first.py        # must print "VERDICT: all expectations hold"
```

Then, as **separate** shell invocations (chaining them trips a
`FileNotFoundError` in Harbor's path resolution):

```bash
cd task5 && harbor run -p . -a nop      # expect 0.0, exactly the 8 fail_to_pass failing
```
```bash
cd task5 && harbor run -p . -a oracle   # expect 1.0
```

Check `jobs/<timestamp>/<trial>/verifier/test-stdout.txt`, not just the reward.
Then `python tools/make_zip.py`, and delete `task5/jobs/` first if it exists.

## Device memory

The full-size fixture is `840000 × 2560` int8 = **2.003 GiB**, allocated once by
a module-scoped fixture. With the CUDA context the suite needs ~2.5 GiB free.
This cannot be reduced: the table has to span more than 2³¹ elements for the
contract to be exercised at all, and int8 is already the narrowest dtype.
Set the compute profile to **GPU** on the submit form.

## Before you submit

`instruction.md` and `task_reasoning.md` in this repo are drafts. The project
rules say instructions and task reasoning must be written by you — rewrite both
in your own words before submitting.
