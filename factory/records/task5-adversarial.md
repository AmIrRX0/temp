# Adversarial run record for task5

reward: 0

solver model: Sonnet via Claude Code, cloud sandbox, no GPU and no docker.io
date: 2026-07-30
package handed over: factory/make_solver_zip.py output -- instruction.md and
environment/ only. No tests/, no solution/, no task_reasoning.md, no task.toml.
The solver confirmed it found no answer files.

## What it did

Read all 13 files. Formed its hypothesis from `cache.py::plan_key` before
running anything: the key carries `is_contiguous()` and `stride(-1)` but not
`stride(-2)`, so two unpacked views of one shape with different row pitches
collide. Installed CPU torch and Triton, monkey-patched only the CUDA gate, and
ran real kernels under the interpreter. Wrote 28 probes derived from the three
contracts, including one added specifically to exercise the multi-column-block
kernel path at `row_len > 1024`. Baseline: 26 pass / 2 fail. After its fix:
28/28. Declared success.

## What it found

Defect B, first attempt, by reading. This is the third solver in a row to find
it that way, which is why the reviewer docs now say plainly that B is the
weaker defect.

## What it missed

Defect A, the 32-bit lane product in `kernels/gather.py`. It never had a probe
that could trigger it, and it did not reason its way there either despite
reading the kernel and deliberately probing the multi-block path. The widening
idiom is visibly present two lines above the defect, which appears to be enough
to stop a reader asking which term the cast belongs to.

## Shortcut used

None. It patched `cache.py` only.

## Graded result

`python tools/verify_harness.py --grade` against the real suite on an RTX 2050:

    reward Harbor would give: 0
      fail_to_pass fixed     4/8
      pass_to_pass intact    21/21
      still failing: the four transposed / strided full-size gather tests

Its patch was reconstructed by `tools/rebuild_solver_patch.py` from the complete
diff the solver printed, and verified byte-identical to the shipped library
everywhere except the one line in `cache.py`.

## Caveat

The run had no GPU, so defect A could not be discovered empirically, only by
reasoning. A Harbor agent has a device and can probe. This result shows the
source does not give A away by reading; it is not proof that A survives an
agent that can allocate 2 GiB and compare a transposed gather against torch.
