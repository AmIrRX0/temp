# Lessons

Binding history. Every entry cost a real cycle. Read this before designing, and
append to it after every round — the loop appends automatically, but a terse,
curated entry is worth more than a transcript.

---

## Why solvers beat a task

**A sibling holding the corrected expression.** task5 v1 put the missing
`.to(tl.int64)` in a neighbouring kernel that did the same multiply. A solver
diffed the two files and fixed it in one line, and said so: "this cast is exactly
what is missing in gather". No sibling may contain the corrected form of the
broken expression.

**A correct twin in the same file.** task5 v1 had a helper returning
`shape[-1]` as a row pitch ten lines under a helper returning `stride(-2)`. Side
by side, that is an answer key, not a decoy. Put the correct helper in another
file, or make it structurally different.

**A symptom that points at the mechanism.** task5 v1 returned whole zero rows.
That sends a reader straight to the load mask, and the only mask term that can be
false for valid input names the cause. Prefer wrong values read from real,
plausible places, or a thin corrupted slice of an otherwise correct result.

**A fix reachable by reflex.** "Cast offset arithmetic to int64" and "call
.contiguous() first" repair a defect without diagnosing it. Make the idiom
visibly present and subtly misplaced instead: task5 v2 casts both halves of the
address, but one cast lands outside the parenthesis, so a blanket widening pass
finds a cast on every term and changes nothing.

**Partial inclusion is a stronger hint than omission.** task5 v2 added
`stride(-1)` to a cache key that was missing `stride(-2)`, expecting a `grep
stride` to look reassuring. The opposite happened: three solvers in a row read
the key's elements and asked why only one stride was there. If a field list is
incomplete, do not draw the eye to it.

---

## Measuring difficulty

**A solver's self-assessment is worth nothing.** Three agents on this project
declared success while scoring zero, each measuring itself against tests it had
written. Always grade the patch against the real suite: `factory/grade.py`.

**Never hand the solver `tests/`.** Good fail_to_pass names describe the input
that triggers the defect, which is the thing the solver is supposed to discover.
Harbor never shows an agent the suite either.

**Never hand the solver the submission archive.** `solution/solve.sh`,
`task_reasoning.md` and `task.toml` each state the answer outright.

**One failure is luck.** Run a sweep of independent sessions, mixing in a
stronger model, and require every one to score 0.

**A solver without the hardware cannot discover a hardware-triggered defect.**
Two task5 runs missed the GPU defect entirely because they ran under
`TRITON_INTERPRET=1` with no CUDA. That shows the source does not give it away by
reading; it is not proof it survives an agent with a device.

---

## Environment traps

**`pytorch/pytorch:*-runtime` has no C compiler.** Triton builds a CPython
extension for its CUDA driver shim on the first kernel launch, so every test
errors with `Failed to find C compiler` before a kernel runs. Install `gcc` and
assert at build time that the compiler and Python headers work together.

**`task.toml` needs `gpus` under `[environment]`.** Without it Harbor brings the
container up with no device and every test errors, which makes the `nop` run
report 0.0 for entirely the wrong reason.

**Harbor's local docker environment cannot run a GPU task at all.**
`DockerEnvironment.capabilities` never sets `gpus`, so `_validate_gpu_support`
raises before a container exists. Its GPU-capable `--env` backends need a cloud
account. Drive the real `test.sh` under plain `docker run --gpus all` instead.

**A Windows checkout leaves CRLF, and `bash` dies on it.** `.gitattributes` only
helps a checkout made after it exists. Normalise to LF when building the archive,
not in the working tree, or QA's Linux run breaks instead of yours.

**Never put backslash-escaped quotes in a Dockerfile `RUN`.** Backslash is
Dockerfile's own escape character. Use outer double quotes with single quotes
inside, and pass anything awkward through a temp file rather than a nested
command substitution.

---

## Triton specifics

- `tl.arange` produces **int32** (`language/semantic.py:arange`).
- `mul(i32, i32)` stays i32; there is no promotion from the consuming context
  (`integer_promote_impl`). A cast applied to the product is too late.
- A Python `int` kernel argument specialises to `i32` when it fits in i32, and to
  `constexpr` when it equals 1 (`runtime/jit.py:299-308`). A pitch of 1 therefore
  never overflows, which is why a contiguous table can be safe while its
  transpose is not.
- Triton emits `device_assert("int32 overflow detected for operation mul")` on
  every i32 add/sub/mul, but `semantic.device_assert` returns early unless
  `options.debug`, which defaults to False. Pin `TRITON_DEBUG=0` in the runner:
  with it set to 1 an overflow defect raises instead of failing silently.
- A masked load turns a wrapped negative offset into zeros instead of an illegal
  memory access. Without that clamp the whole suite dies, not just the
  fail_to_pass set.

---

## Design arithmetic

**Crossing 2^31 element offsets costs 2 GiB.** int8 is the narrowest dtype, so
the allocation floor is fixed. Share one allocation across every full-size case
and use its transpose as the second layout rather than allocating twice.

**Prefer CPU-only ideas.** Five of the six failures on task5 came directly from
it being a GPU task. A CPU task reaches the same quality for a fraction of the
cycle cost.

---

## Appended by the loop

<!-- entries below are written automatically after each round; curate them -->
