You are building a Harbor evaluation task: a small, realistic Python library
with deliberate silent defects, plus a test suite that fails before the fix and
passes after it. It is accepted only if a frontier coding agent cannot repair it.

Read these first, in full, before writing anything:

- `factory/STANDARDS.md` — the acceptance standards. They are binding.
- `task_ideas.md` — the idea pool. I will tell you which idea number to build.
- `task5/` — a finished, verified task. Use it as the shape to copy: file
  layout, harness, `parser.py`, `test.sh` with both reward paths, `solve.sh`
  with match-count assertions.

# The loop

You are not done when you think the task is good. You are done when
`factory/gate.py` says so. Work in this cycle and do not leave it early:

1. **Design.** Write down, before coding, what the two defects are, what
   triggers each, what the symptom looks like, and why neither can be found by
   reading the file it lives in. If you cannot answer the last one, the design
   is wrong — go back.
2. **Verify the mechanism first.** If a defect depends on behaviour you have not
   personally observed — integer widths in generated code, allocator reuse,
   dispatcher order, autograd internals — prove it in a throwaway script before
   building anything around it. Read the library's source if you cannot run it.
   Record what you verified and how.
3. **Build.** Write the defect-free siblings first, then introduce the defects.
4. **Gate.** `python factory/gate.py --task taskN --static-only`, then the full
   run with docker. Fix every FAIL. Read every warn and say what you decided.
5. **Adversarial check.** Package the task with `factory/make_solver_zip.py`,
   then in a *separate* session hand a solver agent only that package and see if
   it repairs both defects. If it does, you have not finished: read its
   reasoning, find the shortcut it used, close that shortcut, return to step 1.

# Hard rules, each one learned from a task that failed

- **No sibling may contain the corrected form of the broken expression.** The
  first version of task 5 put the missing cast in a neighbouring kernel; a solver
  diffed the two files and fixed it in one line without understanding anything.
  The gate reports structurally similar lines — look at every one.
- **No correct twin beside the wrong value in the same file.** A helper
  returning the right thing ten lines above the helper returning the wrong thing
  is an answer key, not a decoy.
- **The symptom must not point at the mechanism.** Whole zero rows send a reader
  straight to the load mask, and the only mask term that can be false for valid
  input names the cause. Prefer wrong values read from real, plausible places.
- **The fix must not be reachable by reflex.** If a well-known rule of thumb
  ("cast offset arithmetic to int64", "call .contiguous() first") repairs it
  without diagnosis, it is too easy. Make the idiom visibly present but subtly
  misplaced.
- **Two independent defects, in different files.** Repairing one must leave the
  other's tests failing. The gate does not check this; you must, by applying
  each half of `solve.sh` alone.
- **`pass_to_pass` must reward wrong hypotheses.** The obvious probe should come
  back clean, so the obvious theory is disproved rather than confirmed.
- **Every assertion compares against a reference computed outside the library** —
  plain Python over `.tolist()`, or a different API family. Never library output
  against library output.
- **Block the trivial bypass.** If rewriting the function in pure torch would
  pass every value test, add a `pass_to_pass` test that pins the real code path.
- **`instruction.md` states the contract and the symptom class, never the
  trigger.** No sentence may say when the defect fires. The gate greps for the
  known-bad phrasings; passing that grep is the floor, not the bar.

# Do not

- Do not claim success without a green gate. Three solver agents in a row
  declared victory on this project while scoring zero, because they measured
  themselves against their own tests.
- Do not make tests brittle or the fix unguessable to raise difficulty. A task a
  competent engineer cannot solve in the estimated hours is rejected.
- Do not write `instruction.md` or `task_reasoning.md` as final copy. Draft them,
  and tell me clearly that I have to rewrite both in my own words.
- Do not touch any other task directory.

# Deliverables

A `taskN/` directory in the shape of `task5/`, a green gate, a written record of
what you verified before building, and a short summary of: the two defects, what
triggers each, why each resists a reflex fix, and what is still unverified and
needs my GPU.

Report honestly. If something is untested, say it is untested.
