# Task factory

Tooling for building further Harbor tasks the way `task5/` was built.

```
factory/
├── BUILD_PROMPT.md      # paste this into a fresh Claude Code session
├── STANDARDS.md         # YOU must add this: the acceptance standards doc
├── preflight.py         # is the factory set up? run this first
├── gate.py              # mechanical acceptance gate -- the loop's stopping rule
└── make_solver_zip.py   # answer-free package for the adversarial run
```

```powershell
python factory\preflight.py                        # before anything
python factory\new_task.py --task task6 --idea 1 --name mylib
python factory\gate.py --task task6 --static-only
python factory\gate.py --task task6
python factory\make_solver_zip.py --task task6     # for the adversarial session
python factory\grade.py --task task6 --patch <dir> # score a solver's patch
```

## Running the whole loop unattended

```powershell
python factory\orchestrate.py --check-cli
python factory\orchestrate.py --task task6 --idea 1 --name mylib --rounds 3
```

Two headless Claude Code sessions per round: a builder in the repo, and a solver
in a clean extract of the answer-free package, so it cannot reach `solve.sh`. The
solver's patch is graded against the real suite; reward 0 ends the loop, reward 1
feeds its own account of the shortcut back to the builder and the next round
starts. Every transcript lands in `factory/runs/<task>/`.

When the gate goes green a third session reviews the task with no stake in it
passing: it judges whether any sibling file hands over the fix, and whether
`instruction.md` names when the defect fires. A change-needed verdict feeds back
to the builder and the loop continues.

It stops and hands back for two things it must not do: writing `instruction.md`
and `task_reasoning.md` as final copy, since project rules treat model-written
copy as grounds for removal, and submitting.

The builder edits files without asking. `--permission-mode acceptEdits` is the
default; `bypassPermissions` also lets it run arbitrary commands. Decide which
before leaving it unattended.

## Before the first run

`BUILD_PROMPT.md` tells the agent to read `factory/STANDARDS.md` and
`task_ideas.md`. Put both in the repo — the standards document you already have,
saved as `factory/STANDARDS.md`, and the idea pool at the repo root. Without them
the agent invents its own bar, which is the failure mode this whole setup exists
to prevent.

## Running it

```powershell
cd C:\Users\RARx\Documents\Github\Tasks\temp
claude
```

Then paste the contents of `factory/BUILD_PROMPT.md`, followed by one line:

```
Build idea 1 as task6.
```

Claude Code on your own machine, not a cloud sandbox: the gate shells out to
`docker`, and the sandboxes have neither a GPU nor access to docker.io.

## The gate

```powershell
python factory\gate.py --task task6 --static-only   # fast, no docker
python factory\gate.py --task task6                 # adds the nop/oracle runs
```

Exit code 0 only when every check passes. What it decides:

| Check | Hard fail on |
|---|---|
| layout | a missing required file |
| task.toml | unparseable, or an empty required field |
| manifest | `config.json` and the defined tests not matching exactly, or duplicates |
| instruction | a phrasing that says *when* the defect fires |
| leakage | a mechanism word inside the library the agent reads |
| solve.sh | not applying, or applying twice |
| no-exemplar | a patched line appearing verbatim in another file |
| isolation | a **sibling** task directory being modified (shared tooling is fine) |
| adversarial | no `factory/records/<task>-adversarial.md`, or a reward other than 0 |
| archive | backslash entries, a wrapper folder, missing root `instruction.md` |
| nop | reward != 0, or anything ERRORing, or the wrong set failing |
| oracle | reward != 1 |

It reports but does not decide: library size, structurally similar lines near
the patch, instruction length. Read those.

## What the gate cannot do

- **Judge whether a sibling gives the answer away.** It prints the patched lines
  and every structurally similar line elsewhere, because the line between "shows
  the idiom" and "hands over the answer" is real judgement. Task 5 v2 keeps the
  same widening idiom two lines above its defect and a solver still missed it;
  task 5 v1 had it in a neighbouring file and a solver copied it immediately.
- **Check defect independence.** Apply each half of `solve.sh` alone and confirm
  the other half's tests still fail. `task5`'s `tools/verify_harness.py
  --independence` shows the shape.
- **Tell you the task is hard.** Only an adversarial solver run does that, and
  only on a GPU if the task needs one.
