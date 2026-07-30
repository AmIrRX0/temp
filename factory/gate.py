#!/usr/bin/env python3
"""Mechanical acceptance gate for a Harbor task.

An agent asked whether its own task is good enough will say yes. This says no
until the task actually earns it, and every check returns a hard verdict rather
than a judgement.

    python factory/gate.py --task task6 --static-only   # no docker needed
    python factory/gate.py --task task6                 # adds the GPU/docker runs

Exit code 0 only when every check passes.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile

REQUIRED_FILES = [
    "instruction.md",
    "task_reasoning.md",
    "task.toml",
    "environment/Dockerfile",
    "solution/solve.sh",
    "tests/test.sh",
    "tests/run_script.sh",
    "tests/parser.py",
    "tests/config.json",
    "tests/test_behavior.py",
]

# Words that name a mechanism. None of these belong in anything the agent under
# evaluation can read.
LEAK_WORDS = re.compile(
    r"\b(TODO|FIXME|HACK|XXX|bug|buggy|broken|defect|workaround|"
    r"overflow|wraps?|wrapped|stale|incomplete|missing|"
    r"be careful|order matters|do not change|don't change)\b",
    re.I,
)

# Phrasings that tell the agent when the defect fires. Adapted from the three
# rejected instruction files.
BANNED_INSTRUCTION = re.compile(
    r"(already work|currently work|works? (only )?when|"
    r"but (is )?wrong after|only when|after the first|the first call|"
    r"on the second|is not cached|is cached|reuses?)",
    re.I,
)

FAILURES: list[str] = []
WARNINGS: list[str] = []


def fail(check: str, detail: str) -> None:
    FAILURES.append(f"{check}: {detail}")
    print(f"  FAIL  {check}: {detail}")


def warn(check: str, detail: str) -> None:
    WARNINGS.append(f"{check}: {detail}")
    print(f"  warn  {check}: {detail}")


def ok(check: str, detail: str = "") -> None:
    print(f"  ok    {check}{(': ' + detail) if detail else ''}")


def header(title: str) -> None:
    print()
    print("=" * 74)
    print(title)
    print("=" * 74)


# ---------------------------------------------------------------------------
# static checks
# ---------------------------------------------------------------------------


def check_layout(task: pathlib.Path) -> None:
    for name in REQUIRED_FILES:
        if (task / name).exists():
            ok(f"present {name}")
        else:
            fail("layout", f"missing {name}")
    if (task / "jobs").exists():
        warn("layout", "jobs/ exists; delete it before packaging")


def check_task_toml(task: pathlib.Path) -> dict:
    import tomllib

    try:
        data = tomllib.loads((task / "task.toml").read_text(encoding="utf-8"))
    except Exception as exc:
        fail("task.toml", f"does not parse: {exc}")
        return {}
    meta = data.get("metadata", {})
    for key in ("category", "difficulty", "difficulty_explanation",
                "solution_explanation", "verification_explanation",
                "expert_time_estimate_hours"):
        if not meta.get(key):
            fail("task.toml", f"metadata.{key} is missing or empty")
    hours = meta.get("expert_time_estimate_hours", 0)
    if hours and not (2 <= float(hours) <= 12):
        warn("task.toml", f"expert_time_estimate_hours={hours} looks out of range")
    env = data.get("environment", {})
    ok("task.toml parses", f"gpus={env.get('gpus', 0)} cpus={env.get('cpus')}")
    return data


def library_files(root: pathlib.Path) -> list[pathlib.Path]:
    """Every python file under an environment directory."""
    return [p for p in root.rglob("*.py") if "__pycache__" not in p.parts]


def check_library_size(task: pathlib.Path) -> None:
    files = library_files(task / "environment")
    lines = sum(len(p.read_text(encoding="utf-8").splitlines()) for p in files)
    if not 5 <= len(files) <= 14:
        warn("library size", f"{len(files)} python files (aim for 6-12)")
    if not 300 <= lines <= 900:
        warn("library size", f"{lines} lines (aim for 400-700)")
    ok("library size", f"{len(files)} files, {lines} lines")


def collected_test_names(task: pathlib.Path) -> list[str]:
    """Module-level test functions, read from the AST so no import is needed."""
    source = (task / "tests" / "test_behavior.py").read_text(encoding="utf-8")
    if "parametrize" in source:
        warn("tests", "uses parametrize; config.json must list the [id] suffixes, "
                      "which this gate cannot derive statically")
    tree = ast.parse(source)
    return [
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test_")
    ]


def check_manifest(task: pathlib.Path) -> tuple[list[str], list[str]]:
    config = json.loads((task / "tests" / "config.json").read_text(encoding="utf-8"))
    f2p = config.get("fail_to_pass", [])
    p2p = config.get("pass_to_pass", [])
    listed = f2p + p2p
    collected = collected_test_names(task)

    if not f2p:
        fail("manifest", "fail_to_pass is empty")
    if not p2p:
        fail("manifest", "pass_to_pass is empty")
    dupes = sorted({n for n in listed if listed.count(n) > 1})
    if dupes:
        fail("manifest", f"duplicate entries: {dupes}")
    missing = sorted(set(listed) - set(collected))
    extra = sorted(set(collected) - set(listed))
    if missing:
        fail("manifest", f"listed but not defined: {missing}")
    if extra:
        fail("manifest", f"defined but not listed: {extra}")
    if not missing and not extra and not dupes:
        ok("manifest", f"{len(f2p)} fail_to_pass, {len(p2p)} pass_to_pass, exact match")
    return f2p, p2p


def check_leakage(task: pathlib.Path) -> None:
    # Scope: only the library, because that is the only thing the agent under
    # evaluation can read. tests/ and solution/ are mounted at verification time
    # and never shown, so scanning them is noise.
    hits = []
    for path in library_files(task / "environment"):
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            match = LEAK_WORDS.search(line)
            if match:
                hits.append(f"{path.relative_to(task)}:{i} {match.group(0)!r}")
    if hits:
        for hit in hits[:12]:
            warn("leakage", hit)
        if len(hits) > 12:
            warn("leakage", f"...and {len(hits) - 12} more")
        fail("leakage", "a word naming the mechanism ships inside the library")
    else:
        ok("leakage", "no mechanism words in the library the agent reads")


def check_instruction(task: pathlib.Path) -> None:
    text = (task / "instruction.md").read_text(encoding="utf-8")
    words = len(text.split())
    if not 120 <= words <= 600:
        warn("instruction", f"{words} words (aim for 200-400)")
    for i, line in enumerate(text.splitlines(), 1):
        match = BANNED_INSTRUCTION.search(line)
        if match:
            fail("instruction", f"line {i} names when the defect fires: {match.group(0)!r}")
    if not FAILURES or not any(f.startswith("instruction") for f in FAILURES):
        ok("instruction", f"{words} words, no trigger-naming phrasing")


def patch_body(task: pathlib.Path) -> str:
    body = (task / "solution" / "solve.sh").read_text(encoding="utf-8")
    if "python3 - <<'PY'" not in body:
        fail("solve.sh", "expected an inline `python3 - <<'PY'` block")
        return ""
    return body.split("python3 - <<'PY'", 1)[1].split("\nPY\n", 1)[0]


def check_solve(task: pathlib.Path) -> pathlib.Path | None:
    """Apply solve.sh to a copy; it must apply once and refuse twice."""
    inner = patch_body(task)
    if not inner:
        return None
    tmp = pathlib.Path(tempfile.mkdtemp())
    target = tmp / "env"
    shutil.copytree(task / "environment", target)
    script = tmp / "patch.py"
    script.write_text(inner, encoding="utf-8")

    first = subprocess.run([sys.executable, str(script)], cwd=target,
                           capture_output=True, text=True)
    if first.returncode != 0:
        fail("solve.sh", f"does not apply cleanly: {first.stdout}{first.stderr}")
        return None
    second = subprocess.run([sys.executable, str(script)], cwd=target,
                            capture_output=True, text=True)
    if second.returncode == 0:
        fail("solve.sh", "applies twice; it must assert on match counts")
    else:
        ok("solve.sh", "applies once, refuses to re-apply")
    return target


KEEP_TOKENS = {
    "tl", "torch", "to", "int64", "int32", "int16", "float32", "stride", "shape",
    "arange", "load", "store", "cast", "astype", "contiguous", "view", "reshape",
    "sum", "numel", "size", "dtype", "device", "program_id",
}


def skeleton(line: str) -> list[str]:
    """Token sequence with local names blanked, so shape survives renaming."""
    tokens = re.findall(r"\w+|\S", line)
    return [t if (t in KEEP_TOKENS or not t.isidentifier()) else "_" for t in tokens]


def similar_lines(needle: str, haystack: str, window: int = 7) -> bool:
    """Does haystack contain a sub-expression shaped like part of needle?"""
    a, b = skeleton(needle), skeleton(haystack)
    if len(a) < window or len(b) < window:
        return False
    grams = {tuple(a[i:i + window]) for i in range(len(a) - window + 1)}
    return any(tuple(b[i:i + window]) in grams for i in range(len(b) - window + 1))


def check_no_exemplar(task: pathlib.Path, fixed: pathlib.Path | None) -> None:
    """Does a corrected form of the fix already exist somewhere else?

    This is the mistake that made the first version of task 5 a one-line diff: a
    sibling file held the expression the broken one was missing, so it could be
    copied across without understanding anything.

    Verbatim duplication is a hard failure. Structural similarity is only
    reported, because the line between "a sibling shows the idiom" and "a sibling
    hands over the answer" is a judgement this cannot make: task 5 v2 keeps the
    same widening idiom two lines above the defect and a solver agent still did
    not find it, because knowing the idiom is not the same as knowing which term
    needs it. Read what is printed and decide.
    """
    if fixed is None:
        return
    added: list[tuple[str, str]] = []
    for new_path in library_files(fixed):
        rel = str(new_path.relative_to(fixed)).replace(os.sep, "/")
        old_path = task / "environment" / rel
        if not old_path.exists():
            continue
        old_lines = {l.strip() for l in old_path.read_text(encoding="utf-8").splitlines()}
        for line in new_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if len(stripped) > 15 and stripped not in old_lines:
                added.append((rel, stripped))

    if not added:
        fail("no-exemplar", "the patch added no new line; solve.sh may be a no-op")
        return

    verbatim, lookalikes = [], []
    for rel, line in added:
        for other in library_files(task / "environment"):
            other_rel = str(other.relative_to(task / "environment")).replace(os.sep, "/")
            if other_rel == rel:
                continue
            text = other.read_text(encoding="utf-8")
            if line in text:
                verbatim.append(f"{line!r} appears verbatim in {other_rel}")
                continue
            for i, candidate in enumerate(text.splitlines(), 1):
                if similar_lines(line, candidate):
                    lookalikes.append(f"{other_rel}:{i}  {candidate.strip()}")

    for item in verbatim:
        fail("no-exemplar", item)
    if verbatim:
        fail("no-exemplar",
             "a sibling holds the corrected expression, so the fix is a copy-paste "
             "away with no understanding required")
    if lookalikes:
        print(f"  ..    no-exemplar: {len(added)} patched line(s); "
              f"structurally similar lines elsewhere, REVIEW BY HAND:")
        for item in added:
            print(f"          patched -> {item[1]}")
        for item in sorted(set(lookalikes))[:10]:
            print(f"          sibling -> {item}")
        warn("no-exemplar",
             "decide whether those siblings show the idiom or hand over the answer")
    elif not verbatim:
        ok("no-exemplar", f"{len(added)} patched line(s), nothing like them elsewhere")


def check_isolation(task: pathlib.Path) -> None:
    """Nothing outside this task and the factory may be modified.

    Every idea gets its own directory. Editing a finished task in place is how a
    verified submission silently stops being verified.
    """
    out = subprocess.run(["git", "status", "--porcelain"],
                         cwd=task.parent, capture_output=True, text=True)
    if out.returncode != 0:
        warn("isolation", "not a git repo; cannot check")
        return
    allowed = (task.name + "/", "factory/")
    stray = []
    for line in out.stdout.splitlines():
        path = line[3:].strip().strip('"')
        if path and not path.startswith(allowed):
            stray.append(path)
    if stray:
        for path in sorted(set(stray))[:10]:
            fail("isolation", f"modified outside the task: {path}")
    else:
        ok("isolation", "no changes outside this task or factory/")


def check_adversarial(task: pathlib.Path) -> None:
    """A task nobody tried to break is not known to be hard."""
    record = task.parent / "factory" / "records" / f"{task.name}-adversarial.md"
    if not record.exists():
        fail("adversarial", f"no record at {record.relative_to(task.parent)}")
        fail("adversarial",
             "hand factory/make_solver_zip.py output to a solver in a SEPARATE "
             "session, grade its patch yourself, and write the result down")
        return
    text = record.read_text(encoding="utf-8")
    match = re.search(r"^reward:\s*(\S+)", text, re.M)
    if not match:
        fail("adversarial", "record has no 'reward:' line")
    elif match.group(1) == "REPLACE_ME":
        fail("adversarial", "record is still a stub")
    elif match.group(1) != "0":
        fail("adversarial",
             f"a solver scored {match.group(1)}; the task did not defeat it. "
             f"Read its reasoning, close the shortcut, rebuild.")
    else:
        ok("adversarial", "a solver attempt scored 0")


def check_archive(task: pathlib.Path) -> None:
    tmp = pathlib.Path(tempfile.mkdtemp()) / "task.zip"
    text_suffixes = {".sh", ".py", ".json", ".toml", ".md"}
    written = []
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as archive:
        for dirpath, dirnames, filenames in os.walk(task):
            dirnames[:] = [d for d in dirnames
                           if d not in {"jobs", "__pycache__", ".pytest_cache"}]
            for name in sorted(filenames):
                if name.endswith((".pyc", ".pyo")):
                    continue
                full = pathlib.Path(dirpath) / name
                arc = str(full.relative_to(task)).replace(os.sep, "/")
                data = full.read_bytes()
                if full.suffix in text_suffixes or name == "Dockerfile":
                    data = data.replace(b"\r\n", b"\n")
                info = zipfile.ZipInfo(arc, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = (0o755 if full.suffix == ".sh" else 0o644) << 16
                archive.writestr(info, data)
                written.append(arc)
    if any("\\" in n for n in written):
        fail("archive", "backslash entries")
    if any(n.startswith(task.name + "/") for n in written):
        fail("archive", "wrapper folder at the archive root")
    if "instruction.md" not in written:
        fail("archive", "instruction.md not at the root")
    ok("archive", f"{len(written)} root-level entries, LF text, .sh at 0o755")


# ---------------------------------------------------------------------------
# dynamic checks
# ---------------------------------------------------------------------------


def docker_build(task: pathlib.Path, tag: str) -> bool:
    out = subprocess.run(["docker", "build", "-t", tag, str(task / "environment")],
                         capture_output=True, text=True)
    if out.returncode != 0:
        fail("docker build", (out.stdout + out.stderr)[-2000:])
        return False
    ok("docker build", tag)
    return True


def run_suite(tag: str, library: pathlib.Path, tests: pathlib.Path,
              logs: pathlib.Path, gpus: bool) -> tuple[str, dict]:
    logs.mkdir(parents=True, exist_ok=True)
    cmd = ["docker", "run", "--rm", "--shm-size=1g"]
    if gpus:
        cmd += ["--gpus", "all"]
    cmd += ["-v", f"{library}:/workspace/{library.name}",
            "-v", f"{tests}:/tests:ro", "-v", f"{logs}:/logs",
            tag, "bash", "/tests/test.sh"]
    out = subprocess.run(cmd, capture_output=True, text=True)
    reward_file = logs / "verifier" / "reward.txt"
    report_file = logs / "verifier" / "report.json"
    if not reward_file.exists() or not report_file.exists():
        sys.stderr.write((out.stdout or "")[-2500:])
        fail("verifier", "no reward.txt / report.json written")
        return "", {}
    return (reward_file.read_text().strip(),
            json.loads(report_file.read_text())["tests"])


def stage(source: pathlib.Path, dest: pathlib.Path) -> pathlib.Path:
    shutil.copytree(source, dest)
    for path in dest.rglob("*.py"):
        path.write_bytes(path.read_bytes().replace(b"\r\n", b"\n"))
    return dest


def check_dynamic(task: pathlib.Path, config: dict, f2p: list[str],
                  p2p: list[str], gpus: bool) -> None:
    tag = f"gate-{task.name}:latest"
    if not docker_build(task, tag):
        return
    tmp = pathlib.Path(tempfile.mkdtemp())
    tests = stage(task / "tests", tmp / "tests")

    package = next(p for p in (task / "environment").iterdir()
                   if p.is_dir() and (p / "__init__.py").exists())

    broken = stage(package, tmp / "broken" / package.name)
    reward, report = run_suite(tag, broken, tests, tmp / "logs_nop", gpus)
    if report:
        errored = [n for n, s in report.items() if s == "ERROR"]
        bad_f2p = [n for n in f2p if report.get(n) != "FAILED"]
        bad_p2p = [n for n in p2p if report.get(n) != "PASSED"]
        if reward != "0":
            fail("nop", f"reward {reward}, expected 0")
        if errored:
            fail("nop", f"{len(errored)} test(s) ERRORed, first {errored[0]}")
        for name in bad_f2p:
            fail("nop", f"{name} -> {report.get(name)}, expected FAILED")
        for name in bad_p2p:
            fail("nop", f"{name} -> {report.get(name)}, expected PASSED")
        if reward == "0" and not errored and not bad_f2p and not bad_p2p:
            ok("nop", "reward 0, exactly the fail_to_pass set failing")

    fixed = stage(package, tmp / "fixed" / package.name)
    inner = patch_body(task)
    script = tmp / "patch.py"
    script.write_text(inner, encoding="utf-8")
    applied = subprocess.run([sys.executable, str(script)], cwd=fixed.parent,
                             capture_output=True, text=True)
    if applied.returncode != 0:
        fail("oracle", f"solve.sh failed against the staged copy: {applied.stderr}")
        return
    reward, report = run_suite(tag, fixed, tests, tmp / "logs_oracle", gpus)
    if report:
        bad = [n for n in f2p + p2p if report.get(n) != "PASSED"]
        if reward != "1":
            fail("oracle", f"reward {reward}, expected 1")
        for name in bad:
            fail("oracle", f"{name} -> {report.get(name)}, expected PASSED")
        if reward == "1" and not bad:
            ok("oracle", "reward 1, everything passing")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--static-only", action="store_true")
    args = parser.parse_args()

    task = pathlib.Path(args.task).resolve()
    if not task.is_dir():
        raise SystemExit(f"no such task directory: {task}")

    header(f"static checks  ({task})")
    check_layout(task)
    config = check_task_toml(task)
    check_library_size(task)
    f2p, p2p = check_manifest(task)
    check_instruction(task)
    check_leakage(task)
    fixed = check_solve(task)
    check_no_exemplar(task, fixed)
    check_isolation(task)
    check_archive(task)

    if not args.static_only:
        gpus = bool(config.get("environment", {}).get("gpus"))
        header(f"dynamic checks  (docker{', gpu' if gpus else ''})")
        check_dynamic(task, config, f2p, p2p, gpus)
        header("difficulty evidence")
        check_adversarial(task)

    header("verdict")
    print(f"  {len(FAILURES)} failure(s), {len(WARNINGS)} warning(s)")
    for item in FAILURES:
        print(f"  FAIL  {item}")
    if FAILURES:
        print("\n  NOT READY. Fix every failure, then run the gate again.")
        return 1
    if args.static_only:
        print("\n  Static checks pass. Run without --static-only before submitting.")
        return 0
    print("\n  READY. Every mechanical check passes; review the warnings by hand.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
