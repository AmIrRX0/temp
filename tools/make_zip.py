#!/usr/bin/env python3
"""Build the submission archive.

Entries are written at the archive root with forward slashes, because
PowerShell's Compress-Archive writes backslashes and the uploader then reports
every file as missing.

    python tools/make_zip.py [-o gatherlib_task.zip]
"""

from __future__ import annotations

import argparse
import os
import pathlib
import zipfile

REPO = pathlib.Path(__file__).resolve().parent.parent
SOURCE = REPO / "task5"

SKIP_DIRS = {"jobs", "__pycache__", ".pytest_cache", ".git"}
SKIP_FILES = {".DS_Store"}
SKIP_SUFFIXES = {".pyc", ".pyo"}

# Everything in this task is text that runs on Linux. A Windows checkout can
# leave CRLF in the working tree, and bash inside the container then dies with
# "$'\r': command not found" on every line of test.sh. Normalise on the way
# into the archive so the submitted artifact never depends on how the tree was
# checked out.
TEXT_SUFFIXES = {".sh", ".py", ".json", ".toml", ".md", ".cfg", ".txt"}
TEXT_NAMES = {"Dockerfile", ".dockerignore", ".gitattributes"}
EXEC_SUFFIXES = {".sh"}

REQUIRED = [
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", default=str(REPO / "gatherlib_task.zip"))
    args = parser.parse_args()

    dest = pathlib.Path(args.output)
    written: list[str] = []
    normalised: list[str] = []
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as archive:
        for dirpath, dirnames, filenames in os.walk(SOURCE):
            dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
            for name in sorted(filenames):
                if name in SKIP_FILES or pathlib.Path(name).suffix in SKIP_SUFFIXES:
                    continue
                full = pathlib.Path(dirpath) / name
                arcname = str(full.relative_to(SOURCE)).replace(os.sep, "/")
                suffix = full.suffix

                data = full.read_bytes()
                if suffix in TEXT_SUFFIXES or name in TEXT_NAMES:
                    fixed = data.replace(b"\r\n", b"\n")
                    if fixed != data:
                        normalised.append(arcname)
                    data = fixed

                # Fixed timestamp and explicit mode: the archive should not
                # depend on the host's filesystem metadata.
                info = zipfile.ZipInfo(arcname, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                mode = 0o755 if suffix in EXEC_SUFFIXES else 0o644
                info.external_attr = mode << 16
                archive.writestr(info, data)
                written.append(arcname)

    backslashed = [name for name in written if "\\" in name]
    if backslashed:
        raise SystemExit(f"archive contains backslash entries: {backslashed}")
    missing = [name for name in REQUIRED if name not in written]
    if missing:
        raise SystemExit(f"archive is missing required files: {missing}")
    wrapped = [name for name in written if name.startswith("task5/")]
    if wrapped:
        raise SystemExit(f"archive has a wrapper folder: {wrapped[:3]}")

    with zipfile.ZipFile(dest) as archive:
        crlf = [n for n in archive.namelist()
                if pathlib.Path(n).suffix in TEXT_SUFFIXES
                and b"\r\n" in archive.read(n)]
    if crlf:
        raise SystemExit(f"archive still contains CRLF text files: {crlf}")

    print(f"wrote {dest} ({len(written)} entries, {dest.stat().st_size} bytes)")
    if normalised:
        print(f"normalised CRLF -> LF in {len(normalised)} file(s):")
        for name in normalised:
            print(f"  ! {name}")
        print("  (your working tree has CRLF; the archive does not)")
    for name in written:
        print(f"  {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
