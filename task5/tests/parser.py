#!/usr/bin/env python3
"""Parse pytest -v output on stdin into a JSON status report on stdout.

Output shape:
    {"tests": {"test_name": "PASSED" | "FAILED" | "ERROR" | "SKIPPED", ...}}

Parametrised tests keep their bracket suffix, so config.json must list them
exactly as pytest prints them (e.g. "test_strides[shape0]").
"""

import json
import re
import sys

LINE = re.compile(
    r"^(?P<file>\S*?::)?(?P<name>test_\w+(?:\[[^\]]*\])?)\s+"
    r"(?P<status>PASSED|FAILED|ERROR|SKIPPED|XFAIL|XPASS)\b"
)

# Collection errors never reach a PASSED/FAILED line; catch them separately so
# a broken import shows up as ERROR instead of silently vanishing.
COLLECT_ERR = re.compile(r"^ERROR\s+\S*?::(?P<name>test_\w+(?:\[[^\]]*\])?)")


def main() -> None:
    results: dict[str, str] = {}
    for raw in sys.stdin:
        line = raw.rstrip("\n")
        match = LINE.match(line.strip())
        if match:
            results[match.group("name")] = match.group("status")
            continue
        match = COLLECT_ERR.match(line.strip())
        if match:
            results.setdefault(match.group("name"), "ERROR")
    json.dump({"tests": results}, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
