#!/usr/bin/env bash
# Harness entrypoint. Runs the suite, parses it, checks every name listed in
# config.json passed, and writes 1/0 as the reward.
# Deliberately does NOT use `set -e`: a failing suite is a normal outcome here.
set -uo pipefail

mkdir -p /logs/verifier/rewards

RAW=/logs/verifier/raw_output.txt
REPORT=/logs/verifier/report.json

bash /tests/run_script.sh | tee "$RAW"
python3 /tests/parser.py < "$RAW" > "$REPORT"

python3 - "$REPORT" /tests/config.json <<'PY'
import json
import pathlib
import sys

report = json.loads(pathlib.Path(sys.argv[1]).read_text())["tests"]
config = json.loads(pathlib.Path(sys.argv[2]).read_text())

required = list(config.get("fail_to_pass", [])) + list(config.get("pass_to_pass", []))
if not required:
    print("VERIFIER: config.json lists no tests", file=sys.stderr)
    sys.exit(1)

missing = [name for name in required if name not in report]
failed = [name for name in required if report.get(name) != "PASSED"]

for name in missing:
    print(f"VERIFIER: test never ran: {name}", file=sys.stderr)
for name in failed:
    if name not in missing:
        print(f"VERIFIER: {name} -> {report[name]}", file=sys.stderr)

sys.exit(1 if (missing or failed) else 0)
PY
STATUS=$?

if [ "$STATUS" -eq 0 ]; then REWARD=1; else REWARD=0; fi

# Written to both known locations: the downloadable template uses
# rewards/reward.txt, the written instructions say reward.txt.
echo "$REWARD" > /logs/verifier/rewards/reward.txt
echo "$REWARD" > /logs/verifier/reward.txt

echo "VERIFIER: reward=$REWARD"
exit 0
