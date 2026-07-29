#!/usr/bin/env bash
set -euo pipefail

cd /workspace

python3 - <<'PY'
import pathlib
import sys

gather_path = pathlib.Path("gatherlib/kernels/gather.py")
source = gather_path.read_text()
old = "    row_base = src_row * row_pitch\n"
new = "    row_base = src_row.to(tl.int64) * row_pitch\n"
if source.count(old) != 1:
    sys.exit("gather.py: expected exactly one match")
gather_path.write_text(source.replace(old, new))

addressing_path = pathlib.Path("gatherlib/addressing.py")
source = addressing_path.read_text()
old = (
    "    if int(table.stride(-1)) == 1:\n"
    "        return int(table.shape[-1]), 1\n"
    "    return element_strides(table)\n"
)
new = (
    "    if int(table.stride(-1)) == 1:\n"
    "        return int(table.stride(-2)), 1\n"
    "    return element_strides(table)\n"
)
if source.count(old) != 1:
    sys.exit("addressing.py: expected exactly one match")
addressing_path.write_text(source.replace(old, new))
PY

echo "solution applied"
