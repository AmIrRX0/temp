#!/usr/bin/env bash
set -euo pipefail

cd /workspace

python3 - <<'PY'
import pathlib
import sys

# The widening happens one step too late: lane and col_pitch are both 32-bit, so
# their product has already wrapped by the time it is cast, once the column pitch
# is large enough for lane * col_pitch to cross 2**31.
gather_path = pathlib.Path("gatherlib/kernels/gather.py")
source = gather_path.read_text()
old = "    in_row = (lane * col_pitch).to(tl.int64)\n"
new = "    in_row = lane.to(tl.int64) * col_pitch\n"
if source.count(old) != 1:
    sys.exit("gather.py: expected exactly one match")
gather_path.write_text(source.replace(old, new))

# A plan is derived from both strides, and the key carries only the trailing one:
# two unpacked views of the same shape can agree on the trailing pitch and differ
# on the leading one.
cache_path = pathlib.Path("gatherlib/cache.py")
source = cache_path.read_text()
old = "        int(table.stride(-1)),\n"
new = "        tuple(int(s) for s in table.stride()),\n"
if source.count(old) != 1:
    sys.exit("cache.py: expected exactly one match")
cache_path.write_text(source.replace(old, new))
PY

echo "solution applied"
