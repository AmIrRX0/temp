#!/usr/bin/env bash
set -euo pipefail

cd /workspace

python3 - <<'PY'
import pathlib
import sys

# The lane index stays 32-bit, so the offset within a row wraps once the column
# pitch is large enough for lane * col_pitch to cross 2**31.
gather_path = pathlib.Path("gatherlib/kernels/gather.py")
source = gather_path.read_text()
old = "    src_off = src_row.to(tl.int64) * row_pitch + lane * col_pitch\n"
new = "    src_off = src_row.to(tl.int64) * row_pitch + lane.to(tl.int64) * col_pitch\n"
if source.count(old) != 1:
    sys.exit("gather.py: expected exactly one match")
gather_path.write_text(source.replace(old, new))

# A plan is derived from the strides, so contiguity is not a fine enough key:
# two unpacked views of the same shape can have different pitches.
cache_path = pathlib.Path("gatherlib/cache.py")
source = cache_path.read_text()
old = "        table.is_contiguous(),\n"
new = "        tuple(int(s) for s in table.stride()),\n"
if source.count(old) != 1:
    sys.exit("cache.py: expected exactly one match")
cache_path.write_text(source.replace(old, new))
PY

echo "solution applied"
