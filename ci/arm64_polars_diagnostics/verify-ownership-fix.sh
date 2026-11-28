#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../python/cudf_polars"
ownership=tests/streaming/test_chunk_ownership.py
original="$RAPIDS_TESTS_DIR/junit-ownership-before.xml"
status=0
timeout --kill-after=15s 4m python -m pytest -x -p no:benchmark \
    --junitxml="$original" "$ownership" || status=$?
test "$status" = 1
python - "$original" <<'PY'
import sys
import xml.etree.ElementTree as ET

failures = ET.parse(sys.argv[1]).findall(".//failure")
assert len(failures) == 1, "Expected one failure on the unpatched owner boundary"
assert "assert copied.data().ptr != original_ptr" in (failures[0].text or "")
PY

python - "$CUDF_POLARS_OWNERSHIP_UTILS" <<'PY'
from pathlib import Path
import sys

path = Path("cudf_polars/streaming/actor_graph/utils.py")
archived = path.read_text()
fixed = Path(sys.argv[1]).read_text()
start = "def _column_buffer_ranges("
end = "def _evaluate_chunk_sync("
helpers = fixed[fixed.index(start):fixed.index(end)]
before = '''    df = evaluate_ir(ir, df_in, context=ir_context)
    return TableChunk.from_pylibcudf_table(
        df.table, df.stream, exclusive_view=True, br=br
    )'''
after = '''    df = evaluate_ir(ir, df_in, context=ir_context)
    output = _copy_shared_output_columns(df, df_in)
    return TableChunk.from_pylibcudf_table(
        output, df.stream, exclusive_view=True, br=br
    )'''
assert before in archived and after in fixed
for marker, text in (
    ("from collections import deque", "from bisect import bisect_left\n"),
    ("from cudf_polars.utils.dtypes import make_empty_column",
     "from cudf_polars.utils.cuda_stream import stream_ordered_after\n"),
    (end, helpers),
):
    assert archived.count(marker) == 1
    archived = archived.replace(marker, text + marker)
path.write_text(archived.replace(before, after))
PY

exec timeout --kill-after=15s 6m python -m pytest -x -p no:benchmark \
    --numprocesses=4 --dist=worksteal \
    --junitxml="$RAPIDS_TESTS_DIR/junit-ownership-after.xml" \
    "$ownership" tests/streaming/test_spilling.py tests/expressions/test_datetime_basic.py \
    -k 'test_evaluated_chunk_owns_shared_output_columns or test_make_spill_function or ((test_datetime_truncate or test_datetime_offset_by) and (spmd or dask))'
