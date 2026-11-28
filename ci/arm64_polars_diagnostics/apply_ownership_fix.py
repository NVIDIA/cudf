# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Apply the current ownership fix to the pinned Python-only CI replay."""

import sys
from pathlib import Path

path = Path("cudf_polars/streaming/actor_graph/utils.py")
archived = path.read_text()
fixed = Path(sys.argv[1]).read_text()
start = "def _column_buffer_ranges("
end = "def _evaluate_chunk_sync("
helpers = fixed[fixed.index(start) : fixed.index(end)]
before = """    df = evaluate_ir(ir, df_in, context=ir_context)
    return TableChunk.from_pylibcudf_table(
        df.table, df.stream, exclusive_view=True, br=br
    )"""
after = """    df = evaluate_ir(ir, df_in, context=ir_context)
    output = _copy_shared_output_columns(df, df_in)
    return TableChunk.from_pylibcudf_table(
        output, df.stream, exclusive_view=True, br=br
    )"""
assert before in archived and after in fixed
for marker, text in (
    ("from collections import deque", "from bisect import bisect_left\n"),
    (
        "from cudf_polars.utils.dtypes import make_empty_column",
        "from cudf_polars.utils.cuda_stream import stream_ordered_after\n",
    ),
    (end, helpers),
):
    assert archived.count(marker) == 1
    archived = archived.replace(marker, text + marker)
path.write_text(archived.replace(before, after))
