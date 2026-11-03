# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import pyarrow as pa
import pytest
from utils import assert_table_eq

import pylibcudf as plc


@pytest.mark.parametrize(
    "arr",
    [
        [],
        [1, 2, 3],
        [1, 2],
        [1],
    ],
)
def test_transpose(arr):
    data = {"a": arr, "b": arr}
    arrow_tbl = pa.table(data)
    plc_tbl = plc.Table.from_arrow(arrow_tbl)
    got = plc.transpose.transpose(plc_tbl)
    expect = pa.table(
        pa.Table.from_pandas(
            arrow_tbl.to_pandas().T, preserve_index=False
        ).rename_columns([""] * len(arr)),
        schema=got.to_arrow().schema,
    )
    assert_table_eq(expect, got)
