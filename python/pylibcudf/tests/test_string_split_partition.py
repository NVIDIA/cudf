# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pyarrow as pa
import pytest
from utils import assert_table_eq

import pylibcudf as plc


@pytest.fixture
def data_col():
    pa_arr = pa.array(["ab_cd", "def_g_h", None])
    plc_column = plc.Column.from_arrow(pa_arr)
    return pa_arr, plc_column


def test_partition(data_col):
    _pa_arr, plc_column = data_col
    got = plc.strings.split.partition.partition(
        plc_column, plc.Scalar.from_arrow(pa.scalar("_"))
    )
    expect = pa.table(
        {
            "a": ["ab", "def", None],
            "b": ["_", "_", None],
            "c": ["cd", "g_h", None],
        }
    )
    assert_table_eq(expect, got)


def test_rpartition(data_col):
    _pa_arr, plc_column = data_col
    got = plc.strings.split.partition.rpartition(
        plc_column, plc.Scalar.from_arrow(pa.scalar("_"))
    )
    expect = pa.table(
        {
            "a": ["ab", "def_g", None],
            "b": ["_", "_", None],
            "c": ["cd", "h", None],
        }
    )
    assert_table_eq(expect, got)
