# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import pyarrow as pa
import pytest

import pylibcudf as plc


@pytest.mark.parametrize(
    "agg",
    [plc.aggregation.any, plc.aggregation.all],
)
def test_groupby_unsupported_bool_aggs_raise(agg):
    pa_table = pa.table(
        {
            "foo": pa.array(["a", "a", "b", "b", "c"]),
            "bar": pa.array([True, False, True, True, False]),
        }
    )
    plc_tbl = plc.Table.from_arrow(pa_table)

    groupby = plc.groupby.GroupBy(plc_tbl)
    request = plc.groupby.GroupByRequest(plc_tbl.columns()[0], [agg()])

    with pytest.raises(
        NotImplementedError,
        match=r"Aggregation(.*) aggregations are not supported by groupby",
    ):
        groupby.aggregate([request])


def test_groupby_min_nan_semantics():
    nan = float("nan")
    keys = plc.Table.from_arrow(pa.table({"k": pa.array([0, 0, 0, 1, 1])}))
    values = plc.Column.from_arrow(pa.array([2.0, nan, 1.0, nan, nan]))
    groupby = plc.groupby.GroupBy(keys, keys_are_sorted=plc.types.Sorted.YES)
    # Aggregations that differ only in NaN semantics have separate results
    request = plc.groupby.GroupByRequest(
        values,
        [
            plc.aggregation.min(),
            plc.aggregation.min(plc.types.NanSemantics.TOTALLY_ORDERED),
        ],
    )
    _, (results,) = groupby.aggregate([request])
    unordered, ordered = (c.to_arrow().to_pylist() for c in results.columns())
    assert all(map(math.isnan, unordered))
    assert ordered[0] == 1.0
    assert math.isnan(ordered[1])
