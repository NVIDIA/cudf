# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

import polars as pl
import polars.testing

from cudf_polars.streaming.benchmarks.asserts import (
    ValidationError,
    assert_tpch_result_equal,
)
from cudf_polars.testing.asserts import (
    assert_gpu_result_equal,
    assert_ir_translation_raises,
    assert_sink_ir_translation_raises,
)


def test_translation_assert_raises(engine: pl.GPUEngine):
    df = pl.LazyFrame(
        {
            "time": pl.datetime_range(
                start=datetime(2021, 12, 16),
                end=datetime(2021, 12, 16, 3),
                interval="30m",
                eager=True,
            ),
            "n": range(7),
        }
    )

    # This should succeed
    assert_gpu_result_equal(df, engine=engine)

    with pytest.raises(AssertionError):
        # This should fail, because we can translate this query.
        assert_ir_translation_raises(df, engine, NotImplementedError)

    class E(Exception):
        pass

    unsupported = df.group_by_dynamic("time", every="1d").agg(pl.col("n").sum())
    # Unsupported query should raise NotImplementedError
    assert_ir_translation_raises(unsupported, engine, NotImplementedError)

    with pytest.raises(AssertionError):
        # This should fail, because we can't translate this query, but it doesn't raise E.
        assert_ir_translation_raises(unsupported, engine, E)


def test_sink_ir_translation_raises_bad_extension(engine: pl.GPUEngine):
    df = pl.LazyFrame({"a": [1, 2, 3]})
    # Should raise because ".foo" is not a recognized file extension
    with pytest.raises(ValueError, match=r"Unsupported file format: .foo"):
        assert_sink_ir_translation_raises(
            df, Path("out.foo"), engine, {}, NotImplementedError
        )


def test_sink_ir_translation_raises_sink_error_before_translation(
    tmp_path: Path, engine: pl.GPUEngine
):
    df = pl.LazyFrame({"a": [1, 2, 3]})
    # Should raise because "foo" is not a valid write kwarg,
    # so the sink_* function fails before IR translation
    with pytest.raises(
        AssertionError,
        match=r"Sink function raised an exception before translation: .*foo",
    ):
        assert_sink_ir_translation_raises(
            df, tmp_path / "out.csv", engine, {"foo": True}, NotImplementedError
        )


def test_assert_tpch_result_equal_ties() -> None:
    epsilon = 1e-5

    left = pl.DataFrame(
        {"a": [1.0, 2.0, 3.0, 3.0, 3.0 + epsilon], "b": ["a", "b", "c", "d", "e"]}
    )
    right = pl.DataFrame(
        {"a": [1.0, 2.0, 3.0 - epsilon, 3.0, 3.0], "b": ["a", "b", "e", "c", "d"]}
    )

    assert_tpch_result_equal(
        left,
        right,
        sort_by=[("a", False)],
        abs_tol=2 * epsilon,
        check_exact=False,
        limit=5,
    )


@pytest.mark.parametrize("descending", [False, True])
def test_assert_tpch_result_equal_ties_non_numeric(*, descending: bool) -> None:
    epsilon = 1e-5
    b = ["a", "b", "c", "c", "c"]

    if descending:
        b = list(reversed(b))

    left = pl.DataFrame({"a": [1.0, 2.0, 3.0, 3.0, 3.0 + epsilon], "b": b})
    right = pl.DataFrame({"a": [1.0, 2.0, 3.0 - epsilon, 3.0, 3.0], "b": b})

    assert_tpch_result_equal(
        left,
        right,
        sort_by=[("b", descending)],
        abs_tol=2 * epsilon,
        check_exact=False,
        limit=5,
    )


def test_assert_tpch_result_equal_ties_non_numeric_non_string() -> None:
    epsilon = 1e-5

    left = pl.DataFrame(
        {"a": [1.0, 2.0, 3.0, 3.0, 3.0 + epsilon], "b": [1, 2, 3, 3, 3]}
    )
    right = pl.DataFrame(
        {"a": [1.0, 2.0, 3.0 - epsilon, 3.0, 3.0], "b": [1, 2, 3, 3, 3]}
    )

    assert_tpch_result_equal(
        left,
        right,
        sort_by=[("b", False)],
        abs_tol=2 * epsilon,
        check_exact=False,
        limit=5,
    )


def test_assert_tpch_result_equal_ties_multi_column_sort_by() -> None:
    epsilon = 1e-5

    left = pl.DataFrame(
        {"a": [1.0, 2.0, 3.0, 3.0, 3.0 + epsilon], "b": ["a", "b", "c", "c", "c"]}
    )
    right = pl.DataFrame(
        {"a": [1.0, 2.0, 3.0 - epsilon, 3.0, 3.0], "b": ["a", "b", "c", "c", "c"]}
    )

    assert_tpch_result_equal(
        left,
        right,
        sort_by=[("b", False), ("a", False)],
        abs_tol=2 * epsilon,
        check_exact=False,
        limit=5,
    )


def test_assert_tpch_result_equal_ties_payload_does_not_drive_order() -> None:
    # Within the ties partition, payload column values must not determine
    # the row order used to compare the sort_by columns. Both sides have
    # the same set of v values in the tolerance band around the split
    # point, but the payload column k is aligned with v in opposite orders.
    # Sorting the full ties frame by k before projecting to v would put the
    # v values in opposite orders on the two sides and fail the
    # approximate comparison even though the result is correct.
    left = pl.DataFrame(
        {
            "v": [1.0, 2.0, 3.099, 3.100, 3.101],
            "k": ["a", "b", "x", "y", "z"],
        }
    )
    right = pl.DataFrame(
        {
            "v": [1.0, 2.0, 3.099, 3.100, 3.101],
            "k": ["a", "b", "z", "y", "x"],
        }
    )
    assert_tpch_result_equal(
        left,
        right,
        sort_by=[("v", False)],
        abs_tol=1e-3,
        check_exact=False,
        limit=5,
    )


@pytest.mark.parametrize("limit", [None, 5])
def test_assert_tpch_result_equal_float_sort_raises(limit: int | None) -> None:
    # Sort on a floating point column with a limit,
    # but the 'key' column *doesn't* match
    epsilon = 1e-5
    left = pl.DataFrame(
        {"a": [1.0, 2.0, 3.0, 3.0, 3.0 + epsilon], "b": ["a", "b", "c", "c", "c"]}
    )
    right = pl.DataFrame(
        {"a": [1.0, 2.0, 3.0 - epsilon, 3.0, 3.0], "b": ["x", "b", "c", "c", "c"]}
    )
    with pytest.raises(ValidationError, match="Result mismatch"):
        assert_tpch_result_equal(
            left,
            right,
            sort_by=[("a", False)],
            limit=limit,
        )


def test_assert_tpch_result_equal_split_at_lexicographic_not_per_column_max() -> None:
    # Previously, we used .max() to find the split point. That was incorrect
    # when the largest value per column came from different rows.

    left = pl.DataFrame({"a": [1, 2], "b": [10, 5], "c": ["x", "y"]})
    right = pl.DataFrame({"a": [1, 2], "b": [10, 5], "c": ["x", "z"]})
    assert_tpch_result_equal(
        left,
        right,
        sort_by=[("a", False), ("b", False)],
        limit=2,
    )


def test_assert_tpch_result_equal_split_at_ascending_so_lt_is_valid() -> None:
    left = pl.DataFrame({"a": [3, 2, 1], "c": ["a", "b", "x"]})
    right = pl.DataFrame({"a": [3, 2, 1], "c": ["a", "c", "x"]})
    with pytest.raises(ValidationError, match="Result mismatch in non-ties part"):
        assert_tpch_result_equal(
            left,
            right,
            sort_by=[("a", True)],  # True = descending
            limit=3,
        )


def test_assert_tpch_result_equal_split_at_uses_query_order_mixed_asc_desc() -> None:
    left = pl.DataFrame({"a": [3, 3, 2], "b": [1, 2, 3], "c": ["a", "b", "c"]})
    right = pl.DataFrame({"a": [3, 3, 2], "b": [1, 2, 3], "c": ["a", "X", "c"]})
    with pytest.raises(ValidationError, match="Result mismatch in non-ties part"):
        assert_tpch_result_equal(
            left,
            right,
            sort_by=[("a", True), ("b", False)],  # a DESC, b ASC
            limit=3,
        )


def test_assert_tpch_result_equal_raises_column_names_mismatch() -> None:
    left = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
    right = pl.DataFrame({"a": [1, 2], "c": [3, 4]})

    with pytest.raises(ValidationError, match="Column names mismatch"):
        assert_tpch_result_equal(left, right, sort_by=[])


def test_assert_tpch_result_equal_raises_schema_mismatch() -> None:
    left = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
    right = pl.DataFrame({"a": [1.0, 2.0], "b": [3, 4]})

    with pytest.raises(ValidationError, match="Schema mismatch"):
        assert_tpch_result_equal(left, right, sort_by=[])


def test_assert_tpch_result_equal_raises_sort_by_columns_mismatch() -> None:
    left = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    right = pl.DataFrame({"a": [1, 2, 99], "b": ["x", "y", "z"]})

    with pytest.raises(ValidationError, match="Result mismatch in ties part"):
        assert_tpch_result_equal(left, right, sort_by=[("a", False)], limit=3)


def test_assert_tpch_result_equal_raises_result_mismatch() -> None:
    left = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    right = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "wrong"]})

    with pytest.raises(ValidationError, match="Result mismatch"):
        assert_tpch_result_equal(left, right, sort_by=[])


def test_assert_tpch_result_equal_raises_result_mismatch_non_ties() -> None:
    # Same sort_by values, but a differing value in the "before" part.
    left = pl.DataFrame({"a": [1.0, 2.0, 3.0, 3.0], "b": ["a", "b", "c", "d"]})
    right = pl.DataFrame({"a": [1.0, 2.0, 3.0, 3.0], "b": ["a", "wrong", "c", "d"]})

    with pytest.raises(ValidationError, match="Result mismatch in non-ties part"):
        assert_tpch_result_equal(left, right, sort_by=[("a", False)], limit=4)


def test_assert_tpch_result_equal_q11_ties():
    # Previously, we claimed q11 was failing validation because of these values.
    # In reality, these *are* considered equal.

    left = pl.DataFrame(
        {
            "ps_partkey": [124439984, 69940887, 118230270, 191696233, 158970051, 0, 0],
            "value": [
                8005096.8,
                8005095.75,
                8005095.75,
                8005093.11,
                8005090.24,
                1.0,
                0.0,
            ],
        }
    )
    right = pl.DataFrame(
        {
            "ps_partkey": [124439984, 118230270, 69940887, 191696233, 158970051, 0, 0],
            "value": [
                8005096.8,
                8005095.75,
                8005095.749999999,
                8005093.109999999,
                8005090.24,
                1.0,
                0.0,
            ],
        }
    )
    sort_by = [("value", True)]  # descending=True

    assert_tpch_result_equal(
        left, right, sort_by=sort_by, check_exact=False, abs_tol=1e-2
    )


def test_assert_tpch_result_equal_float_sort_mixed_sort_columns() -> None:
    left = pl.DataFrame(
        {
            "ps_partkey": [1, 2, 3],
            "value": [8005096.8, 8005095.75, 8005095.75],
        }
    )
    right = pl.DataFrame(
        {
            "ps_partkey": [1, 2, 3],
            "value": [8005096.8, 8005095.75, 8005095.749999999],
        }
    )
    assert_tpch_result_equal(
        left,
        right,
        sort_by=[("value", True), ("ps_partkey", False)],
        check_exact=False,
        abs_tol=1e-2,
    )


def test_assert_tpch_result_equal_float_sort_raises_key_mismatch() -> None:
    left = pl.DataFrame(
        {
            "ps_partkey": [124439984, 69940887, 118230270],
            "value": [8005096.8, 8005095.75, 8005095.75],
        }
    )
    right = pl.DataFrame(
        {
            "ps_partkey": [
                124439984,
                69940887,
                999999999,
            ],  # value matches, but not key
            "value": [8005096.8, 8005095.75, 8005095.749999999],
        }
    )
    with pytest.raises(ValidationError, match="Result mismatch"):
        assert_tpch_result_equal(
            left,
            right,
            sort_by=[("value", True)],
            check_exact=False,
            abs_tol=1e-2,
        )


@pytest.mark.parametrize(
    "sort_by",
    [
        [("value", False)],
        [("value", False), ("key", False)],
    ],
)
def test_assert_tpch_result_equal_float_sort(sort_by: list[tuple[str, bool]]):
    left = pl.DataFrame(
        {
            "key": ["a", "b", "c", "d", "e"],
            "value": [1.0, 1.1, 1.2, 1.21, 1.22],
        }
    )
    # Three immaterial changes:
    # 1. swap the "d" and "e" rows
    # 2. change the value of "c" from 1.20 -> 1.12
    # 3. change the value of "d" from 1.21 -> 1.20
    right = pl.DataFrame(
        {
            "key": ["a", "b", "d", "c", "e"],
            "value": [1.0, 1.1, 1.20, 1.21, 1.22],
        }
    )
    assert_tpch_result_equal(
        left, right, sort_by=sort_by, abs_tol=0.011, check_exact=False
    )


@pytest.mark.parametrize("epsilon", [0.001, 1.0])
def test_assert_tpch_result_float_not_actually_sorted(epsilon: float) -> None:
    # this test verifies that we correctly raise if the
    # result isn't sorted when `sort_by` claims it ought to be.
    left = pl.DataFrame(
        {
            "key": ["a", "b", "c", "d"],
            "value": [1.0, 1.1, 1.20 + epsilon, 1.20],
        }
    )
    right = pl.DataFrame(
        {
            "key": ["a", "b", "c", "d"],
            "value": [1.0, 1.1, 1.20, 1.20],
        }
    )
    with pytest.raises(
        ValidationError, match="left dataframe is not sorted by sort_by columns"
    ):
        assert_tpch_result_equal(
            left, right, sort_by=[("value", False)], abs_tol=0.01, check_exact=False
        )


def test_assert_tpch_result_equal_sort_keys_raises_not_sorted() -> None:
    left = pl.DataFrame(
        {
            "lochierarchy": [0, 0, 0, 1, 1],
            "i_category": ["music", "electronics", "books", "sports", "toys"],
            "total_sales": [150, 200, 100, 300, 250],
        }
    )
    # It does not matter what right is as long as it has the
    # same schema because we should fail before we compare them.
    right = left.clone()
    sort_keys = [
        (
            pl.when(pl.col("lochierarchy") == 0)
            .then(pl.col("i_category"))
            .otherwise(pl.lit("foo")),
            False,
        )
    ]
    with pytest.raises(
        ValidationError, match="left dataframe is not sorted by sort_keys expressions"
    ):
        assert_tpch_result_equal(
            left,
            right,
            sort_by=[("lochierarchy", True), ("i_category", False)],
            sort_keys=sort_keys,
            nulls_last=True,
        )


@pytest.mark.parametrize("sort_by", [[("a", True)], []])
@pytest.mark.parametrize("drop_columns", [[], ["b"], ["a", "b"]])
def test_assert_tpch_result_equal_grouped_float_sort(
    sort_by: list[tuple[str, bool]], drop_columns: list[str]
) -> None:
    # https://github.com/rapidsai/cudf/issues/22129
    # Same non-float values with float values reordered inside each non-float group.
    left = pl.DataFrame({"a": [1, 1, 1], "b": [2, 2, 2], "c": [1.0, 2.0, 3.0]})
    right = pl.DataFrame({"a": [1, 1, 1], "b": [2, 2, 2], "c": [1.0, 2.999, 2.0]})

    if drop_columns:
        left = left.drop(drop_columns)
        right = right.drop(drop_columns)
        if "a" in drop_columns:
            sort_by = []

    assert_tpch_result_equal(
        left, right, sort_by=sort_by, abs_tol=0.01, check_exact=False
    )

    # But this table is different, since row 3.0 - 2.9 > abs_tol.
    right_different = pl.DataFrame(
        {"a": [1, 1, 1], "b": [2, 2, 2], "c": [1.0, 2.90, 2.0]}
    )
    if drop_columns:
        right_different = right_different.drop(drop_columns)
    with pytest.raises(ValidationError, match="Result mismatch"):
        assert_tpch_result_equal(
            left, right_different, sort_by=sort_by, abs_tol=0.01, check_exact=False
        )
