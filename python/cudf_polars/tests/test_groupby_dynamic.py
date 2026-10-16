# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import datetime as dt
import random
from typing import TYPE_CHECKING

import pytest

import polars as pl

from cudf_polars.testing.asserts import (
    assert_gpu_result_equal,
    assert_ir_translation_raises,
)
from cudf_polars.testing.engine_utils import (
    get_blocksize_mode,
    is_streaming_engine,
    warns_on_spmd,
)
from cudf_polars.utils.versions import POLARS_VERSION_LT_142

if TYPE_CHECKING:
    from polars._typing import ClosedInterval, Label, TimeUnit


def _timestamps(n: int, *, seed: int, span: dt.timedelta) -> list[dt.datetime]:
    rng = random.Random(seed)
    start = dt.datetime(2023, 12, 27, 13, 7, 11, 123456)
    values = [
        start
        + dt.timedelta(microseconds=rng.randrange(span // dt.timedelta(microseconds=1)))
        for _ in range(n)
    ]
    # Values exactly on window boundaries: midnights, a Monday, and the
    # first of a month and a year.
    values[::7] = [
        v.replace(hour=0, minute=0, second=0, microsecond=0) for v in values[::7]
    ]
    values += [
        dt.datetime(2024, 1, 1),
        dt.datetime(2024, 3, 4),
        dt.datetime(2025, 2, 1),
    ]
    return sorted(values)


@pytest.fixture(scope="module")
def frame() -> pl.LazyFrame:
    """Datetime-indexed rows sorted by time, with interleaved and null groups."""
    rng = random.Random(42)
    t = _timestamps(400, seed=0, span=dt.timedelta(days=800))
    n = len(t)
    return pl.LazyFrame(
        {
            "t": t,
            "g": [rng.choice(["b", "a", None, "c"]) for _ in range(n)],
            "h": [rng.choice([2, 1, None]) for _ in range(n)],
            "v": [rng.randrange(-50, 50) for _ in range(n)],
            "x": [rng.random() for _ in range(n)],
        }
    )


@pytest.fixture(scope="module")
def short_frame(frame: pl.LazyFrame) -> pl.LazyFrame:
    """
    A prefix of ``frame`` for tests that run on every engine.

    Streaming sorts of many tiny partitions are slow, so these tests keep the
    number of windows small. The option matrices use ``frame`` in memory.
    """
    return frame.head(80)


def _aggs() -> list[pl.Expr]:
    # Aggregations that streaming group-bys decompose across partitions.
    return [
        pl.len().alias("n"),
        pl.col("v").sum().alias("v_sum"),
        pl.col("v").min().alias("v_min"),
        pl.col("x").max().alias("x_max"),
    ]


def _first_last_fallback(engine: pl.GPUEngine):
    """Streaming group-bys run first/last aggregations on one partition."""
    return warns_on_spmd(
        engine,
        UserWarning,
        match="Failed to decompose groupby aggs for multiple partitions",
        when=get_blocksize_mode(engine) == "medium",
    )


@pytest.mark.parametrize(
    "every", ["7m", "10m", "1h", "90m", "36h", "1d", "3d", "1w", "2w"]
)
@pytest.mark.parametrize("offset", ["0ns", "-1us", "5m", "-1h", "2d"])
@pytest.mark.parametrize("closed", ["left", "right"])
def test_fixed_every(
    in_memory_engine: pl.GPUEngine,
    frame: pl.LazyFrame,
    every: str,
    offset: str,
    closed: ClosedInterval,
) -> None:
    q = frame.group_by_dynamic("t", every=every, offset=offset, closed=closed).agg(
        *_aggs()
    )
    assert_gpu_result_equal(q, engine=in_memory_engine)


@pytest.mark.parametrize("every", ["1mo", "3mo", "1q", "1y"])
@pytest.mark.parametrize("closed", ["left", "right"])
def test_calendar_every(
    in_memory_engine: pl.GPUEngine,
    frame: pl.LazyFrame,
    every: str,
    closed: ClosedInterval,
) -> None:
    q = frame.group_by_dynamic(
        "t", every=every, closed=closed, include_boundaries=True
    ).agg(*_aggs())
    assert_gpu_result_equal(q, engine=in_memory_engine)


@pytest.mark.parametrize("time_unit", ["ms", "us", "ns"])
@pytest.mark.parametrize("time_zone", [None, "UTC"])
@pytest.mark.parametrize("every", ["10m", "1w", "1mo"])
def test_datetime_dtypes(
    in_memory_engine: pl.GPUEngine,
    frame: pl.LazyFrame,
    time_unit: TimeUnit,
    time_zone: str | None,
    every: str,
) -> None:
    q = (
        frame.with_columns(pl.col("t").cast(pl.Datetime(time_unit, time_zone)))
        .group_by_dynamic("t", every=every, label="right", include_boundaries=True)
        .agg(*_aggs())
    )
    assert_gpu_result_equal(q, engine=in_memory_engine)


@pytest.mark.parametrize(
    "every, offset",
    [
        ("1d", "0d"),
        ("1d", "-1d"),
        ("3d", "1d"),
        ("3d", "-1d"),
        ("3d", "-2d"),
        ("1w", "0d"),
        ("1w", "-1d"),
        ("1w", "2d"),
        ("1mo", "0d"),
        ("1y", "0d"),
    ],
)
@pytest.mark.parametrize("closed", ["left", "right"])
def test_date_index(
    in_memory_engine: pl.GPUEngine,
    frame: pl.LazyFrame,
    every: str,
    offset: str,
    closed: ClosedInterval,
) -> None:
    q = (
        frame.with_columns(pl.col("t").dt.date())
        .group_by_dynamic("t", every=every, offset=offset, closed=closed, label="right")
        .agg(*_aggs())
    )
    assert_gpu_result_equal(q, engine=in_memory_engine)


@pytest.mark.parametrize("dtype", [pl.Int32, pl.Int64])
@pytest.mark.parametrize("every", ["1i", "3i", "10i"])
@pytest.mark.parametrize("offset", ["0i", "2i", "-1i", "-7i"])
@pytest.mark.parametrize("closed", ["left", "right"])
def test_integer_index(
    in_memory_engine: pl.GPUEngine,
    dtype: pl.DataType,
    every: str,
    offset: str,
    closed: ClosedInterval,
) -> None:
    rng = random.Random(7)
    q = (
        pl.LazyFrame(
            {
                "t": sorted(rng.randrange(-100, 100) for _ in range(200)),
                "v": [rng.randrange(10) for _ in range(200)],
                "x": [rng.random() for _ in range(200)],
            },
            schema_overrides={"t": dtype},
        )
        .group_by_dynamic(
            "t", every=every, offset=offset, closed=closed, include_boundaries=True
        )
        .agg(*_aggs())
    )
    assert_gpu_result_equal(q, engine=in_memory_engine)


@pytest.mark.parametrize("group_by", ["g", ["g", "h"], ["h", "g"]])
@pytest.mark.parametrize(
    "label, include_boundaries", [("left", False), ("right", True), ("datapoint", True)]
)
def test_grouped(
    engine: pl.GPUEngine,
    short_frame: pl.LazyFrame,
    group_by: str | list[str],
    label: Label,
    include_boundaries: bool,  # noqa: FBT001
) -> None:
    # Polars requires the index to be sorted within each group.
    q = short_frame.group_by_dynamic(
        "t",
        every="1w",
        group_by=group_by,
        label=label,
        include_boundaries=include_boundaries,
    ).agg(*_aggs())
    assert_gpu_result_equal(q, engine=engine)


def test_aggregations(engine: pl.GPUEngine, short_frame: pl.LazyFrame) -> None:
    q = short_frame.group_by_dynamic("t", every="1w", group_by="g").agg(
        pl.col("t").min().alias("t_min"),
        pl.col("t").max().alias("t_max"),
        pl.col("v").mean().alias("v_mean"),
        pl.col("v").min().alias("v_min"),
        pl.col("v").n_unique().alias("v_unique"),
        (pl.col("v") * pl.col("x")).sum().alias("vx"),
    )
    assert_gpu_result_equal(q, engine=engine, check_exact=False)


@pytest.mark.parametrize("group_by", [None, "g"])
def test_first_last(
    engine: pl.GPUEngine, short_frame: pl.LazyFrame, group_by: str | None
) -> None:
    q = short_frame.group_by_dynamic("t", every="1w", group_by=group_by).agg(
        pl.col("v").first().alias("v_first"),
        pl.col("v").last().alias("v_last"),
        pl.col("t").first().alias("t_first"),
    )
    with _first_last_fallback(engine):
        assert_gpu_result_equal(q, engine=engine)


def test_no_aggregations(engine: pl.GPUEngine, short_frame: pl.LazyFrame) -> None:
    q = short_frame.group_by_dynamic("t", every="1w", group_by="g").agg()
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("group_by", [None, "g"])
@pytest.mark.parametrize(
    "zlice", [(0, 3), (5, 4), (-3, 3), (-4, 2)], ids=lambda s: f"slice{s}"
)
def test_slice(
    engine: pl.GPUEngine,
    short_frame: pl.LazyFrame,
    group_by: str | None,
    zlice: tuple[int, int],
) -> None:
    # Polars pushes the slice into the dynamic group-by, where it applies to
    # the ordered result.
    q = (
        short_frame.group_by_dynamic("t", every="1w", group_by=group_by)
        .agg(*_aggs())
        .slice(*zlice)
    )
    offset, length = zlice
    top_or_bottom_k = offset == 0 or offset + length == 0
    # Streaming sorts only fuse top-k and bottom-k slices.
    with warns_on_spmd(
        engine,
        UserWarning,
        match="This slice not supported for multiple partitions",
        when=get_blocksize_mode(engine) == "medium" and not top_or_bottom_k,
    ):
        assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("group_by", [None, "g"])
def test_empty(
    engine: pl.GPUEngine, short_frame: pl.LazyFrame, group_by: str | None
) -> None:
    q = (
        short_frame.filter(pl.col("v") > 1000)
        .group_by_dynamic("t", every="1d", group_by=group_by)
        .agg(*_aggs())
    )
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"every": "1w"},
        {"every": "1d", "offset": "-1h", "closed": "right", "label": "right"},
        {"every": "1mo", "include_boundaries": True},
    ],
    ids=["1w", "offset-right", "1mo"],
)
@pytest.mark.parametrize("index", ["datetime", "date", "int"])
def test_streaming(
    engine: pl.GPUEngine, short_frame: pl.LazyFrame, kwargs: dict, index: str
) -> None:
    """Representative windows on every engine, with the full matrix in memory."""
    lf = short_frame
    if index == "date":
        lf = lf.with_columns(pl.col("t").dt.date())
        # Date indexes take whole-day offsets, and boundaries fall back.
        kwargs = {k: v for k, v in kwargs.items() if k != "include_boundaries"}
        kwargs |= {"offset": "-1d"} if "offset" in kwargs else {}
    elif index == "int":
        lf = lf.with_columns(pl.col("t").dt.epoch("d"))
        kwargs = {**kwargs, "every": "7i"}
        kwargs |= {"offset": "-1i"} if "offset" in kwargs else {}
    q = lf.group_by_dynamic("t", **kwargs).agg(*_aggs())
    assert_gpu_result_equal(q, engine=engine)


@pytest.fixture(scope="module")
def quotes() -> pl.LazyFrame:
    rng = random.Random(3)
    n = 200
    open_ = dt.datetime(2026, 1, 5, 9, 30)
    return pl.LazyFrame(
        {
            "time": sorted(
                open_ + dt.timedelta(microseconds=rng.randrange(int(6.5 * 3600e6)))
                for _ in range(n)
            ),
            "ex": [rng.choice("ABCDZ") for _ in range(n)],
            "bid": [rng.uniform(99, 100) for _ in range(n)],
            "ask": [rng.uniform(100, 101) for _ in range(n)],
            "bsize": [float(rng.randrange(0, 5)) for _ in range(n)],
            "asize": [float(rng.randrange(0, 5)) for _ in range(n)],
        }
    )


@pytest.mark.skipif(
    POLARS_VERSION_LT_142,
    reason="Fused arithmetic in group-by aggregations needs polars>=1.42",
)
@pytest.mark.parametrize("every", ["10m", "5m", "1m"])
@pytest.mark.parametrize("group_by", [None, ["ex"]])
def test_taq_mean(
    engine: pl.GPUEngine,
    quotes: pl.LazyFrame,
    every: str,
    group_by: list[str] | None,
) -> None:
    """Window means of the KX NYSE TAQ benchmark (q22-33)."""
    # TAQ applies drop_nans() before mean(), which GroupBy does not support
    # inside aggregations yet. The window computation is the same.
    liquidity = (pl.col("bsize") * pl.col("bid") + pl.col("asize") * pl.col("ask")) / (
        pl.col("asize") + pl.col("bsize")
    )
    q = quotes.group_by_dynamic(
        index_column="time", every=every, closed="left", group_by=group_by
    ).agg(
        liquidity.mean().alias("avgLiqWMid"),
        (pl.col("ask") - pl.col("bid")).mean().alias("avgSpread"),
    )
    # Float means are summed in a different order on the GPU.
    assert_gpu_result_equal(q, engine=engine, check_exact=False)


@pytest.mark.parametrize("every", ["5m", "1m"])
def test_taq_ohlc(engine: pl.GPUEngine, quotes: pl.LazyFrame, every: str) -> None:
    """Open/high/low/close bars of the KX NYSE TAQ benchmark (q64-67)."""
    q = quotes.group_by_dynamic(
        index_column="time", every=every, closed="left", group_by=["ex"]
    ).agg(
        pl.col("bid").first().alias("o"),
        pl.col("bid").max().alias("h"),
        pl.col("bid").min().alias("l"),
        pl.col("bid").last().alias("c"),
        pl.col("bsize").sum().alias("s"),
    )
    with _first_last_fallback(engine):
        assert_gpu_result_equal(q, engine=engine)


_UNSORTED = (pl.exceptions.InvalidOperationError, "is not sorted")
_GROUPED_UNSORTED = (pl.exceptions.ComputeError, "input data is not sorted")
_GROUPED_NULLS = (pl.exceptions.ComputeError, "null values in dynamic group_by")


@pytest.mark.parametrize(
    "t, keys, error",
    [
        ([3, 1, 2, 4], {}, _UNSORTED),
        ([1, None, 3, 4], {}, _UNSORTED),
        ([1, None, 3, 4], {"g": [1, 1, 2, 2]}, _GROUPED_NULLS),
        ([5, 1, 6, 7], {"g": ["a", "a", "b", "b"]}, _GROUPED_UNSORTED),
        ([1, 2, 6, 3], {"g": ["a", "b", "a", "a"]}, _GROUPED_UNSORTED),
        ([1, 5, 3, 6], {"g": [None, None, None, "a"]}, _GROUPED_UNSORTED),
        ([1, 2, 4, 3], {"g": ["a"] * 4, "h": [1, 2, 1, 1]}, _GROUPED_UNSORTED),
    ],
    ids=[
        "unsorted",
        "null",
        "grouped-null",
        "grouped",
        "interleaved",
        "null-key",
        "two-keys",
    ],
)
def test_invalid_index_raises(
    engine: pl.GPUEngine, t: list, keys: dict, error: tuple[type[Exception], str]
) -> None:
    q = (
        pl.LazyFrame({"t": t, **keys, "v": range(len(t))})
        .group_by_dynamic("t", every="2i", group_by=list(keys) or None)
        .agg(pl.col("v").sum())
    )
    exception, match = error
    with pytest.raises(exception, match=match):
        q.collect(engine="in-memory")
    if is_streaming_engine(engine):
        with pytest.RaisesGroup(pytest.RaisesExc(exception, match=match)):
            q.collect(engine=engine)
    else:
        with pytest.raises(exception, match=match):
            q.collect(engine=engine)


@pytest.mark.parametrize(
    "t, keys",
    [
        # Sorted by the keys (nulls first or last), then the index.
        ([6, 7, 1, 5, 2, 9], {"g": [None, None, "a", "a", "b", "b"]}),
        ([1, 5, 2, 9, 6, 7], {"g": ["a", "a", "b", "b", None, None]}),
        # Groups interleaved, sorted within each group only.
        ([5, 6, 1, 7, 2, 9], {"g": ["a", "a", "b", "a", "b", None]}),
        ([5, 1, 6, 2, 0, 3], {"g": ["a"] * 4 + [None] * 2, "h": [1, 2, 1, 2, 1, 1]}),
    ],
    ids=["keys-nulls-first", "keys-nulls-last", "interleaved", "interleaved-two-keys"],
)
def test_sorted_within_groups(
    in_memory_engine: pl.GPUEngine, t: list, keys: dict
) -> None:
    q = (
        pl.LazyFrame({"t": t, **keys, "v": range(len(t))})
        .group_by_dynamic("t", every="2i", group_by=list(keys))
        .agg(pl.col("v").sum(), pl.col("v").first().alias("first"))
    )
    assert_gpu_result_equal(q, engine=in_memory_engine)


@pytest.mark.parametrize(
    "index, kwargs",
    [
        (pl.Datetime("ms"), {"every": "1d", "period": "2d"}),
        (pl.Datetime("ms"), {"every": "2d", "period": "1d"}),
        (pl.Datetime("ms"), {"every": "1d", "closed": "both"}),
        (pl.Datetime("ms"), {"every": "1d", "closed": "none"}),
        (pl.Datetime("ms"), {"every": "1d", "start_by": "datapoint"}),
        (pl.Datetime("ms"), {"every": "1w", "start_by": "monday"}),
        (pl.Datetime("ms"), {"every": "1mo", "offset": "1d"}),
        (pl.Datetime("ms"), {"every": "1d", "offset": "1mo"}),
        (pl.Datetime("ms"), {"every": "1mo2d"}),
        (pl.Datetime("ms"), {"every": "1d12h"}),
        (pl.Datetime("ms"), {"every": "1d", "offset": "1d2h"}),
        (pl.Datetime("ms"), {"every": "10m", "offset": "1ns"}),
        (pl.Datetime("ms"), {"every": "1500us"}),
        (pl.Datetime("us"), {"every": "1ns"}),
        (pl.Datetime("us", "America/New_York"), {"every": "1d"}),
        # Polars' schema says Date, but it computes Datetime boundaries.
        (pl.Date(), {"every": "1w", "include_boundaries": True}),
        # Flooring by a window this wide could overflow INT64.
        (pl.Int64(), {"every": f"{2**62 + 1}i"}),
    ],
    ids=lambda v: (
        ",".join(f"{k}={x}" for k, x in v.items()) if isinstance(v, dict) else str(v)
    ),
)
def test_unsupported_raises(
    in_memory_engine: pl.GPUEngine,
    frame: pl.LazyFrame,
    index: pl.DataType,
    kwargs: dict,
) -> None:
    if index == pl.Int64():
        t = pl.col("t").dt.epoch("d")
    elif isinstance(index, pl.Datetime) and index.time_zone:
        t = pl.col("t").dt.replace_time_zone(index.time_zone)
    else:
        t = pl.col("t").cast(index)
    q = frame.with_columns(t).group_by_dynamic("t", **kwargs).agg(pl.len())
    assert_ir_translation_raises(q, in_memory_engine, NotImplementedError)


@pytest.mark.parametrize("label", ["left", "right", "datapoint"])
@pytest.mark.parametrize("closed", ["left", "right"])
@pytest.mark.parametrize("group_by", [None, "g"])
def test_int32_window_overflow(
    engine: pl.GPUEngine, group_by: str | None, closed: ClosedInterval, label: Label
) -> None:
    # A window may start or end outside Int32's range. Polars nulls that
    # label/boundary, keeping distinct windows distinct, instead of wrapping
    # (which corrupts the sort order) or nulling the grouping key (which
    # collides distinct out-of-range windows into one group).
    i32_min, i32_max = -(2**31), 2**31 - 1
    t = [i32_min, i32_min + 1, i32_min + 3, 0, i32_max - 3, i32_max - 1, i32_max]
    q = (
        pl.LazyFrame(
            {
                "t": pl.Series(t, dtype=pl.Int32),
                "g": [1, 1, 1, 2, 2, 2, 2],
                "v": range(len(t)),
            }
        )
        .group_by_dynamic(
            "t",
            every="2i",
            offset="1i",
            closed=closed,
            label=label,
            group_by=group_by,
            include_boundaries=True,
        )
        .agg(pl.col("v").sum())
    )
    assert_gpu_result_equal(q, engine=engine)


def test_int64_shift_overflow(engine: pl.GPUEngine) -> None:
    # An offset or closed="right" shifts the index before truncating, and for
    # a row within that shift of Int64's range, the shift itself overflows.
    # Null that row's window instead of silently wrapping it into an
    # unrelated, distant one.
    i64_min = -(2**63)
    t = [i64_min, i64_min + 1, i64_min + 5]
    q = (
        pl.LazyFrame({"t": pl.Series(t, dtype=pl.Int64), "v": [1, 2, 3]})
        .group_by_dynamic("t", every="2i", closed="right")
        .agg(pl.col("v").sum())
    )
    result = q.collect(engine=engine)
    assert result.sort("t", nulls_last=False).rows() == [
        (None, 1),
        (i64_min, 2),
        (i64_min + 4, 3),
    ]


@pytest.mark.parametrize(
    "keys",
    [
        {"k": pl.Series(["b", None, "a", "b"])},
        {"k": pl.Series([2, None, 1, 2])},
        {"k": pl.Series([2.0, None, float("nan"), 1.0])},
        {"k": pl.Series([True, None, False, True])},
        {"k": pl.Series(["b", None, "a", None]), "j": pl.Series([None, 2, 1, 1])},
        {"j": pl.Series([None, 2, 1, 1]), "k": pl.Series(["b", None, "a", None])},
    ],
    ids=["str", "int", "float", "bool", "str-int", "int-str"],
)
def test_key_order(engine: pl.GPUEngine, keys: dict[str, pl.Series]) -> None:
    # Polars sorts groups by key, with null keys last for strings and first
    # for other types.
    q = (
        pl.LazyFrame({"t": [0, 1, 2, 3], **keys, "v": [1, 2, 3, 4]})
        .group_by_dynamic("t", every="10i", group_by=list(keys))
        .agg(pl.col("v").sum())
    )
    assert_gpu_result_equal(q, engine=engine)
