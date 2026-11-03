# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for Streamz Dataframes (SDFs) built on top of cuDF DataFrames.
*** Borrowed from streamz.dataframe.tests | License at thirdparty/LICENSE ***
"""

import json
import operator

import numpy as np
import pandas as pd
import pytest

import distributed.gc
from dask.dataframe.utils import assert_eq
from distributed import Client

from streamz import Stream
from streamz.dask import DaskStream
from streamz.dataframe import Aggregation, DataFrame, DataFrames, Series

cudf = pytest.importorskip("cudf")


@pytest.fixture(scope="module")
def client():
    client = Client(processes=False, asynchronous=False)

    # Fix flaky tests seen in workflows like
    # https://github.com/NVIDIA/cudf/actions/runs/15119048978/job/42498435703?pr=18870#step:9:1722
    # These manifest as a RecursionError in https://github.com/dask/distributed/blob/a890b85c8f107f7c8664ef96270ef8c25a2b31e4/distributed/gc.py#L201
    # There isn't a public API for whether it's enabled or disabled. We'll just
    # assume that it's enabled and disable it for the duration of the tests.
    client.run_on_scheduler(distributed.gc.disable_gc_diagnosis)
    client.run(distributed.gc.disable_gc_diagnosis)

    try:
        yield client
    finally:
        client.close()


@pytest.fixture(params=["core", "dask"])
def stream(request, client):
    if request.param == "core":
        return Stream()
    else:
        return DaskStream()


def test_identity(stream):
    df = cudf.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    sdf = DataFrame(example=df, stream=stream)
    L = sdf.stream.gather().sink_to_list()

    sdf.emit(df)

    assert L[0] is df
    assert list(sdf.example.columns) == ["x", "y"]

    x = sdf.x
    assert isinstance(x, Series)
    L2 = x.stream.gather().sink_to_list()
    assert not L2

    sdf.emit(df)
    assert isinstance(L2[0], cudf.Series)
    assert_eq(L2[0], df.x)


def test_dtype(stream):
    df = cudf.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    sdf = DataFrame(example=df, stream=stream)

    assert str(sdf.dtypes) == str(df.dtypes)
    assert sdf.x.dtype == df.x.dtype
    assert sdf.index.dtype == df.index.dtype


def test_attributes():
    df = cudf.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    sdf = DataFrame(example=df)

    assert getattr(sdf, "x", -1) != -1
    assert getattr(sdf, "z", -1) == -1

    sdf.x
    with pytest.raises(AttributeError):
        sdf.z


def test_exceptions(stream):
    df = cudf.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    sdf = DataFrame(example=df, stream=stream)
    with pytest.raises(TypeError):
        sdf.emit(1)

    with pytest.raises(IndexError):
        sdf.emit(cudf.DataFrame())


@pytest.mark.parametrize(
    "func", [lambda x: x.sum(), lambda x: x.mean(), lambda x: x.count()]
)
def test_reductions(stream, func):
    df = cudf.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    for example in [df, df.iloc[:0]]:
        sdf = DataFrame(example=example, stream=stream)

        df_out = func(sdf).stream.gather().sink_to_list()

        x = sdf.x
        x_out = func(x).stream.gather().sink_to_list()

        sdf.emit(df)
        sdf.emit(df)

        assert_eq(df_out[-1], func(cudf.concat([df, df])))
        assert_eq(x_out[-1], func(cudf.concat([df, df]).x))


@pytest.mark.parametrize(
    "op",
    [
        operator.add,
        operator.and_,
        operator.eq,
        operator.floordiv,
        operator.ge,
        operator.gt,
        operator.le,
        operator.lshift,
        operator.lt,
        operator.mod,
        operator.mul,
        operator.ne,
        operator.or_,
        operator.pow,
        operator.rshift,
        operator.sub,
        operator.truediv,
        operator.xor,
    ],
)
@pytest.mark.parametrize("getter", [lambda df: df, lambda df: df.x])
def test_binary_operators(op, getter, stream):
    df = cudf.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    try:
        left = op(getter(df), 2)
        right = op(2, getter(df))
    except Exception:
        return

    a = DataFrame(example=df, stream=stream)
    li = op(getter(a), 2).stream.gather().sink_to_list()
    r = op(2, getter(a)).stream.gather().sink_to_list()

    a.emit(df)

    assert_eq(li[0], left)
    assert_eq(r[0], right)


@pytest.mark.parametrize(
    "op",
    [
        operator.abs,
        operator.inv,
        operator.invert,
        operator.neg,
        lambda x: x.map(lambda x: x + 1),
        lambda x: x.reset_index(),
        lambda x: x.astype(float),
    ],
)
@pytest.mark.parametrize("getter", [lambda df: df, lambda df: df.x])
def test_unary_operators(op, getter):
    df = cudf.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    try:
        expected = op(getter(df))
    except Exception:
        return

    a = DataFrame(example=df)
    b = op(getter(a)).stream.sink_to_list()

    a.emit(df)

    assert_eq(b[0], expected)


@pytest.mark.parametrize(
    "func",
    [
        lambda df: df.query("x > 1 and x < 4"),
        lambda df: df.x.value_counts().nlargest(2).astype(int),
    ],
)
def test_dataframe_simple(func):
    df = cudf.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    expected = func(df)

    a = DataFrame(example=df)
    L = func(a).stream.sink_to_list()

    a.emit(df)

    assert_eq(L[0], expected)


def test_set_index():
    df = cudf.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})

    a = DataFrame(example=df)

    b = a.set_index("x").stream.sink_to_list()
    a.emit(df)
    assert_eq(b[0], df.set_index("x"))

    b = a.set_index(a.y + 1).stream.sink_to_list()
    a.emit(df)
    assert_eq(b[0], df.set_index(df.y + 1))


def test_binary_stream_operators(stream):
    df = cudf.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})

    expected = df.x + df.y

    a = DataFrame(example=df, stream=stream)
    b = (a.x + a.y).stream.gather().sink_to_list()

    a.emit(df)

    assert_eq(b[0], expected)


def test_index(stream):
    df = cudf.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    a = DataFrame(example=df, stream=stream)
    b = a.index + 5
    L = b.stream.gather().sink_to_list()

    a.emit(df)
    a.emit(df)

    assert_eq(L[0], df.index + 5)
    assert_eq(L[1], df.index + 5)


def test_pair_arithmetic(stream):
    df = cudf.DataFrame({"x": list(range(10)), "y": [1] * 10})

    a = DataFrame(example=df.iloc[:0], stream=stream)
    L = ((a.x + a.y) * 2).stream.gather().sink_to_list()

    a.emit(df.iloc[:5])
    a.emit(df.iloc[5:])

    assert len(L) == 2
    assert_eq(cudf.concat(L), (df.x + df.y) * 2)


def test_getitem(stream):
    df = cudf.DataFrame({"x": list(range(10)), "y": [1] * 10})

    a = DataFrame(example=df.iloc[:0], stream=stream)
    L = a[a.x > 4].stream.gather().sink_to_list()

    a.emit(df.iloc[:5])
    a.emit(df.iloc[5:])

    assert len(L) == 2
    assert_eq(cudf.concat(L), df[df.x > 4])


@pytest.mark.parametrize("agg", [lambda x: x.sum(), lambda x: x.mean()])
@pytest.mark.parametrize(
    "grouper",
    [lambda a: a.x % 3, lambda a: "x", lambda a: a.index % 2, lambda a: ["x"]],
)
@pytest.mark.parametrize(
    "indexer", [lambda g: g, lambda g: g[["y"]], lambda g: g[["x", "y"]]]
)
def test_groupby_aggregate(agg, grouper, indexer, stream):
    df = cudf.DataFrame(
        {"x": (np.arange(10) // 2).astype(float), "y": [1.0, 2.0] * 5}
    )

    a = DataFrame(example=df.iloc[:0], stream=stream)

    def f(x):
        return agg(indexer(x.groupby(grouper(x))))

    L = f(a).stream.gather().sink_to_list()

    a.emit(df.iloc[:3])
    a.emit(df.iloc[3:7])
    a.emit(df.iloc[7:])

    first = df.iloc[:3]
    g = f(first)

    h = f(df)

    assert_eq(L[0], g)
    assert_eq(L[-1], h)


def test_repr(stream):
    df = cudf.DataFrame(
        {"x": (np.arange(10) // 2).astype(float), "y": [1.0] * 10}
    )
    a = DataFrame(example=df, stream=stream)

    text = repr(a)
    assert type(a).__name__ in text
    assert "x" in text
    assert "y" in text

    text = repr(a.x)
    assert type(a.x).__name__ in text
    assert "x" in text

    text = repr(a.x.sum())
    assert type(a.x.sum()).__name__ in text


def test_repr_html(stream):
    df = cudf.DataFrame(
        {"x": (np.arange(10) // 2).astype(float), "y": [1.0] * 10}
    )
    a = DataFrame(example=df, stream=stream)

    for x in [a, a.y, a.y.mean()]:
        html = x._repr_html_()
        assert type(x).__name__ in html
        assert "1" in html


def test_setitem(stream):
    df = cudf.DataFrame({"x": list(range(10)), "y": [1] * 10})

    sdf = DataFrame(example=df.iloc[:0], stream=stream)
    stream = sdf.stream

    sdf["z"] = sdf["x"] * 2
    sdf["a"] = 10
    sdf[["c", "d"]] = sdf[["x", "y"]]

    L = sdf.mean().stream.gather().sink_to_list()

    stream.emit(df.iloc[:3])
    stream.emit(df.iloc[3:7])
    stream.emit(df.iloc[7:])

    df["z"] = df["x"] * 2
    df["a"] = 10
    df["c"] = df["x"]
    df["d"] = df["y"]

    assert_eq(L[-1], df.mean())


def test_setitem_overwrites(stream):
    df = cudf.DataFrame({"x": list(range(10))})
    sdf = DataFrame(example=df.iloc[:0], stream=stream)
    stream = sdf.stream

    sdf["x"] = sdf["x"] * 2

    L = sdf.stream.gather().sink_to_list()

    stream.emit(df.iloc[:3])
    stream.emit(df.iloc[3:7])
    stream.emit(df.iloc[7:])

    assert_eq(L[-1], df.iloc[7:] * 2)


@pytest.mark.parametrize(
    "kwargs,op",
    [
        ({}, "sum"),
        ({}, "mean"),
        ({}, "min"),
        pytest.param(
            {},
            "median",
            marks=pytest.mark.xfail(reason="Unavailable for rolling objects"),
        ),
        ({}, "max"),
        ({}, "var"),
        ({}, "count"),
        ({"ddof": 0}, "std"),
        pytest.param(
            {"quantile": 0.5},
            "quantile",
            marks=pytest.mark.xfail(reason="Unavailable for rolling objects"),
        ),
        pytest.param(
            {"arg": {"A": "sum", "B": "min"}},
            "aggregate",
            marks=pytest.mark.xfail(reason="Unavailable for rolling objects"),
        ),
    ],
)
@pytest.mark.parametrize(
    "window",
    [pytest.param(2), 7, pytest.param("3h"), pd.Timedelta("200 minutes")],
)
@pytest.mark.parametrize("m", [2, pytest.param(5)])
@pytest.mark.parametrize(
    "pre_get,post_get",
    [
        (lambda df: df, lambda df: df),
        (lambda df: df.x, lambda x: x),
        (lambda df: df, lambda df: df.x),
    ],
)
def test_rolling_count_aggregations(
    op, window, m, pre_get, post_get, kwargs, stream
):
    index = pd.DatetimeIndex(
        pd.date_range("2000-01-01", "2000-01-03", freq="1h")
    )
    df = cudf.DataFrame({"x": np.arange(len(index))}, index=index)

    expected = getattr(post_get(pre_get(df).rolling(window)), op)(**kwargs)

    sdf = DataFrame(example=df, stream=stream)
    with cudf.option_context("mode.pandas_compatible", True):
        roll = getattr(post_get(pre_get(sdf).rolling(window)), op)(**kwargs)
        L = roll.stream.gather().sink_to_list()
    assert len(L) == 0

    for i in range(0, len(df), m):
        sdf.emit(df.iloc[i : i + m])

    assert len(L) > 1
    with cudf.option_context("mode.pandas_compatible", True):
        assert_eq(cudf.concat(L), expected)


def test_stream_to_dataframe(stream):
    df = cudf.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    source = stream
    L = source.to_dataframe(example=df).x.sum().stream.gather().sink_to_list()

    source.emit(df)
    source.emit(df)
    source.emit(df)

    assert L == [6, 12, 18]


def test_integration_from_stream(stream):
    source = stream
    sdf = (
        source.partition(4)
        .to_batch(example=['{"x": 0, "y": 0}'])
        .map(json.loads)
        .to_dataframe()
    )
    result = sdf.groupby(sdf.x).y.sum().mean()
    L = result.stream.gather().sink_to_list()

    for i in range(12):
        source.emit(json.dumps({"x": i % 3, "y": i}))

    assert L == [2, 28 / 3, 22.0]


def test_to_frame(stream):
    df = cudf.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    sdf = DataFrame(example=df, stream=stream)

    assert sdf.to_frame() is sdf

    a = sdf.x.to_frame()
    assert isinstance(a, DataFrame)
    assert list(a.columns) == ["x"]


@pytest.mark.parametrize("op", ["cumsum", "cummax", "cumprod", "cummin"])
@pytest.mark.parametrize("getter", [lambda df: df, lambda df: df.x])
def test_cumulative_aggregations(op, getter, stream):
    df = cudf.DataFrame({"x": list(range(10)), "y": [1] * 10})
    expected = getattr(getter(df), op)()

    sdf = DataFrame(example=df, stream=stream)

    L = getattr(getter(sdf), op)().stream.gather().sink_to_list()

    for i in range(0, 10, 3):
        sdf.emit(df.iloc[i : i + 3])
    sdf.emit(df.iloc[:0])

    assert len(L) > 1

    assert_eq(cudf.concat(L), expected)


@pytest.mark.xfail(
    reason="IPyWidgets 8.0 broke streamz 0.6.4. "
    "We should remove this xfail when this is fixed in streamz."
)
def test_display(stream):
    pytest.importorskip("ipywidgets")
    pytest.importorskip("IPython")

    df = cudf.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    sdf = DataFrame(example=df, stream=stream)

    s = sdf.x.sum()

    s._ipython_display_()


def test_tail(stream):
    df = cudf.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    sdf = DataFrame(example=df, stream=stream)

    L = sdf.tail(2).stream.gather().sink_to_list()

    sdf.emit(df)
    sdf.emit(df)

    assert_eq(L[0], df.tail(2))
    assert_eq(L[1], df.tail(2))


def test_example_type_error_message():
    try:
        DataFrame(example=[123])
    except Exception as e:
        assert "DataFrame" in str(e)
        assert "[123]" in str(e)


def test_dataframes(stream):
    df = cudf.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    sdf = DataFrames(example=df, stream=stream)
    L = sdf.x.sum().stream.gather().sink_to_list()

    sdf.emit(df)
    sdf.emit(df)

    assert L == [6, 6]


def test_groupby_aggregate_updating(stream):
    df = cudf.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    sdf = DataFrame(example=df, stream=stream)

    assert sdf.groupby("x").y.mean()._stream_type == "updating"
    assert sdf.x.sum()._stream_type == "updating"
    assert (sdf.x.sum() + 1)._stream_type == "updating"


def test_window_sum(stream):
    df = cudf.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    sdf = DataFrame(example=df, stream=stream)
    L = sdf.window(n=4).x.sum().stream.gather().sink_to_list()

    sdf.emit(df)
    assert L == [6]
    sdf.emit(df)
    assert L == [6, 9]
    sdf.emit(df)
    assert L == [6, 9, 9]


def test_window_sum_dataframe(stream):
    df = cudf.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    sdf = DataFrame(example=df, stream=stream)
    L = sdf.window(n=4).sum().stream.gather().sink_to_list()

    sdf.emit(df)
    assert_eq(L[0], cudf.Series([6, 15], index=["x", "y"]))
    sdf.emit(df)
    assert_eq(L[0], cudf.Series([6, 15], index=["x", "y"]))
    assert_eq(L[1], cudf.Series([9, 21], index=["x", "y"]))
    sdf.emit(df)
    assert_eq(L[0], cudf.Series([6, 15], index=["x", "y"]))
    assert_eq(L[1], cudf.Series([9, 21], index=["x", "y"]))
    assert_eq(L[2], cudf.Series([9, 21], index=["x", "y"]))


@pytest.mark.parametrize(
    "func",
    [
        lambda x: x.sum(),
        lambda x: x.mean(),
        lambda x: x.count(),
        lambda x: x.var(ddof=1),
        lambda x: x.std(ddof=1),
        lambda x: x.var(ddof=0),
    ],
)
@pytest.mark.parametrize("n", [2, 4])
@pytest.mark.parametrize("getter", [lambda df: df.x])
def test_windowing_n(func, n, getter):
    df = cudf.DataFrame({"x": list(range(10)), "y": [1, 2] * 5})

    sdf = DataFrame(example=df)
    L = func(getter(sdf).window(n=n)).stream.gather().sink_to_list()

    for i in range(0, 10, 3):
        sdf.emit(df.iloc[i : i + 3])
    sdf.emit(df.iloc[:0])

    assert len(L) == 5

    assert_eq(L[0], func(getter(df).iloc[max(0, 3 - n) : 3]))
    assert_eq(L[-1], func(getter(df).iloc[len(df) - n :]))


@pytest.mark.parametrize("func", [lambda x: x.sum(), lambda x: x.mean()])
@pytest.mark.parametrize("value", ["10h", "1D"])
@pytest.mark.parametrize("getter", [lambda df: df, lambda df: df.x])
@pytest.mark.parametrize(
    "grouper", [lambda a: "y", lambda a: a.index, lambda a: ["y"]]
)
@pytest.mark.parametrize(
    "indexer", [lambda g: g, lambda g: g[["x"]], lambda g: g[["x", "y"]]]
)
def test_groupby_windowing_value(func, value, getter, grouper, indexer):
    index = pd.DatetimeIndex(
        pd.date_range("2000-01-01", "2000-01-03", freq="1h")
    )
    df = cudf.DataFrame(
        {
            "x": np.arange(len(index), dtype=float),
            "y": np.arange(len(index), dtype=float) % 2,
        },
        index=index,
    )

    value = pd.Timedelta(value)

    sdf = DataFrame(example=df)

    def f(x):
        return func(indexer(x.groupby(grouper(x))))

    L = f(sdf.window(value)).stream.gather().sink_to_list()

    diff = 13
    for i in range(0, len(index), diff):
        sdf.emit(df.iloc[i : i + diff])

    assert len(L) == 4

    first = df.iloc[:diff]
    lost = first.loc[first.index.min() + value :]
    first = first.iloc[len(lost) :]

    g = f(first)
    assert_eq(L[0], g)

    last = df.loc[index.max() - value + pd.Timedelta("1s") :]
    h = f(last)
    assert_eq(L[-1], h)


@pytest.mark.parametrize("func", [lambda x: x.sum(), lambda x: x.mean()])
@pytest.mark.parametrize("n", [1, 4])
@pytest.mark.parametrize("getter", [lambda df: df, lambda df: df.x])
@pytest.mark.parametrize(
    "grouper",
    [lambda a: a.x % 3, lambda a: "y", lambda a: a.index % 2, lambda a: ["y"]],
)
@pytest.mark.parametrize("indexer", [lambda g: g, lambda g: g[["x", "y"]]])
def test_groupby_windowing_n(func, n, getter, grouper, indexer):
    df = cudf.DataFrame({"x": np.arange(10, dtype=float), "y": [1.0, 2.0] * 5})

    sdf = DataFrame(example=df)

    def f(x):
        return func(indexer(x.groupby(grouper(x))))

    L = f(sdf.window(n=n)).stream.gather().sink_to_list()

    diff = 3
    for i in range(0, 10, diff):
        sdf.emit(df.iloc[i : i + diff])
    sdf.emit(df.iloc[:0])

    assert len(L) == 5

    first = df.iloc[max(0, diff - n) : diff]

    g = f(first)
    assert_eq(L[0], g)

    last = df.iloc[len(df) - n :]
    h = f(last)
    assert_eq(L[-1], h)


def test_window_full():
    df = cudf.DataFrame({"x": np.arange(10, dtype=float), "y": [1.0, 2.0] * 5})

    sdf = DataFrame(example=df)

    L = sdf.window(n=4).apply(lambda x: x).stream.sink_to_list()

    sdf.emit(df.iloc[:3])
    sdf.emit(df.iloc[3:8])
    sdf.emit(df.iloc[8:])

    assert_eq(L[0], df.iloc[:3])
    assert_eq(L[1], df.iloc[4:8])
    assert_eq(L[2], df.iloc[-4:])


def test_custom_aggregation():
    df = cudf.DataFrame({"x": np.arange(10, dtype=float), "y": [1.0, 2.0] * 5})

    class Custom(Aggregation):
        def initial(self, new):
            return 0

        def on_new(self, state, new):
            return state + 1, state

        def on_old(self, state, new):
            return state - 100, state

    sdf = DataFrame(example=df)
    L = sdf.aggregate(Custom()).stream.sink_to_list()

    sdf.emit(df)
    sdf.emit(df)
    sdf.emit(df)

    assert L == [0, 1, 2]

    sdf = DataFrame(example=df)
    L = sdf.window(n=5).aggregate(Custom()).stream.sink_to_list()

    sdf.emit(df)
    sdf.emit(df)
    sdf.emit(df)

    assert L == [1, -198, -397]


def test_groupby_aggregate_with_start_state(stream):
    example = cudf.DataFrame({"name": [], "amount": []}, dtype="float64")
    sdf = DataFrame(stream, example=example).groupby(["name"])
    output0 = sdf.amount.sum(start=None).stream.gather().sink_to_list()
    output1 = (
        sdf.amount.mean(with_state=True, start=None)
        .stream.gather()
        .sink_to_list()
    )
    output2 = sdf.amount.count(start=None).stream.gather().sink_to_list()

    df = cudf.DataFrame({"name": ["Alice", "Tom"], "amount": [50, 100]})
    stream.emit(df)

    out_df0 = cudf.DataFrame({"name": ["Alice", "Tom"], "amount": [50, 100]})
    out_df1 = cudf.DataFrame(
        {"name": ["Alice", "Tom"], "amount": [50.0, 100.0]}
    )
    out_df2 = cudf.DataFrame({"name": ["Alice", "Tom"], "amount": [1, 1]})
    assert assert_eq(output0[0].reset_index(), out_df0)
    assert assert_eq(output1[0][1].reset_index(), out_df1)
    assert assert_eq(output2[0].reset_index(), out_df2)

    example = cudf.DataFrame({"name": [], "amount": []}, dtype="float64")
    sdf = DataFrame(stream, example=example).groupby(["name"])
    output3 = sdf.amount.sum(start=output0[0]).stream.gather().sink_to_list()
    output4 = (
        sdf.amount.mean(with_state=True, start=output1[0][0])
        .stream.gather()
        .sink_to_list()
    )
    output5 = sdf.amount.count(start=output2[0]).stream.gather().sink_to_list()
    df = cudf.DataFrame(
        {"name": ["Alice", "Tom", "Linda"], "amount": [50, 100, 200]}
    )
    stream.emit(df)

    out_df2 = cudf.DataFrame(
        {"name": ["Alice", "Linda", "Tom"], "amount": [100, 200, 200]}
    )
    out_df3 = cudf.DataFrame(
        {"name": ["Alice", "Linda", "Tom"], "amount": [50.0, 200.0, 100.0]}
    )
    out_df4 = cudf.DataFrame(
        {"name": ["Alice", "Linda", "Tom"], "amount": [2, 1, 2]}
    )
    assert assert_eq(output3[0].reset_index(), out_df2)
    assert assert_eq(output4[0][1].reset_index(), out_df3)
    assert assert_eq(output5[0].reset_index(), out_df4)


def test_reductions_with_start_state(stream):
    example = cudf.DataFrame({"name": [], "amount": []})
    sdf = DataFrame(stream, example=example)
    output0 = sdf.amount.mean(start=(10, 2)).stream.gather().sink_to_list()
    output1 = sdf.amount.count(start=3).stream.gather().sink_to_list()
    output2 = sdf.amount.sum(start=10).stream.gather().sink_to_list()

    df = cudf.DataFrame(
        {"name": ["Alice", "Tom", "Linda"], "amount": [50, 100, 200]}
    )
    stream.emit(df)

    assert output0[0] == 72.0
    assert output1[0] == 6
    assert output2[0] == 360


def test_rolling_aggs_with_start_state(stream):
    example = cudf.DataFrame({"name": [], "amount": []}, dtype="float64")
    sdf = DataFrame(stream, example=example)
    output0 = (
        sdf.rolling(2, with_state=True, start=())
        .amount.sum()
        .stream.gather()
        .sink_to_list()
    )

    df = cudf.DataFrame(
        {"name": ["Alice", "Tom", "Linda"], "amount": [50, 100, 200]}
    )
    stream.emit(df)
    df = cudf.DataFrame({"name": ["Bob"], "amount": [250]})
    stream.emit(df)
    assert assert_eq(
        output0[-1][0].reset_index(drop=True),
        cudf.Series([200, 250], name="amount"),
    )
    assert assert_eq(
        output0[-1][1].reset_index(drop=True),
        cudf.Series([450.0], name="amount"),
    )

    stream = Stream()
    example = cudf.DataFrame({"name": [], "amount": []}, dtype="float64")
    sdf = DataFrame(stream, example=example)
    output1 = (
        sdf.rolling(2, with_state=True, start=output0[-1][0])
        .amount.sum()
        .stream.gather()
        .sink_to_list()
    )
    df = cudf.DataFrame({"name": ["Alice"], "amount": [50]})
    stream.emit(df)
    assert assert_eq(
        output1[-1][0].reset_index(drop=True),
        cudf.Series([250, 50], name="amount"),
    )
    assert assert_eq(
        output1[-1][1].reset_index(drop=True),
        cudf.Series([300.0], name="amount"),
    )


def test_window_aggs_with_start_state(stream):
    example = cudf.DataFrame({"name": [], "amount": []}, dtype="float64")
    sdf = DataFrame(stream, example=example)
    output0 = (
        sdf.window(2, with_state=True, start=None)
        .amount.sum()
        .stream.gather()
        .sink_to_list()
    )

    df = cudf.DataFrame(
        {"name": ["Alice", "Tom", "Linda"], "amount": [50, 100, 200]}
    )
    stream.emit(df)
    df = cudf.DataFrame({"name": ["Bob"], "amount": [250]})
    stream.emit(df)
    assert output0[-1][1] == 450

    stream = Stream()
    example = cudf.DataFrame({"name": [], "amount": []}, dtype="float64")
    sdf = DataFrame(stream, example=example)
    output1 = (
        sdf.window(2, with_state=True, start=output0[-1][0])
        .amount.sum()
        .stream.gather()
        .sink_to_list()
    )
    df = cudf.DataFrame({"name": ["Alice"], "amount": [50]})
    stream.emit(df)
    assert output1[-1][1] == 300


def test_windowed_groupby_aggs_with_start_state(stream):
    example = cudf.DataFrame({"name": [], "amount": []}, dtype="float64")
    sdf = DataFrame(stream, example=example)
    output0 = (
        sdf.window(5, with_state=True, start=None)
        .groupby(["name"])
        .amount.sum()
        .stream.gather()
        .sink_to_list()
    )

    df = cudf.DataFrame(
        {"name": ["Alice", "Tom", "Linda"], "amount": [50, 100, 200]}
    )
    stream.emit(df)
    df = cudf.DataFrame(
        {"name": ["Alice", "Linda", "Bob"], "amount": [250, 300, 350]}
    )
    stream.emit(df)

    stream = Stream()
    example = cudf.DataFrame({"name": [], "amount": []}, dtype="float64")
    sdf = DataFrame(stream, example=example)
    output1 = (
        sdf.window(5, with_state=True, start=output0[-1][0])
        .groupby(["name"])
        .amount.sum()
        .stream.gather()
        .sink_to_list()
    )
    df = cudf.DataFrame(
        {
            "name": ["Alice", "Linda", "Tom", "Bob"],
            "amount": [50, 100, 150, 200],
        }
    )
    stream.emit(df)
    out_df1 = cudf.DataFrame(
        {
            "name": ["Alice", "Bob", "Linda", "Tom"],
            "amount": [50, 550, 100, 150],
        }
    )
    assert_eq(output1[-1][1].reset_index(), out_df1)
