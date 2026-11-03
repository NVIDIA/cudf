# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

"""Benchmarks of DataFrame methods."""

import string

import numpy
import pandas as pd
import pyarrow as pa
import pytest
import pytest_cases
from config import NUM_COLS, NUM_ROWS, cudf, cupy
from utils import benchmark_with_object


@pytest.mark.parametrize("num_rows", NUM_ROWS)
@pytest.mark.parametrize("num_cols", NUM_COLS)
@pytest.mark.parametrize(
    "values_constructor",
    [
        range,
        cupy.random.default_rng(2).standard_normal,
        numpy.random.default_rng(2).standard_normal,
        lambda num_cols: pd.Series(range(num_cols)),
        lambda num_cols: cudf.Series(range(num_cols)),
    ],
    ids=["range", "cupy", "numpy", "pandas", "cudf"],
)
@pytest.mark.parametrize(
    "index",
    [
        lambda num_rows: None,
        lambda num_rows: range(num_rows),
        lambda num_rows: pd.RangeIndex(num_rows),
        lambda num_rows: cudf.RangeIndex(num_rows),
    ],
    ids=["None", "range", "pandas-index", "cudf-index"],
)
@pytest.mark.parametrize(
    "columns",
    [
        lambda num_cols: None,
        lambda num_cols: range(num_cols),
        lambda num_cols: pd.Index(numpy.arange(num_cols)),
        lambda num_cols: cudf.Index(numpy.arange(num_cols)),
    ],
    ids=["None", "range", "pandas-index", "cudf-index"],
)
@pytest.mark.parametrize("dtype", [None, "float32"])
@pytest.mark.pandas_incompatible
def bench_construction_with_mapping(
    benchmark, num_rows, num_cols, values_constructor, index, columns, dtype
):
    benchmark(
        cudf.DataFrame,
        data={i: values_constructor(num_rows) for i in range(num_cols)},
        columns=columns(num_cols),
        index=index(num_rows),
        dtype=dtype,
    )


@pytest.mark.parametrize("num_rows", NUM_ROWS)
@pytest.mark.parametrize("num_cols", NUM_COLS)
@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize(
    "array_constructor",
    [
        lambda num_rows, num_cols, order: numpy.zeros(
            (num_rows, num_cols), order=order
        ),
        lambda num_rows, num_cols, order: cupy.zeros(
            (num_rows, num_cols), order=order
        ),
    ],
    ids=["numpy", "cupy"],
)
@pytest.mark.parametrize(
    "columns",
    [
        lambda num_rows: None,
        lambda num_rows: range(num_rows),
        lambda num_rows: pd.RangeIndex(num_rows),
        lambda num_rows: cudf.Index(numpy.arange(num_rows)),
    ],
    ids=["None", "range", "pandas-index", "cudf-index"],
)
@pytest.mark.parametrize(
    "index",
    [
        lambda num_cols: None,
        lambda num_cols: range(num_cols),
        lambda num_cols: pd.Index(numpy.arange(num_cols)),
        lambda num_cols: cudf.Index(numpy.arange(num_cols)),
    ],
    ids=["None", "range", "pandas-index", "cudf-index"],
)
@pytest.mark.parametrize("dtype", [None, "float32"])
@pytest.mark.pandas_incompatible
def bench_construction_with_array(
    benchmark,
    num_rows,
    num_cols,
    order,
    array_constructor,
    columns,
    index,
    dtype,
):
    benchmark(
        cudf.DataFrame,
        data=array_constructor(num_rows, num_cols, order),
        columns=columns(num_cols),
        index=index(num_rows),
        dtype=dtype,
    )


@pytest.mark.parametrize("num_rows", NUM_ROWS)
@pytest.mark.parametrize(
    "frame_index",
    [
        lambda num_rows: None,
        lambda num_rows: range(num_rows - 1, -1, -1),
    ],
    ids=["None", "reverse-range"],
)
@pytest.mark.parametrize(
    "framelike",
    [
        lambda num_rows, frame_index: pd.Series(range(num_rows), frame_index),
        lambda num_rows, frame_index: cudf.Series(
            range(num_rows), frame_index
        ),
        lambda num_rows, frame_index: pd.DataFrame(
            numpy.zeros((num_rows, 50)), frame_index
        ),
        lambda num_rows, frame_index: cudf.DataFrame(
            numpy.zeros((num_rows, 50)), frame_index
        ),
    ],
    ids=["pandas-series", "cudf-series", "pandas-frame", "cudf-frame"],
)
@pytest.mark.parametrize(
    "index",
    [
        lambda num_rows: None,
        lambda num_rows: range(num_rows),
        lambda num_rows: pd.Index(numpy.arange(num_rows)),
        lambda num_rows: cudf.Index(numpy.arange(num_rows)),
    ],
    ids=["None", "range", "pandas-index", "cudf-index"],
)
@pytest.mark.parametrize("dtype", [None, "float32"])
@pytest.mark.pandas_incompatible
def bench_construction_with_framelike(
    benchmark, framelike, num_rows, frame_index, index, dtype
):
    benchmark(
        cudf.DataFrame,
        data=framelike(num_rows, frame_index(num_rows)),
        index=index(num_rows),
        dtype=dtype,
    )


@pytest.mark.parametrize("N", NUM_ROWS)
def bench_from_arrow(benchmark, N):
    rng = numpy.random.default_rng(seed=10)
    benchmark(cudf.DataFrame, {None: pa.array(rng.random(N))})


@pytest.mark.parametrize("N", NUM_ROWS)
def bench_construction(benchmark, N):
    benchmark(cudf.DataFrame, {None: cupy.random.rand(N)})


@benchmark_with_object(cls="dataframe", dtype="float", cols=6)
@pytest.mark.parametrize(
    "expr", ["a+b", "a+b+c+d+e", "a / (sin(a) + cos(b)) * tanh(d*e*f)"]
)
def bench_eval_func(benchmark, expr, dataframe):
    benchmark(dataframe.eval, expr)


@benchmark_with_object(cls="dataframe", dtype="int", nulls=False, cols=6)
@pytest.mark.parametrize(
    "num_key_cols",
    [2, 3, 4],
)
def bench_merge(benchmark, dataframe, num_key_cols):
    benchmark(
        dataframe.merge, dataframe, on=list(dataframe.columns[:num_key_cols])
    )


# TODO: Some of these cases could be generalized to an IndexedFrame benchmark
# instead of a DataFrame benchmark.
@benchmark_with_object(cls="dataframe", dtype="int")
@pytest.mark.parametrize(
    "values",
    [
        lambda: range(50),
        lambda: {f"{string.ascii_lowercase[i]}": range(50) for i in range(10)},
        lambda: cudf.DataFrame(
            {f"{string.ascii_lowercase[i]}": range(50) for i in range(10)}
        ),
        lambda: cudf.Series(range(50)),
    ],
)
def bench_isin(benchmark, dataframe, values):
    benchmark(dataframe.isin, values())


@pytest.fixture(
    params=[0, numpy.random.RandomState, cupy.random.RandomState],
    ids=["Seed", "NumpyRandomState", "CupyRandomState"],
)
def random_state(request):
    rs = request.param
    return rs if isinstance(rs, int) else rs(seed=42)


@benchmark_with_object(cls="dataframe", dtype="int")
@pytest.mark.parametrize("frac", [0.5])
def bench_sample(benchmark, dataframe, axis, frac, random_state):
    if axis == 1 and isinstance(random_state, cupy.random.RandomState):
        pytest.skip("Unsupported params.")
    benchmark(
        dataframe.sample, frac=frac, axis=axis, random_state=random_state
    )


@benchmark_with_object(cls="dataframe", dtype="int")
@pytest.mark.parametrize("frac", [0, 0.25, 0.5, 0.75, 1])
def bench_iloc_getitem_indices(benchmark, dataframe, frac):
    rs = numpy.random.RandomState(seed=42)
    n = int(len(dataframe) * frac)
    values = rs.choice(len(dataframe), size=n, replace=False)
    benchmark(dataframe.iloc.__getitem__, values)


@benchmark_with_object(cls="dataframe", dtype="int")
@pytest.mark.parametrize("frac", [0, 0.25, 0.5, 0.75, 1])
def bench_iloc_getitem_mask(benchmark, dataframe, frac):
    rs = numpy.random.RandomState(seed=42)
    n = int(len(dataframe) * frac)
    values = rs.choice(len(dataframe), size=n, replace=False)
    mask = numpy.zeros(len(dataframe), dtype=bool)
    mask[values] = True
    benchmark(dataframe.iloc.__getitem__, mask)


@benchmark_with_object(cls="dataframe", dtype="int")
@pytest.mark.parametrize(
    "slice",
    [slice(None), slice(0, 0, 1), slice(1, None, 10), slice(None, -1, -1)],
)
def bench_iloc_getitem_slice(benchmark, dataframe, slice):
    benchmark(dataframe.iloc.__getitem__, slice)


@benchmark_with_object(cls="dataframe", dtype="int")
def bench_iloc_getitem_scalar(benchmark, dataframe):
    benchmark(dataframe.iloc.__getitem__, len(dataframe) // 2)


@benchmark_with_object(cls="dataframe", dtype="int", nulls=False, cols=6)
@pytest.mark.parametrize(
    "num_key_cols",
    [2, 3, 4],
)
def bench_groupby(benchmark, dataframe, num_key_cols):
    benchmark(dataframe.groupby, by=list(dataframe.columns[:num_key_cols]))


@benchmark_with_object(cls="dataframe", dtype="int", nulls=False, cols=6)
@pytest.mark.parametrize(
    "agg",
    [
        "sum",
        ["sum", "mean"],
        {
            f"{string.ascii_lowercase[i]}": ["sum", "mean", "count"]
            for i in range(6)
        },
    ],
)
@pytest.mark.parametrize(
    "num_key_cols",
    [2, 3, 4],
)
@pytest.mark.parametrize("as_index", [True, False])
@pytest.mark.parametrize("sort", [True, False])
def bench_groupby_agg(benchmark, dataframe, agg, num_key_cols, as_index, sort):
    by = list(dataframe.columns[:num_key_cols])
    benchmark(dataframe.groupby(by=by, as_index=as_index, sort=sort).agg, agg)


@benchmark_with_object(cls="dataframe", dtype="int", nulls=False, cols=6)
@pytest.mark.parametrize(
    "num_key_cols",
    [2, 3, 4],
)
@pytest.mark.parametrize("use_frac", [True, False])
@pytest.mark.parametrize("replace", [True, False])
@pytest.mark.parametrize("target_sample_frac", [0.1, 0.5, 1])
def bench_groupby_sample(
    benchmark, dataframe, num_key_cols, use_frac, replace, target_sample_frac
):
    grouper = dataframe.groupby(by=list(dataframe.columns[:num_key_cols]))
    if use_frac:
        kwargs = {"frac": target_sample_frac, "replace": replace}
    else:
        minsize = grouper.size().min()
        target_size = numpy.round(
            target_sample_frac * minsize, decimals=0
        ).astype(int)
        kwargs = {"n": target_size, "replace": replace}

    benchmark(grouper.sample, **kwargs)


@benchmark_with_object(cls="dataframe", dtype="float", nulls=False, cols=6)
@pytest.mark.parametrize("num_key_cols", [1, 2])
@pytest.mark.parametrize("q", [0.5, [0.25, 0.5, 0.75]])
def bench_groupby_quantile(benchmark, dataframe, num_key_cols, q):
    grouper = dataframe.groupby(by=list(dataframe.columns[:num_key_cols]))
    benchmark(grouper.quantile, q)


@benchmark_with_object(cls="dataframe", dtype="int")
@pytest.mark.parametrize("num_cols_to_sort", [1])
def bench_sort_values(benchmark, dataframe, num_cols_to_sort):
    benchmark(
        dataframe.sort_values, list(dataframe.columns[:num_cols_to_sort])
    )


@benchmark_with_object(cls="dataframe", dtype="int")
@pytest.mark.parametrize("num_cols_to_sort", [1])
@pytest.mark.parametrize("n", [10])
def bench_nsmallest(benchmark, dataframe, num_cols_to_sort, n):
    by = list(dataframe.columns[:num_cols_to_sort])
    benchmark(dataframe.nsmallest, n, by)


@pytest_cases.parametrize_with_cases(
    "dataframe, cond, other", prefix="where", cases="cases_dataframe"
)
def bench_where(benchmark, dataframe, cond, other):
    benchmark(dataframe.where, cond, other)


@benchmark_with_object(
    cls="dataframe", dtype="float", nulls=False, cols=20, rows=100
)
@pytest.mark.pandas_incompatible
def bench_to_cupy(benchmark, dataframe):
    benchmark(dataframe.to_cupy)
