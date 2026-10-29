# SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as pd

import cudf

__all__ = ["randomdata", "timeseries"]


# TODO:
# change default of name from category to str type when nvstring are merged
def timeseries(
    start="2000-01-01",
    end="2000-01-31",
    freq="1s",
    dtypes=None,
    nulls_frequency=0,
    seed=None,
):
    """Create timeseries dataframe with random data

    Parameters
    ----------
    start : datetime (or datetime-like string)
        Start of time series
    end : datetime (or datetime-like string)
        End of time series
    dtypes : dict
        Mapping of column names to types.
        Valid types include {float, int, str, 'category'}.
        If none is provided, this defaults to
        ``{"name": "category", "id": int, "x": float, "y": float}``
    freq : string
        String like '2s' or '1H' or '12W' for the time series frequency
    nulls_frequency : float
        Fill the series with the specified proportion of nulls. Default is 0.
    seed : int (optional)
        Randomstate seed

    Examples
    --------
    >>> import cudf
    >>> gdf = cudf.datasets.timeseries()
    >>> gdf.head()  # doctest: +SKIP
              timestamp    id     name         x         y
    2000-01-01 00:00:00   967    Jerry -0.031348 -0.040633
    2000-01-01 00:00:01  1066  Michael -0.262136  0.307107
    2000-01-01 00:00:02   988    Wendy -0.526331  0.128641
    2000-01-01 00:00:03  1016   Yvonne  0.620456  0.767270
    2000-01-01 00:00:04   998   Ursula  0.684902 -0.463278
    """
    if dtypes is None:
        dtypes = {"name": "category", "id": int, "x": float, "y": float}

    index = pd.DatetimeIndex(
        pd.date_range(start, end, freq=freq, name="timestamp")
    )
    state = np.random.RandomState(seed)
    columns = {k: make[dt](len(index), state) for k, dt in dtypes.items()}
    df = pd.DataFrame(columns, index=index, columns=sorted(columns))
    if len(df) > 0 and df.index[-1] == end:
        df = df.iloc[:-1]

    gdf = cudf.from_pandas(df)
    if len(gdf) == 0:
        return gdf
    for col in gdf:
        mask = state.choice(
            [False, True],
            size=len(gdf),
            p=[1 - nulls_frequency, nulls_frequency],
        )
        gdf.loc[mask, col] = None
    return gdf


def randomdata(nrows=10, dtypes=None, seed=None):
    """Create a dataframe with random data

    Parameters
    ----------
    nrows : int
        number of rows in the dataframe
    dtypes : dict
        Mapping of column names to types.
        Valid types include {float, int, str, 'category'}
        If none is provided, this defaults to
        ``{"id": int, "x": float, "y": float}``
    seed : int (optional)
        Randomstate seed

    Examples
    --------
    >>> import cudf
    >>> gdf = cudf.datasets.randomdata()
    >>> cdf.head()  # doctest: +SKIP
            id                  x                   y
    0  1014 0.28361267466770146 -0.44274170661264334
    1  1026 -0.9937981936047235 -0.09433464773262323
    2  1038 -0.1266722796765325 0.20971126368240123
    3  1002 0.9280495300010041  0.5137701393017848
    4   976 0.9089527839187654  0.9881063385586304
    """
    if dtypes is None:
        dtypes = {"id": int, "x": float, "y": float}
    state = np.random.RandomState(seed)
    columns = {k: make[dt](nrows, state) for k, dt in dtypes.items()}
    df = pd.DataFrame(columns, columns=sorted(columns))
    return cudf.from_pandas(df)


def make_float(n, rstate):
    return rstate.rand(n) * 2 - 1


def make_int(n, rstate):
    return rstate.poisson(1000, size=n)


names = [
    "Alice",
    "Bob",
    "Charlie",
    "Dan",
    "Edith",
    "Frank",
    "George",
    "Hannah",
    "Ingrid",
    "Jerry",
    "Kevin",
    "Laura",
    "Michael",
    "Norbert",
    "Oliver",
    "Patricia",
    "Quinn",
    "Ray",
    "Sarah",
    "Tim",
    "Ursula",
    "Victor",
    "Wendy",
    "Xavier",
    "Yvonne",
    "Zelda",
]


def make_string(n, rstate):
    return rstate.choice(names, size=n)


def make_categorical(n, rstate):
    return pd.Categorical.from_codes(
        rstate.randint(0, len(names), size=n), names
    )


def make_bool(n, rstate):
    return rstate.choice([True, False], size=n)


make = {
    float: make_float,
    int: make_int,
    str: make_string,
    object: make_string,
    "category": make_categorical,
    bool: make_bool,
}
