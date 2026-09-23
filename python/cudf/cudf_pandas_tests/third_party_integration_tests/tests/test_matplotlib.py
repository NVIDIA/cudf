# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.axes import Axes
from matplotlib.collections import PathCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from pandas._testing import assert_equal


def assert_plots_equal(expect, got):
    if isinstance(expect, Axes) and isinstance(got, Axes):
        for expect_ch, got_ch in zip(
            expect.get_children(), got.get_children(), strict=True
        ):
            assert type(expect_ch) is type(got_ch)
            if isinstance(expect_ch, Line2D):
                assert_equal(expect_ch.get_xdata(), got_ch.get_xdata())
                assert_equal(expect_ch.get_ydata(), got_ch.get_ydata())
            elif isinstance(expect_ch, Rectangle):
                assert expect_ch.get_height() == got_ch.get_height()
    elif isinstance(expect, PathCollection) and isinstance(
        got, PathCollection
    ):
        assert_equal(expect.get_offsets()[:, 0], got.get_offsets()[:, 0])
        assert_equal(expect.get_offsets()[:, 1], got.get_offsets()[:, 1])
    else:
        assert_equal(expect, got)


pytestmark = pytest.mark.assert_eq(fn=assert_plots_equal)


def test_line():
    df = pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": [2, 4, 6, 8, 10]})
    (_data,) = plt.plot(df["x"], df["y"], marker="o", linestyle="-")

    return plt.gca()


def test_bar():
    data = pd.Series([1, 2, 3, 4, 5], index=["a", "b", "c", "d", "e"])
    ax = data.plot(kind="bar")
    return ax


def test_scatter():
    df = pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": [5, 4, 3, 2, 1]})

    _fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df["x"], df["y"])

    return plt.gca()


def test_dataframe_plot():
    rng = np.random.default_rng(42)
    df = pd.DataFrame(rng.random((10, 5)), columns=["a", "b", "c", "d", "e"])
    ax = df.plot()

    return ax


def test_series_plot():
    sr = pd.Series([1, 2, 3, 4, 5])
    ax = sr.plot()

    return ax
