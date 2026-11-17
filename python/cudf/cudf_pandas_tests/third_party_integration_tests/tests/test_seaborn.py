# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import matplotlib.pyplot as plt
import pandas as pd
import pytest
import seaborn as sns


def assert_plots_equal(expect, got):
    # these are the coordinates of the matplotlib objects.
    assert expect == got


pytestmark = pytest.mark.assert_eq(fn=assert_plots_equal)


@pytest.fixture(autouse=True)
def isolated_figures():
    # Existing axes can make a later test inspect collections from an earlier plot.
    plt.close("all")
    yield
    plt.close("all")


@pytest.fixture(scope="module")
def df():
    df = pd.DataFrame(
        {
            "x": [2, 3, 4, 5, 11],
            "y": [4, 3, 2, 1, 15],
            "hue": ["c", "a", "b", "b", "a"],
        }
    )
    return df


def test_bar(df):
    ax = sns.barplot(data=df, x="x", y="y")
    return [x.get_height().item() for x in ax.patches]


def test_scatter(df):
    ax = sns.scatterplot(data=df, x="x", y="y", hue="hue")
    assert len(ax.collections) == 1
    paths = ax.collections[0].get_paths()
    assert len(paths) == 1
    return paths[0].vertices.tolist()


def test_lineplot_with_sns_data():
    df = sns.load_dataset("flights")
    ax = sns.lineplot(data=df, x="month", y="passengers", seed=0)
    paths = ax.collections[0].get_paths()
    assert len(paths) == 1
    return paths[0].vertices.tolist()
