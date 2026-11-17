# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import cupy as cp
import numpy as np
import pandas as pd
import pytest
from cuml.cluster import KMeans
from cuml.decomposition import PCA
from cuml.ensemble import RandomForestClassifier
from cuml.linear_model import LinearRegression, LogisticRegression
from cuml.metrics import accuracy_score
from cuml.model_selection import train_test_split
from cuml.pipeline import Pipeline
from cuml.preprocessing import StandardScaler


def assert_cuml_equal(expect, got):
    # Coerce GPU arrays to CPU
    if isinstance(expect, cp.ndarray):
        expect = expect.get()
    if isinstance(got, cp.ndarray):
        got = got.get()

    # Handle equality
    if isinstance(expect, KMeans) and isinstance(got, KMeans):
        # same clusters
        np.testing.assert_allclose(
            expect.cluster_centers_, got.cluster_centers_
        )
    elif isinstance(expect, np.ndarray) and isinstance(got, np.ndarray):
        np.testing.assert_allclose(expect, got)
    elif isinstance(expect, tuple) and isinstance(got, tuple):
        assert len(expect) == len(got)
        for e, g in zip(expect, got, strict=True):
            assert_cuml_equal(e, g)
    elif isinstance(expect, pd.DataFrame):
        assert pd.testing.assert_frame_equal(expect, got)
    elif isinstance(expect, pd.Series):
        assert pd.testing.assert_series_equal(expect, got)
    else:
        assert expect == got


pytestmark = pytest.mark.assert_eq(fn=assert_cuml_equal)


@pytest.fixture
def binary_classification_data():
    data = {
        "feature1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        "feature2": [2.0, 4.0, 1.0, 3.0, 5.0, 7.0, 6.0, 8.0, 10.0, 9.0],
        "target": [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    }
    df = pd.DataFrame(data)
    return df


def test_linear_regression():
    lr = LinearRegression(fit_intercept=True, algorithm="eig")
    X = pd.DataFrame()
    X["col1"] = np.array([1, 1, 2, 2], dtype=np.float32)
    X["col2"] = np.array([1, 2, 2, 3], dtype=np.float32)
    y = pd.Series(np.array([6.0, 8.0, 9.0, 11.0], dtype=np.float32))
    lr.fit(X, y)

    X_new = pd.DataFrame()
    X_new["col1"] = np.array([3, 2], dtype=np.float32)
    X_new["col2"] = np.array([5, 5], dtype=np.float32)
    preds = lr.predict(X_new)
    return preds.values


def test_logistic_regression(binary_classification_data):
    X = binary_classification_data[["feature1", "feature2"]]
    y = binary_classification_data["target"]

    (X_train, X_test, y_train, y_test) = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return accuracy


def test_random_forest(binary_classification_data):
    X = binary_classification_data[["feature1", "feature2"]]
    y = binary_classification_data["target"]

    X_train, X_test, y_train, _y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = RandomForestClassifier(
        n_estimators=100, n_bins=len(X_train), max_depth=16
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return preds.values


def test_clustering():
    rng = np.random.default_rng(42)
    nsamps = 300
    X = rng.random(size=(nsamps, 2))
    data = pd.DataFrame(X, columns=["x", "y"])

    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(data)
    return kmeans


def test_data_scaling():
    data = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    scaler = StandardScaler()

    scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))
    return scaled_data


def test_pipeline(binary_classification_data):
    X = binary_classification_data[["feature1", "feature2"]]
    y = binary_classification_data["target"]

    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("pca", PCA()),
            ("random_forest", LogisticRegression()),
        ]
    )

    pipe.fit(X, y)
    results = pipe.predict(X)
    return results.values


@pytest.mark.parametrize(
    "X, y",
    [
        (pd.DataFrame({"a": range(10), "b": range(10)}), pd.Series(range(10))),
        (
            pd.DataFrame({"a": range(10), "b": range(10)}).values,
            pd.Series(range(10)).values,
        ),  # cudf.pandas wrapped numpy arrays
    ],
)
def test_train_test_split(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    # Compare only the size of the data splits
    return len(X_train), len(X_test), len(y_train), len(y_test)
