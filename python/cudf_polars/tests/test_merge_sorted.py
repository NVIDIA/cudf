# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

import polars as pl

from cudf_polars.testing.asserts import (
    assert_gpu_result_equal,
    assert_ir_translation_raises,
)
from cudf_polars.testing.engine_utils import is_streaming_engine
from cudf_polars.utils.versions import POLARS_VERSION_LT_143


@pytest.mark.parametrize("descending", [True, False])
def test_merge_sorted_without_nulls(engine: pl.GPUEngine, descending, request):
    request.applymarker(
        pytest.mark.xfail(
            not is_streaming_engine(engine) and descending,
            reason="https://github.com/pola-rs/polars/issues/21511",
        )
    )
    df0 = pl.LazyFrame(
        {"name": ["steve", "elise", "bob"], "age": [42, 44, 18], "height": [5, 6, 5]}
    ).sort("age", descending=descending)
    df1 = pl.LazyFrame(
        {
            "name": ["anna", "megan", "steve", "thomas"],
            "age": [21, 33, 42, 20],
            "height": [5, 5, 5, 5],
        }
    ).sort("age", descending=descending)
    q = df0.merge_sorted(df1, key="age")
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.skipif(
    POLARS_VERSION_LT_143, reason="Merging on multiple keys added in polars 1.43"
)
def test_merge_sorted_multiple_keys_not_supported(engine: pl.GPUEngine):
    df0 = pl.LazyFrame(
        {"name": ["steve", "elise", "bob"], "age": [42, 44, 18], "height": [5, 6, 5]}
    ).sort(["age", "height"])
    df1 = pl.LazyFrame(
        {
            "name": ["anna", "megan", "steve", "thomas"],
            "age": [21, 33, 42, 20],
            "height": [5, 5, 5, 5],
        }
    ).sort(["age", "height"])
    q = df0.merge_sorted(df1, key=["age", "height"])
    assert_ir_translation_raises(q, engine, NotImplementedError)


@pytest.mark.parametrize(
    "descending",
    [
        pytest.param(
            True,
            marks=pytest.mark.xfail(reason="polars/issues/21511"),
        ),
        False,
    ],
)
def test_merge_sorted_with_nulls(engine: pl.GPUEngine, descending):
    df0 = pl.LazyFrame(
        {
            "name": ["steve", "elise", "bob", "john"],
            "age": [42, 44, 18, None],
            "height": [5, 6, 7, 5],
        }
    ).sort("age", descending=descending)
    df1 = pl.LazyFrame(
        {
            "name": ["anna", "megan", "steve", "thomas", "john"],
            "age": [21, 33, 42, 20, None],
            "height": [5, 5, 5, 5, 5],
        }
    ).sort("age", descending=descending)
    q = df0.merge_sorted(df1, key="age")
    assert_gpu_result_equal(q, engine=engine)
