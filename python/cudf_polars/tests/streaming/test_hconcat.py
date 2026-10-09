# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

import polars as pl

from cudf_polars.engine.options import StreamingOptions
from cudf_polars.testing.asserts import assert_gpu_result_equal


def test_user_hconcat_falls_back_for_multiple_partitions(spmd_engine_factory) -> None:
    engine = spmd_engine_factory(
        StreamingOptions(
            max_rows_per_partition=2,
            dynamic_planning=None,
            fallback_mode="raise",
            raise_on_fail=True,
        )
    )
    left = pl.LazyFrame({"a": [1, 2, 3]})
    right = pl.LazyFrame({"b": [4, 5, 6]})
    q = pl.concat([left, right], how="horizontal")

    with pytest.raises(
        NotImplementedError, match="HConcat is not supported for multiple partitions"
    ):
        assert_gpu_result_equal(q, engine=engine)
