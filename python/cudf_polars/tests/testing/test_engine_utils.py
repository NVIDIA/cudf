# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from cudf_polars.testing.engine_utils import (
    SMALL_MAX_ROWS_PER_PARTITION,
    SMALL_TARGET_PARTITION_SIZE,
    EngineFixtureParam,
    create_streaming_options,
    merge_streaming_options,
)


def test_engine_fixture_param_in_memory():
    param = EngineFixtureParam("in-memory")
    assert param.engine_name == "in-memory"
    assert param.blocksize_mode == "medium"


def test_engine_fixture_param_medium_blocksize():
    param = EngineFixtureParam("spmd")
    assert param.engine_name == "spmd"
    assert param.blocksize_mode == "medium"


def test_engine_fixture_param_small_blocksize():
    param = EngineFixtureParam("spmd-small")
    assert param.engine_name == "spmd"
    assert param.blocksize_mode == "small"


def test_create_streaming_options_medium():
    opts = create_streaming_options("medium")
    assert opts.max_rows_per_partition == 50
    assert opts.target_partition_size == 1_000_000
    assert opts.raise_on_fail is True


def test_create_streaming_options_small():
    opts = create_streaming_options("small")
    assert opts.max_rows_per_partition == SMALL_MAX_ROWS_PER_PARTITION
    assert opts.target_partition_size == SMALL_TARGET_PARTITION_SIZE


def test_create_streaming_options_overrides_merge():
    """Overrides take precedence over the blocksize baseline."""
    from cudf_polars.engine.options import StreamingOptions

    overrides = StreamingOptions(max_rows_per_partition=999)
    merged = merge_streaming_options(create_streaming_options("medium"), overrides)
    # Override wins.
    assert merged.max_rows_per_partition == 999
    # Untouched baseline field is preserved.
    assert merged.target_partition_size == 1_000_000
