# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from cudf_polars_cucascade import ScanCache, ScanCacheOptions


@pytest.fixture
def make_options():
    """Options for a small cache against an endpoint that nothing listens on."""

    def make(**overrides):
        settings = {
            "cache_bytes": 256 << 20,
            "endpoint": "http://127.0.0.1:9",
            "region": "us-east-1",
            "access_key": "test",
            "secret_key": "test",
        }
        settings.update(overrides)
        return ScanCacheOptions(**settings)

    return make


@pytest.fixture(scope="module")
def scan_cache():
    """A scan cache. Creating one pins its memory, so tests share it."""
    cache = ScanCache(
        ScanCacheOptions(
            cache_bytes=256 << 20,
            endpoint="http://127.0.0.1:9",
            region="us-east-1",
            access_key="test",
            secret_key="test",
        )
    )
    yield cache
    del cache
