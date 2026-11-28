# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from cudf_polars_cucascade import ScanCache, ScanCacheConfig


def pytest_configure(config):
    # As in cudf-polars's own tests/conftest.py: Ray leaks `/dev/null` handles and
    # distributed leaves sockets open at shutdown, which would fail the engine tests
    # under `filterwarnings = ["error"]`; Ray also uses a deprecated OpenTelemetry API.
    config.addinivalue_line("filterwarnings", "ignore::ResourceWarning")
    config.addinivalue_line("filterwarnings", "ignore::DeprecationWarning")


@pytest.fixture
def make_config():
    """Config for a small cache against an endpoint that nothing listens on."""

    def make(**overrides):
        settings = {
            "cache_bytes": 256 << 20,
            "endpoint": "http://127.0.0.1:9",
            "region": "us-east-1",
            "access_key": "test",
            "secret_key": "test",
        }
        settings.update(overrides)
        return ScanCacheConfig(**settings)

    return make


@pytest.fixture(scope="module")
def scan_cache():
    """A scan cache. Creating one pins its memory, so tests share it."""
    cache = ScanCache(
        ScanCacheConfig(
            cache_bytes=256 << 20,
            endpoint="http://127.0.0.1:9",
            region="us-east-1",
            access_key="test",
            secret_key="test",
        )
    )
    yield cache
    del cache
