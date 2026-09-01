# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmarks of cudf.pandas proxy argument transformation."""

import pytest

from cudf.pandas.fast_slow_proxy import _transform_arg, make_final_proxy_type


@pytest.fixture(scope="module")
def proxy_object():
    class Fast:
        def __init__(self, x):
            self.x = x

        def to_slow(self):
            return Slow(self.x)

    class Slow:
        def __init__(self, x):
            self.x = x

    Pxy = make_final_proxy_type(
        "Pxy",
        Fast,
        Slow,
        fast_to_slow=lambda fast: fast.to_slow(),
        slow_to_fast=lambda slow: Fast(slow.x),
    )
    return Pxy(1)


@pytest.mark.parametrize("size", [10, 10_000])
def bench_transform_arg_unchanged_list(benchmark, size):
    # No element needs transforming: the identity-scan returns the
    # original container without rebuilding it.
    arg = list(range(size))
    benchmark(lambda: _transform_arg(arg, "_fsproxy_slow", set()))


@pytest.mark.parametrize("size", [10, 10_000])
def bench_transform_arg_unchanged_dict(benchmark, size):
    arg = {i: i for i in range(size)}
    benchmark(lambda: _transform_arg(arg, "_fsproxy_slow", set()))


@pytest.mark.parametrize("size", [10, 10_000])
def bench_transform_arg_list_with_proxy(benchmark, proxy_object, size):
    # One proxy element forces the rebuild path.
    arg = [*range(size - 1), proxy_object]
    benchmark(lambda: _transform_arg(arg, "_fsproxy_slow", set()))


@pytest.mark.parametrize("size", [10, 10_000])
def bench_transform_arg_dict_with_proxy(benchmark, proxy_object, size):
    arg = {i: i for i in range(size - 1)}
    arg["proxy"] = proxy_object
    benchmark(lambda: _transform_arg(arg, "_fsproxy_slow", set()))
