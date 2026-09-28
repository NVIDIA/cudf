# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

import polars as pl

from cudf_polars import Translator
from cudf_polars.dsl import ir
from cudf_polars.dsl.ir import IRExecutionContext
from cudf_polars.dsl.traversal import traversal
from cudf_polars.testing.asserts import assert_gpu_result_equal


@pytest.mark.xfail(reason="python no longer manages cache hits")
def test_cache(engine: pl.GPUEngine):
    df1 = pl.LazyFrame(
        {
            "a": [1, 2, 3, 4, 5, 6, 7],
            "b": [1, 1, 1, 1, 1, 1, 1],
        }
    )
    df2 = pl.LazyFrame({"a": [7, 8], "b": [12, 13]})

    q = pl.concat([df1, df2, df1, df2, df1])
    assert_gpu_result_equal(q, engine=engine)

    t = Translator(q._ldf.visit(), pl.GPUEngine())
    qir = t.translate_ir()
    # There should be two unique cache nodes.
    cache_df2, cache_df1 = (
        node for node in traversal([qir]) if isinstance(node, ir.Cache)
    )
    assert cache_df1 != cache_df2
    assert not cache_df1.is_equal(cache_df2)
    assert cache_df1 == qir.children[0]
    assert cache_df2 == qir.children[1]
    assert cache_df1 == qir.children[2]
    assert cache_df2 == qir.children[3]
    assert cache_df1 == qir.children[4]

    class HitCounter(dict):
        def __init__(self, *args, **kwargs):
            self.hits = 0
            super().__init__(*args, **kwargs)

        def __getitem__(self, key):
            result = super().__getitem__(key)
            self.hits += 1
            return result

    node_cache = HitCounter()
    qir.evaluate(
        cache=node_cache,
        timer=None,
        context=IRExecutionContext(),
    )
    assert len(node_cache) == 0
    assert node_cache.hits == 3


def test_union_cache_nodes():
    df = pl.LazyFrame({"a": [7, 8], "b": [12, 13]})
    q = pl.concat([df, df])
    qir = Translator(q._ldf.visit(), pl.GPUEngine()).translate_ir()
    # Logical plan:
    # UNION ('x', 'y', 'z')
    #   CACHE ('x', 'y', 'z')
    #     PROJECTION ('x', 'y', 'z')
    #       DATAFRAMESCAN ('x', 'y', 'z')
    #   (repeated 2 times)

    # Check that the concatenated Cache nodes are the same object
    # See: https://github.com/NVIDIA/cudf/issues/19766
    assert isinstance(qir, ir.Union)
    assert isinstance(qir.children[0], ir.Cache)
    assert isinstance(qir.children[1], ir.Cache)
    assert hash(qir.children[0]) == hash(qir.children[1])
    assert hash(qir.children[0].children[0]) == hash(qir.children[1].children[0])
