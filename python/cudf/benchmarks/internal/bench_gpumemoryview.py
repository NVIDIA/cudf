# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmarks of gpumemoryview methods."""

import pytest

import pylibcudf as plc


class CudaArray:
    def __init__(self, stream):
        self.__cuda_array_interface__ = {
            "data": (0, False),
            "shape": (10,),
            "typestr": "|u1",
            "version": 3,
        }
        if stream is not None:
            self.__cuda_array_interface__["stream"] = stream


@pytest.mark.parametrize(
    "stream", [None, 42], ids=["without_stream", "with_stream"]
)
def bench_byte_slice(benchmark, stream):
    gpumemview = plc.gpumemoryview(CudaArray(stream))
    benchmark(gpumemview.byte_slice, slice(2, 5))
