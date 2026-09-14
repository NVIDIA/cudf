# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import pyarrow as pa
import pytest
from cuda.bindings import runtime

import pylibcudf as plc


@pytest.mark.parametrize(
    "stream",
    [
        None,
        runtime.cudaStream_t(runtime.cudaStreamDefault),
        runtime.cudaStream_t(runtime.cudaStreamLegacy),
        runtime.cudaStream_t(runtime.cudaStreamPerThread),
    ],
)
def test_empty_like_on_fresh_thread(stream):
    column = plc.Column.from_arrow(pa.array([1, 2, 3]))

    def empty_like():
        result = plc.copying.empty_like(column, stream=stream)
        assert result.size() == 0
        if stream is not None:
            assert plc.utils._get_stream(stream).__cuda_stream__() == (
                0,
                int(stream),
            )

    with ThreadPoolExecutor(max_workers=1) as pool:
        pool.submit(empty_like).result()


def test_get_stream_preserves_current_device():
    status, count = runtime.cudaGetDeviceCount()
    assert status == runtime.cudaError_t.cudaSuccess

    def get_stream(device):
        assert runtime.cudaSetDevice(device) == (
            runtime.cudaError_t.cudaSuccess,
        )
        status, stream = runtime.cudaStreamCreate()
        assert status == runtime.cudaError_t.cudaSuccess
        try:
            plc.utils._get_stream()
            assert plc.utils._get_stream(stream).__cuda_stream__() == (
                0,
                int(stream),
            )
            assert runtime.cudaGetDevice() == (
                runtime.cudaError_t.cudaSuccess,
                device,
            )
        finally:
            assert runtime.cudaStreamDestroy(stream) == (
                runtime.cudaError_t.cudaSuccess,
            )

    with ThreadPoolExecutor(max_workers=1) as pool:
        for device in range(count):
            pool.submit(get_stream, device).result()


def test_get_stream_rejects_integer_handle():
    with pytest.raises(TypeError):
        plc.utils._get_stream(runtime.cudaStreamDefault)


def test_empty_like_initializes_cuda(tmp_path):
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import pylibcudf as plc

column = plc.Column(plc.DataType(plc.TypeId.INT32), 0, None, None, 0, 0, [])
assert plc.copying.empty_like(column).size() == 0
""",
        ],
        check=True,
        cwd=tmp_path,
    )
