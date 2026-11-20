# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import subprocess
import sys


def test_cudf_import_invariants(monkeypatch):
    with monkeypatch.context() as m:
        # Importing cuDF must not require a visible CUDA device.
        m.setenv("CUDA_VISIBLE_DEVICES", "-1")
        output = subprocess.check_call(
            [
                sys.executable,
                "-c",
                (
                    "import cudf, sys; "
                    # Importing cuDF must not eagerly load PyArrow's optional
                    # S3 extension module.
                    "assert 'pyarrow._s3fs' not in sys.modules"
                ),
            ],
            cwd="/",
        )
    assert output == 0
