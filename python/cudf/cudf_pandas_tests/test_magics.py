# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pathlib
import subprocess
import sys

import pytest


# Proxy check for whether the proxy could be referencing GPU objects without
# trying to import cuDF, which could poison the global environment
def _gpu_available():
    try:
        import rmm

        return rmm._cuda.gpu.getDeviceCount() >= 1
    except ImportError:
        return False


LOCATION = pathlib.Path(__file__).absolute().parent


@pytest.mark.skipif(
    not _gpu_available(), reason="Skipping test if a GPU isn't available."
)
def test_magics_gpu():
    sp_completed = subprocess.run(
        [sys.executable, LOCATION / "_magics_gpu_test.py"], capture_output=True
    )
    assert sp_completed.stderr.decode() == ""
