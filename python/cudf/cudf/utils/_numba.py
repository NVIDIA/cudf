# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numba
from numba.cuda import config as numba_config
from packaging import version


def _numba_cuda_mlir_config():
    """Return numba-cuda-mlir's vendored config module, or None.

    The MLIR UDF backend consults its own vendored numba config (not stock
    numba's), so the low-occupancy performance warning must be silenced there
    too -- otherwise it fires (and, under ``filterwarnings = error``, errors)
    for the small-grid kernels cuDF UDFs launch.
    """
    try:
        from numba_cuda_mlir.numba_cuda.core import config as ncm_config
    except Exception:
        return None
    return ncm_config


# Avoids using contextlib.contextmanager due to additional overhead
class _CUDFNumbaConfig:
    def __enter__(self) -> None:
        self.CUDA_LOW_OCCUPANCY_WARNINGS = (
            numba_config.CUDA_LOW_OCCUPANCY_WARNINGS
        )
        numba_config.CUDA_LOW_OCCUPANCY_WARNINGS = 0

        self._ncm_config = _numba_cuda_mlir_config()
        if self._ncm_config is not None:
            self._ncm_low_occupancy = (
                self._ncm_config.CUDA_LOW_OCCUPANCY_WARNINGS
            )
            self._ncm_config.CUDA_LOW_OCCUPANCY_WARNINGS = 0

        self.is_numba_lt_061 = version.parse(
            numba.__version__
        ) < version.parse("0.61")

        if self.is_numba_lt_061:
            self.CAPTURED_ERRORS = numba_config.CAPTURED_ERRORS
            numba_config.CAPTURED_ERRORS = "new_style"

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        numba_config.CUDA_LOW_OCCUPANCY_WARNINGS = (
            self.CUDA_LOW_OCCUPANCY_WARNINGS
        )
        if self._ncm_config is not None:
            self._ncm_config.CUDA_LOW_OCCUPANCY_WARNINGS = (
                self._ncm_low_occupancy
            )
        if self.is_numba_lt_061:
            numba_config.CAPTURED_ERRORS = self.CAPTURED_ERRORS
