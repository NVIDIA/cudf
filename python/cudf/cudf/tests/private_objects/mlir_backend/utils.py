# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from numba_cuda_mlir.numba_cuda.core import config


class MLIRNumbaCudaConfig:
    def __enter__(self) -> None:
        self._low_occupancy_warnings = config.CUDA_LOW_OCCUPANCY_WARNINGS
        config.CUDA_LOW_OCCUPANCY_WARNINGS = 0

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        config.CUDA_LOW_OCCUPANCY_WARNINGS = self._low_occupancy_warnings
