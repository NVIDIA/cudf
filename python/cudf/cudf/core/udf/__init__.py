# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
UDF compilation support.  Typing/lowering registration is loaded from the
MLIR/numba_cuda_mlir backend.
"""

from cudf.core.udf.mlir_backend import (
    cooperative_lowering,
    cooperative_typing,
    groupby_lowering,
    groupby_typing,
    masked_lowering,
    masked_typing,
    strings_lowering,
    strings_typing,
)
