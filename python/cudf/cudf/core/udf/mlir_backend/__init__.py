# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MLIR / numba_cuda_mlir UDF backend.

Typing/lowering modules are registered by ``cudf.core.udf.__init__``, not
here, to avoid circular imports through ``utils.py``.
"""
