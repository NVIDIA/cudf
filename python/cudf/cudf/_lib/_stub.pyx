# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Intentionally empty.
#
# cuDF's former Cython/C++ (its ``_lib`` bindings and the ``strings_udf`` /
# ``cudf_strings_udf`` UDF shim) has moved to pylibcudf, and the device UDF
# path is now pure MLIR (numba-cuda-mlir), so the ``cudf`` package no longer
# contains any real compiled code. This trivial, dependency-free extension is
# built only so the ``cudf`` wheel is still produced as a platform wheel.
# It can be removed once ``cudf`` is packaged as a pure-Python wheel.
