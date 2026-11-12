# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

# cuDF's device UDF path is now pure MLIR (numba-cuda-mlir); its former Cython/C++ (strings_udf /
# cudf_strings_udf) has been removed -- the Unicode character tables are obtained from libcudf
# through the public pylibcudf accessors (pylibcudf.strings.char_types.get_character_*_table_ptr).
# This single trivial, dependency-free module is built only so the cudf wheel remains a platform
# wheel; it can be dropped once cudf is packaged as a pure-Python wheel.
set(cython_sources _stub.pyx)

rapids_cython_create_modules(CXX SOURCE_FILES "${cython_sources}" ASSOCIATED_TARGETS cudf)
