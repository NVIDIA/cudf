# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

# strings_udf.pyx and cudf_strings_udf have been removed -- the Unicode character tables are
# obtained from libcudf through the public pylibcudf accessors
# (pylibcudf.strings.char_types.get_character_*_table_ptr), and all string/groupby device operations
# are pure MLIR via numba-cuda-mlir.
