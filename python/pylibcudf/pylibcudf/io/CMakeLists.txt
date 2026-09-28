# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

set(cython_sources avro.pyx csv.pyx datasource.pyx json.pyx orc.pyx parquet.pyx
                   parquet_io_utils.pyx parquet_metadata.pyx text.pyx timezone.pyx types.pyx
)

set(linked_libraries cudf::cudf)
rapids_cython_create_modules(
  CXX
  SOURCE_FILES "${cython_sources}"
  LINKED_LIBRARIES "${linked_libraries}" MODULE_PREFIX pylibcudf_io_ ASSOCIATED_TARGETS cudf
)

add_subdirectory(experimental)
