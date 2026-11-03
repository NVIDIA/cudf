# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

set(cython_sources
    aggregation.pyx binaryop.pyx copying.pyx datetime.pyx expressions.pyx join.pyx labeling.pyx
    partitioning.pyx reduce.pyx replace.pyx round.pyx stream_compaction.pyx types.pyx unary.pyx
)

set(linked_libraries cudf::cudf)
rapids_cython_create_modules(
  CXX
  SOURCE_FILES "${cython_sources}"
  LINKED_LIBRARIES "${linked_libraries}" ASSOCIATED_TARGETS cudf MODULE_PREFIX cpp
)

add_subdirectory(nvtext)
add_subdirectory(io)
add_subdirectory(lists)
add_subdirectory(strings)
