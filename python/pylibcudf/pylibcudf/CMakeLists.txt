# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

set(cython_sources
    aggregation.pyx
    binaryop.pyx
    column.pyx
    column_factories.pyx
    context.pyx
    contiguous_split.pyx
    concatenate.pyx
    copying.pyx
    datetime.pyx
    decimal_division.pyx
    experimental.pyx
    expressions.pyx
    filling.pyx
    gpumemoryview.pyx
    groupby.pyx
    hashing.pyx
    interop.pyx
    _interop_helpers.pyx
    join.pyx
    json.pyx
    labeling.pyx
    lists.pyx
    merge.pyx
    null_mask.pyx
    partitioning.pyx
    prefetch.pyx
    quantiles.pyx
    reduce.pyx
    replace.pyx
    reshape.pyx
    rolling.pyx
    round.pyx
    scalar.pyx
    search.pyx
    stream_compaction.pyx
    sorting.pyx
    table.pyx
    table_equality.pyx
    traits.pyx
    transform.pyx
    transpose.pyx
    types.pyx
    unary.pyx
    utils.pyx
    utilities.pyx
)
set(linked_libraries cudf::cudf)
rapids_cython_create_modules(
  CXX
  SOURCE_FILES "${cython_sources}"
  LINKED_LIBRARIES "${linked_libraries}" MODULE_PREFIX pylibcudf_ ASSOCIATED_TARGETS cudf
)

include(${rapids-cmake-dir}/export/find_package_root.cmake)
include(../../../cpp/cmake/thirdparty/get_nanoarrow.cmake)

foreach(source interop _interop_helpers table column)
  target_include_directories(pylibcudf_${source} PUBLIC "$<BUILD_INTERFACE:${DLPACK_INCLUDE_DIR}>")
  target_link_libraries(pylibcudf_${source} PUBLIC nanoarrow::nanoarrow)
endforeach()

add_subdirectory(libcudf)
add_subdirectory(strings)
add_subdirectory(io)
add_subdirectory(nvtext)
add_subdirectory(experimental)
