# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

cmake_minimum_required(VERSION 4.0 FATAL_ERROR)

include(../../cmake/rapids_config.cmake)
include(rapids-cpm)
rapids_cpm_init()
include(rapids-cuda)
include(rapids-find)

rapids_cuda_init_architectures(cudf-streaming-python)

project(
  cudf-streaming-python
  VERSION "${RAPIDS_VERSION}"
  LANGUAGES CXX CUDA
)

# For now, disable CMake's automatic module scanning for C++ files. There is an sccache bug in the
# version RAPIDS uses in CI that causes it to handle the resulting -M* flags incorrectly with
# gcc>=14. We can remove this once we upgrade to a newer sccache version.
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)

# ##################################################################################################
# * find C++ libraries from RAPIDS wheels in site-packages -----------------------------------
set_property(GLOBAL PROPERTY FIND_LIBRARY_USE_LIB64_PATHS TRUE)

find_package(cudf_streaming "${RAPIDS_VERSION}" REQUIRED)
find_package(CUDAToolkit REQUIRED)

include(rapids-cython-core)
rapids_cython_init()

set(cython_sources
    cudf_streaming/approx_distinct_count.pyx
    cudf_streaming/bloom_filter.pyx
    cudf_streaming/channel_metadata.pyx
    cudf_streaming/partition.pyx
    cudf_streaming/partition_utils.pyx
    cudf_streaming/table_chunk.pyx
)
set(linked_libraries cudf_streaming::cudf_streaming)

rapids_cython_create_modules(
  CXX ASSOCIATED_TARGETS cudf_streaming
  SOURCE_FILES "${cython_sources}"
  LINKED_LIBRARIES "${linked_libraries}" MODULE_PREFIX cudf_streaming_
)

foreach(target IN LISTS RAPIDS_CYTHON_CREATED_TARGETS)
  set_target_properties(
    ${target}
    PROPERTIES CXX_STANDARD 20
               CXX_STANDARD_REQUIRED ON
               LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/cudf_streaming"
  )
  install(TARGETS ${target} DESTINATION cudf_streaming)
endforeach()
