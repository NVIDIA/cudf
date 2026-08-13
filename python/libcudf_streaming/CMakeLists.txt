# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

cmake_minimum_required(VERSION 4.0 FATAL_ERROR)

include(../../cmake/rapids_config.cmake)
include(rapids-cpm)
include(rapids-cuda)
rapids_cuda_init_architectures(libcudf-streaming-python)
rapids_cpm_init()

project(
  libcudf-streaming-python
  VERSION "${RAPIDS_VERSION}"
  LANGUAGES CXX CUDA
)

# Check if cudf_streaming is already available. If so, it is the user's responsibility to ensure
# that the CMake package is also available at build time of the Python cudf_streaming package.
find_package(cudf_streaming "${RAPIDS_VERSION}")

if(cudf_streaming_FOUND)
  return()
endif()

unset(cudf_streaming_FOUND)

set(BUILD_TESTS OFF)
set(BUILD_BENCHMARKS OFF)
set(CUDF_BUILD_TESTUTIL OFF)
set(CUDF_BUILD_STREAMS_TEST_UTIL OFF)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)

add_subdirectory(../../cpp/libcudf_streaming cudf-streaming-cpp)
