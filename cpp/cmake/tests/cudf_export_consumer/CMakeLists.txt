# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
cmake_minimum_required(VERSION 4.0 FATAL_ERROR)

project(cudf_export_consumer_test LANGUAGES CXX CUDA)

set(CUDF_REPOSITORY_DIR
    ""
    CACHE PATH "Path to the cuDF repository root"
)
if(NOT CUDF_REPOSITORY_DIR)
  cmake_path(SET CUDF_REPOSITORY_DIR NORMALIZE "${CMAKE_CURRENT_LIST_DIR}/../../../..")
endif()

include("${CUDF_REPOSITORY_DIR}/cmake/rapids_config.cmake")
include(rapids-cuda)
include(${rapids-cmake-dir}/cuda/set_architectures.cmake)
rapids_cuda_set_architectures(RAPIDS)
list(GET CMAKE_CUDA_ARCHITECTURES 0 _first_arch)
# Strip any -real/-virtual suffix to get a plain numeric architecture.
string(REGEX REPLACE "[-].*" "" _first_arch "${_first_arch}")

set(cudf_build_dir "${CMAKE_CURRENT_BINARY_DIR}/cudf-build")
set(consumer_build_dir "${CMAKE_CURRENT_BINARY_DIR}/consumer-build")
set(consumer_source_dir "${CMAKE_CURRENT_LIST_DIR}/consumer")

add_custom_target(
  verify_cudf_exports ALL
  COMMAND ${CMAKE_COMMAND} -E rm -rf "${cudf_build_dir}" "${consumer_build_dir}"
  COMMAND
    ${CMAKE_COMMAND} -S "${CUDF_REPOSITORY_DIR}/cpp" -B "${cudf_build_dir}" -G "${CMAKE_GENERATOR}"
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=${_first_arch} -DBUILD_SHARED_LIBS=ON
    -DCUDF_BUILD_STATIC_DEPS=FORCE -DBUILD_TESTS=OFF -DBUILD_BENCHMARKS=OFF
    -DCUDF_BUILD_TESTUTIL=OFF -DCUDF_BUILD_STREAMS_TEST_UTIL=OFF
  COMMAND ${CMAKE_COMMAND} -S "${consumer_source_dir}" -B "${consumer_build_dir}" -G
          "${CMAKE_GENERATOR}" "-Dcudf_DIR=${cudf_build_dir}"
  COMMAND ${CMAKE_COMMAND} --build "${consumer_build_dir}"
  COMMENT "Verifying cuDF build-tree CMake exports"
  VERBATIM
)
