# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
cmake_minimum_required(VERSION 4.0 FATAL_ERROR)

include(../../cmake/rapids_config.cmake)
include(rapids-cmake)
include(rapids-cpm)
include(rapids-cuda)
include(rapids-export)
include(rapids-find)

rapids_cuda_init_architectures(CUDF_STREAMING)

project(
  CUDF_STREAMING
  VERSION "${RAPIDS_VERSION}"
  LANGUAGES CXX CUDA
)

# Set a default build type if none was specified
rapids_cmake_build_type(Release)

# For now, disable CMake's automatic module scanning for C++ files. There is an sccache bug in the
# version RAPIDS uses in CI that causes it to handle the resulting -M* flags incorrectly with
# gcc>=14. We can remove this once we upgrade to a newer sccache version.
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)

# Include CUDA configuration (expt-extended-lambda, expt-relaxed-constexpr, etc.)
include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/Modules/ConfigureCUDA.cmake)

# ##################################################################################################
# * conda environment -----------------------------------------------------------------------------
rapids_cmake_support_conda_env(conda_env MODIFY_PREFIX_PATH)

# ##################################################################################################
# * find C++ libraries from RAPIDS wheels in site-packages -----------------------------------------
find_package(Python COMPONENTS Interpreter)
if(Python_FOUND)
  include(${rapids-cmake-dir}/cython-core/find_prefix_paths.cmake)
  rapids_cython_find_prefix_paths("${Python_EXECUTABLE}" wheel_prefix_paths)
  set_property(GLOBAL PROPERTY FIND_LIBRARY_USE_LIB64_PATHS TRUE)
  list(APPEND CMAKE_PREFIX_PATH ${wheel_prefix_paths})
endif()

# ##################################################################################################
# * Build options
option(BUILD_TESTS "Build tests for libcudf_streaming" OFF)
option(BUILD_BENCHMARKS "Build benchmarks for libcudf_streaming" OFF)
option(BUILD_EXAMPLES "Build examples for libcudf_streaming" OFF)

message(VERBOSE "CUDF_STREAMING: Build gtests: ${BUILD_TESTS}")
message(VERBOSE "CUDF_STREAMING: Build benchmarks: ${BUILD_BENCHMARKS}")
message(VERBOSE "CUDF_STREAMING: Build examples: ${BUILD_EXAMPLES}")

# ##################################################################################################
# * Dependencies

# add third party dependencies using CPM
rapids_cpm_init()
include(cmake/ConfigureOptionalCommunication.cmake)
include(cmake/thirdparty/get_cudf.cmake)
include(cmake/thirdparty/get_rapidsmpf.cmake)
cudf_streaming_configure_optional_communication()
include(../cmake/thirdparty/get_cucollections.cmake)

# ##################################################################################################
# * library target --------------------------------------------------------------------------------
add_library(
  cudf_streaming SHARED
  src/approx_distinct_count.cpp
  src/bloom_filter.cpp
  src/channel_metadata.cpp
  src/detail/approx_distinct_count.cu
  src/detail/device_bloom_filter.cu
  src/parquet.cpp
  src/partition.cpp
  src/partition_utils.cpp
  src/table_chunk.cpp
  src/utils.cpp
)

# ##################################################################################################
# * include paths ---------------------------------------------------------------------------------
target_include_directories(
  cudf_streaming PUBLIC "$<BUILD_INTERFACE:${CUDF_STREAMING_SOURCE_DIR}/include>"
                        "$<INSTALL_INTERFACE:include>"
)

# ##################################################################################################
# * library properties ----------------------------------------------------------------------------
set_target_properties(
  cudf_streaming
  PROPERTIES BUILD_RPATH "\$ORIGIN"
             INSTALL_RPATH "\$ORIGIN"
             CXX_STANDARD 20
             CXX_STANDARD_REQUIRED ON
             CUDA_STANDARD 20
             CUDA_STANDARD_REQUIRED ON
             POSITION_INDEPENDENT_CODE ON
             INTERFACE_POSITION_INDEPENDENT_CODE ON
)

target_compile_options(
  cudf_streaming PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:${CUDF_CUDA_FLAGS};--expt-relaxed-constexpr>"
)

# ##################################################################################################
# * library links ---------------------------------------------------------------------------------
target_link_libraries(
  cudf_streaming
  PUBLIC rapidsmpf::rapidsmpf cudf::cudf
  PRIVATE cuco::cuco
)

# Add Conda library, and include paths if specified
if(TARGET conda_env)
  target_link_libraries(cudf_streaming PRIVATE conda_env)
endif()

add_library(cudf_streaming::cudf_streaming ALIAS cudf_streaming)

# # GTests if enabled
if(BUILD_TESTS)
  # GoogleTest
  include(../cmake/thirdparty/get_gtest.cmake)

  # include CTest module -- automatically calls enable_testing()
  include(CTest)
  add_subdirectory(tests)
endif()

# Benchmarks if enabled
if(BUILD_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()

# Examples if enabled
if(BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

# ##################################################################################################
# * cudf_streaming Install -------------------------------------------------------------------------
rapids_cmake_install_lib_dir(lib_dir)
install(
  TARGETS cudf_streaming
  DESTINATION ${lib_dir}
  EXPORT cudf_streaming-exports
)
install(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  FILES_MATCHING
  PATTERN "*.hpp"
  PATTERN "detail" EXCLUDE
)

rapids_export(
  INSTALL cudf_streaming
  EXPORT_SET cudf_streaming-exports
  GLOBAL_TARGETS cudf_streaming
  NAMESPACE cudf_streaming::
)

rapids_export(
  BUILD cudf_streaming
  EXPORT_SET cudf_streaming-exports
  GLOBAL_TARGETS cudf_streaming
  NAMESPACE cudf_streaming::
)
