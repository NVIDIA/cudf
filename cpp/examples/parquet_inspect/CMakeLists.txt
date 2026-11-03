# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on

cmake_minimum_required(VERSION 4.0 FATAL_ERROR)

include(../set_cuda_architecture.cmake)

# initialize cuda architecture
rapids_cuda_init_architectures(parquet_inspect)

project(
  parquet_inspect
  VERSION 0.0.1
  LANGUAGES CXX CUDA
)

include(../fetch_dependencies.cmake)

include(rapids-cmake)
rapids_cmake_build_type("Release")

# For now, disable CMake's automatic module scanning for C++ files. There is an sccache bug in the
# version RAPIDS uses in CI that causes it to handle the resulting -M* flags incorrectly with
# gcc>=14. We can remove this once we upgrade to a newer sccache version.
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)

add_library(parquet_inspect_utils OBJECT parquet_inspect_utils.cpp)
target_compile_features(parquet_inspect_utils PRIVATE cxx_std_20)
target_link_libraries(parquet_inspect_utils PRIVATE cudf::cudf)

# Build and install parquet_inspect
add_executable(parquet_inspect parquet_inspect.cpp)
target_link_libraries(
  parquet_inspect PRIVATE cudf::cudf $<BUILD_LOCAL_INTERFACE:nvtx3::nvtx3-cpp>
                          $<TARGET_OBJECTS:parquet_inspect_utils>
)
target_compile_features(parquet_inspect PRIVATE cxx_std_20)
install(TARGETS parquet_inspect DESTINATION bin/examples/libcudf/parquet_inspect)

# Install the example.parquet file
install(FILES ${CMAKE_CURRENT_LIST_DIR}/example.parquet
        DESTINATION bin/examples/libcudf/parquet_inspect
)
