# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on

cmake_minimum_required(VERSION 4.0 FATAL_ERROR)

include(../set_cuda_architecture.cmake)

# initialize cuda architecture
rapids_cuda_init_architectures(billion_rows)

project(
  billion_rows
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

list(APPEND CUDF_CUDA_FLAGS --expt-extended-lambda --expt-relaxed-constexpr)

add_library(groupby_results OBJECT groupby_results.cpp)
target_link_libraries(groupby_results PRIVATE cudf::cudf)
target_compile_features(groupby_results PRIVATE cxx_std_20)

# Creates a billion rows example executable and installs it.
#
# Arguments: target_name  - Name of the CMake target and resulting executable source_file  - Source
# file to compile for the example
function(add_billion_rows_example target_name source_file)
  add_executable(${target_name} ${source_file})
  target_link_libraries(
    ${target_name} PRIVATE cudf::cudf $<BUILD_LOCAL_INTERFACE:nvtx3::nvtx3-cpp>
                           $<TARGET_OBJECTS:groupby_results>
  )
  target_compile_features(${target_name} PRIVATE cxx_std_20)
  install(TARGETS ${target_name} DESTINATION bin/examples/libcudf/billion_rows)
endfunction()

add_billion_rows_example(brc brc.cpp)
add_billion_rows_example(brc_chunks brc_chunks.cpp)
add_billion_rows_example(brc_pipeline brc_pipeline.cpp)
