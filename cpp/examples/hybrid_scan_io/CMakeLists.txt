# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on

cmake_minimum_required(VERSION 4.0 FATAL_ERROR)

include(../set_cuda_architecture.cmake)

# initialize cuda architecture
rapids_cuda_init_architectures(hybrid_scan_io)

project(
  hybrid_scan_io
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

add_library(
  hybrid_scan_utils OBJECT common_utils.cpp io_source.cpp io_utils.cpp hybrid_scan_composer.cpp
)
target_compile_features(hybrid_scan_utils PRIVATE cxx_std_20)
target_link_libraries(hybrid_scan_utils PRIVATE cudf::cudf)

# Creates a hybrid scan example executable and installs it.
#
# Arguments: target_name  - Name of the CMake target and resulting executable source_file  - Source
# file to compile for the example
function(add_hybrid_scan_example target_name source_file)
  add_executable(${target_name} ${source_file})
  target_link_libraries(
    ${target_name} PRIVATE cudf::cudf $<BUILD_LOCAL_INTERFACE:nvtx3::nvtx3-cpp>
                           $<TARGET_OBJECTS:hybrid_scan_utils>
  )
  target_compile_features(${target_name} PRIVATE cxx_std_20)
  install(TARGETS ${target_name} DESTINATION bin/examples/libcudf/hybrid_scan_io)
endfunction()

add_hybrid_scan_example(hybrid_scan_io hybrid_scan_io.cpp)
add_hybrid_scan_example(hybrid_scan_pipeline hybrid_scan_pipeline.cpp)
add_hybrid_scan_example(hybrid_scan_multifile_single_step hybrid_scan_multifile_single_step.cpp)
add_hybrid_scan_example(hybrid_scan_multifile_two_step hybrid_scan_multifile_two_step.cpp)

# Install the example.parquet file
install(FILES ${CMAKE_CURRENT_LIST_DIR}/example.parquet
        DESTINATION bin/examples/libcudf/hybrid_scan_io
)
