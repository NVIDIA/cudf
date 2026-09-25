# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on

cmake_minimum_required(VERSION 4.0 FATAL_ERROR)

include(../set_cuda_architecture.cmake)

rapids_cuda_init_architectures(pack_example)

project(
  pack_example
  VERSION 0.0.1
  LANGUAGES CXX CUDA
)

include(../fetch_dependencies.cmake)

include(rapids-cmake)
rapids_cmake_build_type("Release")

set(CMAKE_CXX_SCAN_FOR_MODULES OFF)

add_executable(pack_example pack_example.cu)
target_link_libraries(pack_example PRIVATE cudf::cudf)
target_compile_features(pack_example PRIVATE cxx_std_20)

install(TARGETS pack_example DESTINATION bin/examples/libcudf/pack)
