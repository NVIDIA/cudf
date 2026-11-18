# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

cmake_minimum_required(VERSION 4.0 FATAL_ERROR)

include(../../../cmake/rapids_config.cmake)

include(rapids-cuda)
rapids_cuda_init_architectures(cudf_polars_cucascade)

set(CMAKE_CXX_SCAN_FOR_MODULES OFF)

project(
  cudf-polars-cucascade
  VERSION "${RAPIDS_VERSION}"
  LANGUAGES CXX CUDA
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# libcudf and pylibcudf are the ones installed in this environment: everything here that touches a
# libcudf type is compiled against them, which is why this is built from source and not shipped.
find_package(cudf "${RAPIDS_VERSION}" REQUIRED)

include(rapids-cpm)
rapids_cpm_init()
include(cmake/get_cucascade.cmake)

include(rapids-cython-core)
rapids_cython_init()

add_subdirectory(cudf_polars_cucascade)
