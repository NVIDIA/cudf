# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

cmake_minimum_required(VERSION 4.0 FATAL_ERROR)

include(../../cmake/rapids_config.cmake)

project(
  cudf-kafka-python
  VERSION "${RAPIDS_VERSION}"
  LANGUAGES CXX
)

# Set C++ standard to match other cudf Python packages
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Disable CMake's automatic module scanning for C++ files due to sccache bug with gcc>=14
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)

find_package(cudf_kafka "${RAPIDS_VERSION}" REQUIRED)

if(NOT cudf_kafka_FOUND)
  message(
    FATAL_ERROR
      "cudf_kafka package not found. cudf_kafka C++ is required to build this Python package."
  )
endif()

include(rapids-cython-core)
rapids_cython_init()

add_subdirectory(cudf_kafka/_lib)
