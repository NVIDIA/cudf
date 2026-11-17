# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
cmake_minimum_required(VERSION 4.0 FATAL_ERROR)

project(cudf_export_consumer LANGUAGES CXX)

find_package(cudf REQUIRED CONFIG)

get_target_property(cudf_interface_libraries cudf::cudf INTERFACE_LINK_LIBRARIES)
foreach(forbidden IN ITEMS "rmm::" "rapids_logger::" "spdlog::" "fmt::")
  if(cudf_interface_libraries MATCHES "${forbidden}")
    message(
      FATAL_ERROR
        "cudf::cudf INTERFACE_LINK_LIBRARIES exposes private dependency ${forbidden}: ${cudf_interface_libraries}"
    )
  endif()
endforeach()

add_library(cudf_export_consumer OBJECT main.cpp)
target_link_libraries(cudf_export_consumer PRIVATE cudf::cudf)
set_target_properties(cudf_export_consumer PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)
