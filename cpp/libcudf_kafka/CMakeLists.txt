# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION.
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

project(
  CUDF_KAFKA
  VERSION "${RAPIDS_VERSION}"
  LANGUAGES CXX
)

# Set a default build type if none was specified
rapids_cmake_build_type(Release)

# Set C++ standard globally to ensure all targets use C++20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# For now, disable CMake's automatic module scanning for C++ files. There is an sccache bug in the
# version RAPIDS uses in CI that causes it to handle the resulting -M* flags incorrectly with
# gcc>=14. We can remove this once we upgrade to a newer sccache version.
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)

# ##################################################################################################
# * conda environment -----------------------------------------------------------------------------
rapids_cmake_support_conda_env(conda_env MODIFY_PREFIX_PATH)

# ##################################################################################################
# * Build options
option(BUILD_TESTS "Build tests for libcudf_kafka" ON)

message(VERBOSE "CUDF_KAFKA: Build gtests: ${BUILD_TESTS}")

# ##################################################################################################
# * Dependencies

# add third party dependencies using CPM
rapids_cpm_init()
include(cmake/thirdparty/get_cudf.cmake)
include(cmake/thirdparty/get_rdkafka.cmake)

# # GTests if enabled
if(BUILD_TESTS)
  # GoogleTest
  include(../cmake/thirdparty/get_gtest.cmake)

  # include CTest module -- automatically calls enable_testing()
  include(CTest)
  add_subdirectory(tests)
endif()

# ##################################################################################################
# * library target --------------------------------------------------------------------------------
add_library(cudf_kafka SHARED src/kafka_consumer.cpp src/kafka_callback.cpp)

# ##################################################################################################
# * include paths ---------------------------------------------------------------------------------
target_include_directories(
  cudf_kafka PUBLIC "$<BUILD_INTERFACE:${CUDF_KAFKA_SOURCE_DIR}/include>"
                    "$<INSTALL_INTERFACE:include>"
)

# ##################################################################################################
# * library paths ---------------------------------------------------------------------------------
target_link_libraries(cudf_kafka PUBLIC cudf::cudf RDKAFKA::RDKAFKA)

# Add Conda library, and include paths if specified
if(TARGET conda_env)
  target_link_libraries(cudf_kafka PRIVATE conda_env)
endif()

set_target_properties(
  cudf_kafka PROPERTIES BUILD_RPATH "\$ORIGIN" INSTALL_RPATH
                                               "\$ORIGIN" # set target compile options
)

add_library(cudf_kafka::cudf_kafka ALIAS cudf_kafka)

# ##################################################################################################
# * cudf_kafka Install ----------------------------------------------------------------------------
rapids_cmake_install_lib_dir(lib_dir)
install(
  TARGETS cudf_kafka
  DESTINATION ${lib_dir}
  EXPORT cudf_kafka-exports
)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

rapids_export(
  INSTALL cudf_kafka
  EXPORT_SET cudf_kafka-exports
  GLOBAL_TARGETS cudf_kafka
  NAMESPACE cudf_kafka::
)

rapids_export(
  BUILD cudf_kafka
  EXPORT_SET cudf_kafka-exports
  GLOBAL_TARGETS cudf_kafka
  NAMESPACE cudf_kafka::
)
