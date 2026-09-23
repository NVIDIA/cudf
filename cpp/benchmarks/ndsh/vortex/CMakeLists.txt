# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

add_library(NDSH_VORTEX_IO STATIC EXCLUDE_FROM_ALL vortex_io.cpp writer.cpp)
target_compile_features(NDSH_VORTEX_IO PUBLIC cxx_std_20)
target_compile_options(NDSH_VORTEX_IO PRIVATE ${CUDF_CXX_FLAGS})
# Consumers use vortex/... includes.
target_include_directories(
  NDSH_VORTEX_IO PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>"
)
target_link_libraries(
  NDSH_VORTEX_IO
  PUBLIC cudf::cudf
  PRIVATE Vortex::ffi_static nanoarrow::nanoarrow CUDA::cudart nvtx3::nvtx3-cpp
)

add_executable(NDSH_VORTEX_IO_TEST EXCLUDE_FROM_ALL vortex_io_test.cpp)
target_compile_options(NDSH_VORTEX_IO_TEST PRIVATE ${CUDF_CXX_FLAGS})
set_target_properties(
  NDSH_VORTEX_IO_TEST PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                 "$<BUILD_INTERFACE:${CUDF_BINARY_DIR}/benchmarks>"
)
target_link_libraries(
  NDSH_VORTEX_IO_TEST PRIVATE NDSH_VORTEX_IO nanoarrow::nanoarrow CUDA::cudart nvtx3::nvtx3-cpp
)

# This directory is only added for Vortex-enabled benchmark builds.
if(BUILD_TESTS)
  # Testing is initialized in cpp/tests; load the helpers in this directory explicitly.
  include(rapids-test)
  add_executable(NDSH_VORTEX_WRITER_TEST writer_test.cpp)
  target_compile_options(NDSH_VORTEX_WRITER_TEST PRIVATE ${CUDF_CXX_FLAGS})
  target_link_libraries(
    NDSH_VORTEX_WRITER_TEST PRIVATE NDSH_VORTEX_IO cudf::cudftestutil_objects nanoarrow::nanoarrow
                                    $<TARGET_NAME_IF_EXISTS:conda_env>
  )
  rapids_cuda_set_runtime(NDSH_VORTEX_WRITER_TEST USE_STATIC ON)
  set_target_properties(
    NDSH_VORTEX_WRITER_TEST PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                       "$<BUILD_INTERFACE:${CUDF_BINARY_DIR}/benchmarks>"
  )
  set_target_properties(NDSH_VORTEX_IO_TEST PROPERTIES EXCLUDE_FROM_ALL FALSE)
  foreach(test IN ITEMS NDSH_VORTEX_WRITER_TEST NDSH_VORTEX_IO_TEST)
    rapids_test_add(
      NAME ${test}
      COMMAND ${test}
      GPUS 1
      PERCENT 100
    )
    set_tests_properties(${test} PROPERTIES TIMEOUT 600 RUN_SERIAL TRUE LABELS "ndsh;vortex;smoke")
  endforeach()
endif()
