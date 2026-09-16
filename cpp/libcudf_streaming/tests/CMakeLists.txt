# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

# ##################################################################################################
# enable testing -----------------------------------------------------------------------------------
# ##################################################################################################
enable_testing()

include(rapids-test)
rapids_test_init()

# Create a separate test case for each of these n_ranks.
set(nranks_to_run 1 2 5)

# Register an mpirun-launched test for TEST_TARGET, creating one test case per parallelism in
# nranks_to_run identified by <TEST_TARGET>_<n_ranks>.
function(libcudf_streaming_mpirun_test_add)
  set(options) # no options
  set(one_value TEST_TARGET)
  set(multi_value) # no multi_value args
  cmake_parse_arguments(_MPIRUN_TEST "${options}" "${one_value}" "${multi_value}" ${ARGN})

  if(NOT DEFINED _MPIRUN_TEST_TEST_TARGET)
    message(FATAL_ERROR "libcudf_streaming_mpirun_test_add called without a test prefix")
  endif()

  message(STATUS "Adding mpirun test: ${_MPIRUN_TEST_TEST_TARGET} nranks: ${nranks_to_run}")
  foreach(np IN LISTS nranks_to_run)
    rapids_test_add(
      NAME "${_MPIRUN_TEST_TEST_TARGET}_${np}"
      COMMAND mpirun --oversubscribe --map-by node --bind-to none -np ${np}
              "$<TARGET_FILE:${_MPIRUN_TEST_TEST_TARGET}>"
      INSTALL_COMPONENT_SET testing INSTALL_TARGET ${_MPIRUN_TEST_TEST_TARGET}
    )
    set_tests_properties("${_MPIRUN_TEST_TEST_TARGET}_${np}" PROPERTIES RUN_SERIAL TRUE)
  endforeach(np)
endfunction(libcudf_streaming_mpirun_test_add)

add_library(libcudf_streaming_test_sources OBJECT)
set_target_properties(
  libcudf_streaming_test_sources
  PROPERTIES CXX_STANDARD 20
             CXX_STANDARD_REQUIRED ON
             CXX_EXTENSIONS ON
             CUDA_STANDARD 20
             CUDA_STANDARD_REQUIRED ON
)
target_include_directories(
  libcudf_streaming_test_sources PRIVATE "${CUDF_STREAMING_SOURCE_DIR}/include"
                                         "${CMAKE_CURRENT_SOURCE_DIR}"
)
target_compile_options(
  libcudf_streaming_test_sources
  PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CUDF_CUDA_FLAGS};--expt-extended-lambda>
)
target_link_libraries(
  libcudf_streaming_test_sources
  PRIVATE cudf_streaming rapidsmpf::rapidsmpf cudf::cudftestutil cudf::cudftestutil_impl cuco::cuco
  PUBLIC GTest::gmock GTest::gtest
)
# cudf::cudftestutil_impl injects cudf test-utility .cu sources (via INTERFACE_SOURCES) that are
# compiled into this object library. Those sources use device lambdas, so they require the same CUDA
# flags cudf builds them with (notably --expt-extended-lambda from CUDF_CUDA_FLAGS).
target_compile_options(
  libcudf_streaming_test_sources PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:${CUDF_CUDA_FLAGS}>"
)
target_sources(
  libcudf_streaming_test_sources
  PRIVATE streaming/test_approx_distinct_count.cpp
          streaming/test_bloom_filter.cu
          streaming/test_bloom_filter_config.cpp
          streaming/test_table_chunk.cpp
          streaming/test_read_parquet.cpp
          streaming/test_channel_metadata.cpp
          streaming/test_partition.cpp
          streaming/test_cudf_utils.cpp
          streaming/test_shuffler.cpp
          streaming/test_leaf_actor.cpp
          test_partition.cpp
          test_shuffler.cpp
)

add_executable(libcudf_streaming_single_tests main/single.cpp)
set_target_properties(
  libcudf_streaming_single_tests
  PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CUDF_STREAMING_BINARY_DIR}/gtests"
             INSTALL_RPATH "\$ORIGIN/../../../lib"
             CXX_STANDARD 20
             CXX_STANDARD_REQUIRED ON
             CXX_EXTENSIONS ON
             CUDA_STANDARD 20
             CUDA_STANDARD_REQUIRED ON
             # Custom marker property so this target can be discovered and installed below.
             COMPONENT testing
)
target_include_directories(
  libcudf_streaming_single_tests PRIVATE "${CUDF_STREAMING_SOURCE_DIR}/include"
                                         "${CMAKE_CURRENT_SOURCE_DIR}"
)
target_link_libraries(
  libcudf_streaming_single_tests
  PRIVATE cudf_streaming rapidsmpf::rapidsmpf GTest::gmock GTest::gtest
          $<TARGET_NAME_IF_EXISTS:conda_env> libcudf_streaming_test_sources
)
rapids_test_add(
  NAME libcudf_streaming_single_tests
  COMMAND libcudf_streaming_single_tests
  INSTALL_COMPONENT_SET testing
)
set_tests_properties(libcudf_streaming_single_tests PROPERTIES RUN_SERIAL TRUE)

# ##################################################################################################
# MPI and UCXX based tests -------------------------------------------------------------------------
# These executables reuse libcudf_streaming_test_sources but provide a different test environment
# main (main/mpi.cpp / main/ucxx.cpp), and are launched via mpirun for multiple ranks.
if(CUDF_STREAMING_HAVE_MPI)
  add_executable(libcudf_streaming_mpi_tests main/mpi.cpp)
  set_target_properties(
    libcudf_streaming_mpi_tests
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CUDF_STREAMING_BINARY_DIR}/gtests"
               INSTALL_RPATH "\$ORIGIN/../../../lib"
               CXX_STANDARD 20
               CXX_STANDARD_REQUIRED ON
               CXX_EXTENSIONS ON
               CUDA_STANDARD 20
               CUDA_STANDARD_REQUIRED ON
               # Custom marker property so this target can be discovered and installed below.
               COMPONENT testing
  )
  target_include_directories(
    libcudf_streaming_mpi_tests PRIVATE "${CUDF_STREAMING_SOURCE_DIR}/include"
                                        "${CMAKE_CURRENT_SOURCE_DIR}"
  )
  target_link_libraries(
    libcudf_streaming_mpi_tests
    PRIVATE cudf_streaming rapidsmpf::rapidsmpf cudf_streaming::optional_communication GTest::gmock
            GTest::gtest $<TARGET_NAME_IF_EXISTS:conda_env> libcudf_streaming_test_sources
  )
  libcudf_streaming_mpirun_test_add(TEST_TARGET libcudf_streaming_mpi_tests)

  if(CUDF_STREAMING_HAVE_UCXX)
    add_executable(libcudf_streaming_ucxx_tests main/ucxx.cpp)
    set_target_properties(
      libcudf_streaming_ucxx_tests
      PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CUDF_STREAMING_BINARY_DIR}/gtests"
                 INSTALL_RPATH "\$ORIGIN/../../../lib"
                 CXX_STANDARD 20
                 CXX_STANDARD_REQUIRED ON
                 CXX_EXTENSIONS ON
                 CUDA_STANDARD 20
                 CUDA_STANDARD_REQUIRED ON
                 # Custom marker property so this target can be discovered and installed below.
                 COMPONENT testing
    )
    target_include_directories(
      libcudf_streaming_ucxx_tests PRIVATE "${CUDF_STREAMING_SOURCE_DIR}/include"
                                           "${CMAKE_CURRENT_SOURCE_DIR}"
    )
    target_link_libraries(
      libcudf_streaming_ucxx_tests
      PRIVATE cudf_streaming
              rapidsmpf::rapidsmpf
              cudf_streaming::optional_communication
              ucxx::ucxx
              GTest::gmock
              GTest::gtest
              $<TARGET_NAME_IF_EXISTS:conda_env>
              libcudf_streaming_test_sources
    )
    libcudf_streaming_mpirun_test_add(TEST_TARGET libcudf_streaming_ucxx_tests)
  else()
    cudf_streaming_skip_optional_target(libcudf_streaming_ucxx_tests "UCXX not found")
  endif()
else()
  cudf_streaming_skip_optional_target(libcudf_streaming_mpi_tests "MPI not found")
endif()

rapids_test_install_relocatable(
  INSTALL_COMPONENT_SET testing DESTINATION bin/gtests/libcudf_streaming
)
