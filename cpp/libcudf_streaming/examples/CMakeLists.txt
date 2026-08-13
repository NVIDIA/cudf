# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

if(CUDF_STREAMING_HAVE_MPI)
  add_executable(example_shuffle "example_shuffle.cpp")
  set_target_properties(
    example_shuffle
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CUDF_STREAMING_BINARY_DIR}/examples"
               CXX_STANDARD 20
               CXX_STANDARD_REQUIRED ON
               CUDA_STANDARD 20
               CUDA_STANDARD_REQUIRED ON
  )
  target_link_libraries(
    example_shuffle
    PRIVATE cudf_streaming rapidsmpf::rapidsmpf cudf_streaming::optional_communication
            $<TARGET_NAME_IF_EXISTS:conda_env>
  )
  target_compile_options(example_shuffle PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>)
  target_sources(
    example_shuffle PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../benchmarks/utils/random_data.cu"
  )
  install(
    TARGETS example_shuffle
    COMPONENT examples
    DESTINATION bin/examples/libcudf_streaming
    EXCLUDE_FROM_ALL
  )
else()
  cudf_streaming_skip_optional_target(example_shuffle "requires MPI")
endif()
