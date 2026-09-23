# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

if(CUDF_STREAMING_HAVE_COMM)
  add_executable(bench_streaming_shuffle "bench_streaming_shuffle.cpp")
  set_target_properties(
    bench_streaming_shuffle
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CUDF_STREAMING_BINARY_DIR}/benchmarks"
               OUTPUT_NAME "libcudf_streaming_bench_streaming_shuffle"
               CXX_STANDARD 20
               CXX_STANDARD_REQUIRED ON
               CUDA_STANDARD 20
               CUDA_STANDARD_REQUIRED ON
  )
  target_link_libraries(
    bench_streaming_shuffle
    PRIVATE cudf_streaming rapidsmpf::rapidsmpf cudf_streaming::optional_communication
            $<TARGET_NAME_IF_EXISTS:conda_env> bench_utils
  )
  install(
    TARGETS bench_streaming_shuffle
    COMPONENT benchmarking
    DESTINATION "${CUDF_STREAMING_BENCHMARK_INSTALL_DIR}"
    EXCLUDE_FROM_ALL
  )
else()
  cudf_streaming_skip_optional_target(bench_streaming_shuffle "requires MPI or UCXX")
endif()

add_subdirectory(ndsh)
