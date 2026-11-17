# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

# Find or install GoogleBench
include(${rapids-cmake-dir}/cpm/gbench.cmake)
rapids_cpm_gbench(BUILD_STATIC)

set(CUDF_STREAMING_BENCHMARK_INSTALL_DIR bin)

add_library(bench_utils INTERFACE)
target_sources(bench_utils INTERFACE utils/random_data.cu)
target_compile_options(bench_utils INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>)

if(CUDF_STREAMING_HAVE_COMM)
  add_executable(bench_shuffle "bench_shuffle.cpp")
  set_target_properties(
    bench_shuffle
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CUDF_STREAMING_BINARY_DIR}/benchmarks"
               OUTPUT_NAME "libcudf_streaming_bench_shuffle"
               CXX_STANDARD 20
               CXX_STANDARD_REQUIRED ON
               CXX_EXTENSIONS ON
               CUDA_STANDARD 20
               CUDA_STANDARD_REQUIRED ON
  )
  target_link_libraries(
    bench_shuffle
    PRIVATE cudf_streaming rapidsmpf::rapidsmpf cudf_streaming::optional_communication
            $<TARGET_NAME_IF_EXISTS:conda_env> bench_utils
  )
  install(
    TARGETS bench_shuffle
    COMPONENT benchmarking
    DESTINATION "${CUDF_STREAMING_BENCHMARK_INSTALL_DIR}"
    EXCLUDE_FROM_ALL
  )
else()
  cudf_streaming_skip_optional_target(bench_shuffle "requires MPI or UCXX")
endif()

add_executable(bench_partition "bench_partition.cpp")
set_target_properties(
  bench_partition
  PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CUDF_STREAMING_BINARY_DIR}/benchmarks"
             OUTPUT_NAME "libcudf_streaming_bench_partition"
             CXX_STANDARD 20
             CXX_STANDARD_REQUIRED ON
             CXX_EXTENSIONS ON
             CUDA_STANDARD 20
             CUDA_STANDARD_REQUIRED ON
)
target_link_libraries(
  bench_partition PRIVATE cudf_streaming rapidsmpf::rapidsmpf benchmark::benchmark
                          benchmark::benchmark_main $<TARGET_NAME_IF_EXISTS:conda_env> bench_utils
)
install(
  TARGETS bench_partition
  COMPONENT benchmarking
  DESTINATION "${CUDF_STREAMING_BENCHMARK_INSTALL_DIR}"
  EXCLUDE_FROM_ALL
)

add_executable(bench_pack "bench_pack.cpp")
set_target_properties(
  bench_pack
  PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CUDF_STREAMING_BINARY_DIR}/benchmarks"
             OUTPUT_NAME "libcudf_streaming_bench_pack"
             CXX_STANDARD 20
             CXX_STANDARD_REQUIRED ON
             CXX_EXTENSIONS ON
             CUDA_STANDARD 20
             CUDA_STANDARD_REQUIRED ON
             LINK_FLAGS "-Wl,--allow-shlib-undefined"
)
target_link_libraries(
  bench_pack PRIVATE cudf_streaming rapidsmpf::rapidsmpf benchmark::benchmark
                     benchmark::benchmark_main $<TARGET_NAME_IF_EXISTS:conda_env> bench_utils
)
install(
  TARGETS bench_pack
  COMPONENT benchmarking
  DESTINATION "${CUDF_STREAMING_BENCHMARK_INSTALL_DIR}"
  EXCLUDE_FROM_ALL
)

add_subdirectory(streaming)
