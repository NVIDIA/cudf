# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
enable_language(ASM)

add_library(
  cudf_streaming_ndsh concatenate.cpp groupby.cpp join.cpp parquet_writer.cpp sort.cpp utils.cpp
)

set_target_properties(
  cudf_streaming_ndsh
  PROPERTIES BUILD_RPATH "\$ORIGIN"
             INSTALL_RPATH "\$ORIGIN"
             CXX_STANDARD 20
             CXX_STANDARD_REQUIRED ON
             CUDA_STANDARD 20
             CUDA_STANDARD_REQUIRED ON
             POSITION_INDEPENDENT_CODE ON
             INTERFACE_POSITION_INDEPENDENT_CODE ON
)

target_link_libraries(
  cudf_streaming_ndsh
  PRIVATE cudf_streaming rapidsmpf::rapidsmpf cuco::cuco cudf_streaming::optional_communication
          $<TARGET_NAME_IF_EXISTS:conda_env>
)

set(CUDF_STREAMING_NDSH_QUERIES q01 q03 q04 q09 q21 bench_read)

foreach(query IN ITEMS ${CUDF_STREAMING_NDSH_QUERIES})
  add_executable(${query} "${query}.cpp")
  set_target_properties(
    ${query}
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CUDF_STREAMING_BINARY_DIR}/benchmarks/ndsh"
               OUTPUT_NAME "libcudf_streaming_bench_ndsh_${query}"
               CXX_STANDARD 20
               CXX_STANDARD_REQUIRED ON
               CUDA_STANDARD 20
               CUDA_STANDARD_REQUIRED ON
  )
  target_link_libraries(
    ${query} PRIVATE cudf_streaming_ndsh cudf_streaming rapidsmpf::rapidsmpf
                     cudf_streaming::optional_communication $<TARGET_NAME_IF_EXISTS:conda_env>
  )
endforeach()

# Compile the CUDA UDFs once into an LTO fatbin and embed the module in the shared NDSH library.
include("${CUDF_STREAMING_SOURCE_DIR}/../cmake/thirdparty/get_xxhash.cmake")
include("${CUDF_STREAMING_SOURCE_DIR}/../cmake/thirdparty/get_rtcx.cmake")
include("${CUDF_STREAMING_SOURCE_DIR}/../cmake/Modules/AddFragment.cmake")
get_target_property(cudf_compile_definitions cudf::cudf INTERFACE_COMPILE_DEFINITIONS)
list(FILTER cudf_compile_definitions INCLUDE REGEX "^CUDF_LTO_ARCHITECTURE=")
string(REPLACE "CUDF_LTO_ARCHITECTURE=" "" CUDF_LTO_ARCHITECTURE "${cudf_compile_definitions}")
rtcx_add_embed(streaming_ndsh_fragments)
add_fragment(
  streaming_ndsh_fragments FRAGMENT ndsh_udfs SOURCE fragments.cu LINK_LIBRARIES cudf::cudf
)
rtcx_embed(
  streaming_ndsh_fragments COMPRESSION none OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/embed"
)
target_sources(
  cudf_streaming_ndsh PRIVATE ${streaming_ndsh_fragments_SOURCE_DIR}/streaming_ndsh_fragments.s
)
target_include_directories(cudf_streaming_ndsh PRIVATE ${streaming_ndsh_fragments_SOURCE_DIR})
add_dependencies(cudf_streaming_ndsh streaming_ndsh_fragments)

install(
  TARGETS cudf_streaming_ndsh
  COMPONENT benchmarking
  DESTINATION ${lib_dir}
  EXCLUDE_FROM_ALL
)
install(
  TARGETS ${CUDF_STREAMING_NDSH_QUERIES}
  COMPONENT benchmarking
  DESTINATION "${CUDF_STREAMING_BENCHMARK_INSTALL_DIR}"
  EXCLUDE_FROM_ALL
)
