# Copyright (c) 2025, NVIDIA CORPORATION.

cmake_minimum_required(VERSION 3.30.4 FATAL_ERROR)

include(../set_cuda_architecture.cmake)

# initialize cuda architecture
rapids_cuda_init_architectures(external_sorting)

project(
  external_sorting
  VERSION 0.0.1
  LANGUAGES CXX CUDA
)

include(../fetch_dependencies.cmake)

include(rapids-cmake)
rapids_cmake_build_type("Release")

# For now, disable CMake's automatic module scanning for C++ files. There is an sccache bug in the
# version RAPIDS uses in CI that causes it to handle the resulting -M* flags incorrectly with
# gcc>=14. We can remove this once we upgrade to a newer sccache version.
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)

list(APPEND CUDF_CUDA_FLAGS --expt-extended-lambda --expt-relaxed-constexpr)

add_library(cusort OBJECT cusort.cu parquet_io.cpp random_table_generator.cu)
target_link_libraries(cusort PRIVATE cudf::cudf)
target_compile_features(cusort PRIVATE cxx_std_20 cuda_std_20)
target_compile_options(cusort PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:${CUDF_CUDA_FLAGS}>")

add_executable(external_sort sort.cpp)
target_compile_features(external_sort PRIVATE cxx_std_20)
target_link_libraries(external_sort PRIVATE $<TARGET_OBJECTS:cusort> cudf::cudf)
install(TARGETS external_sort DESTINATION bin/examples/libcudf/external_sorting)
