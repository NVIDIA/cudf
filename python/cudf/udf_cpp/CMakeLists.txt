# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

cmake_minimum_required(VERSION 4.0 FATAL_ERROR)

include(rapids-cmake)
include(rapids-cpm)
include(rapids-find)

rapids_cpm_init()

rapids_find_package(
  CUDAToolkit REQUIRED
  BUILD_EXPORT_SET udf-exports
  INSTALL_EXPORT_SET udf-exports
)

# The below is inspired by legate dataframe
execute_process(
  COMMAND "python" -c
          "from numba.cuda.memory_management.nrt import get_include; print(get_include())"
  OUTPUT_VARIABLE NRT_INCLUDE_DIR
  ERROR_VARIABLE NRT_ERROR
  RESULT_VARIABLE NRT_RESULT
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(${NRT_RESULT})
  message(
    FATAL_ERROR "Error while trying to obtain numba cuda runtime include directory:\n${NRT_ERROR}"
  )
endif()

include(${rapids-cmake-dir}/cpm/cccl.cmake)
rapids_cpm_cccl(BUILD_EXPORT_SET udf-exports INSTALL_EXPORT_SET udf-exports)

add_library(cudf_strings_udf SHARED strings/src/strings/udf/udf_apis.cu)
target_include_directories(
  cudf_strings_udf PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/strings/include>"
                          "${NRT_INCLUDE_DIR}"
)

set_target_properties(
  cudf_strings_udf
  PROPERTIES BUILD_RPATH "\$ORIGIN/../"
             INSTALL_RPATH "\$ORIGIN/../"
             CXX_STANDARD 20
             CXX_STANDARD_REQUIRED ON
             CUDA_STANDARD 20
             CUDA_STANDARD_REQUIRED ON
             POSITION_INDEPENDENT_CODE ON
             INTERFACE_POSITION_INDEPENDENT_CODE ON
)

set(UDF_CXX_FLAGS)
set(UDF_CUDA_FLAGS --expt-extended-lambda --expt-relaxed-constexpr)
target_compile_options(
  cudf_strings_udf PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${UDF_CXX_FLAGS}>"
                           "$<$<COMPILE_LANGUAGE:CUDA>:${UDF_CUDA_FLAGS}>"
)
target_link_libraries(cudf_strings_udf PUBLIC cudf::cudf)
install(TARGETS cudf_strings_udf DESTINATION ./cudf/_lib/)

# Create the shim library for each architecture.
set(SHIM_CUDA_FLAGS --expt-relaxed-constexpr -rdc=true)

# always build a default architecture in case RAPIDS_NO_INITIALIZE is set and the device cc can't be
# safely queried through a context
if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 13.0.0)
  list(INSERT CMAKE_CUDA_ARCHITECTURES 0 "75")
else()
  list(INSERT CMAKE_CUDA_ARCHITECTURES 0 "70")
endif()

list(TRANSFORM CMAKE_CUDA_ARCHITECTURES REPLACE "-real" "")
list(TRANSFORM CMAKE_CUDA_ARCHITECTURES REPLACE "-virtual" "")
list(SORT CMAKE_CUDA_ARCHITECTURES)
list(REMOVE_DUPLICATES CMAKE_CUDA_ARCHITECTURES)

add_library(shim OBJECT shim.cu)
set_target_properties(
  shim
  PROPERTIES CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
             CUDA_SEPARABLE_COMPILATION ON
             CUDA_FATBIN_COMPILATION ON
             POSITION_INDEPENDENT_CODE ON
             INTERPROCEDURAL_OPTIMIZATION ON
)

target_compile_options(shim PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:${SHIM_CUDA_FLAGS}>")
target_link_libraries(shim PUBLIC cudf::cudf)
target_include_directories(
  shim PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/strings/include>" "${NRT_INCLUDE_DIR}"
)

install(
  FILES $<TARGET_OBJECTS:shim>
  DESTINATION cudf/core/udf/
  RENAME shim.fatbin
)
