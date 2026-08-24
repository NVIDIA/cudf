# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

include_guard(GLOBAL)

# This macro is used to create object libraries for JIT compilation fragments, and embed them as
# fatbins in the final library. It compiles the specified source file with the appropriate flags to
# generate a fatbin containing the specified kernel instance, and then embeds that fatbin in the
# final library with metadata that allows it to be looked up at runtime.
macro(add_fragment)
  set(TARGET ${ARGV0})
  set(ONE_VALUE_ARGS FRAGMENT SOURCE KERNEL_ONLY KERNEL_INSTANCE UDF_TYPE)
  set(MULTI_VALUE_ARGS DEFINITIONS ARRAY_IDS ARRAY_VALUES)
  cmake_parse_arguments(ARG "${OPTIONS}" "${ONE_VALUE_ARGS}" "${MULTI_VALUE_ARGS}" ${ARGN})

  if(NOT ARG_FRAGMENT)
    message(FATAL_ERROR "add_fragment requires FRAGMENT argument")
  endif()

  if(NOT ARG_SOURCE)
    message(FATAL_ERROR "add_fragment requires SOURCE argument")
  endif()

  set(OBJECT_ID ${TARGET}_${ARG_FRAGMENT})
  add_library(${OBJECT_ID} OBJECT ${ARG_SOURCE})
  target_compile_options(${OBJECT_ID} PRIVATE --compress-mode=size)

  if(DEFINED ARG_KERNEL_ONLY AND ARG_KERNEL_ONLY)
    # ensure that the FATBIN symbols only contain the specified kernel
    target_compile_options(${OBJECT_ID} PRIVATE -Xnvlink=--kernels-used=cudf_kernel_entry)
  endif()

  set(INSTANTIATION_DIR "${CUDF_GENERATED_INCLUDE_DIR}/${TARGET}/instantiations/${ARG_FRAGMENT}")
  target_include_directories(${OBJECT_ID} PRIVATE ${INSTANTIATION_DIR})

  if(ARG_KERNEL_INSTANCE)
    file(
      GENERATE
      OUTPUT "${INSTANTIATION_DIR}/cudf/detail/kernel_instance.cuh"
      CONTENT "#pragma once\n#define CUDF_KERNEL_INSTANCE ${ARG_KERNEL_INSTANCE}"
    )
  endif()

  if(ARG_UDF_TYPE)
    file(
      GENERATE
      OUTPUT "${INSTANTIATION_DIR}/cudf/detail/operation_udf.cuh"
      CONTENT "#pragma once\n#define CUDF_UDF_TYPE ${ARG_UDF_TYPE}"
    )
  endif()

  target_compile_definitions(${OBJECT_ID} PRIVATE CUDF_DISABLE_EXPORTS ${ARG_DEFINITIONS})
  set_target_properties(
    ${OBJECT_ID}
    PROPERTIES CUDA_SEPARABLE_COMPILATION ON
               CUDA_FATBIN_COMPILATION ON
               POSITION_INDEPENDENT_CODE ON
               INTERPROCEDURAL_OPTIMIZATION ON
               CXX_STANDARD 20
               CXX_STANDARD_REQUIRED ON
               CXX_EXTENSIONS ON
               CXX_VISIBILITY_PRESET hidden
               CUDA_STANDARD 20
               CUDA_STANDARD_REQUIRED ON
               CUDA_VISIBILITY_PRESET hidden
  )
  target_link_libraries(
    ${OBJECT_ID}
    PUBLIC CCCL::CCCL rapids_logger::rapids_logger rmm::rmm $<BUILD_LOCAL_INTERFACE:BS::thread_pool>
    PRIVATE $<BUILD_LOCAL_INTERFACE:nvtx3::nvtx3-cpp> $<BUILD_LOCAL_INTERFACE:cuco::cuco>
            ZLIB::ZLIB nvcomp::nvcomp kvikio::kvikio nanoarrow::nanoarrow zstd
  )
  target_include_directories(
    ${OBJECT_ID} PRIVATE "$<BUILD_INTERFACE:${CUDF_SOURCE_DIR}/include>"
                         "$<BUILD_INTERFACE:${CUDF_SOURCE_DIR}/src>"
  )
  target_compile_options(${OBJECT_ID} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:${CUDF_CUDA_FLAGS}>")

  rtcx_embed_blob(
    ${TARGET} FILE $<TARGET_OBJECTS:${OBJECT_ID}> DEST fragments/${ARG_FRAGMENT}.fatbin ID
    ${ARG_FRAGMENT} ARRAY_IDS ${ARG_ARRAY_IDS} ARRAY_VALUES ${ARG_ARRAY_VALUES}
  )
endmacro()
