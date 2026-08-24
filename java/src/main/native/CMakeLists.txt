# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

cmake_minimum_required(VERSION 4.0 FATAL_ERROR)

include(../../../../cmake/rapids_config.cmake)
include(rapids-cmake)
include(rapids-cuda)
include(rapids-find)
include(rapids-cpm)
rapids_cpm_init(
  OVERRIDE "${CMAKE_CURRENT_SOURCE_DIR}/../../../../cpp/cmake/thirdparty/patches/override.json"
)

rapids_cuda_init_architectures(CUDF_JNI)

project(
  CUDF_JNI
  VERSION "${RAPIDS_VERSION}"
  LANGUAGES CXX CUDA
)

# ##################################################################################################
# * build options ---------------------------------------------------------------------------------

option(USE_NVTX "Build with NVTX support" ON)
option(BUILD_SHARED_LIBS "Build cuDF JNI shared libraries" ON)
option(BUILD_TESTS "Configure CMake to build tests" ON)
option(CUDF_USE_PER_THREAD_DEFAULT_STREAM "Build with per-thread default stream" OFF)
option(USE_GDS "Build with GPUDirect Storage (GDS)/cuFile support" OFF)
option(CUDF_JNI_LIBCUDF_STATIC "Link with libcudf.a" OFF)
option(CUDF_JNI_ENABLE_PROFILING "Build with profiling support" ON)
option(CUDA_STATIC_CUFILE "Statically link cuFile" OFF)

message(VERBOSE "CUDF_JNI: Build with NVTX support: ${USE_NVTX}")
message(VERBOSE "CUDF_JNI: Build cuDF JNI shared libraries: ${BUILD_SHARED_LIBS}")
message(VERBOSE "CUDF_JNI: Configure CMake to build tests: ${BUILD_TESTS}")
message(VERBOSE
        "CUDF_JNI: Build with per-thread default stream: ${CUDF_USE_PER_THREAD_DEFAULT_STREAM}"
)
message(VERBOSE "CUDF_JNI: Build with GPUDirect Storage support: ${USE_GDS}")
message(VERBOSE "CUDF_JNI: Link with libcudf statically: ${CUDF_JNI_LIBCUDF_STATIC}")

# ##################################################################################################
# * compiler options ------------------------------------------------------------------------------

rapids_find_package(CUDAToolkit REQUIRED)

if(CUDF_USE_PER_THREAD_DEFAULT_STREAM)
  message(STATUS "Using per-thread default stream")
  add_compile_definitions(CUDA_API_PER_THREAD_DEFAULT_STREAM CUDF_USE_PER_THREAD_DEFAULT_STREAM)
endif()

# Set a default build type if none was specified
rapids_cmake_build_type("Release")

# ##################################################################################################
# * CUDF ------------------------------------------------------------------------------------------

set(CUDF_SOURCE_DIR "${PROJECT_SOURCE_DIR}/../../../../cpp")

# Search for libcudf in CUDF_INSTALL_DIR if it is specified. Otherwise, search in
# CUDF_CPP_BUILD_DIR.
if(DEFINED ENV{CUDF_INSTALL_DIR})
  set(CUDF_INSTALL_DIR
      "$ENV{CUDF_INSTALL_DIR}"
      CACHE STRING "path to libcudf install root"
  )
  set(CMAKE_PREFIX_PATH ${CUDF_INSTALL_DIR} "${CUDF_INSTALL_DIR}/lib64/cmake")
  message(STATUS "Using dependencies from libcudf installed path: ${CUDF_INSTALL_DIR}")
else()
  if(NOT DEFINED ENV{CUDF_CPP_BUILD_DIR})
    message(FATAL_ERROR "Either CUDF_INSTALL_DIR or CUDF_CPP_BUILD_DIR must be defined.")
  endif()

  set(CUDF_CPP_BUILD_DIR
      "$ENV{CUDF_CPP_BUILD_DIR}"
      CACHE STRING "path to libcudf build root"
  )
  set(cudf_ROOT ${CUDF_CPP_BUILD_DIR})
  message(STATUS "Using pre-configured libcudf from build path: ${CUDF_CPP_BUILD_DIR}")
endif()

rapids_find_package(cudf REQUIRED)

get_target_property(CUDF_CXX_FLAGS cudf::cudf CUDF_CXX_FLAGS)
get_target_property(CUDF_CUDA_FLAGS cudf::cudf CUDF_CUDA_FLAGS)
get_target_property(CUDF_CXX_DEFINITIONS cudf::cudf CUDF_CXX_DEFINITIONS)
get_target_property(CUDF_CUDA_DEFINITIONS cudf::cudf CUDF_CUDA_DEFINITIONS)

# The Java/JNI build always triggers static builds of both libcudf itself and its dependencies, and
# therefore needs all of them to be installed to link against later.
set(CUDF_INSTALL_LIBRARY_DEPS ON)

# ##################################################################################################
# * nvtx3 and nvcomp-------------------------------------------------------------------------------
#   Reuse any cached package that is already available after configuring/installing libcudf. Only
#   fetch (and install) from source if the target does not exist to avoid name conflict when the
#   same target is exported multiple times.

if(NOT TARGET nvtx3::nvtx3-cpp AND NOT DEFINED ENV{CUDF_INSTALL_DIR})
  message(STATUS "Fetch package nvtx3 from source.")
  include(${CUDF_SOURCE_DIR}/cmake/thirdparty/get_nvtx.cmake)
endif()
find_package(nvtx3 REQUIRED)

if(NOT TARGET nvcomp
   AND NOT TARGET nvcomp::nvcomp_static
   AND NOT DEFINED ENV{CUDF_INSTALL_DIR}
)
  include(${CUDF_SOURCE_DIR}/cmake/thirdparty/get_nvcomp.cmake)
endif()

# ##################################################################################################
# * find JNI --------------------------------------------------------------------------------------
find_package(JNI REQUIRED)
if(JNI_FOUND)
  message(STATUS "JDK with JNI in ${JNI_INCLUDE_DIRS}")
else()
  message(FATAL_ERROR "JDK with JNI not found, please check your settings.")
endif()

# ##################################################################################################
# * GDS/cufile ------------------------------------------------------------------------------------

if(USE_GDS)
  message(STATUS "Building with GPUDirect Storage (GDS)/cuFile support")
  if(CUDA_STATIC_CUFILE)
    set(_cufile_suffix _static)
  endif()
  if(NOT TARGET CUDA::cuFile${_cufile_suffix})
    message(FATAL_ERROR "cuFile support not found")
  endif()
endif()

# Set RMM logging level
set(RMM_LOGGING_LEVEL
    "INFO"
    CACHE STRING "Choose the logging level."
)
set_property(
  CACHE RMM_LOGGING_LEVEL PROPERTY STRINGS "TRACE" "DEBUG" "INFO" "WARN" "ERROR" "CRITICAL" "OFF"
)
message(VERBOSE "CUDF_JNI: RMM_LOGGING_LEVEL = '${RMM_LOGGING_LEVEL}'.")

add_compile_definitions("RMM_LOG_ACTIVE_LEVEL=RAPIDS_LOGGER_LOG_LEVEL_${RMM_LOGGING_LEVEL}")

# ##################################################################################################
# * library targets -------------------------------------------------------------------------------

add_library(
  cudfjni
  src/Aggregation128UtilsJni.cpp
  src/AggregationJni.cpp
  src/ChunkedPackJni.cpp
  src/ChunkedReaderJni.cpp
  src/CudfJni.cpp
  src/CudaJni.cpp
  src/ColumnVectorJni.cpp
  src/ColumnViewJni.cpp
  src/ColumnViewJni.cu
  src/CompiledExpression.cpp
  src/ContiguousTableJni.cpp
  src/DataSourceHelperJni.cpp
  src/DeletionVectorJni.cpp
  src/HashJoinJni.cpp
  src/HostMemoryBufferNativeUtilsJni.cpp
  src/HybridScanReaderJni.cpp
  src/HybridScanReaderJniInternal.cpp
  src/HybridScanReaderJniMaterialize.cpp
  src/KeyRemappingJni.cpp
  src/HostUDFWrapperJni.cpp
  src/NvcompJni.cpp
  src/NvtxRangeJni.cpp
  src/NvtxUniqueRangeJni.cpp
  src/PackedColumnMetadataJni.cpp
  src/RmmJni.cpp
  src/ScalarJni.cpp
  src/TableJni.cpp
  src/VariantUtilsJni.cpp
  src/aggregation128_utils.cu
  src/check_nvcomp_output_sizes.cu
  src/maps_column_view.cu
  src/multi_host_buffer_source.cpp
)

# Disable NVTX if necessary
if(NOT USE_NVTX)
  target_compile_definitions(cudfjni PUBLIC NVTX_DISABLE)
endif()

if(CUDF_JNI_ENABLE_PROFILING)
  target_compile_definitions(cudfjni PRIVATE CUDF_JNI_ENABLE_PROFILING)
endif()

if(CUDF_JNI_LIBCUDF_STATIC AND BUILD_SHARED_LIBS)
  # When linking against libcudf.a, the JNI library will include the old libcudf.so. For
  # backwards-compatibility for software that expects to find libcudf.so in the JVM environment
  # after cudf has loaded, the JNI code and libcudf.a will be combined into libcudf.so. A stub
  # library will be created for libcudfjni.so that will simply require libcudf.so for backwards
  # compatibility with software that expects to find libcudfjni.so at runtime.
  set_target_properties(cudfjni PROPERTIES OUTPUT_NAME "cudf")
  add_library(cudfjnistub SHARED src/emptyfile.cpp)
  set_target_properties(cudfjnistub PROPERTIES OUTPUT_NAME "cudfjni")
  target_link_libraries(cudfjnistub -Wl,--no-as-needed cudfjni -Wl,--as-needed)
endif()

# ##################################################################################################
# * include paths ---------------------------------------------------------------------------------

target_include_directories(
  cudfjni PUBLIC "${CMAKE_BINARY_DIR}/include" "${CMAKE_SOURCE_DIR}/include"
                 "${CMAKE_SOURCE_DIR}/src" "${JNI_INCLUDE_DIRS}"
)

# ##################################################################################################
# * compile options
# ---------------------------------------------------------------------------------

# Override RPATH for cudfjni
set_target_properties(
  cudfjni
  PROPERTIES BUILD_RPATH "\$ORIGIN"
             INSTALL_RPATH "\$ORIGIN"
             # set target compile options
             CXX_STANDARD 20
             CXX_STANDARD_REQUIRED ON
             CUDA_STANDARD 20
             CUDA_STANDARD_REQUIRED ON
             POSITION_INDEPENDENT_CODE ON
             INTERFACE_POSITION_INDEPENDENT_CODE ON
)

target_compile_options(
  cudfjni PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${CUDF_CXX_FLAGS}>"
                  "$<$<COMPILE_LANGUAGE:CUDA>:${CUDF_CUDA_FLAGS}>"
)

target_compile_definitions(
  cudfjni PUBLIC "$<$<COMPILE_LANGUAGE:CXX>:${CUDF_CXX_DEFINITIONS}>"
                 "$<$<COMPILE_LANGUAGE:CUDA>:${CUDF_CUDA_DEFINITIONS}>"
)
target_link_options(cudfjni PRIVATE "-Wl,--no-undefined")

set(CUDF_ENABLE_ARROW_PARQUET ON)
include(cmake/thirdparty/get_arrow.cmake)
target_link_libraries(cudfjni PRIVATE ${ARROW_LIBRARIES})

if(USE_GDS)
  add_library(cufilejni src/CuFileJni.cpp)
  set_target_properties(
    cufilejni
    PROPERTIES BUILD_RPATH "\$ORIGIN"
               INSTALL_RPATH "\$ORIGIN"
               # set target compile options
               CXX_STANDARD 20
               CXX_STANDARD_REQUIRED ON
               POSITION_INDEPENDENT_CODE ON
               INTERFACE_POSITION_INDEPENDENT_CODE ON
  )
  target_link_libraries(cufilejni PRIVATE cudfjni CUDA::cuFile${_cufile_suffix} cudf::cudf)
endif()

# ##################################################################################################
# * link libraries --------------------------------------------------------------------------------
if(CUDF_JNI_LIBCUDF_STATIC)
  target_link_libraries(cudfjni PRIVATE $<LINK_LIBRARY:WHOLE_ARCHIVE,cudf::cudf>)
else()
  target_link_libraries(cudfjni PUBLIC cudf::cudf)
endif()

get_target_property(cudf_static_linked_projects cudf::cudf CUDF_STATIC_LINKED_PROJECTS)
if(nvcomp IN_LIST cudf_static_linked_projects)
  target_link_libraries(cudfjni PRIVATE $<COMPILE_ONLY:nvcomp::nvcomp_static>)
else()
  target_link_libraries(cudfjni PRIVATE nvcomp::nvcomp)

  # * install nvcomp for tests
  add_custom_command(
    TARGET cudfjni
    PRE_LINK
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:nvcomp::nvcomp>
            "${PROJECT_BINARY_DIR}/libnvcomp.so"
    COMMENT "Copying nvcomp libraries to ${PROJECT_BINARY_DIR}"
  )
endif()

target_link_libraries(cudfjni PRIVATE nvtx3::nvtx3-cpp)

# ##################################################################################################
# * cudart options --------------------------------------------------------------------------------
#   cudart can be statically linked or dynamically linked. The python ecosystem wants dynamic
#   linking
set_target_properties(cudfjni PROPERTIES CUDA_RUNTIME_LIBRARY Static)
