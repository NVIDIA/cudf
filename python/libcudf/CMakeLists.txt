# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

cmake_minimum_required(VERSION 4.0 FATAL_ERROR)

include(../../cmake/rapids_config.cmake)
include(rapids-cpm)
include(rapids-cuda)
rapids_cuda_init_architectures(libcudf-python)
rapids_cpm_init()

project(
  libcudf-python
  VERSION "${RAPIDS_VERSION}"
  LANGUAGES CXX CUDA
)

option(USE_NVCOMP_RUNTIME_WHEEL "Use the nvcomp wheel at runtime instead of vendoring nvcomp" OFF)

# Check if cudf is already available. If so, it is the user's responsibility to ensure that the
# CMake package is also available at build time of the Python cudf package.
find_package(cudf "${RAPIDS_VERSION}")

if(cudf_FOUND)
  return()
endif()

unset(cudf_FOUND)

set(BUILD_TESTS OFF)
set(BUILD_BENCHMARKS OFF)
set(CUDF_BUILD_TESTUTIL OFF)
set(CUDF_BUILD_STREAMS_TEST_UTIL OFF)

include(../../cpp/cmake/thirdparty/get_nvcomp.cmake)

# Install only the specific libnvcomp.so.* library instead of all nvcomp targets
if(NOT USE_NVCOMP_RUNTIME_WHEEL)
  if(TARGET nvcomp::nvcomp)
    get_target_property(is_imported nvcomp::nvcomp IMPORTED)
    if(is_imported)
      get_target_property(nvcomp_lib_path nvcomp::nvcomp IMPORTED_LOCATION_RELEASE)
      # Compute the SOVERSION from the library path
      get_filename_component(nvcomp_lib_dir ${nvcomp_lib_path} DIRECTORY)
      get_filename_component(nvcomp_lib_name ${nvcomp_lib_path} NAME)
      string(REGEX REPLACE "^libnvcomp\\.so\\.([0-9]+).*$" "\\1" nvcomp_soversion_major
                           ${nvcomp_lib_name}
      )
      install(
        FILES ${nvcomp_lib_path}
        DESTINATION ${SKBUILD_PLATLIB_DIR}/libcudf/lib64/
        RENAME libnvcomp.so.${nvcomp_soversion_major}
      )
    else()
      message(FATAL_ERROR "nvcomp target must be imported")
    endif()
  else()
    message(FATAL_ERROR "nvcomp target not found")
  endif()
endif()

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)

add_subdirectory(../../cpp cudf-cpp)
