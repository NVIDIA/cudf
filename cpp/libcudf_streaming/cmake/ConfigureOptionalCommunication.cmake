# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

include_guard(GLOBAL)

set(CUDF_STREAMING_FOUND_MPI OFF)
set(CUDF_STREAMING_FOUND_UCXX OFF)
set(CUDF_STREAMING_HAVE_MPI OFF)
set(CUDF_STREAMING_HAVE_UCXX OFF)

if(BUILD_BENCHMARKS
   OR BUILD_EXAMPLES
   OR BUILD_TESTS
)
  rapids_find_package(
    MPI QUIET
    BUILD_EXPORT_SET cudf_streaming-exports
    INSTALL_EXPORT_SET cudf_streaming-exports
  )
  if(TARGET MPI::MPI_CXX)
    set(CUDF_STREAMING_FOUND_MPI ON)
  endif()
endif()

if(BUILD_BENCHMARKS OR BUILD_TESTS)
  rapids_find_package(
    ucxx QUIET
    BUILD_EXPORT_SET cudf_streaming-exports
    INSTALL_EXPORT_SET cudf_streaming-exports
  )
  if(TARGET ucxx::ucxx)
    set(CUDF_STREAMING_FOUND_UCXX ON)
  endif()
endif()

# Check whether rapidsmpf exports a compile definition for an optional feature.
function(cudf_streaming_rapidsmpf_exports_feature feature result)
  set(has_feature OFF)
  get_target_property(
    rapidsmpf_compile_definitions rapidsmpf::rapidsmpf INTERFACE_COMPILE_DEFINITIONS
  )
  if(rapidsmpf_compile_definitions)
    foreach(definition IN LISTS rapidsmpf_compile_definitions)
      if(definition STREQUAL "${feature}")
        set(has_feature ON)
      elseif(definition MATCHES "^\\$<\\$<BOOL:([^>]*)>:${feature}>$")
        set(feature_condition "${CMAKE_MATCH_1}")
        if(feature_condition)
          set(has_feature ON)
        endif()
      endif()
    endforeach()
  endif()
  set(${result}
      "${has_feature}"
      PARENT_SCOPE
  )
endfunction()

# Configure the optional communication target after rapidsmpf reports exported feature support.
function(cudf_streaming_configure_optional_communication)
  set(CUDF_STREAMING_HAVE_MPI OFF)
  set(CUDF_STREAMING_HAVE_UCXX OFF)
  set(CUDF_STREAMING_HAVE_COMM OFF)

  cudf_streaming_rapidsmpf_exports_feature(RAPIDSMPF_HAVE_MPI rapidsmpf_have_mpi)
  cudf_streaming_rapidsmpf_exports_feature(RAPIDSMPF_HAVE_UCXX rapidsmpf_have_ucxx)

  if(CUDF_STREAMING_FOUND_MPI AND rapidsmpf_have_mpi)
    set(CUDF_STREAMING_HAVE_MPI ON)
  endif()
  if(CUDF_STREAMING_FOUND_UCXX AND rapidsmpf_have_ucxx)
    set(CUDF_STREAMING_HAVE_UCXX ON)
  endif()
  if(CUDF_STREAMING_HAVE_MPI OR CUDF_STREAMING_HAVE_UCXX)
    set(CUDF_STREAMING_HAVE_COMM ON)
  endif()

  add_library(cudf_streaming_optional_communication INTERFACE)
  target_compile_definitions(
    cudf_streaming_optional_communication
    INTERFACE $<$<BOOL:${CUDF_STREAMING_HAVE_MPI}>:CUDF_STREAMING_HAVE_MPI>
              $<$<BOOL:${CUDF_STREAMING_HAVE_UCXX}>:CUDF_STREAMING_HAVE_UCXX>
  )
  if(CUDF_STREAMING_HAVE_MPI)
    target_link_libraries(cudf_streaming_optional_communication INTERFACE MPI::MPI_CXX)
  endif()
  if(CUDF_STREAMING_HAVE_UCXX)
    target_link_libraries(cudf_streaming_optional_communication INTERFACE ucxx::ucxx)
  endif()
  add_library(cudf_streaming::optional_communication ALIAS cudf_streaming_optional_communication)

  message(STATUS "CUDF_STREAMING: MPI support: ${CUDF_STREAMING_HAVE_MPI}")
  message(STATUS "CUDF_STREAMING: UCXX support: ${CUDF_STREAMING_HAVE_UCXX}")

  set(CUDF_STREAMING_HAVE_MPI
      "${CUDF_STREAMING_HAVE_MPI}"
      PARENT_SCOPE
  )
  set(CUDF_STREAMING_HAVE_UCXX
      "${CUDF_STREAMING_HAVE_UCXX}"
      PARENT_SCOPE
  )
  set(CUDF_STREAMING_HAVE_COMM
      "${CUDF_STREAMING_HAVE_COMM}"
      PARENT_SCOPE
  )
endfunction()

# Emit a standardized status message for targets disabled by unavailable optional dependencies.
function(cudf_streaming_skip_optional_target target reason)
  message(STATUS "CUDF_STREAMING: Skipping ${target}: ${reason}")
endfunction()
