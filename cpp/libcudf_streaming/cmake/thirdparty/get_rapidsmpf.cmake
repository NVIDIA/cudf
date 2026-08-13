# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

# This function finds rapidsmpf and sets any additional necessary environment variables.
function(find_and_configure_rapidsmpf VERSION)
  rapids_cmake_parse_version(MAJOR_MINOR ${VERSION} major_minor)
  set(rapidsmpf_build_mpi_support OFF)
  set(rapidsmpf_build_ucxx_support OFF)
  if(CUDF_STREAMING_FOUND_MPI)
    set(rapidsmpf_build_mpi_support ON)
  endif()
  if(CUDF_STREAMING_FOUND_UCXX)
    set(rapidsmpf_build_ucxx_support ON)
  endif()
  rapids_cpm_find(
    rapidsmpf ${VERSION}
    BUILD_EXPORT_SET cudf_streaming-exports
    INSTALL_EXPORT_SET cudf_streaming-exports
    CPM_ARGS
    GIT_REPOSITORY https://github.com/rapidsai/rapidsmpf.git
    GIT_TAG "${RAPIDS_BRANCH}"
    GIT_SHALLOW TRUE SOURCE_SUBDIR cpp
    OPTIONS "BUILD_MPI_SUPPORT ${rapidsmpf_build_mpi_support}"
            "BUILD_UCXX_SUPPORT ${rapidsmpf_build_ucxx_support}"
            "BUILD_SLURM_SUPPORT OFF"
            "BUILD_TESTS OFF"
            "BUILD_BENCHMARKS OFF"
            "BUILD_EXAMPLES OFF"
  )
endfunction()

set(CUDF_STREAMING_MIN_VERSION
    "${CUDF_STREAMING_VERSION_MAJOR}.${CUDF_STREAMING_VERSION_MINOR}.${CUDF_STREAMING_VERSION_PATCH}"
)
find_and_configure_rapidsmpf(${CUDF_STREAMING_MIN_VERSION})
