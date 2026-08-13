# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

#[=======================================================================[.rst:
find_and_configure_cudf
-----------------------

Find cudf and set any additional necessary environment variables.

``VERSION``
  The cudf version to find.

``EXPORT_SET``
  The CMake export set name to associate with this dependency.

Result Variables
^^^^^^^^^^^^^^^^

  :cmake:variable:`<PackageName>_SOURCE_DIR` is set to the path to the source directory of <PackageName>.
  :cmake:variable:`<PackageName>_BINARY_DIR`  is set to the path to the build directory of  <PackageName>.
  :cmake:variable:`<PackageName>_ADDED`      is set to a true value if <PackageName> has not been added before.

#]=======================================================================]
# cmake-lint: disable=C0112
function(find_and_configure_cudf VERSION EXPORT_SET)
  rapids_cmake_parse_version(MAJOR_MINOR ${VERSION} major_minor)
  set(cudf_components)
  if(BUILD_TESTS)
    list(APPEND cudf_components COMPONENTS testing)
  endif()
  rapids_cpm_find(
    cudf ${VERSION} ${cudf_components}
    BUILD_EXPORT_SET ${EXPORT_SET}
    INSTALL_EXPORT_SET ${EXPORT_SET}
    CPM_ARGS
    GIT_REPOSITORY https://github.com/rapidsai/cudf.git
    GIT_TAG "${RAPIDS_BRANCH}"
    GIT_SHALLOW TRUE SOURCE_SUBDIR cpp
    OPTIONS "BUILD_TESTS OFF" "BUILD_BENCHMARKS OFF"
  )
  # If after loading cudf we now have the CMAKE_CUDA_COMPILER variable we know that we need to
  # re-enable the cuda language.
  if(CMAKE_CUDA_COMPILER)
    set(cudf_REQUIRES_CUDA
        TRUE
        PARENT_SCOPE
    )
  endif()
endfunction()
