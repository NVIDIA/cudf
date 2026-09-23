# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

# This function finds rmm and sets any additional necessary environment variables.
function(find_and_configure_rmm BUILD_SHARED EXCLUDE_FROM_ALL)
  include(${rapids-cmake-dir}/cpm/rmm.cmake)

  if(EXCLUDE_FROM_ALL)
    set(_exclude_flag EXCLUDE_FROM_ALL)
  else()
    set(_exclude_flag)
  endif()

  # Find or install RMM.
  set(_rmm_args)
  if(NOT EXCLUDE_FROM_ALL)
    list(APPEND _rmm_args BUILD_EXPORT_SET cudf-exports INSTALL_EXPORT_SET cudf-exports)
  endif()
  rapids_cpm_rmm(${_rmm_args} ${_exclude_flag} CPM_ARGS OPTIONS "BUILD_SHARED_LIBS ${BUILD_SHARED}")

  # If rmm was found as a pre-existing shared library (e.g. conda), we need find_dependency(rmm) in
  # the build and installed configs even when EXCLUDE_FROM_ALL is set. EXCLUDE_FROM_ALL controls
  # whether CPM-built targets are installed, but a pre-existing shared library is not absorbed into
  # libcudf and must be findable by downstream consumers.
  if(EXCLUDE_FROM_ALL)
    get_target_property(_rmm_type rmm::rmm TYPE)
    if(NOT _rmm_type STREQUAL "STATIC_LIBRARY")
      include("${rapids-cmake-dir}/export/package.cmake")
      rapids_export_package(BUILD rmm cudf-exports VERSION ${rmm_VERSION})
      rapids_export_package(INSTALL rmm cudf-exports VERSION ${rmm_VERSION})
    endif()
  endif()

  # Propagate rmm source/binary dirs to parent scope for header installation
  set(rmm_SOURCE_DIR
      "${rmm_SOURCE_DIR}"
      PARENT_SCOPE
  )
  set(rmm_BINARY_DIR
      "${rmm_BINARY_DIR}"
      PARENT_SCOPE
  )
endfunction()

find_and_configure_rmm(${CUDF_DEPS_BUILD_SHARED} ${CUDF_EXCLUDE_DEPS_FROM_ALL})
