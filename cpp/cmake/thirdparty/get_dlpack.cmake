# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

# This function finds dlpack and sets any additional necessary environment variables.
function(find_and_configure_dlpack VERSION EXCLUDE_FROM_ALL)

  include(${rapids-cmake-dir}/find/generate_module.cmake)
  rapids_find_generate_module(dlpack HEADER_NAMES dlpack/dlpack.h)

  rapids_cpm_find(
    dlpack ${VERSION}
    GIT_REPOSITORY https://github.com/dmlc/dlpack.git
    GIT_TAG v${VERSION}
    GIT_SHALLOW TRUE
    DOWNLOAD_ONLY TRUE
    EXCLUDE_FROM_ALL ${EXCLUDE_FROM_ALL}
    OPTIONS "BUILD_MOCK OFF"
  )

  if(dlpack_SOURCE_DIR)
    set(dlpack_INCLUDE_DIR "${dlpack_SOURCE_DIR}/include")
  else()
    get_target_property(dlpack_INCLUDE_DIR dlpack::dlpack INTERFACE_INCLUDE_DIRECTORIES)
  endif()
  set(DLPACK_INCLUDE_DIR
      "${dlpack_INCLUDE_DIR}"
      PARENT_SCOPE
  )
endfunction()

set(CUDF_MIN_VERSION_dlpack 0.8)

if(NOT DEFINED CUDF_EXCLUDE_DEPS_FROM_ALL)
  set(CUDF_EXCLUDE_DEPS_FROM_ALL OFF)
endif()

find_and_configure_dlpack(${CUDF_MIN_VERSION_dlpack} ${CUDF_EXCLUDE_DEPS_FROM_ALL})
