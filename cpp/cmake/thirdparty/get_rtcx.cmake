# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on

# This function finds rtcx
function(find_and_configure_rtcx VERSION)

  # Ensure rtcx installs its targets unconditionally. In shared builds the static library is
  # absorbed into libcudf.so, but CMake still requires the target to be in an export set for
  # install(EXPORT) validation. In static builds consumers need to link librtcx.a directly.
  set(RTCX_INSTALL ON)

  rapids_cpm_find(
    rtcx ${VERSION}
    GLOBAL_TARGETS rtcx::rtcx
    CPM_ARGS
    GIT_REPOSITORY https://github.com/arhag23/librtcx.git
    GIT_TAG 67b5e05f5282fca9ad04d44b7dc93d75d89f1ad1
    GIT_SHALLOW FALSE
    EXCLUDE_FROM_ALL ${CUDF_EXCLUDE_DEPS_FROM_ALL}
  )

  # When CPM fetches from source (add_subdirectory), embed.cmake is not auto-included. Include it
  # explicitly so add_embed/embed_includes/embed functions are available.
  if(rtcx_ADDED OR DEFINED CPM_rtcx_SOURCE)
    include("${rtcx_SOURCE_DIR}/cmake/modules/embed.cmake")
  endif()
endfunction()

set(RTCX_MIN_VERSION_cudf "0.1")
find_and_configure_rtcx(${RTCX_MIN_VERSION_cudf})
