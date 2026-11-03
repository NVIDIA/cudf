# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

# Need to call rapids_cpm_nvtx3 to get the nvtx3 target available at configure time.
function(find_and_configure_nvtx)
  include(${rapids-cmake-dir}/cpm/nvtx3.cmake)

  # nvtx3 is part of rmm's public interface. Include it in the build export set so that
  # configure-time target resolution works. Only include it in the install export set when rmm is
  # NOT being absorbed — when absorbed, we bundle nvtx3 headers directly and consumers don't need
  # find_dependency(nvtx3).
  set(_nvtx_args BUILD_EXPORT_SET cudf-exports)
  if(CUDF_INSTALL_LIBRARY_DEPS)
    list(APPEND _nvtx_args INSTALL_EXPORT_SET cudf-exports)
  endif()
  rapids_cpm_nvtx3(${_nvtx_args} ${CUDF_EXCLUDE_DEPS_FROM_ALL_FLAG})
endfunction()

find_and_configure_nvtx()
