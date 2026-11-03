# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

# This function finds cuCollections and performs any additional configuration.
function(find_and_configure_cucollections)
  include(${rapids-cmake-dir}/cpm/cuco.cmake)

  rapids_cpm_cuco(${CUDF_EXCLUDE_DEPS_FROM_ALL_FLAG})
endfunction()

find_and_configure_cucollections()
