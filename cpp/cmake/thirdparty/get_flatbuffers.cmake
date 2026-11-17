# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

# Use CPM to find or clone flatbuffers
function(find_and_configure_flatbuffers VERSION EXCLUDE_FROM_ALL)

  rapids_cpm_find(
    flatbuffers ${VERSION}
    GLOBAL_TARGETS flatbuffers
    CPM_ARGS
    GIT_REPOSITORY https://github.com/google/flatbuffers.git
    GIT_TAG v${VERSION}
    GIT_SHALLOW TRUE
    EXCLUDE_FROM_ALL ${EXCLUDE_FROM_ALL}
    OPTIONS "FLATBUFFERS_BUILD_TESTS OFF"
  )

  include("${rapids-cmake-dir}/export/find_package_root.cmake")
  rapids_export_find_package_root(
    BUILD flatbuffers "${flatbuffers_BINARY_DIR}" EXPORT_SET cudf-exports
  )

endfunction()

find_and_configure_flatbuffers(24.3.25 ${CUDF_EXCLUDE_DEPS_FROM_ALL})
