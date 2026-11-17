# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

# This function finds KvikIO
function(find_and_configure_kvikio VERSION)

  rapids_cpm_find(
    kvikio ${VERSION}
    GLOBAL_TARGETS kvikio::kvikio
    CPM_ARGS
    GIT_REPOSITORY https://github.com/rapidsai/kvikio.git
    GIT_TAG "${RAPIDS_BRANCH}"
    GIT_SHALLOW TRUE SOURCE_SUBDIR cpp
    EXCLUDE_FROM_ALL ${CUDF_EXCLUDE_DEPS_FROM_ALL}
    OPTIONS "KvikIO_BUILD_EXAMPLES OFF" "KvikIO_REMOTE_SUPPORT ${CUDF_KVIKIO_REMOTE_IO}"
            "BUILD_SHARED_LIBS OFF"
  )

endfunction()

set(KVIKIO_MIN_VERSION_cudf "${CUDF_VERSION_MAJOR}.${CUDF_VERSION_MINOR}")
find_and_configure_kvikio(${KVIKIO_MIN_VERSION_cudf})
