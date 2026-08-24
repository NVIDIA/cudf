# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

# Use CPM to find or clone xxHash
function(find_and_configure_xxhash)

  include(${rapids-cmake-dir}/find/generate_module.cmake)
  rapids_find_generate_module(xxhash HEADER_NAMES xxhash.h xxh3.h)

  set(CPM_DOWNLOAD_xxhash ON)
  rapids_cpm_find(
    xxhash 0.8.3
    GLOBAL_TARGETS xxhash
    CPM_ARGS
    GIT_REPOSITORY https://github.com/Cyan4973/xxHash.git
    GIT_TAG e626a72bc2321cd320e953a0ccf1584cad60f363
    GIT_SHALLOW TRUE
    DOWNLOAD_ONLY TRUE
    EXCLUDE_FROM_ALL ${CUDF_EXCLUDE_DEPS_FROM_ALL}
  )

  if(xxhash_ADDED AND NOT TARGET xxhash)
    add_library(xxhash INTERFACE)
    target_include_directories(xxhash INTERFACE "${xxhash_SOURCE_DIR}")
  endif()

  if(DEFINED xxhash_SOURCE_DIR)
    set(XXHASH_INCLUDE_DIR
        "${xxhash_SOURCE_DIR}"
        PARENT_SCOPE
    )
    set(xxhash_SOURCE_DIR
        "${xxhash_SOURCE_DIR}"
        PARENT_SCOPE
    )
  endif()

  if(DEFINED xxhash_SOURCE_DIR)
    include("${rapids-cmake-dir}/export/find_package_root.cmake")
    rapids_export_find_package_root(BUILD xxhash "${xxhash_SOURCE_DIR}" EXPORT_SET cudf-exports)
  endif()
endfunction()

find_and_configure_xxhash()
