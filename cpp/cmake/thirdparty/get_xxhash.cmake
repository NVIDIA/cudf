# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

function(find_and_configure_xxhash)

  if(NOT TARGET xxhash
     AND NOT CPM_DOWNLOAD_ALL
     AND NOT CPM_DOWNLOAD_xxhash
     AND NOT CPM_xxhash_SOURCE
  )
    find_path(xxhash_package_include_dir NAMES xxhash.h NO_CACHE)
    if(xxhash_package_include_dir AND EXISTS "${xxhash_package_include_dir}/xxh3.h")
      file(STRINGS "${xxhash_package_include_dir}/xxhash.h" version_parts
           REGEX "^#define XXH_VERSION_(MAJOR|MINOR|RELEASE) +[0-9]+$"
      )
      list(TRANSFORM version_parts REPLACE "^.* +([0-9]+)$" "\\1")
      list(JOIN version_parts "." header_version)
      if(header_version VERSION_EQUAL "0.8.3")
        message(STATUS "Using packaged xxHash headers: ${xxhash_package_include_dir}")
        # cuDF and RTCX use the header-only API, so do not link the packaged library.
        add_library(xxhash INTERFACE IMPORTED GLOBAL)
        set_target_properties(
          xxhash PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${xxhash_package_include_dir}"
        )
        set(XXHASH_INCLUDE_DIR
            "${xxhash_package_include_dir}"
            PARENT_SCOPE
        )
        return()
      endif()
    endif()
  endif()

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
