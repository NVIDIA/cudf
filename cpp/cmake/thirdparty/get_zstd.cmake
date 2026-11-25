# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

# Use CPM to find or clone libzstd
function(find_and_configure_zstd)

  set(CPM_DOWNLOAD_zstd ON)
  if(DEFINED CUDF_DOWNLOAD_ZSTD)
    set(CPM_DOWNLOAD_zstd ${CUDF_DOWNLOAD_ZSTD})
  endif()

  if(NOT CPM_DOWNLOAD_zstd)
    find_package(zstd 1.6.0 EXACT CONFIG REQUIRED)
    add_library(
      cudf_zstd_tracing OBJECT
      "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../../src/io/comp/zstd_tracing.cpp"
    )
    set_target_properties(cudf_zstd_tracing PROPERTIES POSITION_INDEPENDENT_CODE ON)
    add_library(zstd INTERFACE)
    target_compile_definitions(zstd INTERFACE ZSTD_STATIC_LINKING_ONLY=0N)
    # Pass the shim as a link input so intermediate static archives cannot discard it.
    target_link_libraries(zstd INTERFACE zstd::libzstd_static $<TARGET_OBJECTS:cudf_zstd_tracing>)
    get_target_property(ZSTD_INCLUDE_DIR zstd::libzstd_static INTERFACE_INCLUDE_DIRECTORIES)
    set(ZSTD_INCLUDE_DIR
        "${ZSTD_INCLUDE_DIR}"
        PARENT_SCOPE
    )
    return()
  endif()

  rapids_cpm_find(
    zstd 1.6.0
    GLOBAL_TARGETS zstd
    CPM_ARGS
    GIT_REPOSITORY https://github.com/facebook/zstd.git
    GIT_TAG v1.6.0
    GIT_SHALLOW FALSE SOURCE_SUBDIR build/cmake
    EXCLUDE_FROM_ALL ${CUDF_EXCLUDE_DEPS_FROM_ALL}
    OPTIONS "ZSTD_BUILD_STATIC ON" "ZSTD_BUILD_SHARED OFF" "ZSTD_BUILD_TESTS OFF"
            "ZSTD_BUILD_PROGRAMS OFF" "BUILD_SHARED_LIBS OFF"
  )

  if(zstd_ADDED)
    # disable weak symbols support to hide tracing APIs as well
    target_compile_definitions(libzstd_static PRIVATE ZSTD_HAVE_WEAK_SYMBOLS=0)
    # expose experimental API
    target_compile_definitions(libzstd_static PUBLIC ZSTD_STATIC_LINKING_ONLY=0N)
    # suppress warnings from uninitialized variables and redefining ZSTD_STATIC_LINKING_ONLY
    target_compile_options(libzstd_static PRIVATE -w)
    add_library(zstd ALIAS libzstd_static)
  endif()

  if(DEFINED zstd_SOURCE_DIR)
    set(ZSTD_INCLUDE_DIR
        "${zstd_SOURCE_DIR}/lib"
        PARENT_SCOPE
    )
  endif()
  include("${rapids-cmake-dir}/export/find_package_root.cmake")
  rapids_export_find_package_root(BUILD zstd "${zstd_BINARY_DIR}" EXPORT_SET cudf-exports)

endfunction()

find_and_configure_zstd()
