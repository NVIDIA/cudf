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
    # gold can resolve hidden weak tracing references to non-null, invalid callbacks.
    foreach(config IN ITEMS "${CMAKE_BUILD_TYPE}" ${CMAKE_CONFIGURATION_TYPES})
      string(TOUPPER "${config}" config)
      foreach(flag IN ITEMS CMAKE_CXX_FLAGS CMAKE_EXE_LINKER_FLAGS CMAKE_SHARED_LINKER_FLAGS
                            CMAKE_MODULE_LINKER_FLAGS
      )
        if(CMAKE_CXX_COMPILER_LINKER_ID STREQUAL "GNUgold"
           OR CMAKE_LINKER_TYPE STREQUAL "GOLD"
           OR "${${flag}} ${${flag}_${config}}" MATCHES "-fuse-ld=([^ ;]*/)?(ld[.])?gold([ ;]|$)"
        )
          message(
            FATAL_ERROR
              "Packaged Zstd is unsupported with gold. Use BFD or LLD, or set CUDF_DOWNLOAD_ZSTD=ON."
          )
        endif()
      endforeach()
    endforeach()

    find_package(zstd 1.5.7 EXACT CONFIG REQUIRED)
    # Conda omits the static target in 1.5.7. Once we upgrade to 1.6, use zstd::libzstd_static
    # instead of reconstructing its interface here.

    # Clear older cache entries so a changed package prefix cannot reuse the wrong archive.
    unset(CUDF_ZSTD_STATIC_LIBRARY CACHE)
    find_file(
      CUDF_ZSTD_STATIC_LIBRARY libzstd.a
      PATHS "${zstd_DIR}/../.."
      NO_DEFAULT_PATH NO_CACHE REQUIRED
    )
    message(STATUS "Using packaged Zstd archive: ${CUDF_ZSTD_STATIC_LIBRARY}")
    get_target_property(package_includes zstd::libzstd_shared INTERFACE_INCLUDE_DIRECTORIES)
    add_library(
      cudf_zstd_tracing OBJECT
      "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../../src/io/comp/zstd_tracing.cpp"
    )
    set_target_properties(cudf_zstd_tracing PROPERTIES POSITION_INDEPENDENT_CODE ON)
    add_library(zstd STATIC IMPORTED GLOBAL)
    # Pass the shim as a link input so intermediate static archives cannot discard it.
    set_target_properties(
      zstd
      PROPERTIES IMPORTED_LOCATION "${CUDF_ZSTD_STATIC_LIBRARY}"
                 INTERFACE_INCLUDE_DIRECTORIES "${package_includes}"
                 INTERFACE_COMPILE_DEFINITIONS ZSTD_STATIC_LINKING_ONLY=0N
                 INTERFACE_LINK_LIBRARIES "Threads::Threads;$<TARGET_OBJECTS:cudf_zstd_tracing>"
    )
    set(ZSTD_INCLUDE_DIR
        "${package_includes}"
        PARENT_SCOPE
    )
    return()
  endif()

  rapids_cpm_find(
    zstd 1.5.7
    GLOBAL_TARGETS zstd
    CPM_ARGS
    GIT_REPOSITORY https://github.com/facebook/zstd.git
    GIT_TAG v1.5.7
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
