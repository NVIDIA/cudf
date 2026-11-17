# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

# Finding arrow is far more complex than it should be, and as a result we violate multiple linting
# rules aiming to limit complexity. Since all our other CMake scripts conform to expectations
# without undue difficulty, disabling those rules for just this function is our best approach for
# now. The spacing between this comment, the cmake-lint directives, and the function docstring is
# necessary to prevent cmake-format from trying to combine the lines.

# cmake-lint: disable=R0912,R0913,R0915

include_guard(GLOBAL)

# This function finds arrow and sets any additional necessary environment variables.
function(find_and_configure_arrow VERSION BUILD_STATIC EXCLUDE_FROM_ALL ENABLE_PARQUET
         ENABLE_COMPUTE
)
  if(BUILD_STATIC)
    if(TARGET arrow_static)
      set(ARROW_FOUND
          TRUE
          PARENT_SCOPE
      )
      if(ENABLE_COMPUTE)
        set(ARROW_LIBRARIES
            arrow_static arrow_compute_static
            PARENT_SCOPE
        )
      else()
        set(ARROW_LIBRARIES
            arrow_static
            PARENT_SCOPE
        )
      endif()
      return()
    endif()
  else()
    if(TARGET arrow_shared)
      set(ARROW_FOUND
          TRUE
          PARENT_SCOPE
      )
      if(ENABLE_COMPUTE)
        set(ARROW_LIBRARIES
            arrow_shared arrow_compute_shared
            PARENT_SCOPE
        )
      else()
        set(ARROW_LIBRARIES
            arrow_shared
            PARENT_SCOPE
        )
      endif()
      return()
    endif()
  endif()

  if(NOT ARROW_SIMD_LEVEL)
    set(ARROW_SIMD_LEVEL "NONE")
  endif()

  if(BUILD_STATIC)
    set(ARROW_BUILD_STATIC ON)
    set(ARROW_BUILD_SHARED OFF)
    # Turn off CPM using `find_package` so we always download and make sure we get proper static
    # library.
    set(CPM_DOWNLOAD_Arrow TRUE)
    # By default ARROW will try to search for a static version of OpenSSL which is a bad idea given
    # that shared linking is advised for critical components like SSL. If a static build is
    # requested, we honor ARROW's default of static linking, but users may consider setting
    # ARROW_OPENSSL_USE_SHARED even in static builds.
  else()
    set(ARROW_BUILD_SHARED ON)
    set(ARROW_BUILD_STATIC OFF)
    # By default ARROW will try to search for a static version of OpenSSL which is a bad idea given
    # that shared linking is advised for critical components like SSL
    set(ARROW_OPENSSL_USE_SHARED ON)
  endif()

  set(ARROW_PARQUET_OPTIONS "")
  if(ENABLE_PARQUET)
    # Arrow's logic to build Boost from source is busted, so we have to get it from the system.
    list(APPEND ARROW_PARQUET_OPTIONS "BOOST_SOURCE SYSTEM")
    list(APPEND ARROW_PARQUET_OPTIONS "Thrift_SOURCE BUNDLED")
    list(APPEND ARROW_PARQUET_OPTIONS "ARROW_DEPENDENCY_SOURCE AUTO")
  endif()

  rapids_cpm_find(
    Arrow ${VERSION}
    GLOBAL_TARGETS
      arrow_shared parquet_shared arrow_acero_shared arrow_dataset_shared arrow_compute_shared
      arrow_static parquet_static arrow_acero_static arrow_dataset_static arrow_compute_static
    CPM_ARGS
    GIT_REPOSITORY https://github.com/apache/arrow.git
    GIT_TAG apache-arrow-${VERSION}
    GIT_SHALLOW TRUE SOURCE_SUBDIR cpp
    EXCLUDE_FROM_ALL ${EXCLUDE_FROM_ALL}
    OPTIONS "CMAKE_VERBOSE_MAKEFILE ON"
            "ARROW_ACERO ON"
            "ARROW_COMPUTE ${ENABLE_COMPUTE}"
            "ARROW_IPC ON"
            "ARROW_DATASET ON"
            "ARROW_WITH_BACKTRACE ON"
            "ARROW_CXXFLAGS -w"
            "ARROW_JEMALLOC OFF"
            "ARROW_S3 OFF"
            "ARROW_ORC OFF"
            ${ARROW_PARQUET_OPTIONS}
            "ARROW_PARQUET ${ENABLE_PARQUET}"
            "ARROW_FILESYSTEM ON"
            "ARROW_PYTHON OFF"
            # Arrow modifies CMake's GLOBAL RULE_LAUNCH_COMPILE unless this is off
            "ARROW_USE_CCACHE OFF"
            "ARROW_SIMD_LEVEL ${ARROW_SIMD_LEVEL}"
            "ARROW_BUILD_STATIC ${ARROW_BUILD_STATIC}"
            "ARROW_BUILD_SHARED ${ARROW_BUILD_SHARED}"
            "ARROW_POSITION_INDEPENDENT_CODE ON"
            "ARROW_DEPENDENCY_USE_SHARED ${ARROW_BUILD_SHARED}"
            "ARROW_BOOST_USE_SHARED ${ARROW_BUILD_SHARED}"
            "ARROW_BROTLI_USE_SHARED ${ARROW_BUILD_SHARED}"
            "ARROW_GFLAGS_USE_SHARED ${ARROW_BUILD_SHARED}"
            "ARROW_GRPC_USE_SHARED ${ARROW_BUILD_SHARED}"
            "ARROW_PROTOBUF_USE_SHARED ${ARROW_BUILD_SHARED}"
            "ARROW_ZSTD_USE_SHARED ${ARROW_BUILD_SHARED}"
            "xsimd_SOURCE AUTO"
  )

  set(ARROW_FOUND
      TRUE
      PARENT_SCOPE
  )

  if(BUILD_STATIC)
    if(ENABLE_COMPUTE)
      set(ARROW_LIBRARIES arrow_static arrow_compute_static)
    else()
      set(ARROW_LIBRARIES arrow_static)
    endif()
  else()
    if(ENABLE_COMPUTE)
      set(ARROW_LIBRARIES arrow_shared arrow_compute_shared)
    else()
      set(ARROW_LIBRARIES arrow_shared)
    endif()
  endif()

  # Arrow_DIR:   set if CPM found Arrow on the system/conda/etc.
  if(Arrow_DIR)
    # This extra find_package is necessary because rapids_cpm_find does not propagate all the
    # variables from find_package that we might need. This is especially problematic when
    # rapids_cpm_find builds from source.
    find_package(Arrow REQUIRED QUIET)
    if(ENABLE_PARQUET)
      # Setting Parquet_DIR is conditional because parquet may be installed independently of arrow.
      if(NOT Parquet_DIR)
        # Set this to enable `find_package(Parquet)`
        set(Parquet_DIR "${Arrow_DIR}")
      endif()
      # Set this to enable `find_package(ArrowDataset)`. This will call find_package(ArrowAcero) for
      # us
      set(ArrowDataset_DIR "${Arrow_DIR}")
      find_package(ArrowDataset REQUIRED QUIET)
      # Set this to enable `find_package(ArrowCompute)`
      set(ArrowCompute_DIR "${Arrow_DIR}")
      find_package(ArrowCompute REQUIRED QUIET)
    endif()
    # Arrow_ADDED: set if CPM downloaded Arrow from Github
  elseif(Arrow_ADDED)
    # Copy these files so we can avoid adding paths in Arrow_BINARY_DIR to
    # target_include_directories. That defeats ccache.
    file(INSTALL "${Arrow_BINARY_DIR}/src/arrow/util/config.h"
         DESTINATION "${Arrow_SOURCE_DIR}/cpp/src/arrow/util"
    )
    if(ENABLE_PARQUET)
      file(INSTALL "${Arrow_BINARY_DIR}/src/parquet/parquet_version.h"
           DESTINATION "${Arrow_SOURCE_DIR}/cpp/src/parquet"
      )
    endif()
    # Arrow populates INTERFACE_INCLUDE_DIRECTORIES for the `arrow_static` and `arrow_shared`
    # targets in FindArrow, so for static source-builds, we have to do it after-the-fact.
    #
    # This only works because we know exactly which components we're using. Don't forget to update
    # this list if we add more!
    #
    foreach(ARROW_LIBRARY ${ARROW_LIBRARIES})
      target_include_directories(
        ${ARROW_LIBRARY}
        INTERFACE "$<BUILD_INTERFACE:${Arrow_SOURCE_DIR}/cpp/src>"
                  "$<BUILD_INTERFACE:${Arrow_SOURCE_DIR}/cpp/src/generated>"
                  "$<BUILD_INTERFACE:${Arrow_SOURCE_DIR}/cpp/thirdparty/hadoop/include>"
                  "$<BUILD_INTERFACE:${Arrow_SOURCE_DIR}/cpp/thirdparty/flatbuffers/include>"
      )
    endforeach()
  else()
    set(ARROW_FOUND
        FALSE
        PARENT_SCOPE
    )
    message(FATAL_ERROR "CUDF: Arrow library not found or downloaded.")
  endif()

  if(Arrow_ADDED)
    set(arrow_code_string
        [=[
          if (TARGET cudf::arrow_shared AND (NOT TARGET arrow_shared))
              add_library(arrow_shared ALIAS cudf::arrow_shared)
          endif()
          if (TARGET cudf::arrow_static AND (NOT TARGET arrow_static))
              add_library(arrow_static ALIAS cudf::arrow_static)
          endif()
          if (NOT TARGET arrow::flatbuffers)
            add_library(arrow::flatbuffers INTERFACE IMPORTED)
          endif()
          if (NOT TARGET arrow::hadoop)
            add_library(arrow::hadoop INTERFACE IMPORTED)
          endif()
        ]=]
    )
    if(ENABLE_PARQUET)
      string(
        APPEND
        arrow_code_string
        "
          find_package(Boost)
          if (NOT TARGET Boost::headers)
            add_library(Boost::headers INTERFACE IMPORTED)
          endif()
        "
      )
    endif()
    if(NOT TARGET xsimd)
      string(
        APPEND
        arrow_code_string
        "
          if(NOT TARGET arrow::xsimd)
            add_library(arrow::xsimd INTERFACE IMPORTED)
            target_include_directories(arrow::xsimd INTERFACE \"${Arrow_BINARY_DIR}/xsimd_ep/src/xsimd_ep-install/include\")
          endif()
        "
      )
    endif()
    rapids_cmake_install_lib_dir(lib_dir)
    if(TARGET arrow_static)
      get_target_property(interface_libs arrow_static INTERFACE_LINK_LIBRARIES)
      # The `arrow_static` library is leaking a dependency on the object libraries it was built with
      # we need to remove this from the interface, since keeping them around would cause duplicate
      # symbols and CMake export errors
      if(interface_libs MATCHES "arrow_array" AND interface_libs MATCHES "arrow_compute")
        string(REPLACE "BUILD_INTERFACE:" "BUILD_LOCAL_INTERFACE:" interface_libs
                       "${interface_libs}"
        )
        set_target_properties(arrow_static PROPERTIES INTERFACE_LINK_LIBRARIES "${interface_libs}")
        get_target_property(interface_libs arrow_static INTERFACE_LINK_LIBRARIES)
      endif()
    endif()

    include(rapids-export)
    if(NOT EXCLUDE_FROM_ALL)
      rapids_export(
        BUILD Arrow
        VERSION ${VERSION}
        EXPORT_SET arrow_targets
        GLOBAL_TARGETS arrow_shared arrow_static arrow_compute_static arrow_compute_shared
        NAMESPACE cudf::
        FINAL_CODE_BLOCK arrow_code_string
      )

      if(ENABLE_PARQUET)
        set(arrow_acero_code_string
            [=[
                if (TARGET cudf::arrow_acero_shared AND (NOT TARGET arrow_acero_shared))
                    add_library(arrow_acero_shared ALIAS cudf::arrow_acero_shared)
                endif()
                if (TARGET cudf::arrow_acero_static AND (NOT TARGET arrow_acero_static))
                    add_library(arrow_acero_static ALIAS cudf::arrow_acero_static)
                endif()
              ]=]
        )

        rapids_export(
          BUILD ArrowAcero
          VERSION ${VERSION}
          EXPORT_SET arrow_acero_targets
          GLOBAL_TARGETS arrow_acero_shared arrow_acero_static
          NAMESPACE cudf::
          FINAL_CODE_BLOCK arrow_acero_code_string
        )

        set(arrow_dataset_code_string
            [=[
                if (TARGET cudf::arrow_dataset_shared AND (NOT TARGET arrow_dataset_shared))
                    add_library(arrow_dataset_shared ALIAS cudf::arrow_dataset_shared)
                endif()
                if (TARGET cudf::arrow_dataset_static AND (NOT TARGET arrow_dataset_static))
                    add_library(arrow_dataset_static ALIAS cudf::arrow_dataset_static)
                endif()
              ]=]
        )

        rapids_export(
          BUILD ArrowDataset
          VERSION ${VERSION}
          EXPORT_SET arrow_dataset_targets
          GLOBAL_TARGETS arrow_dataset_shared arrow_dataset_static
          NAMESPACE cudf::
          FINAL_CODE_BLOCK arrow_dataset_code_string
        )

        set(parquet_code_string
            [=[
                if (TARGET cudf::parquet_shared AND (NOT TARGET parquet_shared))
                    add_library(parquet_shared ALIAS cudf::parquet_shared)
                endif()
                if (TARGET cudf::parquet_static AND (NOT TARGET parquet_static))
                    add_library(parquet_static ALIAS cudf::parquet_static)
                endif()
              ]=]
        )

        rapids_export(
          BUILD Parquet
          VERSION ${VERSION}
          EXPORT_SET parquet_targets
          GLOBAL_TARGETS parquet_shared parquet_static
          NAMESPACE cudf::
          FINAL_CODE_BLOCK parquet_code_string
        )
      endif()
    endif()
  endif()

  if(NOT EXCLUDE_FROM_ALL)
    # We generate the arrow-configfiles when we built arrow locally, so always do `find_dependency`
    rapids_export_package(BUILD Arrow cudf-exports)
    rapids_export_package(INSTALL Arrow cudf-exports)

    if(ENABLE_PARQUET)
      rapids_export_package(BUILD Parquet cudf-exports)
      rapids_export_package(BUILD ArrowDataset cudf-exports)
    endif()
    rapids_export_package(BUILD ArrowCompute cudf-exports)

    include("${rapids-cmake-dir}/export/find_package_root.cmake")
    rapids_export_find_package_root(
      BUILD Arrow [=[${CMAKE_CURRENT_LIST_DIR}]=] EXPORT_SET cudf-exports
    )
    rapids_export_find_package_root(
      BUILD Parquet [=[${CMAKE_CURRENT_LIST_DIR}]=]
      EXPORT_SET cudf-exports
      CONDITION ENABLE_PARQUET
    )
    rapids_export_find_package_root(
      BUILD ArrowDataset [=[${CMAKE_CURRENT_LIST_DIR}]=]
      EXPORT_SET cudf-exports
      CONDITION ENABLE_PARQUET
    )
    rapids_export_find_package_root(
      BUILD ArrowCompute [=[${CMAKE_CURRENT_LIST_DIR}]=] EXPORT_SET cudf-exports
    )
  endif()

  set(ARROW_LIBRARIES
      "${ARROW_LIBRARIES}"
      PARENT_SCOPE
  )
endfunction()

if(NOT DEFINED CUDF_VERSION_Arrow)
  set(CUDF_VERSION_Arrow
      # This version must be kept in sync with the libarrow version pinned for builds in
      # dependencies.yaml.
      21.0.0
      CACHE STRING "The version of Arrow to find (or build)"
  )
endif()

# Default to static arrow builds
if(NOT DEFINED CUDF_ENABLE_ARROW_PARQUET)
  set(CUDF_ENABLE_ARROW_PARQUET OFF)
endif()

if(NOT DEFINED CUDF_ENABLE_ARROW_COMPUTE)
  set(CUDF_ENABLE_ARROW_COMPUTE OFF)
endif()

# Derive arrow build mode from CUDF_BUILD_STATIC_DEPS
if(CUDF_BUILD_STATIC_DEPS STREQUAL "OFF")
  set(_cudf_arrow_static OFF)
else()
  set(_cudf_arrow_static ON)
endif()
if(CUDF_INSTALL_LIBRARY_DEPS)
  set(_cudf_arrow_exclude_from_all OFF)
else()
  set(_cudf_arrow_exclude_from_all ON)
endif()

find_and_configure_arrow(
  ${CUDF_VERSION_Arrow} ${_cudf_arrow_static} ${_cudf_arrow_exclude_from_all}
  ${CUDF_ENABLE_ARROW_PARQUET} ${CUDF_ENABLE_ARROW_COMPUTE}
)
