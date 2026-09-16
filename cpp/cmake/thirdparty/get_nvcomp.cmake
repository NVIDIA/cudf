# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

# This function finds or downloads nvcomp and sets any additional necessary environment variables.
# The spacing below is necessary to prevent cmake-format from combining the lint directive with the
# surrounding comments.

# cmake-lint: disable=R0915,R0912,C0103

# This function finds nvcomp and sets any additional necessary environment variables.
function(find_and_configure_nvcomp)
  set(options DOWNLOAD_ONLY INSTALL_LIBRARY)
  set(one_value VERSION)
  cmake_parse_arguments(_NVCOMP "${options}" "${one_value}" "" ${ARGN})

  # If DOWNLOAD_ONLY is not set, try searching for an existing installation first
  if(NOT _NVCOMP_DOWNLOAD_ONLY)
    include("${rapids-cmake-dir}/find/package.cmake")
    rapids_find_package(
      nvcomp ${_NVCOMP_VERSION}
      GLOBAL_TARGETS nvcomp::nvcomp nvcomp::nvcomp_static
      FIND_ARGS QUIET
    )
    if(nvcomp_FOUND)
      message(STATUS "Found nvcomp: ${nvcomp_DIR} (found version ${nvcomp_VERSION})")
      return()
    endif()
  endif()

  # Find and download the platform-specific proprietary binary
  include("${rapids-cmake-dir}/cmake/install_lib_dir.cmake")
  rapids_cmake_install_lib_dir(lib_dir)
  find_package(CUDAToolkit REQUIRED)

  if(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
    set(nvcomp_url
        "https://developer.download.nvidia.com/compute/nvcomp/redist/nvcomp/linux-x86_64/nvcomp-linux-x86_64-${_NVCOMP_VERSION}_cuda${CUDAToolkit_VERSION_MAJOR}-archive.tar.xz"
    )
  elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
    set(nvcomp_url
        "https://developer.download.nvidia.com/compute/nvcomp/redist/nvcomp/linux-sbsa/nvcomp-linux-sbsa-${_NVCOMP_VERSION}_cuda${CUDAToolkit_VERSION_MAJOR}-archive.tar.xz"
    )
  else()
    message(FATAL_ERROR "No nvcomp binary available for ${CMAKE_SYSTEM_PROCESSOR}")
  endif()

  include(FetchContent)
  set(pkg_name "nvcomp_proprietary_binary")
  FetchContent_Declare(${pkg_name} URL ${nvcomp_url})
  FetchContent_MakeAvailable(${pkg_name})
  set(nvcomp_ROOT "${nvcomp_proprietary_binary_SOURCE_DIR}")

  # The downloaded nvcomp binary always uses `lib`, but some platforms use `lib64` for 64-bit
  # libraries. If the platform's lib directory is `lib64`, move the contents of the downloaded `lib`
  # directory to a new `lib64` directory and update the CMake config files to point to the new
  # location.
  if(NOT EXISTS "${nvcomp_ROOT}/${lib_dir}/cmake/nvcomp/nvcomp-config.cmake")
    include(GNUInstallDirs)
    cmake_path(GET lib_dir PARENT_PATH lib_dir_parent)
    cmake_path(GET CMAKE_INSTALL_INCLUDEDIR PARENT_PATH include_dir_parent)
    if(NOT lib_dir_parent STREQUAL include_dir_parent)
      message(
        FATAL_ERROR "CMAKE_INSTALL_INCLUDEDIR and CMAKE_INSTALL_LIBDIR must share parent directory"
      )
    endif()

    cmake_path(GET lib_dir FILENAME lib_dir_name)
    set(nvcomp_list_of_target_files
        "nvcomp-targets-common-release.cmake"
        "nvcomp-targets-common.cmake"
        "nvcomp-targets-dynamic-release.cmake"
        "nvcomp-targets-dynamic.cmake"
        "nvcomp-targets-release.cmake"
        "nvcomp-targets-static-release.cmake"
        "nvcomp-targets-static.cmake"
    )
    foreach(filename IN LISTS nvcomp_list_of_target_files)
      if(EXISTS "${nvcomp_ROOT}/lib/cmake/nvcomp/${filename}")
        file(READ "${nvcomp_ROOT}/lib/cmake/nvcomp/${filename}" FILE_CONTENTS)
        string(REPLACE "\$\{_IMPORT_PREFIX\}/lib/" "\$\{_IMPORT_PREFIX\}/${lib_dir_name}/"
                       FILE_CONTENTS ${FILE_CONTENTS}
        )
        file(WRITE "${nvcomp_ROOT}/lib/cmake/nvcomp/${filename}" ${FILE_CONTENTS})
      endif()
    endforeach()
    file(MAKE_DIRECTORY "${nvcomp_ROOT}/${lib_dir_parent}")
    file(RENAME "${nvcomp_ROOT}/lib/" "${nvcomp_ROOT}/${lib_dir}/")
    # Move the `include` dir if necessary as well
    file(RENAME "${nvcomp_ROOT}/include/" "${nvcomp_ROOT}/${CMAKE_INSTALL_INCLUDEDIR}/")
  endif()

  # Now that the downloaded binary is configured correctly, perform a find_package call to generate
  # the necessary targets
  include("${rapids-cmake-dir}/find/package.cmake")
  rapids_find_package(
    nvcomp ${_NVCOMP_VERSION}
    GLOBAL_TARGETS nvcomp::nvcomp nvcomp::nvcomp_static
    FIND_ARGS
    REQUIRED
  )

  include(GNUInstallDirs)
  if(_NVCOMP_INSTALL_LIBRARY)
    install(DIRECTORY "${nvcomp_ROOT}/${lib_dir}/" DESTINATION "${lib_dir}")
    install(DIRECTORY "${nvcomp_ROOT}/${CMAKE_INSTALL_INCLUDEDIR}/"
            DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
    )
    if(EXISTS "${nvcomp_ROOT}/${CMAKE_INSTALL_BINDIR}")
      install(DIRECTORY "${nvcomp_ROOT}/${CMAKE_INSTALL_BINDIR}/"
              DESTINATION "${CMAKE_INSTALL_BINDIR}"
      )
    endif()
    install(
      FILES "${nvcomp_ROOT}/NOTICE"
      DESTINATION info/
      RENAME NVCOMP_NOTICE
    )
    install(
      FILES "${nvcomp_ROOT}/LICENSE"
      DESTINATION info/
      RENAME NVCOMP_LICENSE
    )
  endif()
endfunction()

set(_nvcomp_args VERSION 5.3.0.16)
if(CUDF_BUILD_STATIC_DEPS STREQUAL "FORCE")
  list(APPEND _nvcomp_args DOWNLOAD_ONLY)
endif()
if(CUDF_INSTALL_LIBRARY_DEPS)
  list(APPEND _nvcomp_args INSTALL_LIBRARY)
endif()
find_and_configure_nvcomp(${_nvcomp_args})

if(CUDF_DEPS_BUILD_SHARED OR NOT TARGET nvcomp::nvcomp_static)
  set(CUDF_nvcomp_TARGET nvcomp::nvcomp)
else()
  set(CUDF_nvcomp_TARGET nvcomp::nvcomp_static)
endif()

# Per-thread default stream
if(TARGET nvcomp AND CUDF_USE_PER_THREAD_DEFAULT_STREAM)
  target_compile_definitions(nvcomp PRIVATE CUDA_API_PER_THREAD_DEFAULT_STREAM)
endif()
