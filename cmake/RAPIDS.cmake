# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
#
# This is the preferred entry point for projects using rapids-cmake
#
# Enforce the minimum required CMake version for all users
cmake_minimum_required(VERSION 4.0 FATAL_ERROR)

# Allow users to control which version is used
if(NOT (rapids-cmake-branch OR rapids-cmake-version))
  message(
    FATAL_ERROR "The CMake variable `rapids-cmake-branch` or `rapids-cmake-version` must be defined"
  )
endif()

# Allow users to control which GitHub repo is fetched
if(NOT rapids-cmake-repo)
  # Define a default repo if the user doesn't set one
  set(rapids-cmake-repo rapidsai/rapids-cmake)
endif()

# Allow users to control which branch is fetched
if(NOT rapids-cmake-branch)
  # Define a default branch if the user doesn't set one
  set(rapids-cmake-branch "release/${rapids-cmake-version}")
endif()

# Allow users to control the exact URL passed to FetchContent
if(NOT rapids-cmake-url)
  # Construct a default URL if the user doesn't set one
  set(rapids-cmake-url "https://github.com/${rapids-cmake-repo}/")

  # In order of specificity
  if(rapids-cmake-fetch-via-git)
    if(rapids-cmake-sha)
      # An exact git SHA takes precedence over anything
      set(rapids-cmake-value-to-clone "${rapids-cmake-sha}")
    elseif(rapids-cmake-tag)
      # Followed by a git tag name
      set(rapids-cmake-value-to-clone "${rapids-cmake-tag}")
    else()
      # Or if neither of the above two were defined, use a branch
      set(rapids-cmake-value-to-clone "${rapids-cmake-branch}")
    endif()
  else()
    if(rapids-cmake-sha)
      # An exact git SHA takes precedence over anything
      set(rapids-cmake-value-to-clone "archive/${rapids-cmake-sha}.zip")
    elseif(rapids-cmake-tag)
      # Followed by a git tag name
      set(rapids-cmake-value-to-clone "archive/refs/tags/${rapids-cmake-tag}.zip")
    else()
      # Or if neither of the above two were defined, use a branch
      set(rapids-cmake-value-to-clone "archive/refs/heads/${rapids-cmake-branch}.zip")
    endif()
  endif()
endif()

include(FetchContent)
if(rapids-cmake-fetch-via-git)
  FetchContent_Declare(
    rapids-cmake
    GIT_REPOSITORY "${rapids-cmake-url}"
    GIT_TAG "${rapids-cmake-value-to-clone}"
  )
  message(STATUS "Fetching rapids-cmake from ${rapids-cmake-url}@${rapids-cmake-value-to-clone}")
else()
  string(APPEND rapids-cmake-url "${rapids-cmake-value-to-clone}")
  FetchContent_Declare(rapids-cmake URL "${rapids-cmake-url}")
  message(STATUS "Fetching rapids-cmake from ${rapids-cmake-url}")
endif()
FetchContent_GetProperties(rapids-cmake)
if(rapids-cmake_POPULATED)
  # Something else has already populated rapids-cmake, only thing we need to do is setup the
  # CMAKE_MODULE_PATH
  if(NOT "${rapids-cmake-dir}" IN_LIST CMAKE_MODULE_PATH)
    list(APPEND CMAKE_MODULE_PATH "${rapids-cmake-dir}")
  endif()
else()
  FetchContent_MakeAvailable(rapids-cmake)
endif()
