# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
file(READ "${CMAKE_CURRENT_LIST_DIR}/../VERSION" _rapids_version)
if(_rapids_version MATCHES [[^([0-9][0-9])\.([0-9][0-9])\.([0-9][0-9])]])
  set(RAPIDS_VERSION_MAJOR "${CMAKE_MATCH_1}")
  set(RAPIDS_VERSION_MINOR "${CMAKE_MATCH_2}")
  set(RAPIDS_VERSION_PATCH "${CMAKE_MATCH_3}")
  set(RAPIDS_VERSION_MAJOR_MINOR "${RAPIDS_VERSION_MAJOR}.${RAPIDS_VERSION_MINOR}")
  set(RAPIDS_VERSION "${RAPIDS_VERSION_MAJOR}.${RAPIDS_VERSION_MINOR}.${RAPIDS_VERSION_PATCH}")
else()
  string(REPLACE "\n" "\n  " _rapids_version_formatted "  ${_rapids_version}")
  message(
    FATAL_ERROR
      "Could not determine RAPIDS version. Contents of VERSION file:\n${_rapids_version_formatted}"
  )
endif()

# Use STRINGS to trim whitespace/newlines
file(STRINGS "${CMAKE_CURRENT_LIST_DIR}/../RAPIDS_BRANCH" RAPIDS_BRANCH)
if(NOT RAPIDS_BRANCH)
  message(
    FATAL_ERROR
      "Could not determine branch name to use for checking out rapids-cmake. The file \"${CMAKE_CURRENT_LIST_DIR}/../RAPIDS_BRANCH\" is missing."
  )
endif()

if(NOT rapids-cmake-version)
  set(rapids-cmake-version "${RAPIDS_VERSION_MAJOR_MINOR}")
endif()
if(NOT rapids-cmake-branch)
  set(rapids-cmake-branch "${RAPIDS_BRANCH}")
endif()
include("${CMAKE_CURRENT_LIST_DIR}/RAPIDS.cmake")

# Don't use sccache-dist for CMake's compiler tests
set(ENV{SCCACHE_NO_DIST_COMPILE} "1")
