# =============================================================================
# SPDX-FileCopyrightText: Copyright the Vortex contributors
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
  message(FATAL_ERROR "CUDF_WITH_VORTEX requires Linux and a CUDA toolkit")
endif()

block()
# CPM's DOWNLOAD_ONLY path does not honor FetchContent source overrides itself.
if(FETCHCONTENT_SOURCE_DIR_VORTEX)
  set(vortex_SOURCE_DIR "${FETCHCONTENT_SOURCE_DIR_VORTEX}")
else()
  CPMAddPackage(
    NAME vortex GIT_REPOSITORY https://github.com/vortex-data/vortex.git GIT_TAG
    d196f6010777ba55658133782ad77151307e733a DOWNLOAD_ONLY TRUE
  )
endif()

if(NOT EXISTS "${vortex_SOURCE_DIR}/lang/cpp/CMakeLists.txt")
  message(
    FATAL_ERROR
      "CUDF_WITH_VORTEX requires Vortex sources containing lang/cpp/CMakeLists.txt "
      "(got '${vortex_SOURCE_DIR}'). A local checkout can be selected with "
      "-DFETCHCONTENT_SOURCE_DIR_VORTEX=/path/to/vortex."
  )
endif()

# The benchmark reader and host writer share one FFI archive. CUDA paths still dlopen CUB/nvcomp
# from the Cargo build tree, so Vortex benchmarks must remain NO_INSTALL. This dependency is only
# configured for tests or benchmarks and is not linked into libcudf or its installed interface.
set(VORTEX_ENABLE_CUDA ON)
set(VORTEX_BUILD_TESTS OFF)
set(VORTEX_BUILD_EXAMPLES OFF)
set(VORTEX_WARNINGS_AS_ERRORS OFF)
add_subdirectory(
  "${vortex_SOURCE_DIR}/lang/cpp" "${CMAKE_CURRENT_BINARY_DIR}/_deps/vortex-build" EXCLUDE_FROM_ALL
  SYSTEM
)
endblock()
