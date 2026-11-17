# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

include("${CMAKE_CURRENT_LIST_DIR}/../../../cmake/thirdparty/get_cudf.cmake")

set(CUDF_STREAMING_MIN_VERSION
    "${CUDF_STREAMING_VERSION_MAJOR}.${CUDF_STREAMING_VERSION_MINOR}.${CUDF_STREAMING_VERSION_PATCH}"
)
find_and_configure_cudf(${CUDF_STREAMING_MIN_VERSION} cudf_streaming-exports)

# The cudf-dependent tests link against cudf::cudftestutil{,_impl}, which live in cudf's optional
# `testing` component. The transitive find_dependency(cudf) triggered by find_and_configure_cudf
# does not request it, so request it explicitly here when building tests. This avoids having to
# build cudf from source: the component is provided by the installed (e.g. conda) cudf package.
if(BUILD_TESTS AND NOT TARGET cudf::cudftestutil)
  find_package(cudf ${CUDF_STREAMING_MIN_VERSION} REQUIRED COMPONENTS testing)

  foreach(_target cudf::cudftestutil cudf::cudftestutil_impl)
    if(NOT TARGET ${_target})
      message(WARNING "cudf testing target NOT available: ${_target}")
    endif()
  endforeach()
endif()

if(cudf_REQUIRES_CUDA)
  rapids_cuda_init_architectures(CUDF_STREAMING)

  # Since we are building cudf as part of ourselves we need to enable the CUDA language in the
  # top-most scope
  enable_language(CUDA)

  # Since CUDF_STREAMING only enables CUDA optionally we need to manually include the file that
  # rapids_cuda_init_architectures relies on `project` calling
  if(DEFINED CMAKE_PROJECT_CUDF_STREAMING_INCLUDE)
    include("${CMAKE_PROJECT_CUDF_STREAMING_INCLUDE}")
  endif()
endif()
