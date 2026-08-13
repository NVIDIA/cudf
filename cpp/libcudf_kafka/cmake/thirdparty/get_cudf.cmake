# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

include("${CMAKE_CURRENT_LIST_DIR}/../../../cmake/thirdparty/get_cudf.cmake")

set(CUDF_KAFKA_MIN_VERSION
    "${CUDF_KAFKA_VERSION_MAJOR}.${CUDF_KAFKA_VERSION_MINOR}.${CUDF_KAFKA_VERSION_PATCH}"
)
find_and_configure_cudf(${CUDF_KAFKA_MIN_VERSION} cudf_kafka-exports)

if(cudf_REQUIRES_CUDA)
  rapids_cuda_init_architectures(CUDF_KAFKA)

  # Since we are building cudf as part of ourselves we need to enable the CUDA language in the
  # top-most scope
  enable_language(CUDA)

  # Since CUDF_KAFKA only enables CUDA optionally we need to manually include the file that
  # rapids_cuda_init_architectures relies on `project` calling
  if(DEFINED CMAKE_PROJECT_CUDF_KAFKA_INCLUDE)
    include("${CMAKE_PROJECT_CUDF_KAFKA_INCLUDE}")
  endif()
endif()
