# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

add_library(NDSH_VORTEX_IO STATIC EXCLUDE_FROM_ALL vortex_io.cpp writer.cpp)
target_compile_features(NDSH_VORTEX_IO PUBLIC cxx_std_20)
target_compile_options(NDSH_VORTEX_IO PRIVATE ${CUDF_CXX_FLAGS})
# Consumers use vortex/... includes.
target_include_directories(
  NDSH_VORTEX_IO PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>"
)
target_link_libraries(
  NDSH_VORTEX_IO
  PUBLIC cudf::cudf
  PRIVATE Vortex::ffi_static nanoarrow::nanoarrow CUDA::cudart nvtx3::nvtx3-cpp
)
