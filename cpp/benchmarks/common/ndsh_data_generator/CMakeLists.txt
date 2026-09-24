# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

add_library(
  ndsh_data_generator STATIC ndsh_data_generator.cpp table_helpers.cpp random_column_generator.cu
)
target_compile_features(ndsh_data_generator PUBLIC cxx_std_20 cuda_std_20)

target_compile_options(
  ndsh_data_generator PUBLIC "$<$<COMPILE_LANGUAGE:CXX>:${CUDF_CXX_FLAGS};${CUDF_CXX_TEST_FLAGS}>"
                             "$<$<COMPILE_LANGUAGE:CUDA>:${CUDF_CUDA_FLAGS}>"
)

target_link_libraries(
  ndsh_data_generator
  PUBLIC cudf cudf::cudftestutil nvtx3::nvtx3-cpp
  PRIVATE $<TARGET_NAME_IF_EXISTS:conda_env>
)

target_include_directories(
  ndsh_data_generator
  PUBLIC "$<BUILD_INTERFACE:${CUDF_SOURCE_DIR}/benchmarks>" "$<BUILD_INTERFACE:${CUDF_SOURCE_DIR}>"
         "$<BUILD_INTERFACE:${CUDF_SOURCE_DIR}/src>"
)
