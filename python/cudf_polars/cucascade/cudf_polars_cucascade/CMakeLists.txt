# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

# The C++ that the Cython module wraps: the cudf::io::datasource subclass, and the scan cache that
# opens it. cuCascade's libraries are linked statically, so the module carries what it needs.
add_library(cudf_polars_cucascade_cpp STATIC cpp/cucascade_datasource.cpp cpp/scan_cache.cpp)
target_include_directories(
  cudf_polars_cucascade_cpp PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/cpp"
                                   "${cuCascade_SOURCE_DIR}/include"
)
target_link_libraries(
  cudf_polars_cucascade_cpp PUBLIC cudf::cudf cuCascade::cucascade_io_static
                                   cuCascade::cucascade_static
)
target_compile_features(cudf_polars_cucascade_cpp PUBLIC cxx_std_20)
set_target_properties(cudf_polars_cucascade_cpp PROPERTIES POSITION_INDEPENDENT_CODE ON)

set(cython_sources advising_datasource.pyx build_info.pyx column_admission.pyx scan_cache.pyx)
set(linked_libraries cudf_polars_cucascade_cpp)
rapids_cython_create_modules(
  CXX
  SOURCE_FILES "${cython_sources}"
  LINKED_LIBRARIES "${linked_libraries}" MODULE_PREFIX cudf_polars_cucascade_ ASSOCIATED_TARGETS
                   cudf
)
