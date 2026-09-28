# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

# This function finds nvbench and applies any needed patches.
function(find_and_configure_nvbench)

  include(${rapids-cmake-dir}/cpm/nvbench.cmake)
  include(${rapids-cmake-dir}/cpm/package_override.cmake)

  rapids_cpm_nvbench(BUILD_STATIC)

  # Silence informational -Wpsabi ABI-change notes GCC>=10 emits on aarch64 for std::pair arguments
  # in nvbench's own sources. See CUDF_CXX_TEST_FLAGS in ConfigureCUDA.cmake.
  if(TARGET nvbench AND CUDF_CXX_TEST_FLAGS)
    target_compile_options(nvbench PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${CUDF_CXX_TEST_FLAGS}>")
  endif()

endfunction()

find_and_configure_nvbench()
