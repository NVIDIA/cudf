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

  if(CUDF_PRIVATE_TEST_LIBRARIES)
    # Ship the same build of NVBench and fmt that the benchmark executables link against.
    set(CPM_DOWNLOAD_nvbench ON)
    set(CPM_DOWNLOAD_fmt ON)
    rapids_cpm_nvbench()
    set_target_properties(nvbench PROPERTIES OUTPUT_NAME cudf_nvbench INSTALL_RPATH "\$ORIGIN")
    target_link_options(nvbench PRIVATE "LINKER:-z,defs")
    install(
      TARGETS nvbench
      LIBRARY DESTINATION ${lib_dir}
              COMPONENT benchmarking
              EXCLUDE_FROM_ALL
    )
  else()
    rapids_cpm_nvbench(BUILD_STATIC)
  endif()

  # Silence informational -Wpsabi ABI-change notes GCC>=10 emits on aarch64 for std::pair arguments
  # in nvbench's own sources. See CUDF_CXX_TEST_FLAGS in ConfigureCUDA.cmake.
  if(TARGET nvbench AND CUDF_CXX_TEST_FLAGS)
    target_compile_options(nvbench PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${CUDF_CXX_TEST_FLAGS}>")
  endif()

endfunction()

find_and_configure_nvbench()
