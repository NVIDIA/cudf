# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

if(CMAKE_COMPILER_IS_GNUCXX)
  list(APPEND CUDF_CXX_FLAGS -Wall -Werror -Wno-unknown-pragmas)
endif()

list(APPEND CUDF_CUDA_FLAGS --expt-extended-lambda)

# set warnings as errors
if(CUDA_WARNINGS_AS_ERRORS)
  list(APPEND CUDF_CUDA_FLAGS -Werror=all-warnings)
else()
  list(APPEND CUDF_CUDA_FLAGS -Werror=cross-execution-space-call)
endif()
list(APPEND CUDF_CUDA_FLAGS -Xcompiler=-Wall,-Werror)
# This warning needs to be suppressed because some parts of cudf instantiate templated CCCL
# functions in contexts where the resulting instantiations would have internal linkage (e.g. in
# anonymous namespaces). In such contexts, the visibility attribute on the template is ignored, and
# the compiler issues a warning. This is not a problem and will be fixed in future versions of CCCL.
list(APPEND CUDF_CUDA_FLAGS -diag-suppress=1407)

if(DISABLE_DEPRECATION_WARNINGS)
  list(APPEND CUDF_CXX_FLAGS -Wno-deprecated-declarations)
  list(APPEND CUDF_CUDA_FLAGS -Xcompiler=-Wno-deprecated-declarations)
endif()

# make sure we produce smallest binary size
include(${rapids-cmake-dir}/cuda/enable_fatbin_compression.cmake)
rapids_cuda_enable_fatbin_compression(VARIABLE CUDF_CUDA_FLAGS TUNE_FOR rapids)

# Option to enable line info in CUDA device compilation to allow introspection when profiling /
# memchecking
if(CUDA_ENABLE_LINEINFO)
  list(APPEND CUDF_CUDA_FLAGS -lineinfo)
endif()

# Debug options
if(CMAKE_BUILD_TYPE MATCHES Debug)
  message(VERBOSE "CUDF: Building with debugging flags")
  list(APPEND CUDF_CUDA_FLAGS -Xcompiler=-rdynamic)
endif()
