# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

cmake_minimum_required(VERSION 4.0 FATAL_ERROR)

include(../cmake/rapids_config.cmake)
include(rapids-cmake)
include(rapids-cpm)
include(rapids-cuda)
include(${rapids-cmake-dir}/cuda/set_runtime.cmake)
include(rapids-export)
include(rapids-find)

rapids_cuda_init_architectures(CUDF)

project(
  CUDF
  VERSION "${RAPIDS_VERSION}"
  LANGUAGES C CXX CUDA ASM
)

if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA" AND CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 12.2)
  message(
    FATAL_ERROR
      "libcudf requires CUDA Toolkit 12.2+ to compile (nvcc ${CMAKE_CUDA_COMPILER_VERSION} provided)"
  )
endif()

rapids_cmake_write_version_file(include/cudf/version_config.hpp)

# ##################################################################################################
# * build options ---------------------------------------------------------------------------------
option(USE_NVTX "Build with NVTX support" ON)
option(BUILD_TESTS "Configure CMake to build tests" ON)
option(BUILD_BENCHMARKS "Configure CMake to build benchmarks" OFF)
option(BUILD_SHARED_LIBS "Build cuDF shared libraries" ON)
option(CUDF_BUILD_TESTUTIL "Whether to build the test utilities contained in libcudf" ON)
mark_as_advanced(CUDF_BUILD_TESTUTIL)
option(CUDF_LARGE_STRINGS_DISABLED "Build with large string support disabled" OFF)
mark_as_advanced(CUDF_LARGE_STRINGS_DISABLED)
option(
  CUDF_USE_PER_THREAD_DEFAULT_STREAM
  "Build cuDF with per-thread default stream, including passing the per-thread default
         stream to external libraries."
  OFF
)
option(DISABLE_DEPRECATION_WARNINGS "Disable warnings generated from deprecated declarations." OFF)

# Option to enable line info in CUDA device compilation to allow introspection when profiling /
# memchecking
option(CUDA_ENABLE_LINEINFO
       "Enable the -lineinfo option for nvcc (useful for cuda-memcheck / profiler)" OFF
)
option(CUDA_WARNINGS_AS_ERRORS "Enable -Werror=all-warnings for all CUDA compilation" ON)

# CUDF_BUILD_STATIC_DEPS should almost never be set to OFF since there is no reason to prefer
# building shared libraries for dependencies under normal circumstances. FORCE is a necessary
# setting for building a standalone binary with no external dependencies.
set(CUDF_BUILD_STATIC_DEPS
    "ON"
    CACHE
      STRING
      "Three-state control for static library dependencies. ON (default) = use static when a CPM build is triggered, OFF = prefer shared (rarely useful), FORCE = guarantee using a static build by rebuilding or downloading even when local copies of a library exist"
)
set_property(CACHE CUDF_BUILD_STATIC_DEPS PROPERTY STRINGS "ON" "OFF" "FORCE")

# cudart can be statically linked or dynamically linked. The python ecosystem wants dynamic linking
option(CUDA_STATIC_RUNTIME "Statically link the CUDA runtime" OFF)
# FORCE mode implies static CUDA runtime
if(CUDF_BUILD_STATIC_DEPS STREQUAL "FORCE")
  set(CUDA_STATIC_RUNTIME
      ON
      CACHE BOOL "Statically link the CUDA runtime" FORCE
  )
  set(RTCX_STATIC_LINK_NVRTC
      ON
      CACHE BOOL "Use static linking for NVRTC" FORCE
  )
  set(RTCX_STATIC_LINK_NVJITLINK
      ON
      CACHE BOOL "Use static linking for nvJitLink" FORCE
  )
else()
  set(CUDA_STATIC_RUNTIME
      OFF
      CACHE BOOL "Statically link the CUDA runtime" FORCE
  )
  set(RTCX_STATIC_LINK_NVRTC
      OFF
      CACHE BOOL "Use static linking for NVRTC" FORCE
  )
  set(RTCX_STATIC_LINK_NVJITLINK
      OFF
      CACHE BOOL "Use static linking for nvJitLink" FORCE
  )
endif()

set(DEFAULT_CUDF_BUILD_STREAMS_TEST_UTIL ON)

if(NOT BUILD_SHARED_LIBS)
  set(DEFAULT_CUDF_BUILD_STREAMS_TEST_UTIL OFF)
endif()

option(
  CUDF_BUILD_STREAMS_TEST_UTIL
  "Whether to build the utilities for stream testing contained in libcudf"
  ${DEFAULT_CUDF_BUILD_STREAMS_TEST_UTIL}
)
mark_as_advanced(CUDF_BUILD_STREAMS_TEST_UTIL)
option(CUDF_CLANG_TIDY "Enable clang-tidy during compilation" OFF)
option(CUDF_IWYU "Enable IWYU during compilation" OFF)
option(CUDF_CLANG_TIDY_AUTOFIX "Enable clang-tidy autofixes" OFF)

option(
  CUDF_KVIKIO_REMOTE_IO
  "Enable remote IO (e.g. AWS S3) support through KvikIO. If disabled, cudf-python will still be able to do remote IO through fsspec."
  ON
)

message(VERBOSE "CUDF: Build with NVTX support: ${USE_NVTX}")
message(VERBOSE "CUDF: Configure CMake to build tests: ${BUILD_TESTS}")
message(VERBOSE "CUDF: Configure CMake to build benchmarks: ${BUILD_BENCHMARKS}")
message(VERBOSE "CUDF: Build cuDF shared libraries: ${BUILD_SHARED_LIBS}")
message(VERBOSE "CUDF: Build with per-thread default stream: ${CUDF_USE_PER_THREAD_DEFAULT_STREAM}")
message(
  VERBOSE
  "CUDF: Disable warnings generated from deprecated declarations: ${DISABLE_DEPRECATION_WARNINGS}"
)
message(
  VERBOSE
  "CUDF: Enable the -lineinfo option for nvcc (useful for cuda-memcheck / profiler): ${CUDA_ENABLE_LINEINFO}"
)
message(VERBOSE
        "CUDF: Build with remote IO (e.g. AWS S3) support through KvikIO: ${CUDF_KVIKIO_REMOTE_IO}"
)

# CUDF_INSTALL_LIBRARY_DEPS: Derived variable indicating whether to install library dependencies
if(NOT BUILD_SHARED_LIBS)
  # A static libcudf.a will not contain the code for its dependencies, so we must install them to
  # ensure downstream projects can find them
  set(CUDF_INSTALL_LIBRARY_DEPS ON)
elseif(CUDF_BUILD_STATIC_DEPS STREQUAL "OFF")
  # If dependencies are built as shared they will be needed at runtime
  set(CUDF_INSTALL_LIBRARY_DEPS ON)
else()
  # If dependencies are statically linked into a libcudf shared library they aren't needed at
  # runtime
  set(CUDF_INSTALL_LIBRARY_DEPS OFF)
endif()

# Derived flags for EXCLUDE_FROM_ALL handling across dependency fetching.
# CUDF_EXCLUDE_DEPS_FROM_ALL: ON/OFF value for rapids_cpm_find() calls (key-value style)
# CUDF_EXCLUDE_DEPS_FROM_ALL_FLAG: "EXCLUDE_FROM_ALL" or "" for rapids_cpm_* wrapper calls (flag
# style) We need both because of https://github.com/cpm-cmake/CPM.cmake/issues/693
if(CUDF_INSTALL_LIBRARY_DEPS)
  set(CUDF_EXCLUDE_DEPS_FROM_ALL OFF)
  set(CUDF_EXCLUDE_DEPS_FROM_ALL_FLAG)
else()
  set(CUDF_EXCLUDE_DEPS_FROM_ALL ON)
  set(CUDF_EXCLUDE_DEPS_FROM_ALL_FLAG EXCLUDE_FROM_ALL)
endif()

# CUDF_DEPS_BUILD_SHARED: Boolean reduction of the 3-state CUDF_BUILD_STATIC_DEPS for passing to
# dependency configuration functions that accept a BUILD_SHARED parameter.
if(CUDF_BUILD_STATIC_DEPS STREQUAL "OFF")
  set(CUDF_DEPS_BUILD_SHARED ON)
else()
  set(CUDF_DEPS_BUILD_SHARED OFF)
endif()

message(VERBOSE "CUDF: Build static library dependencies: ${CUDF_BUILD_STATIC_DEPS}")
message(VERBOSE "CUDF: Install library dependencies: ${CUDF_INSTALL_LIBRARY_DEPS}")
message(VERBOSE "CUDF: Build dependencies as shared libraries: ${CUDF_DEPS_BUILD_SHARED}")

# Set a default build type if none was specified
rapids_cmake_build_type("Release")
set(CUDF_BUILD_TESTS ${BUILD_TESTS})
set(CUDF_BUILD_BENCHMARKS ${BUILD_BENCHMARKS})

if(BUILD_TESTS AND NOT CUDF_BUILD_TESTUTIL)
  message(
    FATAL_ERROR
      "Tests cannot be built without building cudf test utils. Please set CUDF_BUILD_TESTUTIL=ON or BUILD_TESTS=OFF"
  )
endif()

set(CUDF_CXX_FLAGS "")
set(CUDF_CUDA_FLAGS "")
set(CUDF_CXX_DEFINITIONS "")
set(CUDF_CUDA_DEFINITIONS "")

# For now, disable CMake's automatic module scanning for C++ files. There is an sccache bug in the
# version RAPIDS uses in CI that causes it to handle the resulting -M* flags incorrectly with
# gcc>=14. We can remove this once we upgrade to a newer sccache version.
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)

# Set logging level
set(LIBCUDF_LOGGING_LEVEL
    "INFO"
    CACHE STRING "Choose the logging level."
)
set_property(
  CACHE LIBCUDF_LOGGING_LEVEL PROPERTY STRINGS "TRACE" "DEBUG" "INFO" "WARN" "ERROR" "CRITICAL"
                                       "OFF"
)
message(VERBOSE "CUDF: LIBCUDF_LOGGING_LEVEL = '${LIBCUDF_LOGGING_LEVEL}'.")

if(NOT CUDF_GENERATED_INCLUDE_DIR)
  set(CUDF_GENERATED_INCLUDE_DIR ${CUDF_BINARY_DIR})
endif()

# ##################################################################################################
# * linter configuration ---------------------------------------------------------------------------
if(CUDF_CLANG_TIDY)
  find_program(
    CLANG_TIDY_EXE
    NAMES "clang-tidy"
    DOC "Path to clang-tidy executable" REQUIRED
  )

  execute_process(
    COMMAND ${CLANG_TIDY_EXE} --version
    OUTPUT_VARIABLE CLANG_TIDY_OUTPUT
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  string(REGEX MATCH "LLVM version ([0-9]+\\.[0-9]+)\\.[0-9]+" LLVM_VERSION_MATCH
               "${CLANG_TIDY_OUTPUT}"
  )

  # Discard the patch version and allow it to float. Empirically, results between patch versions are
  # mostly stable, and looking at available packages on some package managers sometimes patch
  # versions are skipped so we don't want to constrain to a patch version that the user can't
  # install.
  set(LLVM_VERSION "${CMAKE_MATCH_1}")
  set(expected_clang_tidy_version 20.1)

  if(NOT expected_clang_tidy_version VERSION_EQUAL LLVM_VERSION)
    message(
      FATAL_ERROR
        "clang-tidy version ${expected_clang_tidy_version} is required, but found ${LLVM_VERSION}"
    )
  endif()
endif()

if(CUDF_IWYU)
  find_program(IWYU_EXE NAMES include-what-you-use iwyu REQUIRED)
endif()

# Turn on the clang-tidy property for a target excluding the files specified in SKIPPED_FILES.
function(enable_static_checkers target)
  set(_tidy_options IWYU CLANG_TIDY)
  set(_tidy_one_value)
  set(_tidy_multi_value SKIPPED_FILES)
  cmake_parse_arguments(
    _LINT "${_tidy_options}" "${_tidy_one_value}" "${_tidy_multi_value}" ${ARGN}
  )

  if(_LINT_CLANG_TIDY)
    # clang will complain about unused link libraries on the compile line unless we specify
    # -Qunused-arguments.
    if(CUDF_CLANG_TIDY_AUTOFIX)
      set_target_properties(
        ${target} PROPERTIES CXX_CLANG_TIDY
                             "${CLANG_TIDY_EXE};--extra-arg=-Qunused-arguments;--fix"
      )
    else()
      set_target_properties(
        ${target} PROPERTIES CXX_CLANG_TIDY "${CLANG_TIDY_EXE};--extra-arg=-Qunused-arguments"
      )
    endif()
  endif()

  if(_LINT_IWYU)
    # A few extra warnings pop up when building with IWYU. I'm not sure why, but they are not
    # relevant since they don't show up in any other build so it's better to suppress them until we
    # can figure out the cause. Setting this as part of CXX_INCLUDE_WHAT_YOU_USE does not appear to
    # be sufficient, we must also ensure that it is set to the underlying target's CXX compile
    # flags. To do this completely cleanly we should modify the flags on the target rather than the
    # global CUDF_CXX_FLAGS, but this solution is good enough for now since we never run the linters
    # on real builds.
    foreach(_flag -Wno-missing-braces -Wno-unneeded-internal-declaration)
      list(FIND CUDF_CXX_FLAGS "${_flag}" _flag_index)

      if(_flag_index EQUAL -1)
        list(APPEND CUDF_CXX_FLAGS ${_flag})
      endif()
    endforeach()

    set(CUDF_CXX_FLAGS
        "${CUDF_CXX_FLAGS}"
        PARENT_SCOPE
    )
    set_target_properties(${target} PROPERTIES CXX_INCLUDE_WHAT_YOU_USE "${IWYU_EXE}")
  endif()

  foreach(file IN LISTS _LINT_SKIPPED_FILES)
    set_source_files_properties(${file} PROPERTIES SKIP_LINTING ON)
  endforeach()
endfunction()

# ##################################################################################################
# * conda environment -----------------------------------------------------------------------------
rapids_cmake_support_conda_env(conda_env MODIFY_PREFIX_PATH)

# ##################################################################################################
# * find C++ libraries from RAPIDS wheels in in site-packages --------------------------------------
find_package(Python COMPONENTS Interpreter)
if(Python_FOUND)
  include(${rapids-cmake-dir}/cython-core/find_prefix_paths.cmake)
  rapids_cython_find_prefix_paths("${Python_EXECUTABLE}" wheel_prefix_paths)
  set_property(GLOBAL PROPERTY FIND_LIBRARY_USE_LIB64_PATHS TRUE)
  list(APPEND CMAKE_PREFIX_PATH ${wheel_prefix_paths})
endif()

# ##################################################################################################
# * compiler options ------------------------------------------------------------------------------
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules ${CMAKE_MODULE_PATH})
rapids_find_package(
  CUDAToolkit REQUIRED
  BUILD_EXPORT_SET cudf-exports
  INSTALL_EXPORT_SET cudf-exports
)
include(cmake/Modules/ConfigureCUDA.cmake) # set other CUDA compilation flags

# ##################################################################################################
# * dependencies ----------------------------------------------------------------------------------

# find zlib
rapids_find_package(ZLIB REQUIRED)

if(CUDF_BUILD_TESTUTIL)
  # find Threads (needed by cudftestutil)
  rapids_find_package(
    Threads REQUIRED
    BUILD_EXPORT_SET cudf-exports
    INSTALL_EXPORT_SET cudf-exports
  )
endif()

# Set CPM_DOWNLOAD_ALL if CUDF_BUILD_STATIC_DEPS is FORCE to skip find_package for all CPM packages
if(CUDF_BUILD_STATIC_DEPS STREQUAL "FORCE")
  set(CPM_DOWNLOAD_ALL ON)
endif()

# add third party dependencies using CPM
rapids_cpm_init()

include(${rapids-cmake-dir}/cpm/rapids_logger.cmake)
set(_rapids_logger_args)
if(NOT CUDF_BUILD_STATIC_DEPS STREQUAL "OFF")
  list(APPEND _rapids_logger_args ${CUDF_EXCLUDE_DEPS_FROM_ALL_FLAG} CPM_ARGS OPTIONS
       "BUILD_SHARED_LIBS OFF"
  )
endif()
rapids_cpm_rapids_logger(${_rapids_logger_args})

# If dynamically linking to a preexisting rapids_logger library then find_dependency(rapids_logger)
# must be in the installed config when CUDF_INSTALL_LIBRARY_DEPS is OFF. If it's ON then we always
# export
get_target_property(_rapids_logger_type rapids_logger::rapids_logger TYPE)
if(NOT BUILD_SHARED_LIBS OR NOT _rapids_logger_type STREQUAL "STATIC_LIBRARY")
  include("${rapids-cmake-dir}/export/package.cmake")
  rapids_export_package(BUILD rapids_logger cudf-exports VERSION ${rapids_logger_VERSION})
endif()
if(CUDF_INSTALL_LIBRARY_DEPS OR NOT _rapids_logger_type STREQUAL "STATIC_LIBRARY")
  include("${rapids-cmake-dir}/export/package.cmake")
  rapids_export_package(INSTALL rapids_logger cudf-exports VERSION ${rapids_logger_VERSION})
endif()

create_logger_macros(CUDF "cudf::default_logger()" include/cudf)

# find NVTX
include(cmake/thirdparty/get_nvtx.cmake)

# find nvCOMP
include(cmake/thirdparty/get_nvcomp.cmake)

# find CCCL
include(cmake/thirdparty/get_cccl.cmake)

# find rmm, should come after including CCCL to allow overriding the CCCL version used for testing
include(cmake/thirdparty/get_rmm.cmake)

# find croaring, needed by tests and benchmarks
if(CUDF_BUILD_TESTS OR CUDF_BUILD_BENCHMARKS)
  include(cmake/thirdparty/get_croaring.cmake)
endif()

# find flatbuffers
include(cmake/thirdparty/get_flatbuffers.cmake)

# find dlpack
include(cmake/thirdparty/get_dlpack.cmake)

# find cuCollections, should come after including CCCL
include(cmake/thirdparty/get_cucollections.cmake)

# find or install GoogleTest
if(CUDF_BUILD_TESTUTIL)
  include(cmake/thirdparty/get_gtest.cmake)
endif()

# find KvikIO
include(cmake/thirdparty/get_kvikio.cmake)

# find nanoarrow
include(cmake/thirdparty/get_nanoarrow.cmake)

# find thread_pool
include(cmake/thirdparty/get_thread_pool.cmake)

# find xxhash
include(cmake/thirdparty/get_xxhash.cmake)

# find zstd
include(cmake/thirdparty/get_zstd.cmake)

# Fetch librtcx via CPM (zstd and nvtx3 are already available at this point)
include(cmake/thirdparty/get_rtcx.cmake)

# Pre-compiled fragment management helper functions
include(cmake/Modules/AddFragment.cmake)

# Workaround until https://github.com/rapidsai/rapids-cmake/issues/176 is resolved
if(NOT BUILD_SHARED_LIBS)
  include("${rapids-cmake-dir}/export/find_package_file.cmake")
  list(APPEND METADATA_KINDS BUILD INSTALL)
  list(APPEND dependencies KvikIO ZLIB nvcomp nanoarrow zstd rtcx)

  foreach(METADATA_KIND IN LISTS METADATA_KINDS)
    foreach(dep IN LISTS dependencies)
      rapids_export_package(${METADATA_KIND} ${dep} cudf-exports)
    endforeach()
  endforeach()

  if(TARGET conda_env)
    install(TARGETS conda_env EXPORT cudf-exports)
  endif()
endif()

rtcx_add_embed(cudf_cuda_embed)

rtcx_embed_includes(
  cudf_cuda_embed SOURCE_DIRECTORY ${rtcx_SOURCE_DIR}/libcxx DEST_DIRECTORY librtcx/libcxx
  INCLUDE_DIRECTORIES librtcx/libcxx
)

rtcx_embed_includes(
  cudf_cuda_embed SOURCE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/cudf DEST_DIRECTORY
  cudf/cpp/include/cudf INCLUDE_DIRECTORIES cudf/cpp/include
)

rtcx_embed_includes(
  cudf_cuda_embed SOURCE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src/jit DEST_DIRECTORY
  cudf/cpp/src/jit INCLUDE_DIRECTORIES cudf/cpp/src
)

rtcx_embed_includes(
  cudf_cuda_embed SOURCE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src/binaryop/jit DEST_DIRECTORY
  cudf/cpp/src/binaryop/jit INCLUDE_DIRECTORIES cudf/cpp/src
)

rtcx_embed_includes(
  cudf_cuda_embed SOURCE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src/join/jit DEST_DIRECTORY
  cudf/cpp/src/join/jit INCLUDE_DIRECTORIES cudf/cpp/src
)

rtcx_embed_includes(
  cudf_cuda_embed SOURCE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src/rolling DEST_DIRECTORY
  cudf/cpp/src/rolling INCLUDE_DIRECTORIES cudf/cpp/src
)

rtcx_embed_includes(
  cudf_cuda_embed SOURCE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src/transform/jit DEST_DIRECTORY
  cudf/cpp/src/transform/jit INCLUDE_DIRECTORIES cudf/cpp/src
)

get_target_property(LIBCUDACXX_RAW_INCLUDE_DIRS CCCL::libcudacxx INTERFACE_INCLUDE_DIRECTORIES)

foreach(INC_DIR IN LISTS LIBCUDACXX_RAW_INCLUDE_DIRS)
  cmake_path(GET INC_DIR FILENAME INC_DIR_NAME)

  rtcx_embed_includes(
    cudf_cuda_embed SOURCE_DIRECTORY ${INC_DIR} DEST_DIRECTORY CCCL/libcudacxx/${INC_DIR_NAME}
    INCLUDE_DIRECTORIES CCCL/libcudacxx/${INC_DIR_NAME}
  )
endforeach()

rtcx_embed(
  cudf_cuda_embed COMPRESSION zstd OUTPUT_DIRECTORY "${CUDF_GENERATED_INCLUDE_DIR}/rtcx_embed"
)

rtcx_add_embed(cudf_fragments)

list(APPEND CUDF_PRECOMPILE_PHYSICAL_TYPES uint8_t uint16_t uint32_t uint64_t numeric::decimal32
     numeric::decimal64 numeric::decimal128
)

foreach(TYPE IN ITEMS ${CUDF_PRECOMPILE_PHYSICAL_TYPES})
  set(FRAGMENT_NAME transform_kernel)
  get_property(
    FILE_INDEX
    TARGET cudf_fragments__embed_props
    PROPERTY EMBED_FILE_INDEX
  )
  set(VARIANT_NAME transform_kernel_${FILE_INDEX})
  set(INSTANCE
      "cudf::jit::transform_kernel<false, false, cudf::jit::type_list<cudf::jit::column_accessor<0ULL, cudf::column_device_view_core, ${TYPE}, false, 0>>, cudf::jit::type_list<cudf::jit::column_accessor<0ULL, cudf::mutable_column_device_view_core, ${TYPE}, false, 0>>>"
  )
  add_fragment(
    cudf_fragments
    FRAGMENT
    ${VARIANT_NAME}
    SOURCE
    src/transform/jit/kernel.cu
    KERNEL_INSTANCE
    ${INSTANCE}
    UDF_TYPE
    "int(${TYPE} *, ${TYPE})"
    DEFINITIONS
    CUDF_LTO_MODE
    ARRAY_IDS
    ${FRAGMENT_NAME}_FILE_INDEX
    ${FRAGMENT_NAME}_INSTANCE
    ARRAY_VALUES
    ${FILE_INDEX}
    "${INSTANCE}"
  )
endforeach()

foreach(TYPE IN ITEMS ${CUDF_PRECOMPILE_PHYSICAL_TYPES})
  foreach(RHS_IS_SCALAR IN ITEMS "false" "true")
    set(FRAGMENT_NAME transform_kernel)
    get_property(
      FILE_INDEX
      TARGET cudf_fragments__embed_props
      PROPERTY EMBED_FILE_INDEX
    )
    set(VARIANT_NAME transform_kernel_${FILE_INDEX})
    set(INSTANCE
        "cudf::jit::transform_kernel<false, false, cudf::jit::type_list<cudf::jit::column_accessor<0ULL, cudf::column_device_view_core, ${TYPE}, false, 0>, cudf::jit::column_accessor<1ULL, cudf::column_device_view_core, ${TYPE}, ${RHS_IS_SCALAR}, 0>>, cudf::jit::type_list<cudf::jit::column_accessor<0ULL, cudf::mutable_column_device_view_core, ${TYPE}, false, 0>>>"
    )
    add_fragment(
      cudf_fragments
      FRAGMENT
      ${VARIANT_NAME}
      SOURCE
      src/transform/jit/kernel.cu
      KERNEL_INSTANCE
      ${INSTANCE}
      UDF_TYPE
      "int(${TYPE} *, ${TYPE}, ${TYPE})"
      DEFINITIONS
      CUDF_LTO_MODE
      ARRAY_IDS
      ${FRAGMENT_NAME}_FILE_INDEX
      ${FRAGMENT_NAME}_INSTANCE
      ARRAY_VALUES
      ${FILE_INDEX}
      "${INSTANCE}"
    )
  endforeach()
endforeach()

rtcx_embed(
  cudf_fragments COMPRESSION none OUTPUT_DIRECTORY "${CUDF_GENERATED_INCLUDE_DIR}/rtcx_embed"
)

# ##################################################################################################
# * library targets -------------------------------------------------------------------------------
add_library(
  cudf
  src/aggregation/aggregation.cpp
  src/aggregation/aggregation.cu
  src/aggregation/result_cache.cpp
  src/ast/expression_parser.cpp
  src/ast/expressions.cpp
  src/ast/jit/expressions.cpp
  src/ast/operators.cpp
  src/binaryop/binaryop.cpp
  src/binaryop/compiled/ATan2.cu
  src/binaryop/compiled/Add.cu
  src/binaryop/compiled/BitwiseAnd.cu
  src/binaryop/compiled/BitwiseOr.cu
  src/binaryop/compiled/BitwiseXor.cu
  src/binaryop/compiled/Div.cu
  src/binaryop/compiled/FloorDiv.cu
  src/binaryop/compiled/Greater.cu
  src/binaryop/compiled/GreaterEqual.cu
  src/binaryop/compiled/IntPow.cu
  src/binaryop/compiled/Less.cu
  src/binaryop/compiled/LessEqual.cu
  src/binaryop/compiled/LogBase.cu
  src/binaryop/compiled/LogicalAnd.cu
  src/binaryop/compiled/LogicalOr.cu
  src/binaryop/compiled/Mod.cu
  src/binaryop/compiled/Mul.cu
  src/binaryop/compiled/NullEquals.cu
  src/binaryop/compiled/NullNotEquals.cu
  src/binaryop/compiled/NullLogicalAnd.cu
  src/binaryop/compiled/NullLogicalOr.cu
  src/binaryop/compiled/NullMax.cu
  src/binaryop/compiled/NullMin.cu
  src/binaryop/compiled/PMod.cu
  src/binaryop/compiled/Pow.cu
  src/binaryop/compiled/PyMod.cu
  src/binaryop/compiled/ShiftLeft.cu
  src/binaryop/compiled/ShiftRight.cu
  src/binaryop/compiled/ShiftRightUnsigned.cu
  src/binaryop/compiled/Sub.cu
  src/binaryop/compiled/TrueDiv.cu
  src/binaryop/compiled/binary_ops.cu
  src/binaryop/compiled/equality_ops.cu
  src/binaryop/compiled/util.cpp
  src/labeling/label_bins.cu
  src/bitmask/null_mask.cu
  src/bitmask/is_element_valid.cpp
  src/column/column.cu
  src/column/column_device_view.cu
  src/column/column_factories.cpp
  src/column/column_factories.cu
  src/column/column_view.cpp
  src/copying/concatenate.cu
  src/copying/contiguous_split.cu
  src/copying/copy.cpp
  src/copying/copy.cu
  src/copying/copy_range.cu
  src/copying/gather.cu
  src/copying/get_element.cu
  src/copying/pack.cpp
  src/copying/purge_nonempty_nulls.cu
  src/copying/reverse.cu
  src/copying/sample.cu
  src/copying/scatter.cu
  src/copying/shift.cu
  src/copying/slice.cu
  src/copying/split.cpp
  src/copying/segmented_shift.cu
  src/datetime/datetime_ops.cu
  src/dictionary/add_keys.cu
  src/dictionary/decode.cu
  src/dictionary/detail/concatenate.cu
  src/dictionary/detail/merge.cu
  src/dictionary/dictionary_column_view.cpp
  src/dictionary/dictionary_factories.cu
  src/dictionary/encode.cu
  src/dictionary/match_keys.cu
  src/dictionary/remove_keys.cu
  src/dictionary/replace.cu
  src/dictionary/search.cu
  src/dictionary/set_keys.cu
  src/filling/calendrical_month_sequence.cu
  src/filling/fill.cu
  src/filling/repeat.cu
  src/filling/sequence.cu
  src/groupby/common/m2_var_std.cu
  src/groupby/common/utils.cpp
  src/groupby/groupby.cu
  src/groupby/hash/compute_global_memory_aggs.cu
  src/groupby/hash/compute_global_memory_aggs_null.cu
  src/groupby/hash/compute_groupby.cu
  src/groupby/hash/compute_mapping_indices.cu
  src/groupby/hash/compute_mapping_indices_null.cu
  src/groupby/hash/compute_shared_memory_aggs.cu
  src/groupby/hash/compute_single_pass_aggs.cu
  src/groupby/hash/compute_single_pass_aggs_null.cu
  src/groupby/hash/extract_single_pass_aggs.cpp
  src/groupby/hash/groupby.cu
  src/groupby/hash/hash_compound_agg_finalizer.cu
  src/groupby/hash/output_utils.cu
  src/groupby/sort/aggregate.cpp
  src/groupby/sort/group_argmax.cu
  src/groupby/sort/group_argmin.cu
  src/groupby/sort/group_bitwise.cu
  src/groupby/sort/group_collect.cu
  src/groupby/sort/group_correlation.cu
  src/groupby/sort/group_count.cu
  src/groupby/sort/group_count_scan.cu
  src/groupby/sort/group_histogram.cu
  src/groupby/sort/group_m2.cu
  src/groupby/sort/group_max.cu
  src/groupby/sort/group_max_scan.cu
  src/groupby/sort/group_merge_lists.cu
  src/groupby/sort/group_merge_m2.cu
  src/groupby/sort/group_min.cu
  src/groupby/sort/group_min_scan.cu
  src/groupby/sort/group_nth_element.cu
  src/groupby/sort/group_nunique.cu
  src/groupby/sort/group_product.cu
  src/groupby/sort/group_product_scan.cu
  src/groupby/sort/group_quantiles.cu
  src/groupby/sort/group_rank_scan.cu
  src/groupby/sort/group_replace_nulls.cu
  src/groupby/sort/group_std.cu
  src/groupby/sort/group_sum.cu
  src/groupby/sort/group_sum_overflow.cu
  src/groupby/sort/group_sum_scan.cu
  src/groupby/sort/group_topk.cu
  src/groupby/sort/host_udf_aggregation.cpp
  src/groupby/sort/scan.cpp
  src/groupby/sort/sort_helper.cu
  src/groupby/sort/sort_helper_nested.cu
  src/groupby/streaming_groupby.cpp
  src/groupby/streaming_groupby/aggregate.cu
  src/groupby/streaming_groupby/impl.cu
  src/groupby/streaming_groupby/insert.cu
  src/groupby/streaming_groupby/insert_first.cu
  src/groupby/streaming_groupby/insert_first_nested.cu
  src/groupby/streaming_groupby/insert_nested.cu
  src/groupby/streaming_groupby/insert_subsequent.cu
  src/groupby/streaming_groupby/insert_subsequent_nested.cu
  src/groupby/streaming_groupby/merge.cu
  src/hash/md5_hash.cu
  src/hash/murmurhash3_x86_32.cu
  src/hash/murmurhash3_x64_128.cu
  src/hash/sha1_hash.cu
  src/hash/sha224_hash.cu
  src/hash/sha256_hash.cu
  src/hash/sha384_hash.cu
  src/hash/sha512_hash.cu
  src/hash/xxhash_32.cu
  src/hash/xxhash_64.cu
  src/interop/dlpack.cpp
  src/interop/arrow_utilities.cpp
  src/interop/arrow_data_structures.cpp
  src/interop/to_arrow_device.cu
  src/interop/to_arrow_host.cu
  src/interop/from_arrow_device.cu
  src/interop/from_arrow_host.cu
  src/interop/from_arrow_host_strings.cu
  src/interop/from_arrow_stream.cu
  src/interop/to_arrow_schema.cpp
  src/io/avro/avro.cpp
  src/io/avro/avro_gpu.cu
  src/io/avro/reader_impl.cu
  src/io/comp/brotli_dict.cpp
  src/io/comp/compression.cpp
  src/io/comp/compression.cu
  src/io/comp/common.cpp
  src/io/comp/cpu_unbz2.cpp
  src/io/comp/debrotli.cu
  src/io/comp/gpuinflate.cu
  src/io/comp/nvcomp_adapter.cpp
  src/io/comp/nvcomp_adapter.cu
  src/io/comp/snap.cu
  src/io/comp/decompression.cpp
  src/io/comp/unsnap.cu
  src/io/csv/csv_gpu.cu
  src/io/csv/durations.cu
  src/io/csv/reader_impl.cu
  src/io/csv/writer_impl.cu
  src/io/functions.cpp
  src/io/cudftable.cpp
  src/io/json/host_tree_algorithms.cu
  src/io/json/json_column.cu
  src/io/json/column_tree_construction.cu
  src/io/json/json_normalization.cu
  src/io/json/json_tree.cu
  src/io/json/nested_json_gpu.cu
  src/io/json/read_json.cu
  src/io/json/parser_features.cpp
  src/io/json/process_tokens.cu
  src/io/json/write_json.cpp
  src/io/json/write_json.cu
  src/io/orc/aggregate_orc_metadata.cpp
  src/io/orc/dict_enc.cu
  src/io/orc/orc.cpp
  src/io/orc/reader_impl.cu
  src/io/orc/reader_impl_chunking.cu
  src/io/orc/reader_impl_decode.cu
  src/io/orc/reader_impl_helpers.cpp
  src/io/orc/stats_enc.cu
  src/io/orc/stripe_data.cu
  src/io/orc/stripe_enc.cu
  src/io/orc/stripe_init.cu
  src/datetime/timezone.cpp
  src/io/orc/writer_impl.cu
  src/io/parquet/arrow_schema_writer.cpp
  src/io/parquet/bloom_filter_reader.cu
  src/io/parquet/column_path_helpers.cpp
  src/io/parquet/compact_protocol_reader.cpp
  src/io/parquet/compact_protocol_writer.cpp
  src/io/parquet/decode_preprocess.cu
  src/io/parquet/experimental/dictionary_page_filter.cu
  src/io/parquet/experimental/deletion_vectors.cu
  src/io/parquet/experimental/deletion_vectors_helpers.cu
  src/io/parquet/experimental/hybrid_scan.cpp
  src/io/parquet/experimental/hybrid_scan_chunking.cu
  src/io/parquet/experimental/hybrid_scan_helpers.cpp
  src/io/parquet/experimental/hybrid_scan_impl.cpp
  src/io/parquet/experimental/hybrid_scan_multifile.cpp
  src/io/parquet/experimental/hybrid_scan_preprocess.cu
  src/io/parquet/experimental/page_index_filter.cu
  src/io/parquet/experimental/page_index_filter_utils.cu
  src/io/parquet/experimental/variant_extract.cu
  src/io/parquet/experimental/variant_path.cpp
  src/io/parquet/expression_transform_helpers.cpp
  src/io/parquet/io_utils/parquet_io_utils.cpp
  src/io/parquet/page_data.cu
  src/io/parquet/chunk_dict.cu
  src/io/parquet/page_enc.cu
  src/io/parquet/page_hdr.cu
  src/io/parquet/page_delta_decode.cu
  src/io/parquet/page_string_decode.cu
  src/io/parquet/predicate_pushdown.cpp
  src/io/parquet/reader.cpp
  src/io/parquet/reader_impl.cpp
  src/io/parquet/reader_impl_chunking.cu
  src/io/parquet/reader_impl_chunking_utils.cu
  src/io/parquet/reader_impl_helpers.cpp
  src/io/parquet/reader_impl_preprocess.cu
  src/io/parquet/reader_impl_preprocess_utils.cu
  src/io/parquet/stats_filter_helpers.cpp
  src/io/parquet/writer_impl.cu
  src/io/parquet/writer_impl_helpers.cpp
  src/io/parquet/decode_fixed.cu
  src/io/parquet/decode_pruned_pages.cu
  src/io/statistics/orc_column_statistics.cu
  src/io/statistics/parquet_column_statistics.cu
  src/io/text/byte_range_info.cpp
  src/io/text/data_chunk_source_factories.cpp
  src/io/text/bgzip_data_chunk_source.cu
  src/io/text/bgzip_utils.cpp
  src/io/text/multibyte_split.cu
  src/io/utilities/base64_utilities.cpp
  src/io/utilities/column_buffer.cpp
  src/io/utilities/column_buffer_strings.cu
  src/io/utilities/config_utils.cpp
  src/io/utilities/data_casting.cu
  src/io/utilities/data_sink.cpp
  src/io/utilities/datasource.cpp
  src/io/utilities/row_selection.cpp
  src/io/utilities/type_inference.cu
  src/io/utilities/trie.cu
  src/jit/cache.cpp
  src/jit/helpers.cpp
  src/jit/parser.cpp
  src/jit/row_ir.cpp
  src/jit/util.cpp
  src/join/conditional_join.cu
  src/join/cross_join.cu
  src/join/direct_join.cu
  src/join/distinct_hash_join.cu
  src/join/filter_join_indices/filter_join_indices.cu
  src/join/filter_join_indices/filter_join_indices_jit.cu
  src/join/filter_join_indices/filter_join_indices_kernel_complex.cu
  src/join/filter_join_indices/filter_join_indices_kernel_null_complex.cu
  src/join/filter_join_indices/filter_join_indices_kernel_null_primitive.cu
  src/join/filter_join_indices/filter_join_indices_kernel_primitive.cu
  src/join/filter_join_indices/filter_join_indices_output_size_kernel_complex.cu
  src/join/filter_join_indices/filter_join_indices_output_size_kernel_null_complex.cu
  src/join/filter_join_indices/filter_join_indices_output_size_kernel_null_primitive.cu
  src/join/filter_join_indices/filter_join_indices_output_size_kernel_primitive.cu
  src/join/filtered_join/filtered_join.cu
  src/join/filtered_join/filtered_join_flat.cu
  src/join/filtered_join/filtered_join_nested.cu
  src/join/filtered_join/filtered_join_nested_query.cu
  src/join/filtered_join/filtered_join_primitive.cu
  src/join/hash_join/finalize_partitioned_full_join.cpp
  src/join/hash_join/full_join_match_context.cpp
  src/join/hash_join/full_join_retrieve.cu
  src/join/hash_join/full_join_size.cu
  src/join/hash_join/full_join_size_impl.cu
  src/join/hash_join/hash_join.cu
  src/join/hash_join/inner_join_match_context.cpp
  src/join/hash_join/inner_join_retrieve.cu
  src/join/hash_join/inner_join_size.cu
  src/join/hash_join/left_join_match_context.cpp
  src/join/hash_join/left_join_retrieve.cu
  src/join/hash_join/left_join_size.cu
  src/join/hash_join/match_context.cu
  src/join/hash_join/partitioned_count.cu
  src/join/hash_join/partitioned_count_outer.cu
  src/join/hash_join/partitioned_full_join.cu
  src/join/hash_join/partitioned_inner_join.cu
  src/join/hash_join/partitioned_join_retrieve.cu
  src/join/hash_join/partitioned_left_join.cu
  src/join/hash_join/partitioned_retrieve.cu
  src/join/hash_join/partitioned_retrieve_outer.cu
  src/join/join.cu
  src/join/join_utils.cu
  src/join/key_remapping.cu
  src/join/mark_join.cu
  src/join/mixed_join.cu
  src/join/mixed_join_kernels_semi.cu
  src/join/mixed_join_semi.cu
  src/join/sort_merge_join.cu
  src/json/json_path.cu
  src/lists/contains.cu
  src/lists/combine/concatenate_list_elements.cu
  src/lists/combine/concatenate_rows.cu
  src/lists/copying/concatenate.cu
  src/lists/copying/copying.cu
  src/lists/copying/gather.cu
  src/lists/copying/segmented_gather.cu
  src/lists/copying/scatter_helper.cu
  src/lists/count_elements.cu
  src/lists/dremel.cu
  src/lists/explode.cu
  src/lists/extract.cu
  src/lists/interleave_columns.cu
  src/lists/lists_column_factories.cu
  src/lists/lists_column_view.cu
  src/lists/reverse.cu
  src/lists/segmented_sort.cu
  src/lists/sequences.cu
  src/lists/set_operations.cu
  src/lists/stream_compaction/apply_boolean_mask.cu
  src/lists/stream_compaction/distinct.cu
  src/lists/utilities.cu
  src/merge/merge.cu
  src/partitioning/partitioning.cu
  src/partitioning/round_robin.cu
  src/quantiles/tdigest/tdigest.cu
  src/quantiles/tdigest/tdigest_aggregation.cu
  src/quantiles/tdigest/tdigest_column_view.cpp
  src/quantiles/quantile.cu
  src/quantiles/quantiles.cu
  src/reductions/all.cu
  src/reductions/any.cu
  src/reductions/approx_distinct_count.cu
  src/reductions/argmax.cu
  src/reductions/argmin.cu
  src/reductions/bitwise.cu
  src/reductions/collect_ops.cu
  src/reductions/count.cpp
  src/reductions/distinct_count.cu
  src/reductions/histogram.cu
  src/reductions/max.cu
  src/reductions/mean.cu
  src/reductions/min.cu
  src/reductions/minmax.cu
  src/reductions/nth_element.cu
  src/reductions/nunique.cpp
  src/reductions/product.cu
  src/reductions/quantile.cu
  src/reductions/reductions.cpp
  src/reductions/scan/ewm.cu
  src/reductions/scan/rank_scan.cu
  src/reductions/scan/scan.cpp
  src/reductions/scan/scan_exclusive.cu
  src/reductions/scan/scan_inclusive.cu
  src/reductions/segmented/all.cu
  src/reductions/segmented/any.cu
  src/reductions/segmented/counts.cu
  src/reductions/segmented/max.cu
  src/reductions/segmented/mean.cu
  src/reductions/segmented/min.cu
  src/reductions/segmented/nunique.cu
  src/reductions/segmented/product.cu
  src/reductions/segmented/reductions.cpp
  src/reductions/segmented/std.cu
  src/reductions/segmented/sum.cu
  src/reductions/segmented/sum_of_squares.cu
  src/reductions/segmented/update_validity.cu
  src/reductions/segmented/var.cu
  src/reductions/std.cu
  src/reductions/sum.cu
  src/reductions/sum_of_squares.cu
  src/reductions/sum_overflow.cu
  src/reductions/unique_count.cu
  src/reductions/unique_count_column.cu
  src/reductions/var.cu
  src/replace/clamp.cu
  src/replace/nans.cu
  src/replace/nulls.cu
  src/replace/replace.cu
  src/reshape/byte_cast.cu
  src/reshape/interleave_columns.cu
  src/reshape/table_to_array.cu
  src/reshape/tile.cu
  src/rolling/detail/optimized_unbounded_window.cpp
  src/rolling/detail/rolling_collect_list.cu
  src/rolling/detail/rolling_fixed_window.cu
  src/rolling/detail/rolling_utils.cu
  src/rolling/detail/rolling_variable_window.cu
  src/rolling/grouped_rolling.cu
  src/rolling/range_rolling.cu
  src/rolling/range_window_bounds.cpp
  src/rolling/rolling.cpp
  src/round/round.cu
  src/row_operator/primitive_row_operators.cu
  src/row_operator/row_operators.cu
  src/runtime/context.cpp
  src/scalar/scalar.cpp
  src/scalar/scalar_factories.cpp
  src/search/contains_column.cu
  src/search/contains_scalar.cu
  src/search/contains_table.cu
  src/search/contains_table_impl.cu
  src/search/search_ordered.cu
  src/sort/is_sorted.cu
  src/sort/rank.cu
  src/sort/segmented_sort.cu
  src/sort/segmented_top_k.cu
  src/sort/sort.cu
  src/sort/sort_column.cu
  src/sort/sort_radix.cu
  src/sort/sorted_order_radix.cu
  src/sort/stable_segmented_sort.cu
  src/sort/stable_sort_column.cu
  src/sort/stable_sort.cu
  src/sort/top_k.cu
  src/stream_compaction/apply_boolean_mask.cu
  src/stream_compaction/distinct.cu
  src/stream_compaction/distinct_helpers.cu
  src/stream_compaction/distinct_helpers_flat_nan_equal_any.cu
  src/stream_compaction/distinct_helpers_flat_nan_equal_ordered.cu
  src/stream_compaction/distinct_helpers_flat_nan_unequal_any.cu
  src/stream_compaction/distinct_helpers_flat_nan_unequal_ordered.cu
  src/stream_compaction/distinct_helpers_nested_nan_equal_any.cu
  src/stream_compaction/distinct_helpers_nested_nan_equal_ordered.cu
  src/stream_compaction/distinct_helpers_nested_nan_unequal_any.cu
  src/stream_compaction/distinct_helpers_nested_nan_unequal_ordered.cu
  src/stream_compaction/drop_nans.cu
  src/stream_compaction/drop_nulls.cu
  src/stream_compaction/filter/filter.cu
  src/stream_compaction/stable_distinct.cu
  src/stream_compaction/unique.cu
  src/strings/attributes.cu
  src/strings/capitalize.cu
  src/strings/case.cu
  src/strings/char_types/char_cases.cu
  src/strings/char_types/char_types.cu
  src/strings/combine/concatenate.cu
  src/strings/combine/join.cu
  src/strings/combine/join_list_elements.cu
  src/strings/contains.cu
  src/strings/convert/convert_booleans.cu
  src/strings/convert/convert_datetime.cu
  src/strings/convert/convert_durations.cu
  src/strings/convert/convert_fixed_point.cu
  src/strings/convert/convert_floats.cu
  src/strings/convert/convert_hex.cu
  src/strings/convert/convert_integers.cu
  src/strings/convert/convert_ipv4.cu
  src/strings/convert/convert_urls.cu
  src/strings/convert/convert_lists.cu
  src/strings/convert/int_cast.cu
  src/strings/copying/concatenate.cu
  src/strings/copying/copying.cu
  src/strings/copying/copy_range.cu
  src/strings/copying/shift.cu
  src/strings/count_matches.cu
  src/strings/extract/extract.cu
  src/strings/extract/extract_all.cu
  src/strings/filling/fill.cu
  src/strings/filter_chars.cu
  src/strings/like.cu
  src/strings/merge/merge.cu
  src/strings/padding.cu
  src/strings/positions.cu
  src/strings/regex/gkexec.cpp
  src/strings/regex/glushkov_regcomp.cpp
  src/strings/regex/regcomp.cpp
  src/strings/regex/regexec.cpp
  src/strings/regex/regex_program.cpp
  src/strings/repeat_strings.cu
  src/strings/replace/backref_re.cu
  src/strings/replace/find_replace.cu
  src/strings/replace/multi.cu
  src/strings/replace/replace.cu
  src/strings/replace/replace_nulls.cu
  src/strings/replace/replace_re.cu
  src/strings/replace/replace_slice.cu
  src/strings/reverse.cu
  src/strings/scan/scan_inclusive.cu
  src/strings/search/contains_multiple.cu
  src/strings/search/count.cu
  src/strings/search/findall.cu
  src/strings/search/find.cu
  src/strings/search/find_instance.cu
  src/strings/search/find_multiple.cu
  src/strings/slice.cu
  src/strings/split/partition.cu
  src/strings/split/split.cu
  src/strings/split/split_part.cu
  src/strings/split/split_re.cu
  src/strings/split/split_record.cu
  src/strings/strings_column_factories.cu
  src/strings/strings_column_view.cpp
  src/strings/strings_scalar_factories.cpp
  src/strings/strip.cu
  src/strings/translate.cu
  src/strings/utilities.cu
  src/strings/wrap.cu
  src/structs/copying/concatenate.cu
  src/structs/scan/scan_inclusive.cu
  src/structs/structs_column_factories.cu
  src/structs/structs_column_view.cpp
  src/structs/utilities.cu
  src/table/table.cpp
  src/table/table_device_view.cu
  src/table/table_equal.cu
  src/table/table_view.cpp
  src/text/deduplicate.cu
  src/text/detokenize.cu
  src/text/edit_distance.cu
  src/text/generate_ngrams.cu
  src/text/jaccard.cu
  src/text/minhash.cu
  src/text/ngrams_tokenize.cu
  src/text/normalize.cu
  src/text/replace.cu
  src/text/stemmer.cu
  src/text/bpe/byte_pair_encoding.cu
  src/text/bpe/load_merge_pairs.cu
  src/text/tokenize.cu
  src/text/vocabulary_tokenize.cu
  src/text/wordpiece_tokenize.cu
  src/transform/bools_to_mask.cu
  src/transform/compute_column.cu
  src/transform/compute_column_kernel_complex.cu
  src/transform/compute_column_kernel_null_complex.cu
  src/transform/compute_column_kernel_null_primitive.cu
  src/transform/compute_column_kernel_primitive.cu
  src/transform/encode.cu
  src/transform/mask_to_bools.cu
  src/transform/nans_to_nulls.cu
  src/transform/one_hot_encode.cu
  src/transform/row_bit_count.cu
  src/transform/transform.cu
  src/transpose/transpose.cu
  src/unary/cast_ops.cu
  src/unary/math_ops.cu
  src/unary/nan_ops.cu
  src/unary/null_ops.cu
  src/utilities/cuda.cpp
  src/utilities/cuda_memcpy.cu
  src/utilities/default_stream.cpp
  src/utilities/getenv_or.cpp
  src/utilities/grid_1d.cu
  src/utilities/host_memory.cpp
  src/utilities/host_worker_pool.cpp
  src/utilities/linked_column.cpp
  src/utilities/logger.cpp
  src/utilities/prefetch.cpp
  src/utilities/roaring_bitmap.cu
  src/utilities/stream_pool.cpp
  src/utilities/traits.cpp
  src/utilities/type_checks.cpp
  src/utilities/type_dispatcher.cpp
  ${cudf_cuda_embed_SOURCE_DIR}/cudf_cuda_embed.s
  ${cudf_fragments_SOURCE_DIR}/cudf_fragments.s
)

add_dependencies(cudf cudf_cuda_embed cudf_fragments)

set_property(
  SOURCE src/io/parquet/writer_impl.cu
  APPEND
  PROPERTY COMPILE_DEFINITIONS "CUDF_VERSION=${PROJECT_VERSION}"
)

set_target_properties(
  cudf
  PROPERTIES BUILD_RPATH "\$ORIGIN"
             INSTALL_RPATH "\$ORIGIN"
             # set target compile options
             CXX_STANDARD 20
             CXX_STANDARD_REQUIRED ON
             # For std:: support of __int128_t. Can be removed once using cuda::std
             CXX_EXTENSIONS ON
             CXX_VISIBILITY_PRESET hidden
             CUDA_STANDARD 20
             CUDA_STANDARD_REQUIRED ON
             CUDA_VISIBILITY_PRESET hidden
             POSITION_INDEPENDENT_CODE ON
             INTERFACE_POSITION_INDEPENDENT_CODE ON
             LINK_FLAGS "-Wl,--exclude-libs,libzstd.a"
)

# Export the compiler flags and definitions so the downstream applications can optionally retrieve
# and use them.
set_target_properties(
  cudf
  PROPERTIES
    EXPORT_PROPERTIES
    "CUDF_CXX_FLAGS;CUDF_CUDA_FLAGS;CUDF_CXX_DEFINITIONS;CUDF_CUDA_DEFINITIONS;CUDF_STATIC_LINKED_PROJECTS"
    CUDF_CXX_FLAGS "${CUDF_CXX_FLAGS}"
    CUDF_CUDA_FLAGS "${CUDF_CUDA_FLAGS}"
    CUDF_CXX_DEFINITIONS "${CUDF_CXX_DEFINITIONS}"
    CUDF_CUDA_DEFINITIONS "${CUDF_CUDA_DEFINITIONS}"
)

# Note: This must come before the target_compile_options below so that the function can modify the
# flags if necessary.
if(CUDF_CLANG_TIDY OR CUDF_IWYU)
  set(linters)

  if(CUDF_CLANG_TIDY)
    list(APPEND linters CLANG_TIDY)
  endif()

  if(CUDF_IWYU)
    list(APPEND linters IWYU)
  endif()

  enable_static_checkers(
    cudf SKIPPED_FILES src/io/comp/cpu_unbz2.cpp src/io/comp/brotli_dict.cpp ${linters}
  )
endif()

target_compile_options(
  cudf PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${CUDF_CXX_FLAGS}>"
               "$<$<COMPILE_LANGUAGE:CUDA>:${CUDF_CUDA_FLAGS}>"
)

# Specify include paths for the current target and dependents
target_include_directories(
  cudf
  PUBLIC "$<BUILD_INTERFACE:${DLPACK_INCLUDE_DIR}>" "$<BUILD_INTERFACE:${CUDF_SOURCE_DIR}/include>"
         "$<BUILD_INTERFACE:${CUDF_GENERATED_INCLUDE_DIR}/include>"
  PRIVATE "$<BUILD_INTERFACE:${CUDF_SOURCE_DIR}/src>"
          "$<BUILD_INTERFACE:${nanoarrow_SOURCE_DIR}/src>"
          "$<BUILD_INTERFACE:${FlatBuffers_SOURCE_DIR}/include>"
          "$<BUILD_INTERFACE:${ZSTD_INCLUDE_DIR}>"
          "$<BUILD_INTERFACE:${cudf_cuda_embed_INCLUDE_DIRS}>"
          "$<BUILD_INTERFACE:${cudf_fragments_INCLUDE_DIRS}>"
  INTERFACE "$<INSTALL_INTERFACE:include>"
)

target_compile_definitions(
  cudf PUBLIC "$<$<COMPILE_LANGUAGE:CXX>:${CUDF_CXX_DEFINITIONS}>"
              "$<BUILD_INTERFACE:$<$<COMPILE_LANGUAGE:CUDA>:${CUDF_CUDA_DEFINITIONS}>>"
)

# Per-thread default stream
if(CUDF_USE_PER_THREAD_DEFAULT_STREAM)
  target_compile_definitions(
    cudf PUBLIC CUDA_API_PER_THREAD_DEFAULT_STREAM CUDF_USE_PER_THREAD_DEFAULT_STREAM
  )
endif()

# Disable NVTX if necessary
if(NOT USE_NVTX)
  target_compile_definitions(cudf PUBLIC NVTX_DISABLE)
endif()

# Disable large strings support
if(CUDF_LARGE_STRINGS_DISABLED)
  target_compile_definitions(cudf PRIVATE CUDF_LARGE_STRINGS_DISABLED)
endif()

# Define logging level
target_compile_definitions(
  cudf PRIVATE "CUDF_LOG_ACTIVE_LEVEL=RAPIDS_LOGGER_LOG_LEVEL_${LIBCUDF_LOGGING_LEVEL}"
)

# Enable remote IO through KvikIO
target_compile_definitions(cudf PRIVATE $<$<BOOL:${CUDF_KVIKIO_REMOTE_IO}>:CUDF_KVIKIO_REMOTE_IO>)

# Remove this after upgrading to a CCCL that has a proper CMake option. See
# https://github.com/NVIDIA/cccl/pull/2844
target_compile_definitions(cudf PRIVATE THRUST_FORCE_32_BIT_OFFSET_TYPE=1 CCCL_AVOID_SORT_UNROLL=1)

# When rmm is a static library being absorbed via whole-archive, strip nvtx3 from its public
# interface. We bundle nvtx3 headers directly into cudf's install tree, so consumers get them from
# cudf's include path without needing the nvtx3 target or find_dependency(nvtx3).
get_target_property(_rmm_type rmm::rmm TYPE)
if(BUILD_SHARED_LIBS AND _rmm_type STREQUAL "STATIC_LIBRARY")
  get_target_property(_rmm_real rmm::rmm ALIASED_TARGET)
  if(_rmm_real)
    get_target_property(_rmm_libs ${_rmm_real} INTERFACE_LINK_LIBRARIES)
    if(_rmm_libs)
      list(REMOVE_ITEM _rmm_libs nvtx3::nvtx3-cpp)
      set_property(TARGET ${_rmm_real} PROPERTY INTERFACE_LINK_LIBRARIES ${_rmm_libs})
    endif()
  endif()
endif()

# When building a shared libcudf with static deps, we absorb them via whole-archive linking so that
# their symbols are available to downstream users linking to libcudf.so without needing to link
# against them directly. Public absorbed dependencies still promote their public transitive
# dependencies and export set metadata into cudf's interface.
set(_absorbed_deps rmm rapids_logger)
set(_absorbed_nvcomp FALSE)
foreach(_dep IN LISTS _absorbed_deps)
  if(NOT TARGET "${_dep}::${_dep}")
    continue()
  endif()
  get_target_property(_dep_link "${_dep}::${_dep}" NAME)
  get_target_property(_target_type ${_dep_link} TYPE)
  if(NOT BUILD_SHARED_LIBS OR NOT _target_type STREQUAL "STATIC_LIBRARY")
    # If not building a shared library, or the dependency is not a static library, link normally.
    target_link_libraries(cudf PUBLIC ${_dep_link})
    continue()
  endif()

  # Promote the absorbed library's public transitive deps into cudf's public interface. Filter out
  # deps that are themselves absorbed — they are already linked via whole-archive. Also filter
  # LINK_ONLY entries — those are private link deps already statically linked in.
  get_target_property(_iface_libs ${_dep_link} INTERFACE_LINK_LIBRARIES)
  if(_iface_libs)
    foreach(_absorbed IN LISTS _absorbed_deps)
      list(REMOVE_ITEM _iface_libs ${_absorbed}::${_absorbed})
    endforeach()
    list(FILTER _iface_libs EXCLUDE REGEX "\\\$<LINK_ONLY:")
    target_link_libraries(cudf PUBLIC ${_iface_libs})
  endif()

  # Promote the absorbed library's public include directories into cudf's public interface so that
  # in-tree consumers (tests, Python bindings) can find the absorbed library's headers.
  get_target_property(_dep_includes ${_dep_link} INTERFACE_INCLUDE_DIRECTORIES)
  if(_dep_includes)
    target_include_directories(cudf PUBLIC ${_dep_includes})
  endif()

  # Wrap with WHOLE_ARCHIVE + BUILD_INTERFACE so the static lib is absorbed into libcudf.so and
  # hidden from the installed interface.
  target_link_libraries(
    cudf PRIVATE "$<BUILD_INTERFACE:$<LINK_LIBRARY:WHOLE_ARCHIVE,${_dep_link}>>"
  )
  set_property(
    TARGET cudf
    APPEND
    PROPERTY CUDF_STATIC_LINKED_PROJECTS "${_dep_link}"
  )

  # nvcomp_static has done a WHOLE_ARCHIVE on rmm and rapids_logger itself. Therefore it needs to
  # come after our WHOLE_ARCHIVE otherwise we will get duplicate symbols for rmm and/or
  # rapids_logger
  if(TARGET nvcomp::nvcomp_static)
    target_link_libraries(${_dep_link} INTERFACE $<BUILD_LOCAL_INTERFACE:nvcomp::nvcomp_static>)
    set(_absorbed_nvcomp TRUE)
  endif()

  # Merge the absorbed library's export set metadata into cudf-exports so the installed config has
  # find_dependency() calls for all transitive deps (e.g. nvtx3, CUDAToolkit from rmm).
  foreach(_mode build install)
    set(_src_target rapids_export_${_mode}_${_dep}-exports)
    if(NOT TARGET ${_src_target})
      continue()
    endif()
    get_property(
      _pkg_names
      TARGET ${_src_target}
      PROPERTY "PACKAGE_NAMES"
    )
    get_property(
      _global_tgts
      TARGET ${_src_target}
      PROPERTY "GLOBAL_TARGETS"
    )
    foreach(_pkg IN LISTS _pkg_names)
      # Skip packages that are themselves absorbed into libcudf or bundled as headers — they won't
      # be findable as separate packages.
      if(_pkg IN_LIST _absorbed_deps OR _pkg STREQUAL "nvtx3")
        continue()
      endif()
      rapids_export_package(${_mode} ${_pkg} cudf-exports GLOBAL_TARGETS ${_global_tgts})
    endforeach()
  endforeach()
endforeach()

if(TARGET spdlog::spdlog)
  get_target_property(_spdlog_link spdlog::spdlog NAME)
  get_target_property(_spdlog_type ${_spdlog_link} TYPE)
  if(BUILD_SHARED_LIBS AND _spdlog_type STREQUAL "STATIC_LIBRARY")
    target_link_libraries(
      cudf PRIVATE "$<BUILD_INTERFACE:$<LINK_LIBRARY:WHOLE_ARCHIVE,${_spdlog_link}>>"
    )
    set_property(
      TARGET cudf
      APPEND
      PROPERTY CUDF_STATIC_LINKED_PROJECTS "${_spdlog_link}"
    )
  else()
    target_link_libraries(cudf PRIVATE ${_spdlog_link})
  endif()
endif()

# Specify the target module library dependencies We do this after all the WHOLE_ARCHIVES to protect
# against any of these libraries ever adding a rapids_logger / rmm hidden dependency
target_link_libraries(
  cudf
  PUBLIC CCCL::CCCL $<BUILD_LOCAL_INTERFACE:BS::thread_pool>
  PRIVATE $<BUILD_LOCAL_INTERFACE:nvtx3::nvtx3-cpp> $<BUILD_LOCAL_INTERFACE:cuco::cuco> ZLIB::ZLIB
          kvikio::kvikio ${CUDF_nanoarrow_TARGET} zstd $<BUILD_LOCAL_INTERFACE:xxhash> rtcx::rtcx
)

if(NOT _absorbed_nvcomp)
  target_link_libraries(cudf PRIVATE ${CUDF_nvcomp_TARGET})
  if(CUDF_nvcomp_TARGET STREQUAL "nvcomp::nvcomp_static")
    set_property(
      TARGET cudf
      APPEND
      PROPERTY CUDF_STATIC_LINKED_PROJECTS "nvcomp"
    )
  endif()
endif()

# Add Conda library, and include paths if specified
if(TARGET conda_env)
  target_link_libraries(cudf PRIVATE conda_env)
endif()

rapids_cuda_set_runtime(cudf USE_STATIC ON)

file(
  WRITE "${CUDF_BINARY_DIR}/fatbin.ld"
  [=[
SECTIONS
{
  .nvFatBinSegment : { *(.nvFatBinSegment) }
  .nv_fatbin : { *(.nv_fatbin) }
}
]=]
)
target_link_options(cudf PRIVATE "$<HOST_LINK:${CUDF_BINARY_DIR}/fatbin.ld>")

add_library(cudf::cudf ALIAS cudf)

# ##################################################################################################
# * tests and benchmarks --------------------------------------------------------------------------
# ##################################################################################################

# ##################################################################################################
# * build cudftestutil ----------------------------------------------------------------------------
if(CUDF_BUILD_TESTUTIL)
  add_library(
    cudftest_default_stream
    # When compiled as a dynamic library allows us to use LD_PRELOAD injection of symbols. We
    # currently leverage this for stream-related library validation and may make use of it for
    # other similar features in the future.
    tests/utilities/default_stream.cpp
  )
  set_target_properties(
    cudftest_default_stream
    PROPERTIES BUILD_RPATH "\$ORIGIN"
               INSTALL_RPATH "\$ORIGIN"
               # set target compile options
               CXX_STANDARD 20
               CXX_STANDARD_REQUIRED ON
               CUDA_STANDARD 20
               CUDA_STANDARD_REQUIRED ON
               POSITION_INDEPENDENT_CODE ON
               INTERFACE_POSITION_INDEPENDENT_CODE ON
  )
  target_link_libraries(
    cudftest_default_stream
    PUBLIC cudf
    PRIVATE $<TARGET_NAME_IF_EXISTS:conda_env>
  )
  rapids_cuda_set_runtime(cudftest_default_stream USE_STATIC ON)

  add_library(cudf::cudftest_default_stream ALIAS cudftest_default_stream)

  add_library(cudftestutil INTERFACE)

  set_target_properties(
    cudftestutil
    PROPERTIES BUILD_RPATH "\$ORIGIN"
               INSTALL_RPATH "\$ORIGIN"
               # set target compile options
               CXX_STANDARD 20
               CXX_STANDARD_REQUIRED ON
               CUDA_STANDARD 20
               CUDA_STANDARD_REQUIRED ON
  )

  target_compile_options(
    cudftestutil INTERFACE "$<BUILD_INTERFACE:$<$<COMPILE_LANGUAGE:CXX>:${CUDF_CXX_FLAGS}>>"
                           "$<BUILD_INTERFACE:$<$<COMPILE_LANGUAGE:CUDA>:${CUDF_CUDA_FLAGS}>>"
  )

  target_link_libraries(
    cudftestutil INTERFACE $<BUILD_LOCAL_INTERFACE:cuco::cuco> Threads::Threads cudf rmm::rmm
                           cudftest_default_stream $<TARGET_NAME_IF_EXISTS:conda_env>
  )

  target_include_directories(
    cudftestutil INTERFACE "$<BUILD_INTERFACE:${CUDF_SOURCE_DIR}>"
                           "$<BUILD_INTERFACE:${CUDF_SOURCE_DIR}/src>"
  )
  rapids_cuda_set_runtime(cudftestutil USE_STATIC ON)
  add_library(cudf::cudftestutil ALIAS cudftestutil)

  add_library(cudftestutil_impl INTERFACE)
  add_library(cudf::cudftestutil_impl ALIAS cudftestutil_impl)
  target_sources(
    cudftestutil_impl
    INTERFACE
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/tests/io/metadata_utilities.cpp>
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/tests/utilities/column_utilities.cu>
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/tests/utilities/debug_utilities.cu>
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/tests/utilities/memory_resource_utilities.cpp>
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/tests/utilities/random_seed.cpp>
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/tests/utilities/table_utilities.cu>
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/tests/utilities/tdigest_utilities.cpp>
      $<INSTALL_INTERFACE:src/cudftestutil/io/metadata_utilities.cpp>
      $<INSTALL_INTERFACE:src/cudftestutil/utilities/column_utilities.cu>
      $<INSTALL_INTERFACE:src/cudftestutil/utilities/debug_utilities.cu>
      $<INSTALL_INTERFACE:src/cudftestutil/utilities/memory_resource_utilities.cpp>
      $<INSTALL_INTERFACE:src/cudftestutil/utilities/random_seed.cpp>
      $<INSTALL_INTERFACE:src/cudftestutil/utilities/table_utilities.cu>
      $<INSTALL_INTERFACE:src/cudftestutil/utilities/tdigest_utilities.cpp>
  )
  target_link_libraries(cudftestutil_impl INTERFACE cudf::cudftestutil)
  target_compile_features(cudftestutil_impl INTERFACE cxx_std_20)

  # Base library for linking to cudf::cudftestutil, Note that targets should not directly link to
  # cudf::cudftestutil_impl and should instead link to an OBJECT library that contains the compiled
  # test utilities first
  add_library(cudftestutil_objects OBJECT)
  target_link_libraries(
    cudftestutil_objects
    PUBLIC cudf::cudftestutil GTest::gmock GTest::gmock_main GTest::gtest GTest::gtest_main
    PRIVATE cudf::cudftestutil_impl
  )
  add_library(cudf::cudftestutil_objects ALIAS cudftestutil_objects)

  install(FILES tests/io/metadata_utilities.cpp DESTINATION src/cudftestutil/io)
  install(
    FILES tests/utilities/column_utilities.cu
          tests/utilities/debug_utilities.cu
          tests/utilities/memory_resource_utilities.cpp
          tests/utilities/random_seed.cpp
          tests/utilities/table_utilities.cu
          tests/utilities/tdigest_utilities.cpp
    DESTINATION src/cudftestutil/utilities
  )
endif()

# * build cudf_identify_stream_usage --------------------------------------------------------------
if(CUDF_BUILD_STREAMS_TEST_UTIL)
  # Libraries for stream-related testing. We build the library twice, one with STREAM_MODE_TESTING
  # on and one with it set to off. Each test will then be configured to use the appropriate library
  # depending via ctest and whether it has been updated to expose public stream APIs.
  foreach(_mode cudf testing)
    set(_tgt "cudf_identify_stream_usage_mode_${_mode}")
    add_library(${_tgt} SHARED tests/utilities/identify_stream_usage.cpp)

    if(CUDF_USE_PER_THREAD_DEFAULT_STREAM)
      target_compile_definitions(
        ${_tgt} PUBLIC CUDA_API_PER_THREAD_DEFAULT_STREAM CUDF_USE_PER_THREAD_DEFAULT_STREAM
      )
    endif()

    set(sanitizer_relative_genex
        "$<PATH:RELATIVE_PATH,$<TARGET_FILE_DIR:CUDA::sanitizer>,$<TARGET_FILE_DIR:CUDA::cudart_static>>"
    )
    set_target_properties(
      ${_tgt}
      PROPERTIES # set target compile options
                 CXX_STANDARD 20
                 CXX_STANDARD_REQUIRED ON
                 POSITION_INDEPENDENT_CODE ON
                 BUILD_RPATH "\$ORIGIN"
                 INSTALL_RPATH "\$ORIGIN;\$ORIGIN/${sanitizer_relative_genex}"
    )
    target_compile_options(
      ${_tgt} PRIVATE "$<BUILD_INTERFACE:$<$<COMPILE_LANGUAGE:CXX>:${CUDF_CXX_FLAGS}>>"
    )
    target_include_directories(${_tgt} PRIVATE "$<BUILD_INTERFACE:${CUDF_SOURCE_DIR}/include>")
    target_link_libraries(${_tgt} PUBLIC CUDA::cudart_static rmm::rmm CUDA::sanitizer)

    rapids_cuda_set_runtime(${_tgt} USE_STATIC ON)
    add_library(cudf::${_tgt} ALIAS ${_tgt})

    if("${_mode}" STREQUAL "testing")
      target_compile_definitions(${_tgt} PUBLIC STREAM_MODE_TESTING)
    endif()
  endforeach()
endif()

# ##################################################################################################
# * add tests -------------------------------------------------------------------------------------
if(CUDF_BUILD_TESTS)
  # include CTest module -- automatically calls enable_testing()
  include(CTest)

  # ctest cuda memcheck
  find_program(CUDA_SANITIZER compute-sanitizer)
  set(MEMORYCHECK_COMMAND ${CUDA_SANITIZER})
  set(MEMORYCHECK_TYPE CudaSanitizer)
  set(CUDA_SANITIZER_COMMAND_OPTIONS "--tool memcheck")

  # Always print verbose output when tests fail if run using `make test`.
  list(APPEND CMAKE_CTEST_ARGUMENTS "--output-on-failure")
  add_subdirectory(tests)
endif()

# ##################################################################################################
# * add benchmarks --------------------------------------------------------------------------------
if(CUDF_BUILD_BENCHMARKS)
  # Find or install nvbench
  include(cmake/thirdparty/get_nvbench.cmake)

  add_subdirectory(benchmarks)
endif()

# ##################################################################################################
# * install targets -------------------------------------------------------------------------------
rapids_cmake_install_lib_dir(lib_dir)
include(CPack)
include(GNUInstallDirs)

set(CMAKE_INSTALL_DEFAULT_COMPONENT_NAME cudf)

# install target for cudf_base and the proxy libcudf.so
install(
  TARGETS cudf
  DESTINATION ${lib_dir}
  EXPORT cudf-exports
)
install(FILES ${CUDF_BINARY_DIR}/include/cudf/version_config.hpp
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/cudf
)

set(_components_export_string)

if(TARGET cudftestutil)
  install(
    TARGETS cudftest_default_stream cudftestutil cudftestutil_impl
    DESTINATION ${lib_dir}
    EXPORT cudf-testing-exports
  )
  set(_components_export_string COMPONENTS testing COMPONENTS_EXPORT_SET cudf-testing-exports)
endif()

install(DIRECTORY ${CUDF_SOURCE_DIR}/include/cudf ${CUDF_SOURCE_DIR}/include/cudf_test
                  ${CUDF_SOURCE_DIR}/include/nvtext DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# In standalone DSO mode, rmm and rapids_logger are absorbed into libcudf via whole-archive. Their
# headers are still needed by consumers, so install them manually.
if(NOT CUDF_INSTALL_LIBRARY_DEPS)
  # rmm uses SOURCE_SUBDIR cpp, so headers are at ${rmm_SOURCE_DIR}/cpp/include/rmm/ Generated
  # headers (version_config.hpp, logger_macros.hpp) are at ${rmm_BINARY_DIR}/include/rmm/
  if(rmm_SOURCE_DIR)
    install(DIRECTORY ${rmm_SOURCE_DIR}/cpp/include/rmm DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
    install(DIRECTORY ${rmm_BINARY_DIR}/include/rmm DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
  endif()
  if(rapids_logger_SOURCE_DIR)
    install(DIRECTORY ${rapids_logger_SOURCE_DIR}/include/rapids_logger
            DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    )
  endif()
  # nvtx3 headers: nvtx3_SOURCE_DIR resolves to the cache variable set by FetchContent, which
  # includes the SOURCE_SUBDIR (c/), so the headers are at ${nvtx3_SOURCE_DIR}/include/nvtx3/.
  if(nvtx3_SOURCE_DIR)
    install(DIRECTORY ${nvtx3_SOURCE_DIR}/include/nvtx3 DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
  endif()
endif()

if(CUDF_BUILD_STREAMS_TEST_UTIL)
  install(
    TARGETS cudf_identify_stream_usage_mode_cudf
    DESTINATION ${lib_dir}
    COMPONENT testing
  )
  install(
    TARGETS cudf_identify_stream_usage_mode_testing
    DESTINATION ${lib_dir}
    COMPONENT testing
  )
endif()

set(doc_string
    [=[
Provide targets for the cudf library.

Built based on the Apache Arrow columnar memory format, cuDF is a GPU DataFrame
library for loading, joining, aggregating, filtering, and otherwise
manipulating data.

cuDF provides a pandas-like API that will be familiar to data engineers &
data scientists, so they can use it to easily accelerate their workflows
without going into the details of CUDA programming.


Imported Targets
^^^^^^^^^^^^^^^^

If cudf is found, this module defines the following IMPORTED GLOBAL
targets:

 cudf::cudf             - The main cudf library.

This module offers an optional testing component which defines the
following IMPORTED GLOBAL  targets:

 cudf::cudftestutil          - The main cudf testing library
 cudf::cudftestutil_impl     - C++ and CUDA sources to compile for definitions in cudf::cudftestutil
    ]=]
)

rapids_export(
  INSTALL cudf
  EXPORT_SET cudf-exports ${_components_export_string}
  GLOBAL_TARGETS cudf cudftestutil cudftestutil_impl
  NAMESPACE cudf::
  DOCUMENTATION doc_string
)

# ##################################################################################################
# * build export -------------------------------------------------------------------------------
set(build_code_string
    [=[
if(EXISTS "${CMAKE_CURRENT_LIST_DIR}/cudf-testing-dependencies.cmake")
  include("${CMAKE_CURRENT_LIST_DIR}/cudf-testing-dependencies.cmake")
endif()
if(EXISTS "${CMAKE_CURRENT_LIST_DIR}/cudf-testing-targets.cmake")
  include("${CMAKE_CURRENT_LIST_DIR}/cudf-testing-targets.cmake")
endif()
]=]
)

rapids_export(
  BUILD cudf
  EXPORT_SET cudf-exports ${_components_export_string}
  GLOBAL_TARGETS cudf cudftestutil cudftestutil_impl
  NAMESPACE cudf::
  DOCUMENTATION doc_string
  FINAL_CODE_BLOCK build_code_string
)

# ##################################################################################################
# * make documentation ----------------------------------------------------------------------------

# doc targets for cuDF
add_custom_command(
  OUTPUT CUDF_DOXYGEN
  WORKING_DIRECTORY ${CUDF_SOURCE_DIR}/doxygen
  COMMAND ${CMAKE_COMMAND} -E env "RAPIDS_VERSION=${RAPIDS_VERSION}"
          "RAPIDS_VERSION_MAJOR_MINOR=${RAPIDS_VERSION_MAJOR_MINOR}" doxygen Doxyfile
  VERBATIM
  COMMENT "Custom command for building cudf doxygen docs."
)

add_custom_target(
  docs_cudf
  DEPENDS CUDF_DOXYGEN
  COMMENT "Custom command for building cudf doxygen docs."
)

# ##################################################################################################
# * make gdb helper scripts ------------------------------------------------------------------------

# build pretty-printer load script
if(Thrust_SOURCE_DIR AND rmm_SOURCE_DIR)
  configure_file(scripts/load-pretty-printers.in load-pretty-printers @ONLY)
endif()
