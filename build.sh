#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# cuDF build script

# This script is used to build the component(s) in this repo from
# source, and can be called with various options to customize the
# build as needed (see the help output for details)
# Abort script on first error
set -e

NUMARGS=$#
ARGS=$*

# NOTE: ensure all dir changes are relative to the location of this
# script, and that this script resides in the repo dir!
REPODIR=$(cd "$(dirname "$0")"; pwd)

VALIDARGS="clean libcudf pylibcudf cudf cudf_polars dask_cudf benchmarks tests libcudf_kafka cudf_kafka custreamz libcudf_streaming cudf_streaming -v -g -n --pydevelop -l --allgpuarch --disable_nvtx --opensource_nvcomp  --hide_depr_warn --ptds -h --build_metrics --incl_cache_stats --disable_large_strings"
HELP="$0 [clean] [libcudf] [pylibcudf] [cudf] [cudf_polars] [dask_cudf] [benchmarks] [tests] [libcudf_kafka] [cudf_kafka] [custreamz] [libcudf_streaming] [cudf_streaming] [-v] [-g] [-n] [-h] [--cmake-args=\\\"<args>\\\"]
   clean                         - remove all existing build artifacts and configuration (start
                                   over)
   libcudf                       - build the cudf C++ code only
   pylibcudf                     - build the pylibcudf Python package
   cudf                          - build the cudf Python package
   cudf_polars                   - build the cudf_polars Python package
   dask_cudf                     - build the dask_cudf Python package
   benchmarks                    - build benchmarks
   tests                         - build tests
   libcudf_kafka                 - build the libcudf_kafka C++ code only
   cudf_kafka                    - build the cudf_kafka Python package
   custreamz                     - build the custreamz Python package
   libcudf_streaming              - build the libcudf_streaming C++ code only
   cudf_streaming                - build the cudf_streaming Python package
   -v                            - verbose build mode
   -g                            - build for debug
   -n                            - no install step (does not affect Python)
   --pydevelop                   - Install Python packages in editable mode
   --allgpuarch                  - build for all supported GPU architectures
   --disable_nvtx                - disable inserting NVTX profiling ranges
   --opensource_nvcomp           - disable use of proprietary nvcomp extensions
   --hide_depr_warn              - hide cmake deprecation warnings
   --ptds                        - enable per-thread default stream
   --disable_large_strings       - disable large strings support
   --build_metrics               - generate build metrics report for libcudf
   --incl_cache_stats            - include cache statistics in build metrics report
   --cmake-args=\\\"<args>\\\"   - pass arbitrary list of CMake configuration options (escape all quotes in argument)
   -h | --h[elp]                 - print this text

   default action (no args) is to build and install 'libcudf', 'pylibcudf', 'cudf', 'libcudf_streaming', 'cudf_streaming', 'cudf_polars', and 'dask_cudf' targets
"
LIB_BUILD_DIR=${LIB_BUILD_DIR:=${REPODIR}/cpp/build}
KAFKA_LIB_BUILD_DIR=${KAFKA_LIB_BUILD_DIR:=${REPODIR}/cpp/libcudf_kafka/build}
STREAMING_LIB_BUILD_DIR=${STREAMING_LIB_BUILD_DIR:=${REPODIR}/cpp/libcudf_streaming/build}
CUDF_KAFKA_BUILD_DIR=${REPODIR}/python/cudf_kafka/build
CUDF_BUILD_DIR=${REPODIR}/python/cudf/build
DASK_CUDF_BUILD_DIR=${REPODIR}/python/dask_cudf/build
PYLIBCUDF_BUILD_DIR=${REPODIR}/python/pylibcudf/build
CUSTREAMZ_BUILD_DIR=${REPODIR}/python/custreamz/build
CUDF_JAR_JAVA_BUILD_DIR="$REPODIR/java/target"

BUILD_DIRS="${LIB_BUILD_DIR} ${CUDF_BUILD_DIR} ${DASK_CUDF_BUILD_DIR} ${KAFKA_LIB_BUILD_DIR} ${CUDF_KAFKA_BUILD_DIR} ${CUSTREAMZ_BUILD_DIR} ${CUDF_JAR_JAVA_BUILD_DIR} ${PYLIBCUDF_BUILD_DIR} ${STREAMING_LIB_BUILD_DIR}"

CUDA_VERSION="${RAPIDS_CUDA_VERSION:-$(nvcc --version | sed -E -n 's/^.*release ([0-9]+\.[0-9]+).*$/\1/p')}"
if [[ -z "$CUDA_VERSION" ]]; then
    echo "Could not determine CUDA version. Please set RAPIDS_CUDA_VERSION or make sure your \$PATH contains a valid nvcc."
    exit 1
fi

# Set defaults for vars modified by flags to this script
VERBOSE_FLAG=""
BUILD_TYPE=Release
INSTALL_TARGET=install
BUILD_BENCHMARKS=OFF
BUILD_ALL_GPU_ARCH=0
BUILD_NVTX=ON
BUILD_TESTS=OFF
BUILD_DISABLE_DEPRECATION_WARNINGS=OFF
BUILD_PER_THREAD_DEFAULT_STREAM=OFF
BUILD_REPORT_METRICS=OFF
BUILD_REPORT_INCL_CACHE_STATS=OFF
BUILD_DISABLE_LARGE_STRINGS=OFF
PYTHON_ARGS_FOR_INSTALL=(
    --no-build-isolation
    --no-deps
    --config-settings="rapidsai.disable-cuda=true"
    --config-settings="rapidsai.matrix-entry=cuda=${CUDA_VERSION};cuda_suffixed=false;use_cuda_wheels=false"
)

# Set defaults for vars that may not have been defined externally
#  FIXME: if INSTALL_PREFIX is not set, check PREFIX, then check
#         CONDA_PREFIX, but there is no fallback from there!
INSTALL_PREFIX=${INSTALL_PREFIX:=${PREFIX:=${CONDA_PREFIX}}}
PARALLEL_LEVEL=${PARALLEL_LEVEL:=$(nproc)}

function hasArg {
    (( NUMARGS != 0 )) && (echo " ${ARGS} " | grep -q " $1 ")
}

function cmakeArgs {
    # Check for multiple cmake args options
    if [[ $(echo "$ARGS" | { grep -Eo "\-\-cmake\-args" || true; } | wc -l ) -gt 1 ]]; then
        echo "Multiple --cmake-args options were provided, please provide only one: ${ARGS}"
        exit 1
    fi

    # Check for cmake args option
    if [[ -n $(echo "$ARGS" | { grep -E "\-\-cmake\-args" || true; } ) ]]; then
        # There are possible weird edge cases that may cause this regex filter to output nothing and fail silently
        # the true pipe will catch any weird edge cases that may happen and will cause the program to fall back
        # on the invalid option error
        EXTRA_CMAKE_ARGS=$(echo "$ARGS" | { grep -Eo "\-\-cmake\-args=\".+\"" || true; })
        if [[ -n ${EXTRA_CMAKE_ARGS} ]]; then
            # Remove the full  EXTRA_CMAKE_ARGS argument from list of args so that it passes validArgs function
            ARGS=${ARGS//$EXTRA_CMAKE_ARGS/}
            # Filter the full argument down to just the extra string that will be added to cmake call
            EXTRA_CMAKE_ARGS=$(echo "$EXTRA_CMAKE_ARGS" | grep -Eo "\".+\"" | sed -e 's/^"//' -e 's/"$//')
        fi
    fi
    read -ra EXTRA_CMAKE_ARGS <<< "$EXTRA_CMAKE_ARGS"
}

function buildAll {
    (( NUMARGS == 0 )) || ! (echo " ${ARGS} " | grep -q " [^-]\+ ")
}

if hasArg -h || hasArg --h || hasArg --help; then
    echo "${HELP}"
    exit 0
fi

# Check for valid usage
if (( NUMARGS != 0 )); then
    # Check for cmake args
    cmakeArgs
    for a in ${ARGS}; do
    if ! (echo " ${VALIDARGS} " | grep -q " ${a} "); then
        echo "Invalid option or formatting, check --help: ${a}"
        exit 1
    fi
    done
fi

# Process flags
if hasArg -v; then
    VERBOSE_FLAG="-v"
fi
if hasArg -g; then
    BUILD_TYPE=Debug
fi
if hasArg -n; then
    INSTALL_TARGET=""
    LIBCUDF_BUILD_DIR=${LIB_BUILD_DIR}
fi
if hasArg --allgpuarch; then
    BUILD_ALL_GPU_ARCH=1
fi
if hasArg benchmarks; then
    BUILD_BENCHMARKS=ON
fi
if hasArg tests; then
    BUILD_TESTS=ON
fi
if hasArg --disable_nvtx; then
    BUILD_NVTX="OFF"
fi
if hasArg --hide_depr_warn; then
    BUILD_DISABLE_DEPRECATION_WARNINGS=ON
fi
if hasArg --ptds; then
    BUILD_PER_THREAD_DEFAULT_STREAM=ON
fi
if hasArg --build_metrics; then
    BUILD_REPORT_METRICS=ON
fi

if hasArg --incl_cache_stats; then
    BUILD_REPORT_INCL_CACHE_STATS=ON
fi
if hasArg --pydevelop; then
    PYTHON_ARGS_FOR_INSTALL+=("-e")
fi

if hasArg --disable_large_strings; then
    BUILD_DISABLE_LARGE_STRINGS="ON"
fi

# If clean given, run it prior to any other steps
if hasArg clean; then
    # If the dirs to clean are mounted dirs in a container, the
    # contents should be removed but the mounted dirs will remain.
    # The find removes all contents but leaves the dirs, the rmdir
    # attempts to remove the dirs but can fail safely.
    for bd in ${BUILD_DIRS}; do
    if [ -d "${bd}" ]; then
        find "${bd}" -mindepth 1 -delete
        rmdir "${bd}" || true
    fi
    done

    # Cleaning up python artifacts
    find "${REPODIR}"/python/ | grep -E "(__pycache__|\.pyc|\.pyo|\.so|\_skbuild$)"  | xargs rm -rf

fi


################################################################################
# Configure, build, and install libcudf

if buildAll || hasArg libcudf || hasArg libcudf_streaming || hasArg pylibcudf || hasArg cudf ; then
    if (( BUILD_ALL_GPU_ARCH == 0 )); then
        CUDF_CMAKE_CUDA_ARCHITECTURES="${CUDF_CMAKE_CUDA_ARCHITECTURES:-NATIVE}"
        if [[ "$CUDF_CMAKE_CUDA_ARCHITECTURES" == "NATIVE" ]]; then
            echo "Building for the architecture of the GPU in the system..."
        else
            echo "Building for the GPU architecture(s) $CUDF_CMAKE_CUDA_ARCHITECTURES ..."
        fi
    else
        CUDF_CMAKE_CUDA_ARCHITECTURES="RAPIDS"
        echo "Building for *ALL* supported GPU architectures..."
    fi
fi

if buildAll || hasArg libcudf; then
    # get the current count before the compile starts
    if [[ "$BUILD_REPORT_INCL_CACHE_STATS" == "ON" && -x "$(command -v sccache)" ]]; then
        # zero the sccache statistics
        sccache --zero-stats
    fi

    cmake -S "$REPODIR"/cpp -B "${LIB_BUILD_DIR}" \
          -DCMAKE_INSTALL_PREFIX="${INSTALL_PREFIX}" \
          -DCMAKE_CUDA_ARCHITECTURES="${CUDF_CMAKE_CUDA_ARCHITECTURES}" \
          -DUSE_NVTX=${BUILD_NVTX} \
          -DBUILD_TESTS=${BUILD_TESTS} \
          -DBUILD_BENCHMARKS=${BUILD_BENCHMARKS} \
          -DDISABLE_DEPRECATION_WARNINGS=${BUILD_DISABLE_DEPRECATION_WARNINGS} \
          -DCUDF_USE_PER_THREAD_DEFAULT_STREAM=${BUILD_PER_THREAD_DEFAULT_STREAM} \
          -DCUDF_LARGE_STRINGS_DISABLED=${BUILD_DISABLE_LARGE_STRINGS} \
          -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
          "${EXTRA_CMAKE_ARGS[@]}"

    cd "${LIB_BUILD_DIR}"

    compile_start=$(date +%s)
    cmake --build . -j"${PARALLEL_LEVEL}" ${VERBOSE_FLAG}
    compile_end=$(date +%s)
    compile_total=$(( compile_end - compile_start ))

    # Record build times
    if [[ "$BUILD_REPORT_METRICS" == "ON" && -f "${LIB_BUILD_DIR}/.ninja_log" ]]; then
        echo "Formatting build metrics"
        MSG=""
        # get some sccache stats after the compile
        if [[ "$BUILD_REPORT_INCL_CACHE_STATS" == "ON" && -x "$(command -v sccache)" ]]; then
           COMPILE_REQUESTS=$(sccache -s | grep "Compile requests \+ [0-9]\+$" | awk '{ print $NF }')
           CACHE_HITS=$(sccache -s | grep "Cache hits \+ [0-9]\+$" | awk '{ print $NF }')
           HIT_RATE=$(echo - | awk "{printf \"%.2f\n\", $CACHE_HITS / $COMPILE_REQUESTS * 100}")
           MSG="${MSG}<br/>cache hit rate ${HIT_RATE} %"
        fi
        MSG="${MSG}<br/>parallel setting: $PARALLEL_LEVEL"
        MSG="${MSG}<br/>parallel build time: $compile_total seconds"
        if [[ -f "${LIB_BUILD_DIR}/libcudf.so" ]]; then
           LIBCUDF_FS=$(find "${LIB_BUILD_DIR}" -name libcudf.so -printf '%s')
           MSG="${MSG}<br/>libcudf.so size: $LIBCUDF_FS"
        fi
        BMR_DIR=${RAPIDS_ARTIFACTS_DIR:-"${LIB_BUILD_DIR}"}
        echo "Metrics output dir: [$BMR_DIR]"
        mkdir -p "${BMR_DIR}"
        MSG_OUTFILE="$(mktemp)"
        echo "$MSG" > "${MSG_OUTFILE}"
        python "${REPODIR}/cpp/scripts/sort_ninja_log.py" "${LIB_BUILD_DIR}/.ninja_log" --fmt html --msg "${MSG_OUTFILE}" > "${BMR_DIR}/ninja_log.html"
        cp "${LIB_BUILD_DIR}/.ninja_log" "${BMR_DIR}/ninja.log"
    fi

    if [[ ${INSTALL_TARGET} != "" ]]; then
        cmake --build . -j"${PARALLEL_LEVEL}" --target install ${VERBOSE_FLAG}
    fi
fi

# If `RAPIDS_PY_VERSION` is set, use that as the lower-bound for the stable ABI CPython version
# This is only applied to pylibcudf, cudf, and cudf_kafka, not other Python packages
PY_API_ARGS=()
if [ -n "${RAPIDS_PY_VERSION:-}" ]; then
    RAPIDS_PY_API="cp${RAPIDS_PY_VERSION//./}"
    PY_API_ARGS+=("--config-settings" "skbuild.wheel.py-api=${RAPIDS_PY_API}")
fi

# Build and install the pylibcudf Python package
if buildAll || hasArg pylibcudf; then

    cd "${REPODIR}/python/pylibcudf"
    SKBUILD_CMAKE_ARGS="-DCMAKE_PREFIX_PATH=${INSTALL_PREFIX};-DCMAKE_LIBRARY_PATH=${LIBCUDF_BUILD_DIR};-DCMAKE_CUDA_ARCHITECTURES=${CUDF_CMAKE_CUDA_ARCHITECTURES};${EXTRA_CMAKE_ARGS[*]}" \
        python -m pip install \
            "${PYTHON_ARGS_FOR_INSTALL[@]}" \
            "${PY_API_ARGS[@]}" \
            .
fi

# Build and install the cudf Python package
if buildAll || hasArg cudf; then

    cd "${REPODIR}/python/cudf"
    SKBUILD_CMAKE_ARGS="-DCMAKE_PREFIX_PATH=${INSTALL_PREFIX};-DCMAKE_LIBRARY_PATH=${LIBCUDF_BUILD_DIR};-DCMAKE_CUDA_ARCHITECTURES=${CUDF_CMAKE_CUDA_ARCHITECTURES};${EXTRA_CMAKE_ARGS[*]}" \
        python -m pip install \
            "${PYTHON_ARGS_FOR_INSTALL[@]}" \
            "${PY_API_ARGS[@]}" \
            .
fi

# Build libcudf_streaming library
if buildAll || hasArg libcudf_streaming; then
    cmake -S "$REPODIR/cpp/libcudf_streaming" -B "${STREAMING_LIB_BUILD_DIR}" \
          -DCMAKE_INSTALL_PREFIX="${INSTALL_PREFIX}" \
          -DCMAKE_CUDA_ARCHITECTURES="${CUDF_CMAKE_CUDA_ARCHITECTURES}" \
          -DBUILD_TESTS=${BUILD_TESTS} \
          -DBUILD_BENCHMARKS=${BUILD_BENCHMARKS} \
          -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
          "${EXTRA_CMAKE_ARGS[@]}"


    cd "${STREAMING_LIB_BUILD_DIR}"
    cmake --build . -j"${PARALLEL_LEVEL}" ${VERBOSE_FLAG}

    if [[ ${INSTALL_TARGET} != "" ]]; then
        cmake --build . -j"${PARALLEL_LEVEL}" --target install ${VERBOSE_FLAG}
    fi
fi

# build cudf_streaming Python package
if buildAll || hasArg cudf_streaming; then
    cd "${REPODIR}/python/cudf_streaming"
    SKBUILD_CMAKE_ARGS="-DCMAKE_PREFIX_PATH=${INSTALL_PREFIX};-DCMAKE_LIBRARY_PATH=${STREAMING_LIB_BUILD_DIR};${EXTRA_CMAKE_ARGS[*]}" \
        python -m pip install \
            "${PYTHON_ARGS_FOR_INSTALL[@]}" \
            "${PY_API_ARGS[@]}" \
            .
fi

# Build and install the cudf_polars Python package
if buildAll || hasArg cudf_polars; then

    cd "${REPODIR}/python/cudf_polars"
    python -m pip install "${PYTHON_ARGS_FOR_INSTALL[@]}" .
fi

# Build and install the dask_cudf Python package
if buildAll || hasArg dask_cudf; then

    cd "${REPODIR}/python/dask_cudf"
    python -m pip install "${PYTHON_ARGS_FOR_INSTALL[@]}" .
fi

# Build libcudf_kafka library
if hasArg libcudf_kafka; then
    cmake -S "$REPODIR/cpp/libcudf_kafka" -B "${KAFKA_LIB_BUILD_DIR}" \
          -DCMAKE_INSTALL_PREFIX="${INSTALL_PREFIX}" \
          -DBUILD_TESTS=${BUILD_TESTS} \
          -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
          "${EXTRA_CMAKE_ARGS[@]}"


    cd "${KAFKA_LIB_BUILD_DIR}"
    cmake --build . -j"${PARALLEL_LEVEL}" ${VERBOSE_FLAG}

    if [[ ${INSTALL_TARGET} != "" ]]; then
        cmake --build . -j"${PARALLEL_LEVEL}" --target install ${VERBOSE_FLAG}
    fi
fi

# build cudf_kafka Python package
if hasArg cudf_kafka; then
    cd "${REPODIR}/python/cudf_kafka"
    # shellcheck disable=2034
    SKBUILD_CMAKE_ARGS="-DCMAKE_PREFIX_PATH=${INSTALL_PREFIX};-DCMAKE_LIBRARY_PATH=${LIBCUDF_BUILD_DIR};${EXTRA_CMAKE_ARGS[*]}"
        python -m pip install \
            "${PYTHON_ARGS_FOR_INSTALL[@]}" \
            "${PY_API_ARGS[@]}" \
            .
fi

# build custreamz Python package
if hasArg custreamz; then
    cd "${REPODIR}/python/custreamz"
    python -m pip install "${PYTHON_ARGS_FOR_INSTALL[@]}" .
fi
