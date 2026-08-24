#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Toolchain for Java builds inside a RAPIDS ci-wheel container.
#
# Installs/configures JDK 8 + 17, ninja, cmake, Boost.Filesystem/System static
# archives, the RAPIDS GCC toolset, and sccache. Safe to source repeatedly
# (JAVA_ENV_READY short-circuits after the first load).

TOOLSET_VERSION="${TOOLSET_VERSION:-14}"
BOOST_VERSION="${BOOST_VERSION:-1.79.0}"
BOOST_PREFIX="${BOOST_PREFIX:-/usr/local}"

if [[ -n "${JAVA_ENV_READY:-}" ]]; then
  return 0
fi

if ! type rapids-logger >/dev/null 2>&1; then
  rapids-logger() {
    echo ">>>> $*" >&2
  }
fi

# JDK 8 for javac; JDK 17 for the -Pjavadoc-jdk17 Maven profile.
dnf install -y maven java-1.8.0-openjdk-devel java-17-openjdk-devel

# ci-wheel images do not ship ninja or a modern cmake. Install both from PyPI.
if command -v rapids-pip-retry >/dev/null 2>&1; then
  rapids-pip-retry install cmake ninja
else
  pip install cmake ninja
fi

# Refresh pyenv shims so the cmake/ninja we just installed are on PATH.
if command -v pyenv >/dev/null 2>&1; then
  pyenv rehash || true
fi

# Static libcudf needs Boost.Filesystem / Boost.System. ci-wheel does not ship
# those static archives, so build them from source into BOOST_PREFIX when missing.
has_boost_static_lib() {
  local name=$1
  [[ -f "${BOOST_PREFIX}/lib/${name}" || -f "${BOOST_PREFIX}/lib64/${name}" ]]
}

if ! has_boost_static_lib libboost_filesystem.a || ! has_boost_static_lib libboost_system.a; then
  BOOST_DIR="boost_${BOOST_VERSION//./_}"
  wget -q -O /tmp/boost.tgz \
    "https://archives.boost.io/release/${BOOST_VERSION}/source/${BOOST_DIR}.tar.gz"
  tar -xzf /tmp/boost.tgz -C /tmp
  (
    if ! cd "/tmp/${BOOST_DIR}"; then
      exit 1
    fi
    ./bootstrap.sh --prefix="${BOOST_PREFIX}"
    ./b2 install --prefix="${BOOST_PREFIX}" --with-filesystem --with-system -j"$(nproc)"
  )
  rm -rf "/tmp/${BOOST_DIR}" /tmp/boost.tgz
fi

if [[ -z "${CUDACXX:-}" ]]; then
  if [[ -x /usr/local/cuda/bin/nvcc ]]; then
    export CUDACXX=/usr/local/cuda/bin/nvcc
  elif command -v nvcc >/dev/null 2>&1; then
    CUDACXX="$(command -v nvcc)"
    export CUDACXX
  fi
fi

export JAVA_HOME="${JAVA_HOME:-/usr/lib/jvm/java-1.8.0-openjdk}"
if [[ -z "${JDK17_HOME:-}" ]]; then
  if [[ -d /usr/lib/jvm/java-17-openjdk ]]; then
    export JDK17_HOME=/usr/lib/jvm/java-17-openjdk
  else
    JDK17_HOME="$(ls -d /usr/lib/jvm/java-17-openjdk* 2>/dev/null | head -1)"
    export JDK17_HOME
  fi
fi

export LIBCUDF_KERNEL_CACHE_PATH="${LIBCUDF_KERNEL_CACHE_PATH:-/tmp/rapids-kernel-cache}"
mkdir -p "${LIBCUDF_KERNEL_CACHE_PATH}"
export CMAKE_GENERATOR="${CMAKE_GENERATOR:-Ninja}"
export PARALLEL_LEVEL="${PARALLEL_LEVEL:-$(nproc)}"

# Prefer gcc-toolset over Rocky's default system compiler.
GCC_TOOLSET_ROOT="/opt/rh/gcc-toolset-${TOOLSET_VERSION}/root"
export CC="${CC:-${GCC_TOOLSET_ROOT}/usr/bin/gcc}"
export CXX="${CXX:-${GCC_TOOLSET_ROOT}/usr/bin/g++}"
export CMAKE_CUDA_HOST_COMPILER="${CMAKE_CUDA_HOST_COMPILER:-${CC}}"

if command -v rapids-configure-sccache >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source rapids-configure-sccache || true
fi

if command -v sccache >/dev/null 2>&1; then
  export CMAKE_C_COMPILER_LAUNCHER="${CMAKE_C_COMPILER_LAUNCHER:-sccache}"
  export CMAKE_CXX_COMPILER_LAUNCHER="${CMAKE_CXX_COMPILER_LAUNCHER:-sccache}"
  export CMAKE_CUDA_COMPILER_LAUNCHER="${CMAKE_CUDA_COMPILER_LAUNCHER:-sccache}"
  # Restart so the next compile picks up the launcher env we just set.
  sccache --stop-server 2>/dev/null || true
fi

# Rocky SCL wrapper: child build steps run under gcc-toolset-${TOOLSET_VERSION}.
cudf_java_scl() {
  scl enable "gcc-toolset-${TOOLSET_VERSION}" -- "$@"
}
export -f cudf_java_scl
export TOOLSET_VERSION BOOST_PREFIX
export JAVA_ENV_READY=1
