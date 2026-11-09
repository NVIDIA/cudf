# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

# cuCascade is fetched at a pinned commit and built from source as static libraries that belong to
# this extension alone: nothing of it is installed, and there is no libcucascade package to keep in
# step with libcudf. Only its io core is built, which has no libcudf dependency
# (CUCASCADE_BUILD_CUDF is OFF), so the one libcudf-facing class is the datasource subclass in this
# extension.
#
# To build against a local checkout, for example while changing cuCascade, pass
# -DCPM_cuCascade_SOURCE=/path/to/cuCascade.
function(find_and_configure_cucascade)
  # CPMAddPackage and not rapids_cpm_find: an installed cuCascade, which would carry its own
  # libcudf-facing code, must never be picked up in place of this build.
  CPMAddPackage(
    NAME
    cuCascade
    VERSION
    0.1.0
    # While NVIDIA/cuCascade#231 is open its commits are on a fork, which a plain clone can fetch (a
    # clone of NVIDIA/cuCascade does not include pull request refs). Once it is merged, use
    # https://github.com/NVIDIA/cuCascade.git and the merge commit.
    GIT_REPOSITORY
    https://github.com/Matt711/cuCascade.git
    GIT_TAG
    6aa45ad9bcb14e85ce3b06e708b675cf70e02485
    GIT_SHALLOW
    FALSE
    OPTIONS
    "CUCASCADE_BUILD_TESTS OFF"
    "CUCASCADE_BUILD_BENCHMARKS OFF"
    "CUCASCADE_BUILD_CUDF OFF"
    "CUCASCADE_BUILD_IO ON"
    "CUCASCADE_BUILD_SHARED_LIBS OFF"
    "CUCASCADE_BUILD_STATIC_LIBS ON"
    "CUCASCADE_WARNINGS_AS_ERRORS OFF"
    EXCLUDE_FROM_ALL
    ON
  )
  set(cuCascade_SOURCE_DIR
      "${cuCascade_SOURCE_DIR}"
      PARENT_SCOPE
  )
endfunction()

find_and_configure_cucascade()
