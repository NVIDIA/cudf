# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

set(cython_sources
    byte_pair_encode.pyx
    deduplicate.pyx
    edit_distance.pyx
    generate_ngrams.pyx
    jaccard.pyx
    minhash.pyx
    ngrams_tokenize.pyx
    normalize.pyx
    replace.pyx
    stemmer.pyx
    tokenize.pyx
    unicode_normalize.pyx
    wordpiece_tokenize.pyx
)

set(linked_libraries cudf::cudf)
rapids_cython_create_modules(
  CXX
  SOURCE_FILES "${cython_sources}"
  LINKED_LIBRARIES "${linked_libraries}" MODULE_PREFIX pylibcudf_nvtext_ ASSOCIATED_TARGETS cudf
)
