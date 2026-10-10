/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <vector>

namespace cudf::hashing::detail::jit {

/**
 * @brief Canonical host-side schema used to specialize row hashing kernels.
 *
 * Each entry in `columns` is the sequence of types visited by the row hasher for one column in the
 * input table. LIST and STRUCT entries are followed by their selected child type, while a
 * DICTIONARY32 entry is followed by its keys type.
 */
struct row_hash_schema {
  std::vector<std::vector<type_id>> columns;

  // Index storage type for the dictionary in each column's type path, or EMPTY.
  // Kept separate from the logical types so dispatch can specialize index loads.
  std::vector<type_id> dictionary_index_types;
};

/** @brief Builds specialization metadata only when a JIT row operator is requested. */
CUDF_EXPORT row_hash_schema make_row_hash_schema(table_view const& table);

}  // namespace cudf::hashing::detail::jit
