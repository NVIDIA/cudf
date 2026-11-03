/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

/**
 * @file column_child_offsets.hpp
 * @brief Constants for child column indices within compound column types
 */

#include <cudf/types.hpp>
#include <cudf/utilities/export.hpp>

namespace CUDF_EXPORT cudf {

static constexpr size_type offsets_column_index = 0;  ///< Child index of the offsets column

static constexpr size_type dictionary_indices_column_index =
  0;  ///< Child index of the dictionary indices column

static constexpr size_type dictionary_keys_column_index =
  1;  ///< Child index of the dictionary key column

}  // namespace CUDF_EXPORT cudf
