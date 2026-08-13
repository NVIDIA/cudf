/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/binary/binary_column_factories.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/utilities/error.hpp>

#include <vector>

namespace cudf {

std::unique_ptr<column> make_empty_binary_column()
{
  return make_empty_column(data_type{type_id::BINARY});
}

std::unique_ptr<column> make_binary_column(size_type num_rows,
                                           std::unique_ptr<column> offsets_column,
                                           rmm::device_buffer&& bytes_buffer,
                                           size_type null_count,
                                           rmm::device_buffer&& null_mask)
{
  CUDF_EXPECTS(num_rows >= 0, "Number of binary rows cannot be negative");
  CUDF_EXPECTS(offsets_column != nullptr, "Offsets column must not be null");
  CUDF_EXPECTS(offsets_column->type().id() == type_id::INT32 or
                 offsets_column->type().id() == type_id::INT64,
               "Binary offsets must have type INT32 or INT64");
  CUDF_EXPECTS(offsets_column->null_count() == 0, "Binary offsets must not contain nulls");
  CUDF_EXPECTS(null_count >= 0 and null_count <= num_rows, "Invalid binary null count");
  if (null_count > 0) {
    CUDF_EXPECTS(null_mask.size() > 0, "Binary column with nulls must be nullable");
  }

  if (num_rows == 0) {
    CUDF_EXPECTS(offsets_column->size() == 0 or offsets_column->size() == 1,
                 "Empty binary offsets must contain zero or one element");
    CUDF_EXPECTS(bytes_buffer.size() == 0, "Empty binary column cannot contain payload bytes");
    return make_empty_binary_column();
  }

  CUDF_EXPECTS(offsets_column->size() == num_rows + 1,
               "Binary offsets size must equal the row count plus one");

  std::vector<std::unique_ptr<column>> children;
  children.emplace_back(std::move(offsets_column));
  return std::make_unique<column>(data_type{type_id::BINARY},
                                  num_rows,
                                  std::move(bytes_buffer),
                                  std::move(null_mask),
                                  null_count,
                                  std::move(children));
}

}  // namespace cudf
