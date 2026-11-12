// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/strings/utilities.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/traits.hpp>

#include <rmm/device_buffer.hpp>

#include <memory>

namespace cudf::detail::partition_identity {
inline bool is_uniform(column_view const& input)
{
  bool const flat = is_fixed_width(input.type()) || input.type().id() == type_id::STRING;
  return flat && (input.null_count() == input.size() ||
                  (input.type().id() == type_id::STRING && input.null_count() == 0 &&
                   input.head<char>() == nullptr));
}

inline std::unique_ptr<column> uniform_column(column_view const& input,
                                              cuda::stream_ref stream,
                                              rmm::device_async_resource_ref mr)
{
  auto const rows = input.size();
  if (input.type().id() != type_id::STRING)
    return make_fixed_width_column(input.type(), rows, mask_state::ALL_NULL, stream, mr);
  auto offsets =
    make_numeric_column(data_type{type_id::INT32}, rows + 1, mask_state::UNALLOCATED, stream, mr);
  CUDF_CUDA_TRY(cudaMemsetAsync(offsets->mutable_view().data<size_type>(),
                                0,
                                (static_cast<std::size_t>(rows) + 1) * sizeof(size_type),
                                stream.get()));
  return make_strings_column(
    rows,
    std::move(offsets),
    rmm::device_buffer(0, stream, mr),
    input.null_count(),
    cudf::detail::create_null_mask(
      rows, input.null_count() ? mask_state::ALL_NULL : mask_state::UNALLOCATED, stream, mr));
}

inline std::unique_ptr<table> copy(table_view const& input,
                                   cuda::stream_ref stream,
                                   rmm::device_async_resource_ref mr)
{
  return std::make_unique<table>(input, stream, mr);
}
}  // namespace cudf::detail::partition_identity
