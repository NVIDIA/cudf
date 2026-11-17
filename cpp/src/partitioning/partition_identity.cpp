// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "partition_identity.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/traits.hpp>

namespace cudf::detail::partition_identity {

bool is_uniform(column_view const& input)
{
  bool const flat = is_fixed_width(input.type()) || input.type().id() == type_id::STRING;
  return flat && input.null_count() == input.size();
}

std::unique_ptr<column> uniform_column(column_view const& input,
                                       cuda::stream_ref stream,
                                       cudf::memory_resources mr)
{
  auto const scalar = make_empty_scalar_like(input, stream, mr.as_temporary());
  return make_column_from_scalar(*scalar, input.size(), stream, mr);
}

std::unique_ptr<table> copy(table_view const& input,
                            cuda::stream_ref stream,
                            cudf::memory_resources mr)
{
  return std::make_unique<table>(input, stream, mr);
}

}  // namespace cudf::detail::partition_identity
