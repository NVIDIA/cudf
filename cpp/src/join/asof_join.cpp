/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "asof_join.hpp"

#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_checks.hpp>

#include <memory>
#include <stdexcept>
#include <string>

namespace cudf::experimental::detail {
namespace {

bool is_supported_on_type(data_type type)
{
  return cudf::is_integral_not_bool(type) || cudf::is_timestamp(type) || cudf::is_duration(type);
}

bool is_supported_by_type(data_type type)
{
  return type.id() == type_id::STRING || cudf::is_fixed_width(type);
}

void validate_by_types(table_view const& by)
{
  for (auto const& column : by) {
    CUDF_EXPECTS(is_supported_by_type(column.type()),
                 "As-of join grouping keys must be strings or fixed-width scalar columns",
                 cudf::data_type_error);
  }
}

void validate_key_sizes(table_view const& by, column_view const& on, char const* side)
{
  CUDF_EXPECTS(by.num_columns() == 0 || by.num_rows() == on.size(),
               std::string{"As-of join "} + side +
                 " grouping and ordered keys must have the same number of rows",
               std::invalid_argument);
}

}  // namespace

asof_join_impl::asof_join_impl(table_view const& right_by,
                               column_view const& right_on,
                               cuda::stream_ref stream,
                               cudf::memory_resources mr)
  : _right_by{right_by},
    _right_on{right_on},
    _right_group_offsets{stream, mr.get_output_mr(), 0, cuda::no_init}
{
  cudf::scoped_range range{"asof_join::asof_join"};
  validate_key_sizes(right_by, right_on, "right");
  CUDF_EXPECTS(is_supported_on_type(right_on.type()),
               "As-of join ordered keys must be integer, timestamp, or duration columns",
               cudf::data_type_error);
  validate_by_types(right_by);

  build_groups(stream, mr);
}

cuda::device_buffer<size_type> asof_join_impl::join(table_view const& left_by,
                                                    column_view const& left_on,
                                                    asof_join_strategy strategy,
                                                    asof_join_exact_matches exact_matches,
                                                    cuda::stream_ref stream,
                                                    cudf::memory_resources mr) const
{
  cudf::scoped_range range{"asof_join::join"};
  CUDF_EXPECTS(strategy == asof_join_strategy::BACKWARD,
               "Only backward as-of joins are currently supported");
  CUDF_EXPECTS(exact_matches == asof_join_exact_matches::YES,
               "As-of joins without exact matches are currently unsupported");
  validate_key_sizes(left_by, left_on, "left");
  CUDF_EXPECTS(left_by.num_columns() == _right_by.num_columns(),
               "As-of join inputs must have the same number of grouping keys",
               std::invalid_argument);
  CUDF_EXPECTS(cudf::have_same_types(left_by, _right_by),
               "As-of join grouping-key types must match",
               cudf::data_type_error);
  CUDF_EXPECTS(cudf::have_same_types(left_on, _right_on),
               "As-of join ordered-key types must match",
               cudf::data_type_error);

  return probe(left_by, left_on, stream, mr);
}

}  // namespace cudf::experimental::detail

namespace cudf::experimental {

asof_join::~asof_join() = default;

asof_join::asof_join(table_view const& right_by,
                     column_view const& right_on,
                     cuda::stream_ref stream,
                     cudf::memory_resources mr)
  : _impl{std::make_unique<impl_type>(right_by, right_on, stream, mr)}
{
}

cuda::device_buffer<size_type> asof_join::join(table_view const& left_by,
                                               column_view const& left_on,
                                               asof_join_strategy strategy,
                                               asof_join_exact_matches exact_matches,
                                               cuda::stream_ref stream,
                                               cudf::memory_resources mr) const
{
  return _impl->join(left_by, left_on, strategy, exact_matches, stream, mr);
}

}  // namespace cudf::experimental
