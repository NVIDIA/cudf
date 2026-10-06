/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column.hpp>
#include <cudf/detail/aggregation/result_cache.hpp>
#include <cudf/utilities/error.hpp>

#include <string>

namespace cudf {
namespace detail {

bool result_cache::has_result(column_view const& input, aggregation const& agg) const
{
  return _cache.count({input, agg});
}

void result_cache::add_result(column_view const& input,
                              aggregation const& agg,
                              std::unique_ptr<column>&& col)
{
  // We can't guarantee that agg will outlive the cache, so we need to take ownership of a copy.
  // To allow lookup by reference, make the key a reference and keep the owner in the value pair.
  auto owned_agg  = agg.clone();
  auto const& key = *owned_agg;
  // try_emplace doesn't update/insert if already present
  _cache.try_emplace({input, key}, std::move(owned_agg), std::move(col));
}

column_view result_cache::get_result(column_view const& input, aggregation const& agg) const
{
  auto result_it = _cache.find({input, agg});
  CUDF_EXPECTS(result_it != _cache.end(), "Result does not exist in cache");
  return result_it->second.second->view();
}

std::unique_ptr<column> result_cache::release_result(column_view const& input,
                                                     aggregation const& agg)
{
  auto node = _cache.extract({input, agg});
  CUDF_EXPECTS(not node.empty(), "Result does not exist in cache");
  return std::move(node.mapped().second);
}

}  // namespace detail
}  // namespace cudf
