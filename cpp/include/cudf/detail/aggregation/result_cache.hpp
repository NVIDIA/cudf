/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/aggregation.hpp>
#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/hashing/detail/hashing.hpp>

#include <cstddef>
#include <functional>
#include <memory>
#include <unordered_map>
#include <utility>

namespace cudf {
namespace detail {
struct pair_column_aggregation_equal_to {
  bool operator()(std::pair<column_view, aggregation const&> const& lhs,
                  std::pair<column_view, aggregation const&> const& rhs) const
  {
    return is_shallow_equivalent(lhs.first, rhs.first) and lhs.second.is_equal(rhs.second);
  }
};

struct pair_column_aggregation_hash {
  size_t operator()(std::pair<column_view, aggregation const&> const& key) const
  {
    return cudf::hashing::detail::hash_combine(shallow_hash(key.first), key.second.do_hash());
  }
};

class result_cache {
 public:
  result_cache()                                     = delete;
  ~result_cache()                                    = default;
  result_cache(result_cache const&)                  = delete;
  result_cache& operator=(result_cache const& other) = delete;

  result_cache(size_t num_columns) : _cache(num_columns) {}

  [[nodiscard]] bool has_result(column_view const& input, aggregation const& agg) const;

  void add_result(column_view const& input, aggregation const& agg, std::unique_ptr<column>&& col);

  [[nodiscard]] column_view get_result(column_view const& input, aggregation const& agg) const;

  std::unique_ptr<column> release_result(column_view const& input, aggregation const& agg);

 private:
  std::unordered_map<std::pair<column_view, std::reference_wrapper<aggregation const>>,
                     std::pair<std::unique_ptr<aggregation>, std::unique_ptr<column>>,
                     pair_column_aggregation_hash,
                     pair_column_aggregation_equal_to>
    _cache;
};

}  // namespace detail
}  // namespace cudf
