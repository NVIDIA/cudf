/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column_device_view.cuh>
#include <cudf/detail/concatenate.hpp>
#include <cudf/detail/cuco_helpers.hpp>
#include <cudf/detail/gather.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/row_operator/equality.cuh>
#include <cudf/detail/row_operator/hashing.cuh>
#include <cudf/dictionary/detail/update_keys.hpp>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/dictionary/update_keys.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>
#include <rmm/mr/polymorphic_allocator.hpp>

#include <cuco/static_set.cuh>
#include <cuda/iterator>
#include <cuda/std/iterator>

#include <algorithm>
#include <iterator>
#include <vector>

namespace cudf {
namespace dictionary {
namespace detail {
namespace {

struct unique_keys_dispatch_fn {
  template <typename T>
  std::unique_ptr<cudf::column> operator()(cudf::column_view const& all_keys,
                                           rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr)
    requires(cudf::is_dictionary_key<T>())
  {
    using probe_t = cuco::linear_probing<
      1,
      cudf::detail::row::hash::device_row_hasher<cudf::hashing::detail::default_hash,
                                                 cudf::nullate::DYNAMIC>>;

    auto const has_nulls  = nullate::DYNAMIC{false};
    auto const keys_tv    = table_view({all_keys});
    auto const row_hash   = cudf::detail::row::hash::row_hasher(keys_tv, stream);
    auto const row_equal  = cudf::detail::row::equality::self_comparator(keys_tv, stream);
    auto const comparator = cudf::detail::row::equality::nan_equal_physical_equality_comparator{};
    auto const d_equal    = row_equal.equal_to<false>(has_nulls, null_equality::EQUAL, comparator);
    auto const empty_key  = cuco::empty_key{cudf::detail::CUDF_SIZE_TYPE_SENTINEL};
    auto probe            = probe_t{row_hash.device_hasher(has_nulls)};
    auto allocator        = rmm::mr::polymorphic_allocator<char>{};
    auto set              = cuco::static_set{
      all_keys.size(), 0.5, empty_key, d_equal, probe, {}, {}, allocator, stream.value()};

    // use a static_set to find the unique elements of all_keys
    auto const iter = cuda::counting_iterator<cudf::size_type>{0};
    set.insert_async(iter, iter + all_keys.size(), stream.value());

    // retrieve the indices of all the unique keys
    auto keys_indices = rmm::device_uvector<size_type>(all_keys.size(), stream);
    auto keys_end     = set.retrieve_all(keys_indices.begin(), stream.value());
    keys_indices.resize(cuda::std::distance(keys_indices.begin(), keys_end), stream);

    // gather the unique keys using the keys_indices
    auto const oob_policy   = cudf::out_of_bounds_policy::DONT_CHECK;
    auto const index_policy = cudf::negative_index_policy::NOT_ALLOWED;
    auto keys_column =
      std::move(cudf::detail::gather(keys_tv, keys_indices, oob_policy, index_policy, stream, mr)
                  ->release()
                  .front());
    return keys_column;
  }

  template <typename T>
  std::unique_ptr<cudf::column> operator()(cudf::column_view const&,
                                           rmm::cuda_stream_view,
                                           rmm::device_async_resource_ref)
    requires(not cudf::is_dictionary_key<T>())
  {
    CUDF_UNREACHABLE("invalid dictionary key type");
  }
};
}  // namespace

std::vector<std::unique_ptr<column>> match_dictionaries(
  std::span<dictionary_column_view const> input,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(not input.empty(), "expect at least one dictionary", std::invalid_argument);

  auto temp_mr = cudf::get_current_device_resource_ref();
  std::vector<cudf::column_view> keys(input.size());
  std::transform(input.begin(), input.end(), keys.begin(), [](auto& col) { return col.keys(); });
  auto all_keys = cudf::detail::concatenate(keys, stream, temp_mr);

  auto new_keys = cudf::type_dispatcher(
    keys.front().type(), unique_keys_dispatch_fn{}, all_keys->view(), stream, temp_mr);
  auto keys_view = new_keys->view();

  std::vector<std::unique_ptr<column>> result(input.size());
  std::transform(input.begin(), input.end(), result.begin(), [keys_view, mr, stream](auto& col) {
    return set_keys(col, keys_view, stream, mr);
  });
  return result;
}

std::pair<std::vector<std::unique_ptr<column>>, std::vector<table_view>> match_dictionaries(
  std::vector<table_view> tables, rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr)
{
  // Make a copy of all the column views from each table_view
  std::vector<std::vector<column_view>> updated_columns;
  std::transform(tables.begin(), tables.end(), std::back_inserter(updated_columns), [](auto& t) {
    return std::vector<column_view>(t.begin(), t.end());
  });

  // Each column in a table must match in type.
  // Once a dictionary column is found, all the corresponding column_views in the
  // other table_views are matched. The matched column_views then replace the originals.
  std::vector<std::unique_ptr<column>> dictionary_columns;
  auto first_table = tables.front();
  for (size_type col_idx = 0; col_idx < first_table.num_columns(); ++col_idx) {
    auto col = first_table.column(col_idx);
    if (col.type().id() == type_id::DICTIONARY32) {
      std::vector<dictionary_column_view> dict_views;  // hold all column_views at col_idx
      std::transform(
        tables.begin(), tables.end(), std::back_inserter(dict_views), [col_idx](auto& t) {
          return dictionary_column_view(t.column(col_idx));
        });
      // now match the keys in these dictionary columns
      auto dict_cols = dictionary::detail::match_dictionaries(dict_views, stream, mr);
      // replace the updated_columns vector entries for the set of columns at col_idx
      auto dict_col_idx = 0;
      for (auto& v : updated_columns)
        v[col_idx] = dict_cols[dict_col_idx++]->view();
      // move the updated dictionary columns into the main output vector
      std::move(dict_cols.begin(), dict_cols.end(), std::back_inserter(dictionary_columns));
    }
  }
  // All the new column_views are in now updated_columns.

  // Rebuild the table_views from the column_views.
  std::vector<table_view> updated_tables;
  std::transform(updated_columns.begin(),
                 updated_columns.end(),
                 std::back_inserter(updated_tables),
                 [](auto& v) { return table_view{v}; });

  // Return the new dictionary columns and table_views
  return {std::move(dictionary_columns), std::move(updated_tables)};
}

}  // namespace detail

// external API

std::vector<std::unique_ptr<column>> match_dictionaries(
  std::span<dictionary_column_view const> input,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::match_dictionaries(input, stream, mr);
}

}  // namespace dictionary
}  // namespace cudf
