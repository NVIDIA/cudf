/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column_device_view.cuh>
#include <cudf/detail/concatenate.hpp>
#include <cudf/detail/copy_if.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/search.hpp>
#include <cudf/dictionary/dictionary_factories.hpp>
#include <cudf/dictionary/update_keys.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/type_checks.hpp>

namespace cudf {
namespace dictionary {
namespace detail {

std::unique_ptr<column> add_keys(dictionary_column_view const& input,
                                 column_view const& new_keys,
                                 cuda::stream_ref stream,
                                 rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(!new_keys.has_nulls(), "Keys must not have nulls", std::invalid_argument);
  auto old_keys = input.keys();
  CUDF_EXPECTS(
    cudf::have_same_types(new_keys, old_keys), "Keys must be the same type", cudf::data_type_error);

  // first, find duplicate keys; returns bools where the keys intersect
  auto intersect   = cudf::detail::contains(old_keys, new_keys, stream, mr);
  auto d_intersect = column_device_view::create(intersect->view(), stream);
  // filter out duplicates from the input
  auto negate_fn = [d_intersect = d_intersect->data<bool>()] __device__(size_type idx) {
    return !d_intersect[idx];
  };
  auto nks = std::move(
    cudf::detail::copy_if(cudf::table_view({new_keys}), negate_fn, stream, mr)->release().front());
  // build new keys by concatenating new ones to the end
  auto keys_column =
    (nks->size() > 0)
      ? cudf::detail::concatenate(std::vector<column_view>{old_keys, nks->view()}, stream, mr)
      : std::make_unique<column>(old_keys, stream, mr);
  // this leaves the indices untouched so just copy them
  auto indices_column = std::make_unique<column>(input.get_indices_annotated(), stream, mr);
  indices_column->set_null_mask(rmm::device_buffer{}, 0);
  return make_dictionary_column(std::move(keys_column),
                                std::move(indices_column),
                                cudf::detail::copy_bitmask(input.parent(), stream, mr),
                                input.null_count());
}

}  // namespace detail

std::unique_ptr<column> add_keys(dictionary_column_view const& dictionary_column,
                                 column_view const& keys,
                                 cuda::stream_ref stream,
                                 rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::add_keys(dictionary_column, keys, stream, mr);
}

}  // namespace dictionary
}  // namespace cudf
