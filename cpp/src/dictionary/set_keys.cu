/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/indexalator.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/sorting.hpp>
#include <cudf/detail/valid_if.cuh>
#include <cudf/dictionary/detail/iterator.cuh>
#include <cudf/dictionary/detail/update_keys.hpp>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/dictionary/dictionary_factories.hpp>
#include <cudf/dictionary/update_keys.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_checks.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/iterator>
#include <cuda/std/iterator>
#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>
#include <thrust/transform.h>

namespace cudf {
namespace dictionary {
namespace detail {
namespace {

template <typename T, typename Iterator>
struct create_indices_map_fn {
  cudf::column_device_view d_old_keys;
  Iterator d_begin;
  Iterator d_end;
  cudf::size_type const* d_indices;

  __device__ size_type operator()(size_type idx)
  {
    auto const key = d_old_keys.element<T>(idx);
    auto const itr = thrust::lower_bound(thrust::seq, d_begin, d_end, key);
    return (itr != d_end && key == *itr) ? d_indices[cuda::std::distance(d_begin, itr)] : -1;
  }
};

struct apply_indices_map_fn {
  cudf::column_device_view d_input;
  cudf::size_type const* d_map;

  __device__ size_type operator()(cudf::size_type idx)
  {
    if (d_input.is_null(idx)) { return -1; }
    auto const indices = d_input.child(cudf::dictionary_column_view::indices_column_index);
    auto const indices_itr =
      cudf::detail::input_indexalator(indices.head(), indices.type(), d_input.offset());
    return d_map[indices_itr[idx]];
  }
};

struct set_keys_dispatch_fn {
  template <typename T>
  std::unique_ptr<cudf::column> operator()(cudf::dictionary_column_view const& input,
                                           cudf::column_view const& new_keys,
                                           rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr)
    requires(cudf::is_dictionary_key<T>())
  {
    // compute sorted-order so the new_keys can be searched more quickly
    auto sorted_indices = cudf::detail::sorted_order(
      table_view({new_keys}), {}, {}, stream, cudf::get_current_device_resource_ref());
    auto d_sorted_indices = sorted_indices->view().data<size_type>();

    auto const old_keys   = input.keys();
    auto const d_old_keys = column_device_view::create(old_keys, stream);
    auto const d_new_keys = column_device_view::create(new_keys, stream);
    auto const keys_itr = cuda::make_permutation_iterator(d_new_keys->begin<T>(), d_sorted_indices);
    auto const iota     = cuda::counting_iterator<cudf::size_type>{0};

    // create a map from the old key indices to the new ones
    auto indices_map = rmm::device_uvector<size_type>(old_keys.size(), stream);
    create_indices_map_fn<T, decltype(keys_itr)> map_fn{
      *d_old_keys, keys_itr, keys_itr + new_keys.size(), d_sorted_indices};
    thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                      iota,
                      iota + old_keys.size(),
                      indices_map.begin(),
                      map_fn);

    // map the old indices to the new set
    auto indices_column = cudf::make_numeric_column(
      input.indices().type(), input.size(), cudf::mask_state::UNALLOCATED, stream, mr);
    auto d_new_indices =
      cudf::detail::indexalator_factory::make_output_iterator(indices_column->mutable_view());
    auto d_input = cudf::column_device_view::create(input.parent(), stream);
    apply_indices_map_fn apply_fn{*d_input, indices_map.data()};
    thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                      iota,
                      iota + input.size(),
                      d_new_indices,
                      apply_fn);

    // compute the nulls (any indices < 0)
    auto d_indices = cudf::detail::indexalator_factory::make_input_iterator(indices_column->view());
    auto [null_mask, null_count] = cudf::detail::valid_if(
      iota,
      iota + input.size(),
      [d_indices] __device__(size_type idx) { return d_indices[idx] >= 0; },
      stream,
      mr);

    auto keys_column = std::make_unique<cudf::column>(new_keys, stream, mr);
    return make_dictionary_column(
      std::move(keys_column), std::move(indices_column), std::move(null_mask), null_count);
  }

  template <typename T>
  std::unique_ptr<cudf::column> operator()(cudf::dictionary_column_view const&,
                                           cudf::column_view const&,
                                           rmm::cuda_stream_view,
                                           rmm::device_async_resource_ref)
    requires(not cudf::is_dictionary_key<T>())
  {
    CUDF_UNREACHABLE("not a valid dictionary key type");
  }
};
}  // namespace

std::unique_ptr<column> set_keys(dictionary_column_view const& input,
                                 column_view const& new_keys,
                                 rmm::cuda_stream_view stream,
                                 rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(!new_keys.has_nulls(), "keys parameter must not have nulls", std::invalid_argument);
  CUDF_EXPECTS(!new_keys.is_empty(), "keys cannot be empty", std::invalid_argument);
  CUDF_EXPECTS(
    cudf::have_same_types(input.keys(), new_keys), "keys types must match", cudf::data_type_error);

  return type_dispatcher<dispatch_storage_type>(
    new_keys.type(), set_keys_dispatch_fn{}, input, new_keys, stream, mr);
}

}  // namespace detail

// external API

std::unique_ptr<column> set_keys(dictionary_column_view const& dictionary_column,
                                 column_view const& keys,
                                 rmm::cuda_stream_view stream,
                                 rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::set_keys(dictionary_column, keys, stream, mr);
}

}  // namespace dictionary
}  // namespace cudf
