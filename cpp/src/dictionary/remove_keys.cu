/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/copy_if.cuh>
#include <cudf/detail/gather.hpp>
#include <cudf/detail/indexalator.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/search.hpp>
#include <cudf/detail/valid_if.cuh>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/dictionary/dictionary_factories.hpp>
#include <cudf/dictionary/update_keys.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/type_checks.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/iterator>
#include <thrust/fill.h>
#include <thrust/scatter.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>

namespace cudf {
namespace dictionary {
namespace detail {
namespace {

/**
 * @brief Return a new dictionary by removing identified keys from the provided dictionary.
 *
 * This is a common utility for `remove_keys` and `remove_unused_keys` detail functions.
 * It will create a new dictionary with the remaining keys and create new indices values
 * to go with these new keys.
 *
 * @tparam KeysKeeper Function bool(size_type) that takes keys position index
 *                    and returns true if that key is to be used in the output dictionary.
 * @param dictionary_column The column to use for creating the new dictionary.
 * @param keys_to_keep_fn Called to determine which keys in `dictionary_column` to keep.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 */
template <typename KeysKeeper>
std::unique_ptr<column> remove_keys_fn(dictionary_column_view const& dictionary_column,
                                       KeysKeeper keys_to_keep_fn,
                                       rmm::cuda_stream_view stream,
                                       rmm::device_async_resource_ref mr)
{
  auto const keys_view    = dictionary_column.keys();
  auto const indices_type = dictionary_column.indices().type();
  auto const max_size     = dictionary_column.size();

  // create/init indices map array
  auto map_indices = make_fixed_width_column(indices_type,
                                             keys_view.size(),
                                             mask_state::UNALLOCATED,
                                             stream,
                                             cudf::get_current_device_resource_ref());
  auto map_itr =
    cudf::detail::indexalator_factory::make_output_iterator(map_indices->mutable_view());
  // init to max to identify new nulls
  thrust::fill(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
               map_itr,
               map_itr + keys_view.size(),
               max_size);  // all valid indices are less than this value

  // build keys column and indices map
  std::unique_ptr<column> keys_column = [&] {
    // create keys positions column to identify original key positions after removing they keys
    auto keys_positions = [&] {
      auto positions = make_fixed_width_column(indices_type,
                                               keys_view.size(),
                                               cudf::mask_state::UNALLOCATED,
                                               stream,
                                               cudf::get_current_device_resource_ref());
      auto itr = cudf::detail::indexalator_factory::make_output_iterator(positions->mutable_view());
      thrust::sequence(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                       itr,
                       itr + keys_view.size());
      return positions;
    }();
    // copy the non-removed keys ( keys_to_keep_fn(idx)==true )
    auto table_keys =
      cudf::detail::copy_if(
        table_view{{keys_view, keys_positions->view()}}, keys_to_keep_fn, stream, mr)
        ->release();
    auto const filtered_view = table_keys[1]->view();
    auto filtered_itr = cudf::detail::indexalator_factory::make_input_iterator(filtered_view);
    auto positions_itr =
      cudf::detail::indexalator_factory::make_input_iterator(keys_positions->view());
    // build indices mapper
    // Example scatter([0,1,2][0,2,4][max,max,max,max,max]) => [0,max,1,max,2]
    thrust::scatter(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                    positions_itr,
                    positions_itr + filtered_view.size(),
                    filtered_itr,
                    map_itr);
    return std::move(table_keys.front());
  }();

  // create non-nullable indices view with offset applied -- this is used as a gather map
  column_view indices_view(dictionary_column.indices().type(),
                           dictionary_column.size(),
                           dictionary_column.indices().head(),
                           nullptr,
                           0,
                           dictionary_column.offset());
  // create new indices column
  // Example: gather([0,max,1,max,2],[4,0,3,1,2,2,2,4,0]) => [2,0,max,max,1,1,1,2,0]
  auto table_indices = cudf::detail::gather(table_view{{map_indices->view()}},
                                            indices_view,
                                            cudf::out_of_bounds_policy::NULLIFY,
                                            cudf::negative_index_policy::NOT_ALLOWED,
                                            stream,
                                            mr)
                         ->release();
  std::unique_ptr<column> indices_column(std::move(table_indices.front()));
  indices_column->set_null_mask(rmm::device_buffer{}, 0);

  // compute new nulls -- merge the existing nulls with the newly created ones (value<0)
  auto const offset = dictionary_column.offset();
  auto d_null_mask  = dictionary_column.null_mask();
  auto indices_itr = cudf::detail::indexalator_factory::make_input_iterator(indices_column->view());
  auto new_nulls   = cudf::detail::valid_if(
    cuda::counting_iterator<size_type>{0},
    cuda::counting_iterator<size_type>{dictionary_column.size()},
    [offset, d_null_mask, indices_itr, max_size] __device__(size_type idx) {
      if (d_null_mask && !bit_is_set(d_null_mask, idx + offset)) return false;
      return (indices_itr[idx] < max_size);  // new nulls have max values
    },
    stream,
    mr);
  rmm::device_buffer new_null_mask =
    (new_nulls.second > 0) ? std::move(new_nulls.first) : rmm::device_buffer{0, stream, mr};

  // create column with keys_column and indices_column
  return make_dictionary_column(
    std::move(keys_column), std::move(indices_column), std::move(new_null_mask), new_nulls.second);
}

}  // namespace

std::unique_ptr<column> remove_keys(dictionary_column_view const& dictionary_column,
                                    column_view const& keys_to_remove,
                                    rmm::cuda_stream_view stream,
                                    rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(!keys_to_remove.has_nulls(), "keys_to_remove must not have nulls");
  auto const keys_view = dictionary_column.keys();
  CUDF_EXPECTS(cudf::have_same_types(keys_view, keys_to_remove),
               "keys types must match",
               cudf::data_type_error);

  // locate keys to remove by searching the keys column
  auto const matches = cudf::detail::contains(keys_to_remove, keys_view, stream, mr);
  auto d_matches     = matches->view().data<bool>();
  // call common utility method to keep the keys not matched to keys_to_remove
  auto key_matcher = [d_matches] __device__(size_type idx) -> bool { return !d_matches[idx]; };
  return remove_keys_fn(dictionary_column, key_matcher, stream, mr);
}

std::unique_ptr<column> remove_unused_keys(dictionary_column_view const& dictionary_column,
                                           rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr)
{
  // locate the keys to remove
  auto const keys_size     = dictionary_column.keys_size();
  column_view indices_view = dictionary_column.get_indices_annotated();

  // search the indices values with key indices to look for any holes
  auto const matches = [&] {
    // build keys index to verify against indices values
    rmm::device_uvector<int32_t> keys_positions(keys_size, stream);
    thrust::sequence(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                     keys_positions.begin(),
                     keys_positions.end());
    // wrap the indices for comparison in contains()
    column_view keys_positions_view(
      data_type{type_id::INT32}, keys_size, keys_positions.data(), nullptr, 0);
    return cudf::detail::contains(indices_view, keys_positions_view, stream, mr);
  }();
  auto d_matches = matches->view().data<bool>();

  // call common utility method to keep the keys that match
  auto key_matcher = [d_matches] __device__(size_type idx) -> bool { return d_matches[idx]; };
  return remove_keys_fn(dictionary_column, key_matcher, stream, mr);
}

}  // namespace detail

// external APIs

std::unique_ptr<column> remove_keys(dictionary_column_view const& dictionary_column,
                                    column_view const& keys_to_remove,
                                    rmm::cuda_stream_view stream,
                                    rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::remove_keys(dictionary_column, keys_to_remove, stream, mr);
}

std::unique_ptr<column> remove_unused_keys(dictionary_column_view const& dictionary_column,
                                           rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::remove_unused_keys(dictionary_column, stream, mr);
}

}  // namespace dictionary
}  // namespace cudf
