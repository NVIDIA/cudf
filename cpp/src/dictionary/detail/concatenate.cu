/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/concatenate.hpp>
#include <cudf/detail/cuco_helpers.hpp>
#include <cudf/detail/gather.hpp>
#include <cudf/detail/indexalator.cuh>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/row_operator/equality.cuh>
#include <cudf/detail/row_operator/hashing.cuh>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/dictionary/detail/concatenate.hpp>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/dictionary/dictionary_factories.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/type_checks.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>
#include <rmm/mr/polymorphic_allocator.hpp>

#include <cuco/static_set.cuh>
#include <cuda/functional>
#include <cuda/iterator>
#include <cuda/std/utility>
#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>
#include <thrust/gather.h>
#include <thrust/transform.h>
#include <thrust/transform_scan.h>

#include <algorithm>
#include <vector>

namespace cudf {
namespace dictionary {
namespace detail {
namespace {

/**
 * @brief Functor for inserting keys into the set and recording the resulting index.
 */
template <typename SetRef>
struct insert_keys_fn {
  SetRef set_ref;
  column_device_view d_keys;
  __device__ size_type operator()(size_type idx)
  {
    return *cuda::std::get<0>(set_ref.insert_and_find(idx));
  }
};

/**
 * @brief Keys and indices offsets values.
 *
 * The first value is the keys offsets and the second values is the indices offsets.
 * These are offsets to the beginning of each input column after concatenating.
 */
using offsets_pair = cuda::std::pair<size_type, size_type>;

/**
 * @brief Utility for calculating the offsets for the concatenated child columns
 *        of the output dictionary column.
 */
struct compute_children_offsets_fn {
  /**
   * @brief Create the utility functor.
   *
   * The columns vector is converted into vector of column_view pointers so they
   * can be used in thrust::transform_exclusive_scan without causing the
   * compiler warning/error: "host/device function calling host function".
   *
   * @param columns The input dictionary columns.
   */
  compute_children_offsets_fn(host_span<column_view const> columns) : columns_ptrs{columns.size()}
  {
    std::transform(
      columns.begin(), columns.end(), columns_ptrs.begin(), [](auto& cv) { return &cv; });
  }

  /**
   * @brief Return the first keys() of the dictionary columns.
   */
  column_view get_keys()
  {
    auto const view(*std::find_if(
      columns_ptrs.begin(), columns_ptrs.end(), [](auto pcv) { return pcv->size() > 0; }));
    return dictionary_column_view(*view).keys();
  }

  /**
   * @brief Create the offsets pair for the concatenated columns.
   *
   * Both vectors have the length of the number of input columns.
   * The sizes of each child (keys and indices) of the individual columns
   * are used to create the offsets.
   *
   * @param stream Stream used for allocating the output rmm::device_uvector.
   * @return Vector of offsets_pair objects for keys and indices.
   */
  rmm::device_uvector<offsets_pair> create_children_offsets(rmm::cuda_stream_view stream)
  {
    auto offsets = cudf::detail::make_host_vector<offsets_pair>(columns_ptrs.size(), stream);
    thrust::transform_exclusive_scan(
      thrust::host,
      columns_ptrs.begin(),
      columns_ptrs.end(),
      offsets.begin(),
      [](auto pcv) {
        dictionary_column_view view(*pcv);
        return offsets_pair{view.keys_size(), view.size()};
      },
      offsets_pair{0, 0},
      [](auto lhs, auto rhs) {
        return offsets_pair{lhs.first + rhs.first, lhs.second + rhs.second};
      });
    return cudf::detail::make_device_uvector(
      offsets, stream, cudf::get_current_device_resource_ref());
  }

 private:
  std::vector<column_view const*> columns_ptrs;  ///< pointer version of input column_view vector
};

/**
 * @brief Functor for mapping the old indices values to the new indices values
 *        based on the new keys arrangement after concatenation
 */
struct map_indices_fn {
  cuda::std::span<offsets_pair const> d_offsets;
  cuda::std::span<size_type const> d_keys_remap;
  column_device_view d_indices;

  __device__ size_type operator()(size_type idx) const
  {
    if (d_indices.is_null(idx)) { return 0; }
    auto cmp = [] __device__(auto const& lhs, auto const& rhs) { return lhs.second < rhs.second; };
    auto col_iter = thrust::upper_bound(
                      thrust::seq, d_offsets.begin(), d_offsets.end(), offsets_pair{0, idx}, cmp) -
                    1;
    auto col_idx    = cuda::std::distance(d_offsets.begin(), col_iter);
    auto key_offset = d_offsets[col_idx].first;
    return d_keys_remap[key_offset + d_indices.element<size_type>(idx)];
  }
};

}  // namespace

std::unique_ptr<column> concatenate(host_span<column_view const> columns,
                                    rmm::cuda_stream_view stream,
                                    rmm::device_async_resource_ref mr)
{
  // exception here is the same behavior as in cudf::concatenate
  CUDF_EXPECTS(not columns.empty(), "Unexpected empty list of columns to concatenate.");

  // concatenate the keys (and check the keys match)
  compute_children_offsets_fn child_offsets_fn{columns};
  auto expected_keys = child_offsets_fn.get_keys();
  std::vector<column_view> keys_views(columns.size());
  std::transform(columns.begin(), columns.end(), keys_views.begin(), [expected_keys](auto cv) {
    auto dict_view = dictionary_column_view(cv);
    // empty column may not have keys so we create an empty column_view place-holder
    if (dict_view.is_empty()) return column_view{expected_keys.type(), 0, nullptr, nullptr, 0};
    auto keys = dict_view.keys();
    CUDF_EXPECTS(cudf::have_same_types(keys, expected_keys),
                 "key types of all dictionary columns must match",
                 cudf::data_type_error);
    return keys;
  });

  // TODO: Overload function to accept multiple vectors to do the concatenate at once, with a 2D
  // kernel. The keys concatenate below and the indices concatenate further down are two separate
  // launches over the same set of input columns and could be fused into a single batched call.
  // first, concatenate all the keys
  auto all_keys =
    cudf::detail::concatenate(keys_views, stream, cudf::get_current_device_resource_ref());
  // compute the unique set of keys to better help map the new indices values
  using encode_probe_t = cuco::linear_probing<
    1,
    cudf::detail::row::hash::device_row_hasher<cudf::hashing::detail::default_hash,
                                               cudf::nullate::NO>>;
  auto const tv         = cudf::table_view({all_keys->view()});
  auto const row_hash   = cudf::detail::row::hash::row_hasher(tv, stream);
  auto const row_equal  = cudf::detail::row::equality::self_comparator(tv, stream);
  auto const comparator = cudf::detail::row::equality::nan_equal_physical_equality_comparator{};
  auto const d_equal =
    row_equal.equal_to<false>(cudf::nullate::NO{}, null_equality::EQUAL, comparator);
  auto const empty_key = cuco::empty_key{cudf::detail::CUDF_SIZE_TYPE_SENTINEL};
  auto probe           = encode_probe_t{row_hash.device_hasher(cudf::nullate::NO{})};
  auto allocator = rmm::mr::polymorphic_allocator<char>(cudf::get_current_device_resource_ref());
  auto set       = cuco::static_set{
    all_keys->size(), 0.5, empty_key, d_equal, probe, {}, {}, allocator, stream.value()};
  auto set_ref    = set.ref(cuco::insert_and_find);
  using set_ref_t = decltype(set_ref);

  auto policy = rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref());
  auto iota   = cuda::counting_iterator<size_type>{0};

  auto d_indices = rmm::device_uvector<size_type>(
    all_keys->size(), stream, cudf::get_current_device_resource_ref());
  auto d_all_keys = column_device_view::create(all_keys->view(), stream);
  thrust::transform(
    policy, iota, iota + all_keys->size(), d_indices.begin(), insert_keys_fn{set_ref, *d_all_keys});
  auto keys_indices = rmm::device_uvector<size_type>(
    all_keys->size(), stream, cudf::get_current_device_resource_ref());
  auto keys_end = set.retrieve_all(keys_indices.begin(), stream.value());
  keys_indices.resize(cuda::std::distance(keys_indices.begin(), keys_end), stream);

  // use keys_indices to retrieve the keys (gather)
  auto const oob_policy   = cudf::out_of_bounds_policy::DONT_CHECK;
  auto const index_policy = cudf::negative_index_policy::NOT_ALLOWED;
  auto keys_column =
    std::move(cudf::detail::gather(tv, keys_indices, oob_policy, index_policy, stream, mr)
                ->release()
                .front());

  // build an all_keys_remap: abs position in all_keys to new key index
  // use scatter to assign new index values: all_keys_remap[keys_indices[i]] = i
  auto all_keys_remap = rmm::device_uvector<size_type>(
    all_keys->size(), stream, cudf::get_current_device_resource_ref());
  thrust::scatter(
    policy, iota, iota + keys_indices.size(), keys_indices.begin(), all_keys_remap.begin());
  // use gather to propagate new indices values to all duplicate positions
  auto final_remap = rmm::device_uvector<size_type>(
    all_keys->size(), stream, cudf::get_current_device_resource_ref());
  thrust::gather(
    policy, d_indices.begin(), d_indices.end(), all_keys_remap.begin(), final_remap.begin());

  // next, concatenate the indices
  std::vector<column_view> indices_views(columns.size());
  std::transform(columns.begin(), columns.end(), indices_views.begin(), [](auto cv) {
    auto dict_view = dictionary_column_view(cv);
    if (dict_view.is_empty()) {
      return column_view{data_type{type_id::INT32}, 0, nullptr, nullptr, 0};
    }
    return dict_view.get_indices_annotated();  // nicely includes validity mask and view offset
  });
  auto all_indices = cudf::detail::concatenate(indices_views, stream, mr);

  // remap the input indices values to the new indices for the new keys order
  auto indices_column = make_numeric_column(
    all_indices->type(), all_indices->size(), mask_state::UNALLOCATED, stream, mr);
  auto output_view      = indices_column->mutable_view();
  auto input_view       = column_device_view::create(all_indices->view(), stream);
  auto children_offsets = child_offsets_fn.create_children_offsets(stream);
  auto map_fn           = map_indices_fn{children_offsets, final_remap, *input_view};
  thrust::transform(
    policy, iota, iota + all_indices->size(), output_view.begin<size_type>(), map_fn);

  // remove the bitmask from the all_indices
  auto null_count = all_indices->null_count();  // get before release()
  auto contents   = all_indices->release();     // all_indices will now be empty

  // finally, frankenstein that dictionary column together
  return make_dictionary_column(std::move(keys_column),
                                std::move(indices_column),
                                std::move(*(contents.null_mask.release())),
                                null_count);
}

}  // namespace detail
}  // namespace dictionary
}  // namespace cudf
