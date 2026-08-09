/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "sort_merge_join.hpp"

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/algorithms/copy_if.cuh>
#include <cudf/detail/algorithms/reduce.cuh>
#include <cudf/detail/device_scalar.hpp>
#include <cudf/detail/null_mask.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/row_operator/lexicographic.cuh>
#include <cudf/detail/sizes_to_offsets_iterator.cuh>
#include <cudf/join/join.hpp>
#include <cudf/join/sort_merge_join.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/sorting.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cub/device/device_copy.cuh>
#include <cub/device/device_reduce.cuh>
#include <cub/device/device_scan.cuh>
#include <cub/device/device_select.cuh>
#include <cub/device/device_transform.cuh>
#include <cuda/functional>
#include <cuda/iterator>
#include <cuda/std/iterator>
#include <cuda/std/tuple>
#include <thrust/binary_search.h>
#include <thrust/for_each.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/unique.h>

#include <memory>
#include <utility>

namespace cudf {

namespace {

/**
 * @brief Functor to map indices through a provided mapping container.
 *
 * Non-negative indices are mapped through the container, while negative indices
 * (e.g., JoinNoMatch) are passed through unchanged.
 *
 * @tparam T Type of the mapping container
 */
template <typename T>
struct index_mapping {
  T mapping;  ///< Mapping container that translates indices

  __device__ size_type operator()(size_type idx) const noexcept
  {
    return idx >= 0 ? mapping[idx] : idx;
  }
};

/**
 * @brief Functor to filter and update validity masks for list columns.
 *
 * Propagates null information from a reduced validity mask to specific child positions
 * in the output validity mask for nested list columns.
 */
struct list_nonnull_filter {
  bitmask_type* validity_mask;                   ///< Output validity mask to update
  bitmask_type const* reduced_validity_mask;     ///< Input reduced validity mask
  device_span<size_type const> child_positions;  ///< Positions in the child column
  size_type subset_offset;                       ///< Offset into child_positions

  __device__ void operator()(size_type idx) const noexcept
  {
    if (!bit_is_set(reduced_validity_mask, idx)) {
      clear_bit(validity_mask, child_positions[idx + subset_offset]);
    }
  };
};

/**
 * @brief Functor to check if a row is valid in an unprocessed table.
 *
 * Maps table row indices to boolean values based on the validity mask.
 */
struct is_row_valid {
  bitmask_type const* _validity_mask;  ///< Validity mask for the table

  __device__ auto operator()(size_type idx) const noexcept
  {
    return bit_is_set(_validity_mask, idx);
  }
};

/**
 * @brief Functor to identify null rows for left join with unequal null semantics.
 *
 * Returns true for rows that have nulls and were filtered out during preprocessing.
 */
struct is_row_null {
  bitmask_type const* const _validity_mask;  ///< Validity mask for the table

  __device__ auto operator()(size_type idx) const noexcept
  {
    return !cudf::bit_is_set(_validity_mask, idx);
  }
};

template <typename InputIt, typename ScalarType>
cudf::detail::device_scalar<ScalarType> reduce(InputIt input,
                                               cudf::detail::device_scalar<ScalarType>&& output,
                                               size_type num_items,
                                               rmm::cuda_stream_view stream)
{
  size_t temp_storage_bytes = 0;
  cub::DeviceReduce::Sum(
    nullptr, temp_storage_bytes, input, output.data(), num_items, stream.value());
  rmm::device_buffer temp_storage(temp_storage_bytes, stream);
  cub::DeviceReduce::Sum(
    temp_storage.data(), temp_storage_bytes, input, output.data(), num_items, stream.value());
  return output;
}

template <typename InputIts, typename OutputIts, typename SizeIt>
void batched_copy(InputIts input_iterators,
                  OutputIts output_iterators,
                  SizeIt sizes,
                  size_type num_ranges,
                  rmm::cuda_stream_view stream)
{
  size_t temp_storage_bytes = 0;
  cub::DeviceCopy::Batched(nullptr,
                           temp_storage_bytes,
                           input_iterators,
                           output_iterators,
                           sizes,
                           num_ranges,
                           stream.value());
  rmm::device_buffer temp_storage(temp_storage_bytes, stream);
  cub::DeviceCopy::Batched(temp_storage.data(),
                           temp_storage_bytes,
                           input_iterators,
                           output_iterators,
                           sizes,
                           num_ranges,
                           stream.value());
}

template <typename LargerIterator, typename SmallerIterator>
class merge {
 private:
  table_view smaller;
  table_view larger;
  SmallerIterator sorted_smaller_order_begin;
  SmallerIterator sorted_smaller_order_end;
  LargerIterator sorted_larger_order_begin;
  LargerIterator sorted_larger_order_end;
  std::unique_ptr<detail::row::lexicographic::two_table_comparator> tt_comparator;

 public:
  merge(table_view const& smaller,
        SmallerIterator sorted_smaller_order_begin,
        SmallerIterator sorted_smaller_order_end,
        table_view const& larger,
        LargerIterator sorted_larger_order_begin,
        LargerIterator sorted_larger_order_end,
        rmm::cuda_stream_view stream)
    : smaller{smaller},
      sorted_smaller_order_begin{sorted_smaller_order_begin},
      sorted_smaller_order_end{sorted_smaller_order_end},
      larger{larger},
      sorted_larger_order_begin{sorted_larger_order_begin},
      sorted_larger_order_end{sorted_larger_order_end}
  {
    std::vector<cudf::order> column_order(smaller.num_columns(), cudf::order::ASCENDING);
    std::vector<cudf::null_order> null_precedence(smaller.num_columns(), cudf::null_order::BEFORE);
    tt_comparator = std::make_unique<detail::row::lexicographic::two_table_comparator>(
      smaller, larger, column_order, null_precedence, stream);
  }

  std::unique_ptr<rmm::device_uvector<size_type>> matches_per_row(
    rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr);

  std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
            std::unique_ptr<rmm::device_uvector<size_type>>>
  inner(rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr);

  std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
            std::unique_ptr<rmm::device_uvector<size_type>>>
  left(rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr);
};

template <typename LargerIterator, typename SmallerIterator>
std::unique_ptr<rmm::device_uvector<size_type>>
merge<LargerIterator, SmallerIterator>::matches_per_row(rmm::cuda_stream_view stream,
                                                        rmm::device_async_resource_ref mr)
{
  // naive: iterate through larger table and binary search on smaller table
  auto const has_nulls       = has_nested_nulls(smaller) or has_nested_nulls(larger);
  auto const larger_numrows  = larger.num_rows();
  auto const smaller_numrows = smaller.num_rows();
  auto match_counts =
    cudf::detail::make_zeroed_device_uvector_async<size_type>(larger_numrows + 1, stream, mr);

  auto const comparator = tt_comparator->less<true>(nullate::DYNAMIC{has_nulls});
  auto match_counts_it  = match_counts.begin();
  auto smaller_it =
    cuda::transform_iterator(sorted_smaller_order_begin,
                             cuda::proclaim_return_type<detail::row::lhs_index_type>(
                               [] __device__(size_type idx) -> detail::row::lhs_index_type {
                                 return static_cast<detail::row::lhs_index_type>(idx);
                               }));
  thrust::upper_bound(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                      smaller_it,
                      smaller_it + smaller_numrows,
                      cudf::detail::row::rhs_iterator(0),
                      cudf::detail::row::rhs_iterator(0) + larger_numrows,
                      match_counts_it,
                      comparator);

  auto match_counts_update_it =
    cuda::tabulate_output_iterator([match_counts = match_counts.begin()] __device__(
                                     size_type idx, size_type val) { match_counts[idx] -= val; });
  thrust::lower_bound(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                      smaller_it,
                      smaller_it + smaller_numrows,
                      cudf::detail::row::rhs_iterator(0),
                      cudf::detail::row::rhs_iterator(0) + larger_numrows,
                      match_counts_update_it,
                      comparator);

  return std::make_unique<rmm::device_uvector<size_type>>(std::move(match_counts));
}

template <typename LargerIterator, typename SmallerIterator>
std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
merge<LargerIterator, SmallerIterator>::inner(rmm::cuda_stream_view stream,
                                              rmm::device_async_resource_ref mr)
{
  auto temp_mr               = cudf::get_current_device_resource_ref();
  auto const has_nulls       = has_nested_nulls(smaller) or has_nested_nulls(larger);
  auto const larger_numrows  = larger.num_rows();
  auto const smaller_numrows = smaller.num_rows();

  // naive: iterate through larger table and binary search on smaller table
  auto match_counts = matches_per_row(stream, temp_mr);

  auto count_matches_it = cuda::transform_iterator(
    match_counts->begin(),
    cuda::proclaim_return_type<size_type>([] __device__(auto c) -> size_type { return c != 0; }));
  auto const count_matches =
    thrust::reduce(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                   count_matches_it,
                   count_matches_it + larger_numrows);
  rmm::device_uvector<size_type> nonzero_matches(count_matches, stream, temp_mr);
  cudf::detail::copy_if_async(cuda::counting_iterator<size_type>{0},
                              cuda::counting_iterator<size_type>{larger_numrows},
                              match_counts->begin(),
                              nonzero_matches.begin(),
                              cuda::std::identity{},
                              stream);

  // Use 64-bit prefix sums to handle large output sizes (> INT32_MAX rows)
  // The prefix sums can exceed INT32_MAX even though individual match counts are small
  auto match_offsets =
    cudf::detail::make_zeroed_device_uvector_async<int64_t>(match_counts->size(), stream, temp_mr);
  // Use pinned memory as bounce buffer for efficient device-to-host transfer of the last element
  auto last_element =
    cudf::detail::device_scalar<int64_t>(0, stream, cudf::get_current_device_resource_ref());
  auto output_itr = cudf::detail::make_sizes_to_offsets_iterator(
    match_offsets.begin(), match_offsets.end(), last_element.data());
  thrust::exclusive_scan(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                         match_counts->begin(),
                         match_counts->end(),
                         output_itr,
                         int64_t{0});
  auto const total_matches = static_cast<std::size_t>(last_element.value(stream));

  // populate larger indices
  auto larger_indices =
    cudf::detail::make_zeroed_device_uvector_async<size_type>(total_matches, stream, mr);

  {
    auto const input_iterators = cuda::transform_iterator{
      nonzero_matches.begin(),
      cuda::proclaim_return_type<cuda::constant_iterator<size_type>>(
        [] __device__(auto val) { return cuda::constant_iterator<size_type>(val); })};
    auto const output_iterators = cuda::transform_iterator{
      cuda::permutation_iterator{match_offsets.begin(), nonzero_matches.begin()},
      cuda::proclaim_return_type<rmm::device_uvector<size_type>::iterator>(
        [larger_indices = larger_indices.begin()] __device__(auto val) {
          return larger_indices + val;
        })};
    auto const sizes = cuda::permutation_iterator{match_counts->begin(), nonzero_matches.begin()};

    batched_copy(input_iterators, output_iterators, sizes, count_matches, stream);
  }

  // populate smaller indices
  rmm::device_uvector<size_type> smaller_indices(total_matches, stream, mr);

  // Use cub API to handle large arrays (> INT32_MAX).
  cub::DeviceTransform::Fill(smaller_indices.begin(), smaller_indices.size(), 1, stream.value());

  {
    auto const comparator    = tt_comparator->less<true>(nullate::DYNAMIC{has_nulls});
    auto smaller_tabulate_it = cuda::tabulate_output_iterator(
      [nonzero_matches = nonzero_matches.begin(),
       match_offsets   = match_offsets.begin(),
       smaller_indices = smaller_indices.begin()] __device__(size_type idx, size_type lb) {
        auto const lhs_idx   = nonzero_matches[idx];
        auto const pos       = match_offsets[lhs_idx];
        smaller_indices[pos] = lb;
      });
    auto smaller_it = cuda::transform_iterator(
      sorted_smaller_order_begin,
      cuda::proclaim_return_type<detail::row::lhs_index_type>(
        [] __device__(size_type idx) { return static_cast<detail::row::lhs_index_type>(idx); }));
    auto larger_it = cuda::transform_iterator(
      nonzero_matches.begin(),
      cuda::proclaim_return_type<detail::row::rhs_index_type>(
        [] __device__(size_type idx) { return static_cast<detail::row::rhs_index_type>(idx); }));
    thrust::lower_bound(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                        smaller_it,
                        smaller_it + smaller_numrows,
                        larger_it,
                        larger_it + nonzero_matches.size(),
                        smaller_tabulate_it,
                        comparator);
  }

  // Use cub API to handle large arrays (> INT32_MAX)
  {
    std::size_t temp_storage_bytes = 0;
    cub::DeviceScan::InclusiveSumByKey(nullptr,
                                       temp_storage_bytes,
                                       larger_indices.begin(),
                                       smaller_indices.begin(),
                                       smaller_indices.begin(),
                                       total_matches,
                                       cuda::std::equal_to<>{},
                                       stream.value());
    rmm::device_buffer tmp_storage(temp_storage_bytes, stream);
    cub::DeviceScan::InclusiveSumByKey(tmp_storage.data(),
                                       temp_storage_bytes,
                                       larger_indices.begin(),
                                       smaller_indices.begin(),
                                       smaller_indices.begin(),
                                       total_matches,
                                       cuda::std::equal_to<>{},
                                       stream.value());
  }

  // Use cub API to handle large arrays (> INT32_MAX)
  cub::DeviceTransform::Transform(smaller_indices.begin(),
                                  smaller_indices.begin(),
                                  smaller_indices.size(),
                                  index_mapping<SmallerIterator>{sorted_smaller_order_begin},
                                  stream.value());

  return {std::make_unique<rmm::device_uvector<size_type>>(std::move(smaller_indices)),
          std::make_unique<rmm::device_uvector<size_type>>(std::move(larger_indices))};
}

/**
 * @brief Performs a left join between the larger and smaller tables.
 *
 * This method performs a sort-merge left join, ensuring all rows from the larger table
 * are included in the output. Rows with no matches in the smaller table are paired with
 * JoinNoMatch sentinel values. The output indices are organized with unmatched rows first,
 * followed by matched rows.
 *
 * @param stream CUDA stream used for device operations
 * @param mr Device memory resource for allocations
 * @return Pair of device vectors containing (smaller_indices, larger_indices)
 */
template <typename LargerIterator, typename SmallerIterator>
std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
merge<LargerIterator, SmallerIterator>::left(rmm::cuda_stream_view stream,
                                             rmm::device_async_resource_ref mr)
{
  auto temp_mr               = cudf::get_current_device_resource_ref();
  auto const has_nulls       = has_nested_nulls(smaller) or has_nested_nulls(larger);
  auto const larger_numrows  = larger.num_rows();
  auto const smaller_numrows = smaller.num_rows();

  // Compute match counts per row and identify rows with at least one match
  auto match_counts = matches_per_row(stream, temp_mr);

  cudf::detail::device_scalar<size_type> count_matches(stream, temp_mr);
  auto count_matches_it = cuda::transform_iterator(
    match_counts->begin(),
    cuda::proclaim_return_type<size_type>([] __device__(auto c) -> size_type { return c != 0; }));
  count_matches = reduce(count_matches_it, std::move(count_matches), larger_numrows, stream);
  auto const h_count_matches = count_matches.value(stream);
  rmm::device_uvector<size_type> nonzero_matches(h_count_matches, stream, temp_mr);
  cudf::detail::copy_if_async(cuda::counting_iterator<size_type>{0},
                              cuda::counting_iterator<size_type>{larger_numrows},
                              match_counts->begin(),
                              nonzero_matches.begin(),
                              cuda::std::identity{},
                              stream);

  cudf::detail::device_scalar<int64_t> total_matches(stream, temp_mr);
  auto match_offsets =
    cudf::detail::make_zeroed_device_uvector_async<int64_t>(match_counts->size(), stream, temp_mr);
  auto output_itr = cudf::detail::make_sizes_to_offsets_iterator(
    match_offsets.begin(), match_offsets.end(), total_matches.data());
  thrust::exclusive_scan(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                         match_counts->begin(),
                         match_counts->end(),
                         output_itr,
                         int64_t{0});
  auto const inner_join_matches     = total_matches.value(stream);
  auto const left_join_only_matches = static_cast<int64_t>(larger_numrows - h_count_matches);
  auto larger_indices               = cudf::detail::make_zeroed_device_uvector_async<size_type>(
    inner_join_matches + left_join_only_matches, stream, mr);
  rmm::device_uvector<size_type> smaller_indices(
    inner_join_matches + left_join_only_matches, stream, mr);

  // Fill in unmatched entries (left-join-only rows)
  // These rows exist in the larger table but have no matches in the smaller table
  cudf::detail::copy_if_async(
    cuda::counting_iterator<size_type>{0},
    cuda::counting_iterator<size_type>{larger_numrows},
    match_counts->begin(),
    larger_indices.begin(),
    [] __device__(auto c) -> bool { return c == 0; },
    stream);
  cub::DeviceTransform::Fill(
    smaller_indices.begin(), left_join_only_matches, JoinNoMatch, stream.value());

  // Fill in matched entries (same algorithm as inner join)
  {
    auto const input_iterators = cuda::transform_iterator{
      nonzero_matches.begin(),
      cuda::proclaim_return_type<cuda::constant_iterator<size_type>>(
        [] __device__(auto val) { return cuda::constant_iterator<size_type>(val); })};
    auto const output_iterators = cuda::transform_iterator{
      cuda::permutation_iterator{match_offsets.begin(), nonzero_matches.begin()},
      cuda::proclaim_return_type<rmm::device_uvector<size_type>::iterator>(
        [larger_indices = larger_indices.begin() + left_join_only_matches] __device__(auto val) {
          return larger_indices + val;
        })};
    auto const sizes = cuda::permutation_iterator{match_counts->begin(), nonzero_matches.begin()};

    batched_copy(input_iterators, output_iterators, sizes, h_count_matches, stream);
  }

  // Populate smaller indices for matched rows using binary search
  cub::DeviceTransform::Fill(
    smaller_indices.begin() + left_join_only_matches, inner_join_matches, 1, stream.value());

  {
    auto const comparator    = tt_comparator->less<true>(nullate::DYNAMIC{has_nulls});
    auto smaller_tabulate_it = cuda::tabulate_output_iterator(
      [nonzero_matches = nonzero_matches.begin(),
       match_offsets   = match_offsets.begin(),
       smaller_indices =
         smaller_indices.begin() + left_join_only_matches] __device__(size_type idx, size_type lb) {
        auto const lhs_idx   = nonzero_matches[idx];
        auto const pos       = match_offsets[lhs_idx];
        smaller_indices[pos] = lb;
      });
    auto smaller_it = cuda::transform_iterator(
      sorted_smaller_order_begin,
      cuda::proclaim_return_type<detail::row::lhs_index_type>(
        [] __device__(size_type idx) { return static_cast<detail::row::lhs_index_type>(idx); }));
    auto larger_it = cuda::transform_iterator(
      nonzero_matches.begin(),
      cuda::proclaim_return_type<detail::row::rhs_index_type>(
        [] __device__(size_type idx) { return static_cast<detail::row::rhs_index_type>(idx); }));
    thrust::lower_bound(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                        smaller_it,
                        smaller_it + smaller_numrows,
                        larger_it,
                        larger_it + nonzero_matches.size(),
                        smaller_tabulate_it,
                        comparator);
  }

  // Use cub API to handle large arrays (> INT32_MAX)
  {
    std::size_t temp_storage_bytes = 0;
    cub::DeviceScan::InclusiveSumByKey(nullptr,
                                       temp_storage_bytes,
                                       larger_indices.begin() + left_join_only_matches,
                                       smaller_indices.begin() + left_join_only_matches,
                                       smaller_indices.begin() + left_join_only_matches,
                                       inner_join_matches,
                                       cuda::std::equal_to<>{},
                                       stream.value());
    rmm::device_buffer tmp_storage(temp_storage_bytes, stream);
    cub::DeviceScan::InclusiveSumByKey(tmp_storage.data(),
                                       temp_storage_bytes,
                                       larger_indices.begin() + left_join_only_matches,
                                       smaller_indices.begin() + left_join_only_matches,
                                       smaller_indices.begin() + left_join_only_matches,
                                       inner_join_matches,
                                       cuda::std::equal_to<>{},
                                       stream.value());
  }

  // Use cub API to handle large arrays (> INT32_MAX)
  cub::DeviceTransform::Transform(smaller_indices.begin() + left_join_only_matches,
                                  smaller_indices.begin() + left_join_only_matches,
                                  inner_join_matches,
                                  index_mapping<SmallerIterator>{sorted_smaller_order_begin},
                                  stream.value());

  return {std::make_unique<rmm::device_uvector<size_type>>(std::move(smaller_indices)),
          std::make_unique<rmm::device_uvector<size_type>>(std::move(larger_indices))};
}

}  // anonymous namespace

namespace detail {

void sort_merge_join::preprocessed_table::populate_nonnull_filter(rmm::cuda_stream_view stream)
{
  auto table   = this->_table_view;
  auto temp_mr = cudf::get_current_device_resource_ref();
  // remove rows that have nulls at any nesting level
  // step 1: identify nulls at root level
  auto [validity_mask, num_nulls] = cudf::bitmask_and(table, stream, temp_mr);

  // If the table has no nullable top-level columns, then we need to create
  // an all-valid bitmask that is passed to subsequent operations. This bitmask
  // is updated if any of the nested struct/list children columns have nulls.
  if (validity_mask.is_empty())
    validity_mask =
      cudf::create_null_mask(table.num_rows(), mask_state::ALL_VALID, stream, temp_mr);

  // step 2: identify nulls at non-root levels
  for (size_type col_idx = 0; col_idx < table.num_columns(); col_idx++) {
    auto col = table.column(col_idx);
    if (col.type().id() == type_id::LIST) {
      auto lcv     = lists_column_view(col);
      auto offsets = lcv.offsets();
      auto child   = lcv.child();

      rmm::device_uvector<size_type> offsets_subset(offsets.size(), stream, temp_mr);
      rmm::device_uvector<size_type> child_positions(offsets.size(), stream, temp_mr);
      auto unique_end = thrust::unique_by_key_copy(
        rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
        cuda::std::reverse_iterator(lcv.offsets_end()),
        cuda::std::reverse_iterator(lcv.offsets_end()) + offsets.size(),
        cuda::std::reverse_iterator(cuda::counting_iterator{offsets.size()}),
        cuda::std::reverse_iterator(offsets_subset.end()),
        cuda::std::reverse_iterator(child_positions.end()));
      auto subset_size   = cuda::std::distance(cuda::std::reverse_iterator(offsets_subset.end()),
                                             cuda::std::get<0>(unique_end));
      auto subset_offset = offsets.size() - subset_size;

      auto [reduced_validity_mask, num_nulls] =
        detail::segmented_null_mask_reduction(lcv.child().null_mask(),
                                              offsets_subset.data() + subset_offset,
                                              offsets_subset.data() + offsets_subset.size() - 1,
                                              offsets_subset.data() + subset_offset + 1,
                                              null_policy::INCLUDE,
                                              std::nullopt,
                                              stream,
                                              temp_mr);

      thrust::for_each(
        rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
        cuda::counting_iterator<cudf::size_type>{0},
        cuda::counting_iterator<cudf::size_type>{0} + subset_size,
        list_nonnull_filter{static_cast<bitmask_type*>(validity_mask.data()),
                            static_cast<bitmask_type const*>(reduced_validity_mask.data()),
                            child_positions,
                            static_cast<size_type>(subset_offset)});
    } else if (col.type().id() == type_id::STRUCT) {
      // Recursive lambda to traverse struct hierarchy and accumulate null information.
      // This lambda ANDs the column's null mask with the accumulated mask in-place and recursively
      // processes all child columns to capture nested nulls.
      auto and_bitmasks = [&](auto&& self, bitmask_type* mask, column_view const& colview) -> void {
        auto const num_rows = colview.size();
        if (colview.type().id() == cudf::type_id::EMPTY) { return; }

        if (colview.nullable()) {
          // AND this column's null mask with the accumulated mask
          auto colmask = colview.null_mask();
          std::vector masks{reinterpret_cast<bitmask_type const*>(colmask),
                            reinterpret_cast<bitmask_type const*>(mask)};
          std::vector<size_type> begin_bits{0, 0};
          cudf::detail::inplace_bitmask_and(
            device_span<bitmask_type>(mask, num_bitmask_words(num_rows)),
            masks,
            begin_bits,
            num_rows,
            stream);
        }

        if (colview.type().id() == cudf::type_id::STRUCT ||
            colview.type().id() == cudf::type_id::LIST) {
          CUDF_EXPECTS(
            std::all_of(colview.child_begin(),
                        colview.child_end(),
                        [&](auto const& child_col) { return num_rows == child_col.size(); }),
            "Child columns must have the same number of rows as the Struct column.");

          // Recursively process child columns to capture nulls at deeper nesting levels.
          for (auto it = colview.child_begin(); it != colview.child_end(); it++) {
            auto& child = *it;
            self(self, mask, child);
          }
        }
      };
      // Process all children of the struct column
      for (auto it = col.child_begin(); it != col.child_end(); it++) {
        auto& child = *it;
        and_bitmasks(and_bitmasks, static_cast<bitmask_type*>(validity_mask.data()), child);
      }
    }
  }
  this->_num_nulls =
    null_count(static_cast<bitmask_type*>(validity_mask.data()), 0, table.num_rows(), stream);
  this->_validity_mask = std::move(validity_mask);
}

void sort_merge_join::preprocessed_table::apply_nonnull_filter(rmm::cuda_stream_view stream)
{
  auto temp_mr = cudf::get_current_device_resource_ref();
  // construct bool column to apply mask
  cudf::scalar_type_t<bool> true_scalar(true, true, stream, temp_mr);
  auto bool_mask =
    cudf::make_column_from_scalar(true_scalar, _table_view.num_rows(), stream, temp_mr);
  CUDF_EXPECTS(_validity_mask.has_value() && _num_nulls.has_value(),
               "Something went wrong while dropping nulls in the unprocessed tables");
  bool_mask->set_null_mask(_validity_mask.value(), _num_nulls.value(), stream);

  _null_processed_table      = apply_boolean_mask(_table_view, *bool_mask, stream, temp_mr);
  _null_processed_table_view = _null_processed_table.value()->view();
}

void sort_merge_join::preprocessed_table::preprocess_unprocessed_table(rmm::cuda_stream_view stream)
{
  populate_nonnull_filter(stream);
  apply_nonnull_filter(stream);
}

void sort_merge_join::preprocessed_table::compute_sorted_order(rmm::cuda_stream_view stream)
{
  auto temp_mr = cudf::get_current_device_resource_ref();
  std::vector<cudf::order> column_order(_null_processed_table_view.num_columns(),
                                        cudf::order::ASCENDING);
  std::vector<cudf::null_order> null_precedence(_null_processed_table_view.num_columns(),
                                                cudf::null_order::BEFORE);
  this->_null_processed_table_sorted_order =
    cudf::sorted_order(_null_processed_table_view, column_order, null_precedence, stream, temp_mr);
}

sort_merge_join::preprocessed_table sort_merge_join::preprocessed_table::create(
  table_view const& table,
  null_equality compare_nulls,
  sorted is_sorted,
  rmm::cuda_stream_view stream)
{
  preprocessed_table result;
  result._table_view = table;

  if (compare_nulls == null_equality::EQUAL) {
    result._null_processed_table_view = table;
  } else {
    // if a table has no nullable column, then there's no preprocessing to be done
    if (has_nested_nulls(table)) {
      result.preprocess_unprocessed_table(stream);
    } else {
      result._null_processed_table_view = table;
    }
  }

  if (is_sorted == cudf::sorted::NO) { result.compute_sorted_order(stream); }

  return result;
}

sort_merge_join::sort_merge_join(table_view const& right,
                                 sorted is_right_sorted,
                                 null_equality compare_nulls,
                                 rmm::cuda_stream_view stream)
  : preprocessed_right{preprocessed_table::create(right, compare_nulls, is_right_sorted, stream)},
    compare_nulls{compare_nulls}
{
  cudf::scoped_range range{"sort_merge_join::sort_merge_join"};
  // Sanity checks
  CUDF_EXPECTS(right.num_columns() != 0,
               "Number of columns the keys table must be non-zero for a join",
               std::invalid_argument);
}

rmm::device_uvector<size_type> sort_merge_join::preprocessed_table::map_table_to_unprocessed(
  rmm::cuda_stream_view stream) const
{
  CUDF_EXPECTS(_validity_mask.has_value() && _num_nulls.has_value(), "Mapping is not possible");
  auto temp_mr                  = cudf::get_current_device_resource_ref();
  auto const table_mapping_size = _table_view.num_rows() - _num_nulls.value();
  rmm::device_uvector<size_type> table_mapping(table_mapping_size, stream, temp_mr);
  cudf::detail::copy_if_async(
    cuda::counting_iterator<size_type>{0},
    cuda::counting_iterator<size_type>{_table_view.num_rows()},
    cuda::counting_iterator<size_type>{0},
    table_mapping.begin(),
    is_row_valid{static_cast<bitmask_type const*>(_validity_mask.value().data())},
    stream);
  return table_mapping;
}

void sort_merge_join::postprocess_indices(preprocessed_table const& preprocessed_left,
                                          device_span<size_type> smaller_indices,
                                          device_span<size_type> larger_indices,
                                          rmm::cuda_stream_view stream) const
{
  if (compare_nulls == null_equality::UNEQUAL) {
    // if a table has no nullable column, then there's no postprocessing to be done
    if (has_nested_nulls(preprocessed_left._table_view)) {
      auto left_mapping = preprocessed_left.map_table_to_unprocessed(stream);
      // Use cub API to handle large arrays (> INT32_MAX)
      cub::DeviceTransform::Transform(larger_indices.begin(),
                                      larger_indices.begin(),
                                      larger_indices.size(),
                                      index_mapping<device_span<size_type>>{left_mapping},
                                      stream.value());
    }
    if (has_nested_nulls(preprocessed_right._table_view)) {
      auto right_mapping = preprocessed_right.map_table_to_unprocessed(stream);
      // Use cub API to handle large arrays (> INT32_MAX)
      cub::DeviceTransform::Transform(smaller_indices.begin(),
                                      smaller_indices.begin(),
                                      smaller_indices.size(),
                                      index_mapping<device_span<size_type>>{right_mapping},
                                      stream.value());
    }
  }
}

template <typename MergeOperation>
auto sort_merge_join::invoke_merge(preprocessed_table const& preprocessed_left,
                                   table_view right_view,
                                   table_view left_view,
                                   MergeOperation&& op,
                                   rmm::cuda_stream_view stream) const
{
  auto has_right_sorting_order = preprocessed_right._null_processed_table_sorted_order.has_value();
  auto has_left_sorting_order  = preprocessed_left._null_processed_table_sorted_order.has_value();
  if (has_right_sorting_order && has_left_sorting_order) {
    // Both sorted
    auto r_view = preprocessed_right._null_processed_table_sorted_order.value()->view();
    auto l_view = preprocessed_left._null_processed_table_sorted_order.value()->view();
    merge obj(right_view,
              r_view.begin<size_type>(),
              r_view.end<size_type>(),
              left_view,
              l_view.begin<size_type>(),
              l_view.end<size_type>(),
              stream);
    return op(obj);
  } else if (has_right_sorting_order && !has_left_sorting_order) {
    // preprocessed_right sorted, preprocessed_left unsorted
    auto r_view = preprocessed_right._null_processed_table_sorted_order.value()->view();
    merge obj(right_view,
              r_view.begin<size_type>(),
              r_view.end<size_type>(),
              left_view,
              cuda::counting_iterator<cudf::size_type>{0},
              cuda::counting_iterator{left_view.num_rows()},
              stream);
    return op(obj);
  } else if (!has_right_sorting_order && has_left_sorting_order) {
    // preprocessed_right unsorted, preprocessed_left sorted
    auto l_view = preprocessed_left._null_processed_table_sorted_order.value()->view();
    merge obj(right_view,
              cuda::counting_iterator<cudf::size_type>{0},
              cuda::counting_iterator{preprocessed_right._null_processed_table_view.num_rows()},
              left_view,
              l_view.begin<size_type>(),
              l_view.end<size_type>(),
              stream);
    return op(obj);
  }
  // Both unsorted
  merge obj(right_view,
            cuda::counting_iterator<cudf::size_type>{0},
            cuda::counting_iterator{preprocessed_right._null_processed_table_view.num_rows()},
            left_view,
            cuda::counting_iterator<cudf::size_type>{0},
            cuda::counting_iterator{left_view.num_rows()},
            stream);
  return op(obj);
}

std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
sort_merge_join::inner_join(table_view const& left,
                            sorted is_left_sorted,
                            rmm::cuda_stream_view stream,
                            rmm::device_async_resource_ref mr) const
{
  cudf::scoped_range range{"sort_merge_join::inner_join"};
  // Sanity checks
  CUDF_EXPECTS(left.num_columns() != 0,
               "Number of columns in left keys must be non-zero for a join",
               std::invalid_argument);
  CUDF_EXPECTS(left.num_columns() == preprocessed_right._null_processed_table_view.num_columns(),
               "Number of columns must match for a join",
               std::invalid_argument);

  // Create preprocessed left table locally for thread safety
  auto preprocessed_left = preprocessed_table::create(left, compare_nulls, is_left_sorted, stream);

  return invoke_merge(
    preprocessed_left,
    preprocessed_right._null_processed_table_view,
    preprocessed_left._null_processed_table_view,
    [this, &preprocessed_left, stream, mr](auto& obj) {
      auto [preprocessed_right_indices, preprocessed_left_indices] = obj.inner(stream, mr);
      postprocess_indices(
        preprocessed_left, *preprocessed_right_indices, *preprocessed_left_indices, stream);
      return std::pair{std::move(preprocessed_left_indices), std::move(preprocessed_right_indices)};
    },
    stream);
}

std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
sort_merge_join::left_join(table_view const& left,
                           sorted is_left_sorted,
                           rmm::cuda_stream_view stream,
                           rmm::device_async_resource_ref mr) const
{
  cudf::scoped_range range{"sort_merge_join::left_join"};
  // Sanity checks
  CUDF_EXPECTS(left.num_columns() != 0,
               "Number of columns in left keys must be non-zero for a join",
               std::invalid_argument);
  CUDF_EXPECTS(left.num_columns() == preprocessed_right._null_processed_table_view.num_columns(),
               "Number of columns must match for a join",
               std::invalid_argument);

  // Create preprocessed left table locally for thread safety
  auto preprocessed_left = preprocessed_table::create(left, compare_nulls, is_left_sorted, stream);

  return invoke_merge(
    preprocessed_left,
    preprocessed_right._null_processed_table_view,
    preprocessed_left._null_processed_table_view,
    [this, &preprocessed_left, left, stream, mr](auto& obj) {
      auto [preprocessed_right_indices, preprocessed_left_indices] = obj.left(stream, mr);
      postprocess_indices(
        preprocessed_left, *preprocessed_right_indices, *preprocessed_left_indices, stream);

      //  For left join with UNEQUAL nulls, we need to add back rows that were filtered out.
      //  Remaining configs can return directly
      if (compare_nulls == null_equality::EQUAL ||
          !has_nested_nulls(preprocessed_left._table_view)) {
        return std::pair{std::move(preprocessed_left_indices),
                         std::move(preprocessed_right_indices)};
      }

      // Special handling for UNEQUAL null semantics with nested nulls:
      // Rows containing nulls were filtered during preprocessing and must be reinserted.
      // These rows have no matches by definition (nulls are unequal), so they're added
      // to the output with JoinNoMatch sentinel values for the right side.

      auto const num_filtered_nulls = preprocessed_left._num_nulls.value();
      auto const total_output_size =
        preprocessed_left_indices->size() + static_cast<int64_t>(num_filtered_nulls);

      // Create new result vectors with space for filtered rows
      rmm::device_uvector<size_type> left_result_indices(total_output_size, stream, mr);
      rmm::device_uvector<size_type> right_result_indices(total_output_size, stream, mr);

      // Copy existing join results
      {
        using Iterator       = decltype(preprocessed_left_indices->begin());
        auto input_iterators = cudf::detail::make_pinned_vector_async<Iterator>(2, stream);
        input_iterators[0]   = preprocessed_left_indices->begin();
        input_iterators[1]   = preprocessed_right_indices->begin();

        auto output_iterators = cudf::detail::make_pinned_vector_async<Iterator>(2, stream);
        output_iterators[0]   = left_result_indices.begin();
        output_iterators[1]   = right_result_indices.begin();

        auto sizes = cudf::detail::make_pinned_vector_async<size_t>(2, stream);
        sizes[0]   = preprocessed_left_indices->size();
        sizes[1]   = preprocessed_right_indices->size();

        batched_copy(input_iterators.begin(), output_iterators.begin(), sizes.begin(), 2, stream);
        stream.synchronize();  // ensures the vectors are not destroyed before the copy is completed
      }

      // Append filtered null rows with JoinNoMatch for right side
      auto const validity_mask =
        static_cast<bitmask_type const*>(preprocessed_left._validity_mask.value().data());
      cudf::detail::copy_if_async(cuda::counting_iterator<size_type>{0},
                                  cuda::counting_iterator<size_type>{left.num_rows()},
                                  cuda::counting_iterator<size_type>{0},
                                  left_result_indices.begin() + preprocessed_left_indices->size(),
                                  is_row_null{validity_mask},
                                  stream);
      cub::DeviceTransform::Fill(right_result_indices.begin() + preprocessed_right_indices->size(),
                                 num_filtered_nulls,
                                 JoinNoMatch,
                                 stream.value());

      return std::pair{
        std::make_unique<rmm::device_uvector<size_type>>(std::move(left_result_indices)),
        std::make_unique<rmm::device_uvector<size_type>>(std::move(right_result_indices))};
    },
    stream);
}

std::unique_ptr<cudf::join_match_context> sort_merge_join::inner_join_match_context(
  table_view const& left,
  sorted is_left_sorted,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr) const
{
  cudf::scoped_range range{"sort_merge_join::inner_join_match_context"};
  // Sanity checks
  CUDF_EXPECTS(left.num_columns() != 0,
               "Number of columns in left keys must be non-zero for a join",
               std::invalid_argument);
  CUDF_EXPECTS(left.num_columns() == preprocessed_right._null_processed_table_view.num_columns(),
               "Number of columns must match for a join",
               std::invalid_argument);

  // Create preprocessed left table locally for thread safety
  auto preprocessed_left = preprocessed_table::create(left, compare_nulls, is_left_sorted, stream);

  return invoke_merge(
    preprocessed_left,
    preprocessed_right._null_processed_table_view,
    preprocessed_left._null_processed_table_view,
    [this, left, &preprocessed_left, stream, mr](auto& obj) mutable {
      auto matches_per_row = obj.matches_per_row(stream, cudf::get_current_device_resource_ref());
      matches_per_row->resize(matches_per_row->size() - 1, stream);
      if (compare_nulls == null_equality::UNEQUAL &&
          has_nested_nulls(preprocessed_left._table_view)) {
        // Now we need to post-process the matches i.e. insert zero counts for all the null
        // positions
        auto unprocessed_matches_per_row =
          cudf::detail::make_zeroed_device_uvector_async<size_type>(
            preprocessed_left._table_view.num_rows(), stream, mr);
        auto mapping = preprocessed_left.map_table_to_unprocessed(stream);
        thrust::scatter(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                        matches_per_row->begin(),
                        matches_per_row->end(),
                        mapping.begin(),
                        unprocessed_matches_per_row.begin());
        return std::make_unique<sort_merge_join_match_context>(
          left,
          std::make_unique<rmm::device_uvector<size_type>>(std::move(unprocessed_matches_per_row)),
          std::move(preprocessed_left));
      }
      return std::make_unique<sort_merge_join_match_context>(
        left, std::move(matches_per_row), std::move(preprocessed_left));
    },
    stream);
}

// left_partition_end exclusive
std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
sort_merge_join::partitioned_inner_join(cudf::join_partition_context const& context,
                                        rmm::cuda_stream_view stream,
                                        rmm::device_async_resource_ref mr) const
{
  cudf::scoped_range range{"sort_merge_join::partitioned_inner_join"};

  // Extract preprocessed_left from the context
  auto const& preprocessed_left =
    static_cast<sort_merge_join_match_context const*>(context.left_table_context.get())
      ->preprocessed_left;

  auto const left_partition_start_idx = context.left_start_idx;
  auto const left_partition_end_idx   = context.left_end_idx;
  auto null_processed_table_start_idx = left_partition_start_idx;
  auto null_processed_table_end_idx   = left_partition_end_idx;
  if (compare_nulls == null_equality::UNEQUAL && has_nested_nulls(preprocessed_left._table_view)) {
    auto left_mapping              = preprocessed_left.map_table_to_unprocessed(stream);
    null_processed_table_start_idx = cuda::std::distance(
      left_mapping.begin(),
      thrust::lower_bound(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                          left_mapping.begin(),
                          left_mapping.end(),
                          left_partition_start_idx));
    null_processed_table_end_idx = cuda::std::distance(
      left_mapping.begin(),
      thrust::upper_bound(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                          left_mapping.begin(),
                          left_mapping.end(),
                          left_partition_end_idx - 1));
  }
  auto null_processed_left_partition =
    cudf::slice(preprocessed_left._null_processed_table_view,
                {null_processed_table_start_idx, null_processed_table_end_idx},
                stream)[0];

  auto [preprocessed_right_indices, preprocessed_left_indices] = invoke_merge(
    preprocessed_left,
    preprocessed_right._null_processed_table_view,
    null_processed_left_partition,
    [this, left_partition_start_idx, stream, mr](auto& obj) { return obj.inner(stream, mr); },
    stream);
  // Map from slice to total null processed table
  // Use cub API to handle large arrays (> INT32_MAX)
  cub::DeviceTransform::Transform(
    preprocessed_left_indices->begin(),
    preprocessed_left_indices->begin(),
    preprocessed_left_indices->size(),
    [left_partition_start_idx] __device__(auto idx) { return left_partition_start_idx + idx; },
    stream.value());
  // Map from total null processed table to unprocessed table
  postprocess_indices(
    preprocessed_left, *preprocessed_right_indices, *preprocessed_left_indices, stream);
  return std::pair{std::move(preprocessed_left_indices), std::move(preprocessed_right_indices)};
}

}  // namespace detail

sort_merge_join::~sort_merge_join() = default;

sort_merge_join::sort_merge_join(table_view const& right,
                                 sorted is_right_sorted,
                                 null_equality compare_nulls,
                                 rmm::cuda_stream_view stream)
  : _impl{std::make_unique<impl_type>(right, is_right_sorted, compare_nulls, stream)}
{
}

std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
sort_merge_join::inner_join(table_view const& left,
                            sorted is_left_sorted,
                            rmm::cuda_stream_view stream,
                            rmm::device_async_resource_ref mr) const
{
  return _impl->inner_join(left, is_left_sorted, stream, mr);
}

std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
sort_merge_join::left_join(table_view const& left,
                           sorted is_left_sorted,
                           rmm::cuda_stream_view stream,
                           rmm::device_async_resource_ref mr) const
{
  return _impl->left_join(left, is_left_sorted, stream, mr);
}

std::unique_ptr<join_match_context> sort_merge_join::inner_join_match_context(
  table_view const& left,
  sorted is_left_sorted,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr) const
{
  return _impl->inner_join_match_context(left, is_left_sorted, stream, mr);
}

std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
sort_merge_join::partitioned_inner_join(cudf::join_partition_context const& context,
                                        rmm::cuda_stream_view stream,
                                        rmm::device_async_resource_ref mr) const
{
  return _impl->partitioned_inner_join(context, stream, mr);
}

}  // namespace cudf
