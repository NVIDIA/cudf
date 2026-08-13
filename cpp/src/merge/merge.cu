/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/copying.hpp>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/gather.cuh>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/merge.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/row_operator/lexicographic.cuh>
#include <cudf/detail/search.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/dictionary/detail/merge.hpp>
#include <cudf/dictionary/detail/update_keys.hpp>
#include <cudf/lists/detail/concatenate.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/merge.hpp>
#include <cudf/strings/detail/merge.hpp>
#include <cudf/structs/structs_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_device_view.cuh>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_checks.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/functional>
#include <cuda/iterator>
#include <cuda/std/iterator>
#include <cuda/std/utility>
#include <thrust/binary_search.h>
#include <thrust/merge.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>

#include <limits>
#include <numeric>
#include <queue>
#include <vector>

namespace cudf {
namespace detail {

namespace {

template <bool has_nulls>
struct row_lexicographic_tagged_comparator {
  row_lexicographic_tagged_comparator(table_device_view const lhs,
                                      table_device_view const rhs,
                                      device_span<order const> const column_order,
                                      device_span<null_order const> const null_precedence)
    : _lhs{lhs}, _rhs{rhs}, _column_order{column_order}, _null_precedence{null_precedence}
  {
  }

  __device__ bool operator()(index_type lhs_tagged_index,
                             index_type rhs_tagged_index) const noexcept
  {
    auto const [l_side, l_indx] = lhs_tagged_index;
    auto const [r_side, r_indx] = rhs_tagged_index;

    table_device_view const* ptr_left_dview{l_side == side::LEFT ? &_lhs : &_rhs};
    table_device_view const* ptr_right_dview{r_side == side::LEFT ? &_lhs : &_rhs};
    auto const comparator = [&]() {
      if constexpr (has_nulls) {
        return cudf::detail::row::lexicographic::device_row_comparator<false, bool>{
          has_nulls, *ptr_left_dview, *ptr_right_dview, _column_order, _null_precedence};
      } else {
        return cudf::detail::row::lexicographic::device_row_comparator<false, bool>{
          has_nulls, *ptr_left_dview, *ptr_right_dview, _column_order};
      }
    }();

    return comparator(l_indx, r_indx) == weak_ordering::LESS;
  }

 private:
  table_device_view const _lhs;
  table_device_view const _rhs;
  device_span<null_order const> const _null_precedence;
  device_span<order const> const _column_order;
};

using detail::side;
using index_type = detail::index_type;

/**
 * @brief Merges the bits of two validity bitmasks.
 *
 * Merges the bits from two column_device_views into the destination validity buffer
 * according to `merged_indices` map such that bit `i` in `out_validity`
 * will be equal to bit `cuda::std::get<1>(merged_indices[i])` from `left_dcol`
 * if `cuda::std::get<0>(merged_indices[i])` equals `side::LEFT`; otherwise,
 * from `right_dcol`.
 *
 * `left_dcol` and `right_dcol` must not overlap.
 *
 * @tparam left_have_valids Indicates whether left_dcol mask is unallocated (hence, ALL_VALID)
 * @tparam right_have_valids Indicates whether right_dcol mask is unallocated (hence ALL_VALID)
 * @param[in] left_dcol The left column_device_view whose bits will be merged
 * @param[in] right_dcol The right column_device_view whose bits will be merged
 * @param[out] out_validity The output validity buffer after merging the left and right buffers
 * @param[in] num_destination_rows The number of rows in the out_validity buffer
 * @param[in] merged_indices The map that indicates the source of the input and index
 * to be copied to the output. Length must be equal to `num_destination_rows`
 */
template <bool left_have_valids, bool right_have_valids>
CUDF_KERNEL void materialize_merged_bitmask_kernel(
  column_device_view left_dcol,
  column_device_view right_dcol,
  bitmask_type* out_validity,
  size_type const num_destination_rows,
  index_type const* __restrict__ const merged_indices)
{
  auto const stride = detail::grid_1d::grid_stride();

  auto tid = detail::grid_1d::global_thread_id();

  auto active_threads = __ballot_sync(0xffff'ffffu, tid < num_destination_rows);

  while (tid < num_destination_rows) {
    auto const destination_row     = static_cast<size_type>(tid);
    auto const [src_side, src_row] = merged_indices[destination_row];
    bool const from_left{src_side == side::LEFT};
    bool source_bit_is_valid{true};
    if (left_have_valids && from_left) {
      source_bit_is_valid = left_dcol.is_valid_nocheck(src_row);
    } else if (right_have_valids && !from_left) {
      source_bit_is_valid = right_dcol.is_valid_nocheck(src_row);
    }

    // Use ballot to find all valid bits in this warp and create the output
    // bitmask element
    bitmask_type const result_mask{__ballot_sync(active_threads, source_bit_is_valid)};

    // Only one thread writes output
    if (0 == threadIdx.x % warpSize) { out_validity[word_index(destination_row)] = result_mask; }

    tid += stride;
    active_threads = __ballot_sync(active_threads, tid < num_destination_rows);
  }
}

void materialize_bitmask(column_view const& left_col,
                         column_view const& right_col,
                         bitmask_type* out_validity,
                         size_type num_elements,
                         index_type const* merged_indices,
                         rmm::cuda_stream_view stream)
{
  constexpr size_type BLOCK_SIZE{256};
  detail::grid_1d grid_config{num_elements, BLOCK_SIZE};

  auto p_left_dcol  = column_device_view::create(left_col, stream);
  auto p_right_dcol = column_device_view::create(right_col, stream);

  auto left_valid  = *p_left_dcol;
  auto right_valid = *p_right_dcol;

  if (left_col.has_nulls()) {
    if (right_col.has_nulls()) {
      materialize_merged_bitmask_kernel<true, true>
        <<<grid_config.num_blocks, grid_config.num_threads_per_block, 0, stream.value()>>>(
          left_valid, right_valid, out_validity, num_elements, merged_indices);
      CUDF_CUDA_TRY(cudaGetLastError());
    } else {
      materialize_merged_bitmask_kernel<true, false>
        <<<grid_config.num_blocks, grid_config.num_threads_per_block, 0, stream.value()>>>(
          left_valid, right_valid, out_validity, num_elements, merged_indices);
      CUDF_CUDA_TRY(cudaGetLastError());
    }
  } else {
    if (right_col.has_nulls()) {
      materialize_merged_bitmask_kernel<false, true>
        <<<grid_config.num_blocks, grid_config.num_threads_per_block, 0, stream.value()>>>(
          left_valid, right_valid, out_validity, num_elements, merged_indices);
      CUDF_CUDA_TRY(cudaGetLastError());
    } else {
      CUDF_FAIL("materialize_merged_bitmask_kernel<false, false>() should never be called.");
    }
  }

  CUDF_CHECK_CUDA(stream.value());
}

struct side_index_generator {
  side _side;

  __device__ index_type operator()(size_type i) const noexcept { return index_type{_side, i}; }
};

/**
 * @brief Generates the row indices and source side (left or right) in accordance with the index
 * columns.
 *
 *
 * @tparam index_type Indicates the type to be used to collect index and side information;
 * @param[in] left_table The left table_view to be merged
 * @param[in] right_table The right table_view to be merged
 * @param[in] column_order Sort order types of index columns
 * @param[in] null_precedence Array indicating the order of nulls with respect to non-nulls for the
 * index columns
 * @param[in] nullable Flag indicating if at least one of the table_view arguments has nulls
 * (defaults to true)
 * @param[in] stream CUDA stream used for device memory operations and kernel launches.
 *
 * @return A device_uvector of merged indices
 */
index_vector generate_merged_indices(table_view const& left_table,
                                     table_view const& right_table,
                                     std::vector<order> const& column_order,
                                     std::vector<null_order> const& null_precedence,
                                     bool nullable,
                                     rmm::cuda_stream_view stream)
{
  size_type const left_size  = left_table.num_rows();
  size_type const right_size = right_table.num_rows();
  size_type const total_size = left_size + right_size;

  auto left_gen    = side_index_generator{side::LEFT};
  auto right_gen   = side_index_generator{side::RIGHT};
  auto left_begin  = cudf::detail::make_counting_transform_iterator(0, left_gen);
  auto right_begin = cudf::detail::make_counting_transform_iterator(0, right_gen);

  index_vector merged_indices(total_size, stream);

  auto const has_nulls =
    nullate::DYNAMIC{cudf::has_nulls(left_table) or cudf::has_nulls(right_table)};

  auto lhs_device_view = table_device_view::create(left_table, stream);
  auto rhs_device_view = table_device_view::create(right_table, stream);

  auto d_column_order = cudf::detail::make_device_uvector_async(
    column_order, stream, cudf::get_current_device_resource_ref());

  if (has_nulls) {
    auto const new_null_precedence = [&]() {
      if (null_precedence.size() > 0) {
        CUDF_EXPECTS(static_cast<size_type>(null_precedence.size()) == left_table.num_columns(),
                     "Null precedence vector size mismatched");
        return null_precedence;
      } else {
        return std::vector<null_order>(left_table.num_columns(), null_order::BEFORE);
      }
    }();

    auto d_null_precedence = cudf::detail::make_device_uvector_async(
      new_null_precedence, stream, cudf::get_current_device_resource_ref());

    auto ineq_op = detail::row_lexicographic_tagged_comparator<true>(
      *lhs_device_view, *rhs_device_view, d_column_order, d_null_precedence);
    thrust::merge(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                  left_begin,
                  left_begin + left_size,
                  right_begin,
                  right_begin + right_size,
                  merged_indices.begin(),
                  ineq_op);
  } else {
    auto ineq_op = detail::row_lexicographic_tagged_comparator<false>(
      *lhs_device_view, *rhs_device_view, d_column_order, {});
    thrust::merge(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                  left_begin,
                  left_begin + left_size,
                  right_begin,
                  right_begin + right_size,
                  merged_indices.begin(),
                  ineq_op);
  }

  CUDF_CHECK_CUDA(stream.value());

  return merged_indices;
}

index_vector generate_merged_indices_nested(table_view const& left_table,
                                            table_view const& right_table,
                                            std::vector<order> const& column_order,
                                            std::vector<null_order> const& null_precedence,
                                            bool nullable,
                                            rmm::cuda_stream_view stream)
{
  size_type const left_size  = left_table.num_rows();
  size_type const right_size = right_table.num_rows();
  size_type const total_size = left_size + right_size;

  index_vector merged_indices(total_size, stream);

  auto const left_indices_col     = cudf::detail::lower_bound(right_table,
                                                          left_table,
                                                          column_order,
                                                          null_precedence,
                                                          stream,
                                                          cudf::get_current_device_resource_ref());
  auto const left_indices         = left_indices_col->view();
  auto left_indices_mutable       = left_indices_col->mutable_view();
  auto const left_indices_begin   = left_indices.begin<cudf::size_type>();
  auto const left_indices_end     = left_indices.end<cudf::size_type>();
  auto left_indices_mutable_begin = left_indices_mutable.begin<cudf::size_type>();

  auto const total_counter = cuda::counting_iterator<cudf::size_type>{0};
  thrust::for_each(
    rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
    total_counter,
    total_counter + total_size,
    [merged = merged_indices.data(), left = left_indices_begin, left_size, right_size] __device__(
      auto const idx) {
      // We split threads into two groups, so only one kernel is needed.
      // Threads in [0, right_size) will insert right indices in sorted order.
      // Threads in [right_size, total_size) will insert left indices in sorted order.
      if (idx < right_size) {
        // this tells us between which segments of left elements a right element
        // would fall
        auto const r_bound      = thrust::upper_bound(thrust::seq, left, left + left_size, idx);
        auto const r_segment    = cuda::std::distance(left, r_bound);
        merged[r_segment + idx] = cuda::std::make_pair(side::RIGHT, idx);
      } else {
        auto const left_idx               = idx - right_size;
        merged[left[left_idx] + left_idx] = cuda::std::make_pair(side::LEFT, left_idx);
      }
    });

  return merged_indices;
}

/**
 * @brief Generate merged column given row-order of merged tables
 *  (ordered according to indices of key_cols) and the 2 columns to merge.
 */
struct column_merger {
  explicit column_merger(index_vector const& row_order) : row_order_(row_order) {}

  template <typename Element, CUDF_ENABLE_IF(not is_rep_layout_compatible<Element>())>
  std::unique_ptr<column> operator()(column_view const&,
                                     column_view const&,
                                     rmm::cuda_stream_view,
                                     rmm::device_async_resource_ref) const
  {
    CUDF_FAIL("Unsupported type for merge.");
  }

  // column merger operator;
  //
  template <typename Element>
  std::unique_ptr<column> operator()(column_view const& lcol,
                                     column_view const& rcol,
                                     rmm::cuda_stream_view stream,
                                     rmm::device_async_resource_ref mr) const
    requires(is_rep_layout_compatible<Element>())
  {
    auto lsz         = lcol.size();
    auto merged_size = lsz + rcol.size();
    auto merged_col  = cudf::detail::allocate_like(lcol.has_nulls() ? lcol : rcol,
                                                  merged_size,
                                                  cudf::mask_allocation_policy::RETAIN,
                                                  stream,
                                                  mr);

    //"gather" data from lcol, rcol according to row_order_ "map"
    //(directly calling gather() won't work because
    // lcol, rcol indices overlap!)
    //
    cudf::mutable_column_view merged_view = merged_col->mutable_view();

    // initialize null_mask to all valid:
    //
    // Note: this initialization in conjunction with
    // _conditionally_ calling materialize_bitmask() below covers
    // the case materialize_merged_bitmask_kernel<false, false>()
    // which won't be called anymore (because of the _condition_
    // below)
    //
    cudf::detail::set_null_mask(merged_view.null_mask(), 0, merged_view.size(), true, stream);

    // set the null count:
    //
    merged_col->set_null_count(lcol.null_count() + rcol.null_count());

    // to resolve view.data()'s types use: Element
    //
    auto const d_lcol = lcol.data<Element>();
    auto const d_rcol = rcol.data<Element>();

    // capture lcol, rcol
    // and "gather" into merged_view.data()[indx_merged]
    // from lcol or rcol, depending on side;
    //
    thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                      row_order_.begin(),
                      row_order_.end(),
                      merged_view.begin<Element>(),
                      cuda::proclaim_return_type<Element>(
                        [d_lcol, d_rcol] __device__(index_type const& index_pair) {
                          auto const [side, index] = index_pair;
                          return side == side::LEFT ? d_lcol[index] : d_rcol[index];
                        }));

    // CAVEAT: conditional call below is erroneous without
    // set_null_mask() call (see TODO above):
    //
    if (lcol.has_nulls() || rcol.has_nulls()) {
      // resolve null mask:
      //
      materialize_bitmask(
        lcol, rcol, merged_view.null_mask(), merged_view.size(), row_order_.data(), stream);
    }

    return merged_col;
  }

 private:
  index_vector const& row_order_;
};

// specialization for strings
template <>
std::unique_ptr<column> column_merger::operator()<cudf::string_view>(
  column_view const& lcol,
  column_view const& rcol,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr) const
{
  return strings::detail::merge(
    strings_column_view(lcol), strings_column_view(rcol), row_order_, stream, mr);
}

// specialization for dictionary
template <>
std::unique_ptr<column> column_merger::operator()<cudf::dictionary32>(
  column_view const& lcol,
  column_view const& rcol,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr) const
{
  auto result = cudf::dictionary::detail::merge(
    cudf::dictionary_column_view(lcol), cudf::dictionary_column_view(rcol), row_order_, stream, mr);

  // set the validity mask
  if (lcol.has_nulls() || rcol.has_nulls()) {
    auto merged_view = result->mutable_view();
    materialize_bitmask(
      lcol, rcol, merged_view.null_mask(), merged_view.size(), row_order_.data(), stream);
  }
  return result;
}

// specialization for lists
template <>
std::unique_ptr<column> column_merger::operator()<cudf::list_view>(
  column_view const& lcol,
  column_view const& rcol,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr) const
{
  std::vector<column_view> columns{lcol, rcol};
  auto concatenated_list = cudf::lists::detail::concatenate(columns, stream, mr);

  auto const iter_gather = cudf::detail::make_counting_transform_iterator(
    0,
    cuda::proclaim_return_type<cudf::size_type>(
      [row_order = row_order_.data(), lsize = lcol.size()] __device__(auto const idx) {
        auto const [side, index] = row_order[idx];
        return side == side::LEFT ? index : lsize + index;
      }));

  auto result = cudf::detail::gather(table_view{{concatenated_list->view()}},
                                     iter_gather,
                                     iter_gather + concatenated_list->size(),
                                     out_of_bounds_policy::DONT_CHECK,
                                     stream,
                                     mr);
  return std::move(result->release()[0]);
}

// specialization for structs
template <>
std::unique_ptr<column> column_merger::operator()<cudf::struct_view>(
  column_view const& lcol,
  column_view const& rcol,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr) const
{
  // merge each child.
  auto const lhs = structs_column_view{lcol};
  auto const rhs = structs_column_view{rcol};

  auto it = cudf::detail::make_counting_transform_iterator(
    0, [&, merger = column_merger{row_order_}](size_type i) {
      return cudf::type_dispatcher<dispatch_storage_type>(lhs.child(i).type(),
                                                          merger,
                                                          lhs.get_sliced_child(i, stream),
                                                          rhs.get_sliced_child(i, stream),
                                                          stream,
                                                          mr);
    });

  auto merged_children   = std::vector<std::unique_ptr<column>>(it, it + lhs.num_children());
  auto const merged_size = lcol.size() + rcol.size();

  // materialize the output buffer
  rmm::device_buffer validity =
    lcol.has_nulls() || rcol.has_nulls()
      ? detail::create_null_mask(merged_size, mask_state::UNINITIALIZED, stream, mr)
      : rmm::device_buffer{};
  if (lcol.has_nulls() || rcol.has_nulls()) {
    materialize_bitmask(lcol,
                        rcol,
                        static_cast<bitmask_type*>(validity.data()),
                        merged_size,
                        row_order_.data(),
                        stream);
  }

  return make_structs_column(merged_size,
                             std::move(merged_children),
                             lcol.null_count() + rcol.null_count(),
                             std::move(validity),
                             stream,
                             mr);
}

using table_ptr_type = std::unique_ptr<cudf::table>;

table_ptr_type merge(cudf::table_view const& left_table,
                     cudf::table_view const& right_table,
                     std::vector<cudf::size_type> const& key_cols,
                     std::vector<cudf::order> const& column_order,
                     std::vector<cudf::null_order> const& null_precedence,
                     rmm::cuda_stream_view stream,
                     rmm::device_async_resource_ref mr)
{
  // collect index columns for lhs, rhs, resp.
  //
  cudf::table_view index_left_view{left_table.select(key_cols)};
  cudf::table_view index_right_view{right_table.select(key_cols)};
  bool const nullable = cudf::has_nulls(index_left_view) || cudf::has_nulls(index_right_view);

  // extract merged row order according to indices:
  //
  auto const merged_indices = [&]() {
    if (cudf::detail::has_nested_columns(left_table) or
        cudf::detail::has_nested_columns(right_table)) {
      return generate_merged_indices_nested(
        index_left_view, index_right_view, column_order, null_precedence, nullable, stream);
    } else {
      return generate_merged_indices(
        index_left_view, index_right_view, column_order, null_precedence, nullable, stream);
    }
  }();
  // create merged table:
  //
  auto const n_cols = left_table.num_columns();
  std::vector<std::unique_ptr<column>> merged_cols;
  merged_cols.reserve(n_cols);

  column_merger merger{merged_indices};
  transform(left_table.begin(),
            left_table.end(),
            right_table.begin(),
            std::back_inserter(merged_cols),
            [&](auto const& left_col, auto const& right_col) {
              return cudf::type_dispatcher<dispatch_storage_type>(
                left_col.type(), merger, left_col, right_col, stream, mr);
            });

  return std::make_unique<cudf::table>(std::move(merged_cols));
}

struct merge_queue_item {
  table_view view;
  table_ptr_type table;
  // Priority is a separate member to ensure that moving from an object
  // does not change its priority (which would ruin the queue invariant)
  cudf::size_type priority = 0;

  merge_queue_item(table_view const& view, table_ptr_type&& table)
    : view{view}, table{std::move(table)}, priority{-view.num_rows()}
  {
  }

  bool operator<(merge_queue_item const& other) const { return priority < other.priority; }
};

// Helper function to ensure that moving out of the priority_queue is "atomic"
template <typename T>
T top_and_pop(std::priority_queue<T>& q)
{
  auto moved = std::move(const_cast<T&>(q.top()));
  q.pop();
  return moved;
}

}  // anonymous namespace

table_ptr_type merge(std::vector<table_view> const& tables_to_merge,
                     std::vector<cudf::size_type> const& key_cols,
                     std::vector<cudf::order> const& column_order,
                     std::vector<cudf::null_order> const& null_precedence,
                     rmm::cuda_stream_view stream,
                     rmm::device_async_resource_ref mr)
{
  if (tables_to_merge.empty()) { return std::make_unique<cudf::table>(); }

  auto const& first_table = tables_to_merge.front();
  auto const n_cols       = first_table.num_columns();

  CUDF_EXPECTS(std::all_of(tables_to_merge.cbegin(),
                           tables_to_merge.cend(),
                           [n_cols](auto const& tbl) { return n_cols == tbl.num_columns(); }),
               "Mismatched number of columns");
  CUDF_EXPECTS(
    std::all_of(tables_to_merge.cbegin(),
                tables_to_merge.cend(),
                [&](auto const& tbl) { return cudf::have_same_types(first_table, tbl); }),
    "Mismatched column types");

  CUDF_EXPECTS(!key_cols.empty(), "Empty key_cols");
  CUDF_EXPECTS(key_cols.size() <= static_cast<size_t>(n_cols), "Too many values in key_cols");

  CUDF_EXPECTS(key_cols.size() == column_order.size(),
               "Mismatched size between key_cols and column_order");
  CUDF_EXPECTS(
    std::accumulate(tables_to_merge.cbegin(),
                    tables_to_merge.cend(),
                    std::size_t{0},
                    [](auto const& running_sum, auto const& tbl) {
                      return running_sum + static_cast<std::size_t>(tbl.num_rows());
                    }) <= static_cast<std::size_t>(std::numeric_limits<cudf::size_type>::max()),
    "Total number of merged rows exceeds row limit");

  // This utility will ensure all corresponding dictionary columns have matching keys.
  // It will return any new dictionary columns created as well as updated table_views.
  auto matched = cudf::dictionary::detail::match_dictionaries(
    tables_to_merge, stream, cudf::get_current_device_resource_ref());
  auto merge_tables = matched.second;

  // A queue of (table view, table) pairs
  std::priority_queue<merge_queue_item> merge_queue;
  // The table pointer is null if we do not own the table (input tables)
  std::for_each(merge_tables.begin(), merge_tables.end(), [&](auto const& table) {
    if (table.num_rows() > 0) merge_queue.emplace(table, table_ptr_type());
  });

  // If there is only one non-empty table_view, return its copy
  if (merge_queue.size() == 1) {
    return std::make_unique<cudf::table>(merge_queue.top().view, stream, mr);
  }
  // No inputs have rows, return a table with same columns as the first one
  if (merge_queue.empty()) { return empty_like(first_table); }

  // Pick the two smallest tables and merge them
  // Until there is only one table left in the queue
  while (merge_queue.size() > 1) {
    // To delete the intermediate table at the end of the block
    auto const left_table = top_and_pop(merge_queue);
    // Deallocated at the end of the block
    auto const right_table = top_and_pop(merge_queue);

    // Only use mr for the output table
    auto const& new_tbl_mr = merge_queue.empty() ? mr : cudf::get_current_device_resource_ref();
    auto merged_table      = merge(left_table.view,
                              right_table.view,
                              key_cols,
                              column_order,
                              null_precedence,
                              stream,
                              new_tbl_mr);

    auto const merged_table_view = merged_table->view();
    merge_queue.emplace(merged_table_view, std::move(merged_table));
  }

  return std::move(top_and_pop(merge_queue).table);
}

}  // namespace detail

std::unique_ptr<cudf::table> merge(std::vector<table_view> const& tables_to_merge,
                                   std::vector<cudf::size_type> const& key_cols,
                                   std::vector<cudf::order> const& column_order,
                                   std::vector<cudf::null_order> const& null_precedence,
                                   rmm::cuda_stream_view stream,
                                   rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::merge(tables_to_merge, key_cols, column_order, null_precedence, stream, mr);
}

}  // namespace cudf
