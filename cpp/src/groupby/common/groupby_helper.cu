/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "groupby/common/utils.hpp"
#include "groupby/hash/compute_single_pass_aggs.hpp"
#include "groupby/hash/group_keys.hpp"
#include "groupby_helper_group_offsets.cuh"

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/algorithms/copy_if.cuh>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/gather.hpp>
#include <cudf/detail/groupby/groupby_helper.hpp>
#include <cudf/detail/labeling/label_segments.cuh>
#include <cudf/detail/sorting.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/exec_policy.hpp>

#include <cuda/buffer>
#include <cuda/iterator>
#include <cuda/std/bit>
#include <cuda/std/cstddef>
#include <cuda/std/functional>
#include <cuda/std/iterator>
#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/gather.h>
#include <thrust/scatter.h>
#include <thrust/sequence.h>

#include <cstddef>
#include <utility>

namespace cudf::groupby::detail {

size_type compact_group_offsets(bool* flags,
                                size_type size,
                                size_type* offsets,
                                cuda::stream_ref stream,
                                cudf::memory_resources mr)
{
  auto const begin = cuda::counting_iterator<size_type>{0};
  auto const end =
    cudf::detail::copy_if(begin, begin + size, flags, offsets, cuda::std::identity{}, stream, mr);
  return static_cast<size_type>(end - offsets);
}

namespace {

rmm::device_uvector<size_type> included_rows(table_view const& keys,
                                             null_policy include_null_keys,
                                             cuda::stream_ref stream,
                                             cudf::memory_resources mr)
{
  auto const policy = rmm::exec_policy_nosync(stream, mr.get_temporary_mr());
  auto rows         = rmm::device_uvector<size_type>(keys.num_rows(), stream, mr.get_output_mr());
  auto [row_bitmask_data, row_bitmask] =
    include_null_keys == null_policy::EXCLUDE
      ? compute_row_bitmask(
          keys, stream, cudf::memory_resources{mr.get_temporary_mr(), mr.get_temporary_mr()})
      : std::pair<cuda::device_buffer<std::byte>, bitmask_type const*>{
          cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED, stream, mr), nullptr};
  if (row_bitmask == nullptr) {
    thrust::sequence(policy, rows.begin(), rows.end(), size_type{0});
  } else {
    // Stable compaction retains the input order while dropping null key rows.
    auto const end = thrust::copy_if(
      policy,
      cuda::counting_iterator<size_type>{0},
      cuda::counting_iterator<size_type>{keys.num_rows()},
      rows.begin(),
      [row_bitmask] __device__(size_type row) { return cudf::bit_is_set(row_bitmask, row); });
    rows.resize(cuda::std::distance(rows.begin(), end), stream);
  }
  return rows;
}

void stabilize_rows(hash::grouped_keys& groups,
                    rmm::device_uvector<size_type> const* cached_labels,
                    size_type num_rows,
                    bool keep_labels,
                    cuda::stream_ref stream,
                    cudf::memory_resources mr)
{
  auto const size       = groups.num_grouped_rows;
  auto const num_groups = groups.num_groups;
  // HashCSR determines group membership. Stably sorting those group IDs in input-row order
  // preserves the existing group offsets while restoring input order within each group.
  auto const temp_mr   = mr.get_temporary_mr();
  auto const policy    = rmm::exec_policy_nosync(stream, temp_mr);
  auto const filtered  = size != num_rows;
  auto const labels_mr = keep_labels && cached_labels == nullptr ? mr.get_output_mr() : temp_mr;
  auto input_labels    = rmm::device_uvector<size_type>(num_rows, stream, labels_mr);
  auto ordered_labels  = rmm::device_uvector<size_type>(num_rows, stream, labels_mr);
  auto ordered_rows =
    rmm::device_uvector<size_type>(num_rows, stream, filtered ? temp_mr : mr.get_output_mr());
  if (filtered) {
    auto const initialize_rows = cuda::tabulate_output_iterator{
      [labels = input_labels.data(), rows = ordered_rows.data(), num_groups] __device__(
        cuda::std::ptrdiff_t index, size_type row) {
        labels[index] = num_groups;
        rows[index]   = row;
      }};
    thrust::sequence(policy, initialize_rows, initialize_rows + num_rows, size_type{0});
  }

  // Write into separate storage while the original CSR permutation is still being read.
  auto const write_label = [labels = input_labels.data(),
                            rows   = ordered_rows.data(),
                            filtered] __device__(cuda::std::ptrdiff_t row, size_type label) {
    labels[row] = label;
    if (!filtered) { rows[row] = static_cast<size_type>(row); }
  };
  auto const label_output = cuda::tabulate_output_iterator{
    [grouped_rows = groups.grouped_rows.data(), write_label] __device__(
      cuda::std::ptrdiff_t position, size_type label) {
      write_label(grouped_rows[position], label);
    }};
  if (cached_labels) {
    thrust::copy(policy, cached_labels->begin(), cached_labels->end(), label_output);
  } else {
    // A CSR position's group ID is the number of segment ends at or before that position.
    // Compute it directly instead of materializing labels that only this ordering needs.
    auto const& offsets = groups.group_offsets;
    thrust::upper_bound(policy,
                        offsets.begin() + 1,
                        offsets.end(),
                        cuda::counting_iterator<size_type>{0},
                        cuda::counting_iterator<size_type>{size},
                        label_output);
  }

  // The unfiltered CSR buffer is now free to serve as the alternate sorting buffer. With
  // exclusions, retain its exact-sized allocation for the sorted prefix instead.
  auto alternate_rows = rmm::device_uvector<size_type>(filtered ? num_rows : 0, stream, temp_mr);
  auto* alternate     = filtered ? alternate_rows.data() : groups.grouped_rows.data();
  // Only excluded rows use the sentinel group ID, which sorts after every included group.
  auto const end_bit =
    cuda::std::bit_width(static_cast<uint32_t>(filtered ? num_groups : num_groups - 1));
  auto const sorted = hash::stable_sort_group_rows({input_labels.data(), ordered_rows.data()},
                                                   {ordered_labels.data(), alternate},
                                                   num_rows,
                                                   end_bit,
                                                   stream,
                                                   mr);
  if (filtered) {
    thrust::copy_n(policy, sorted.rows, size, groups.grouped_rows.begin());
  } else if (sorted.rows == ordered_rows.data()) {
    groups.grouped_rows = std::move(ordered_rows);
  }
  // The sorted labels are the group of each grouped row; keep them for group_labels().
  if (keep_labels && cached_labels == nullptr) {
    groups.group_labels =
      sorted.labels == input_labels.data() ? std::move(input_labels) : std::move(ordered_labels);
    groups.group_labels.resize(size, stream);
    groups.group_labels.shrink_to_fit(stream);
  }
}

}  // namespace

groupby_helper::groupby_helper(table_view const& keys,
                               null_policy include_null_keys,
                               sorted keys_pre_sorted,
                               cudf::memory_resources mr)
  : _mr{mr.get_temporary_mr()},
    _keys{keys},
    _keys_pre_sorted{keys_pre_sorted},
    _include_null_keys{include_null_keys},
    _is_presorted{keys_pre_sorted == sorted::YES &&
                  (include_null_keys == null_policy::INCLUDE || !has_nulls(keys))}
{
}

groupby_helper::~groupby_helper() = default;

void groupby_helper::build_groups(bool stable_rows,
                                  bool keep_labels,
                                  bool need_grouped_rows,
                                  cuda::stream_ref stream,
                                  cudf::memory_resources mr)
{
  if (_groups) {
    if (!need_grouped_rows ||
        _groups->grouped_rows.size() == static_cast<std::size_t>(_groups->num_grouped_rows)) {
      return;
    }
    // A previous counts-only aggregation needed no row permutation. Build it only when a later
    // operation needs rows; no result cache is shared across aggregation calls.
    _groups.reset();
    _stable = false;
  }
  if (_keys_pre_sorted == sorted::NO && _keys.num_rows() != 0) {
    _groups = std::make_unique<hash::grouped_keys>(
      hash::group_keys(_keys,
                       _include_null_keys,
                       true,
                       need_grouped_rows,
                       stable_rows,
                       keep_labels,
                       stream,
                       cudf::memory_resources{_mr, mr.get_temporary_mr()}));
    _stable = stable_rows;
    return;
  }

  // Sorted keys already form contiguous groups. Null exclusion only needs stable compaction,
  // after which adjacent equality identifies boundaries without sorting or hashing.
  auto rows = included_rows(
    _keys, _include_null_keys, stream, cudf::memory_resources{_mr, mr.get_temporary_mr()});
  auto const size = static_cast<size_type>(rows.size());
  auto offsets    = rmm::device_uvector<size_type>(static_cast<std::size_t>(size) + 1, stream, _mr);
  auto const num_groups =
    size == 0 ? size_type{0}
    : cudf::detail::has_nested_columns(_keys)
      ? compute_nested_group_offsets(_keys, rows.data(), size, offsets, stream, mr)
      : compute_group_offsets<false>(_keys, rows.data(), size, offsets, stream, mr);
  offsets.set_element_async(num_groups, size, stream);
  offsets.resize(static_cast<std::size_t>(num_groups) + 1, stream);
  // Without filtered rows, each group offset already identifies its representative input row.
  auto key_rows = rmm::device_uvector<size_type>(_is_presorted ? 0 : num_groups, stream, _mr);
  if (!_is_presorted) {
    thrust::gather(rmm::exec_policy_nosync(stream, mr.get_temporary_mr()),
                   offsets.begin(),
                   offsets.begin() + num_groups,
                   rows.begin(),
                   key_rows.begin());
  }
  _groups = std::make_unique<hash::grouped_keys>(
    hash::grouped_keys{num_groups,
                       size,
                       std::move(key_rows),
                       std::move(offsets),
                       std::move(rows),
                       rmm::device_uvector<size_type>{0, stream, _mr}});
  _stable = true;
}

void groupby_helper::make_stable(bool keep_labels,
                                 cuda::stream_ref stream,
                                 cudf::memory_resources mr)
{
  build_groups(true, keep_labels, true, stream, mr);
  if (_stable) { return; }
  // Scheduling contains spans into the current row buffer and hints derived from its order.
  _reduction_groups.reset();
  auto const size       = _groups->num_grouped_rows;
  auto const num_groups = _groups->num_groups;
  if (num_groups == 1) {
    // A single group is the stable compaction of all included input rows.
    _groups->grouped_rows = included_rows(
      _keys, _include_null_keys, stream, cudf::memory_resources{_mr, mr.get_temporary_mr()});
  } else if (num_groups > 1 && num_groups < size) {
    stabilize_rows(*_groups,
                   _group_labels.get(),
                   _keys.num_rows(),
                   keep_labels,
                   stream,
                   cudf::memory_resources{_mr, mr.get_temporary_mr()});
  }
  _stable = true;
}

size_type groupby_helper::num_keys(cuda::stream_ref stream, cudf::memory_resources mr)
{
  build_groups(false, false, false, stream, mr);
  return _groups->num_grouped_rows;
}

column_view groupby_helper::grouped_order(bool keep_labels,
                                          cuda::stream_ref stream,
                                          cudf::memory_resources mr)
{
  make_stable(keep_labels, stream, mr);
  return column_view(device_span<size_type const>{_groups->grouped_rows});
}

device_span<size_type const> groupby_helper::unordered_grouped_order(cuda::stream_ref stream,
                                                                     cudf::memory_resources mr)
{
  build_groups(false, false, true, stream, mr);
  return _groups->grouped_rows;
}

hash::group_reduction_plan const& groupby_helper::reduction_groups(cuda::stream_ref stream,
                                                                   cudf::memory_resources mr)
{
  if (!_reduction_groups) {
    build_groups(false, false, true, stream, mr);
    _reduction_groups = std::make_unique<hash::group_reduction_plan>(
      hash::make_group_reduction_plan(_groups->grouped_rows,
                                      _groups->group_offsets,
                                      stream,
                                      cudf::memory_resources{_mr, mr.get_temporary_mr()}));
  }
  return *_reduction_groups;
}

rmm::device_uvector<size_type> const& groupby_helper::group_offsets(cuda::stream_ref stream,
                                                                    cudf::memory_resources mr)
{
  build_groups(false, false, false, stream, mr);
  return _groups->group_offsets;
}

rmm::device_uvector<size_type> const& groupby_helper::group_labels(cuda::stream_ref stream,
                                                                   cudf::memory_resources mr)
{
  // Labels must use the final grouping order before any row consumer can observe them.
  build_groups(false, false, true, stream, mr);
  if (_group_labels) { return *_group_labels; }
  if (num_keys(stream, mr) != 0 && !_groups->group_labels.is_empty()) {
    // The stable build sorted the labels along with the rows.
    _group_labels =
      std::make_unique<rmm::device_uvector<size_type>>(std::move(_groups->group_labels));
    return *_group_labels;
  }
  auto labels = std::make_unique<rmm::device_uvector<size_type>>(num_keys(stream, mr), stream, _mr);
  if (!labels->is_empty()) {
    auto const& offsets = group_offsets(stream, mr);
    cudf::detail::label_segments(offsets.begin(),
                                 offsets.end(),
                                 labels->begin(),
                                 labels->end(),
                                 stream,
                                 cudf::memory_resources{_mr, mr.get_temporary_mr()});
  }
  _group_labels = std::move(labels);
  return *_group_labels;
}

rmm::device_uvector<size_type> const& groupby_helper::input_labels(cuda::stream_ref stream,
                                                                   cudf::memory_resources mr)
{
  if (_input_labels) { return *_input_labels; }
  auto const policy  = rmm::exec_policy_nosync(stream, mr.get_temporary_mr());
  auto const& labels = group_labels(stream, mr);
  auto input = std::make_unique<rmm::device_uvector<size_type>>(_keys.num_rows(), stream, _mr);
  // Excluded rows take the label past the last group so that they sort after every group.
  if (labels.size() != input->size()) {
    thrust::fill(policy, input->begin(), input->end(), num_groups(stream, mr));
  }
  // Labels are constant within each group, so the unordered CSR permutation is sufficient.
  thrust::scatter(
    policy, labels.begin(), labels.end(), _groups->grouped_rows.begin(), input->begin());
  _input_labels = std::move(input);
  return *_input_labels;
}

std::unique_ptr<column> groupby_helper::sorted_values(column_view const& values,
                                                      cuda::stream_ref stream,
                                                      cudf::memory_resources mr)
{
  // Group labels avoid sorting the original key columns. Starting with input rows also preserves
  // input order among equal values without materializing the stable CSR row permutation first.
  auto const labels = column_view(device_span<size_type const>{input_labels(stream, mr)});
  auto order        = cudf::detail::stable_sorted_order(table_view{{labels, values}},
                                                        {},
                                                 std::vector<null_order>(2, null_order::AFTER),
                                                 stream,
                                                 mr.get_temporary_mr());
  auto rows         = cudf::detail::slice(order->view(), 0, num_keys(stream, mr), stream);
  auto result       = cudf::detail::gather(table_view{{values}},
                                     rows,
                                     out_of_bounds_policy::DONT_CHECK,
                                     negative_index_policy::NOT_ALLOWED,
                                     stream,
                                     mr);
  return std::move(result->release()[0]);
}

std::unique_ptr<column> groupby_helper::grouped_values(column_view const& values,
                                                       cuda::stream_ref stream,
                                                       cudf::memory_resources mr)
{
  auto result = cudf::detail::gather(table_view{{values}},
                                     grouped_order(false, stream, mr),
                                     out_of_bounds_policy::DONT_CHECK,
                                     negative_index_policy::NOT_ALLOWED,
                                     stream,
                                     mr);
  return std::move(result->release()[0]);
}

std::unique_ptr<column> groupby_helper::unordered_grouped_values(column_view const& values,
                                                                 cuda::stream_ref stream,
                                                                 cudf::memory_resources mr)
{
  build_groups(false, false, true, stream, mr);
  auto result = cudf::detail::gather(table_view{{values}},
                                     _groups->grouped_rows,
                                     out_of_bounds_policy::DONT_CHECK,
                                     negative_index_policy::NOT_ALLOWED,
                                     stream,
                                     mr);
  return std::move(result->release()[0]);
}

std::unique_ptr<table> groupby_helper::distinct_keys(cuda::stream_ref stream,
                                                     cudf::memory_resources mr)
{
  std::unique_ptr<hash::grouped_keys> key_groups;
  if (!_groups && _keys_pre_sorted == sorted::NO && _keys.num_rows() != 0) {
    // Empty aggregation requests need only representatives. Keep this metadata local so a
    // later operation can populate the helper with complete grouped rows.
    auto const temp_mr = mr.get_temporary_mr();
    key_groups         = std::make_unique<hash::grouped_keys>(
      hash::group_keys(_keys,
                       _include_null_keys,
                       false,
                       false,
                       false,
                       false,
                       stream,
                       cudf::memory_resources{temp_mr, temp_mr}));
  } else {
    build_groups(false, false, false, stream, mr);
  }
  auto const& groups = key_groups ? *key_groups : *_groups;
  auto const key_rows =
    groups.key_rows.is_empty()
      ? device_span<size_type const>{groups.group_offsets.data(),
                                     static_cast<std::size_t>(groups.num_groups)}
      : device_span<size_type const>{groups.key_rows};
  return cudf::detail::gather(_keys,
                              key_rows,
                              out_of_bounds_policy::DONT_CHECK,
                              negative_index_policy::NOT_ALLOWED,
                              stream,
                              mr);
}

std::unique_ptr<table> groupby_helper::grouped_keys(cuda::stream_ref stream,
                                                    cudf::memory_resources mr)
{
  return cudf::detail::gather(_keys,
                              grouped_order(false, stream, mr),
                              out_of_bounds_policy::DONT_CHECK,
                              negative_index_policy::NOT_ALLOWED,
                              stream,
                              mr);
}

}  // namespace cudf::groupby::detail
