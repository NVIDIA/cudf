/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "groupby/common/utils.hpp"
#include "groupby/hash/compute_single_pass_aggs.hpp"
#include "groupby/hash/group_keys.hpp"
#include "groupby_helper_group_offsets.cuh"

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/gather.hpp>
#include <cudf/detail/groupby/groupby_helper.hpp>
#include <cudf/detail/labeling/label_segments.cuh>
#include <cudf/detail/scatter.hpp>
#include <cudf/detail/sorting.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/exec_policy.hpp>

#include <cub/device/device_radix_sort.cuh>
#include <cuda/iterator>
#include <cuda/std/bit>
#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/gather.h>
#include <thrust/sequence.h>

#include <utility>

namespace cudf::groupby::detail {

namespace {

rmm::device_uvector<size_type> included_rows(table_view const& keys,
                                             null_policy include_null_keys,
                                             cuda::stream_ref stream)
{
  auto const mr     = cudf::get_current_device_resource_ref();
  auto const policy = rmm::exec_policy_nosync(stream, mr);
  auto rows         = rmm::device_uvector<size_type>(keys.num_rows(), stream, mr);
  auto [row_bitmask_data, row_bitmask] =
    include_null_keys == null_policy::EXCLUDE
      ? compute_row_bitmask(keys, stream, cudf::memory_resources{mr, mr})
      : std::pair<rmm::device_buffer, bitmask_type const*>{rmm::device_buffer{0, stream, mr},
                                                           nullptr};
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
                    groupby_helper::index_vector const* cached_labels,
                    size_type num_rows,
                    cuda::stream_ref stream)
{
  using index_vector    = groupby_helper::index_vector;
  auto const size       = groups.num_grouped_rows;
  auto const num_groups = groups.num_groups;
  // HashCSR determines group membership. Stably sorting those group IDs in input-row order
  // preserves the existing group offsets while restoring input order within each group.
  auto const mr       = cudf::get_current_device_resource_ref();
  auto const policy   = rmm::exec_policy_nosync(stream, mr);
  auto const filtered = size != num_rows;
  auto input_labels   = index_vector(num_rows, stream, mr);
  auto ordered_labels = index_vector(num_rows, stream, mr);
  auto ordered_rows   = index_vector(num_rows, stream, mr);
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
  auto alternate_rows = index_vector(filtered ? num_rows : 0, stream, mr);
  auto* alternate     = filtered ? alternate_rows.data() : groups.grouped_rows.data();
  auto group_ids      = cub::DoubleBuffer<size_type>{input_labels.data(), ordered_labels.data()};
  auto rows           = cub::DoubleBuffer<size_type>{ordered_rows.data(), alternate};
  // Only excluded rows use the sentinel group ID, which sorts after every included group.
  auto const end_bit =
    cuda::std::bit_width(static_cast<uint32_t>(filtered ? num_groups : num_groups - 1));
  std::size_t temp_bytes = 0;
  CUDF_CUDA_TRY(cub::DeviceRadixSort::SortPairs(
    nullptr, temp_bytes, group_ids, rows, num_rows, 0, end_bit, stream.get()));
  auto temp = rmm::device_buffer(temp_bytes, stream, mr);
  CUDF_CUDA_TRY(cub::DeviceRadixSort::SortPairs(
    temp.data(), temp_bytes, group_ids, rows, num_rows, 0, end_bit, stream.get()));
  if (filtered) {
    thrust::copy_n(policy, rows.Current(), size, groups.grouped_rows.begin());
  } else if (rows.Current() == ordered_rows.data()) {
    groups.grouped_rows = std::move(ordered_rows);
  }
}

}  // namespace

groupby_helper::groupby_helper(table_view const& keys,
                               null_policy include_null_keys,
                               sorted keys_pre_sorted)
  : _keys{keys},
    _keys_pre_sorted{keys_pre_sorted},
    _include_null_keys{include_null_keys},
    _is_presorted{keys_pre_sorted == sorted::YES &&
                  (include_null_keys == null_policy::INCLUDE || !has_nulls(keys))}
{
}

groupby_helper::~groupby_helper()                                    = default;
groupby_helper::groupby_helper(groupby_helper&&) noexcept            = default;
groupby_helper& groupby_helper::operator=(groupby_helper&&) noexcept = default;

void groupby_helper::build_groups(cuda::stream_ref stream, bool stable_rows)
{
  if (_groups) { return; }
  auto const mr = cudf::get_current_device_resource_ref();
  if (_keys_pre_sorted == sorted::NO && _keys.num_rows() != 0) {
    _groups = std::make_unique<hash::grouped_keys>(hash::group_keys(
      _keys, _include_null_keys, true, stream, cudf::memory_resources{mr, mr}, stable_rows));
    _stable = stable_rows;
    return;
  }

  // Sorted keys already form contiguous groups. Null exclusion only needs stable compaction,
  // after which adjacent equality identifies boundaries without sorting or hashing.
  auto rows       = included_rows(_keys, _include_null_keys, stream);
  auto const size = static_cast<size_type>(rows.size());
  auto offsets    = index_vector(static_cast<std::size_t>(size) + 1, stream, mr);
  auto const num_groups =
    size == 0 ? size_type{0}
    : cudf::detail::has_nested_columns(_keys)
      ? compute_nested_group_offsets(_keys, rows.data(), size, offsets, stream)
      : compute_group_offsets<false>(_keys, rows.data(), size, offsets, stream);
  offsets.set_element_async(num_groups, size, stream);
  offsets.resize(static_cast<std::size_t>(num_groups) + 1, stream);
  auto key_rows = index_vector(num_groups, stream, mr);
  thrust::gather(rmm::exec_policy_nosync(stream, mr),
                 offsets.begin(),
                 offsets.begin() + num_groups,
                 rows.begin(),
                 key_rows.begin());
  _groups = std::make_unique<hash::grouped_keys>(
    hash::grouped_keys{num_groups, size, std::move(key_rows), std::move(offsets), std::move(rows)});
  _stable = true;
}

void groupby_helper::make_stable(cuda::stream_ref stream)
{
  build_groups(stream, true);
  if (_stable) { return; }
  // Scheduling contains spans into the current row buffer and hints derived from its order.
  _reduction_groups.reset();
  auto const size       = _groups->num_grouped_rows;
  auto const num_groups = _groups->num_groups;
  if (num_groups == 1) {
    // A single group is the stable compaction of all included input rows.
    _groups->grouped_rows = included_rows(_keys, _include_null_keys, stream);
  } else if (num_groups > 1 && num_groups < size) {
    stabilize_rows(*_groups, _group_labels.get(), _keys.num_rows(), stream);
  }
  _stable = true;
}

size_type groupby_helper::num_keys(cuda::stream_ref stream)
{
  build_groups(stream);
  return _groups->num_grouped_rows;
}

column_view groupby_helper::grouped_order(cuda::stream_ref stream)
{
  make_stable(stream);
  return column_view(device_span<size_type const>{_groups->grouped_rows});
}

device_span<size_type const> groupby_helper::unordered_grouped_order(cuda::stream_ref stream)
{
  build_groups(stream);
  return _groups->grouped_rows;
}

hash::grouped_rows const& groupby_helper::reduction_groups(cuda::stream_ref stream)
{
  if (!_reduction_groups) {
    build_groups(stream);
    auto const mr     = cudf::get_current_device_resource_ref();
    _reduction_groups = std::make_unique<hash::grouped_rows>(hash::make_grouped_rows(
      _groups->grouped_rows, _groups->group_offsets, stream, cudf::memory_resources{mr, mr}));
  }
  return *_reduction_groups;
}

groupby_helper::index_vector const& groupby_helper::group_offsets(cuda::stream_ref stream)
{
  build_groups(stream);
  return _groups->group_offsets;
}

groupby_helper::index_vector const& groupby_helper::group_labels(cuda::stream_ref stream)
{
  if (_group_labels) { return *_group_labels; }
  auto labels = std::make_unique<index_vector>(num_keys(stream), stream);
  if (!labels->is_empty()) {
    auto const& offsets = group_offsets(stream);
    cudf::detail::label_segments(
      offsets.begin(), offsets.end(), labels->begin(), labels->end(), stream);
  }
  _group_labels = std::move(labels);
  return *_group_labels;
}

column_view groupby_helper::ungrouped_keys_labels(cuda::stream_ref stream)
{
  if (_unsorted_keys_labels) { return _unsorted_keys_labels->view(); }
  auto const mr     = cudf::get_current_device_resource_ref();
  auto empty_labels = make_numeric_column(
    data_type{type_to_id<size_type>()}, _keys.num_rows(), mask_state::ALL_NULL, stream, mr);
  auto labels = column_view(device_span<size_type const>{group_labels(stream)});
  // Labels are constant within each group, so the unordered CSR permutation is sufficient.
  auto rows      = column_view(device_span<size_type const>{_groups->grouped_rows});
  auto scattered = cudf::detail::scatter(
    table_view{{labels}}, rows, table_view{{empty_labels->view()}}, stream, mr);
  _unsorted_keys_labels = std::move(scattered->release()[0]);
  return _unsorted_keys_labels->view();
}

groupby_helper::column_ptr groupby_helper::sorted_values(column_view const& values,
                                                         cuda::stream_ref stream,
                                                         rmm::device_async_resource_ref mr)
{
  // Group labels avoid sorting the original key columns. Starting with input rows also preserves
  // input order among equal values without materializing the stable CSR row permutation first.
  auto order =
    cudf::detail::stable_sorted_order(table_view{{ungrouped_keys_labels(stream), values}},
                                      {},
                                      std::vector<null_order>(2, null_order::AFTER),
                                      stream,
                                      mr);
  auto rows   = cudf::detail::slice(order->view(), 0, num_keys(stream), stream);
  auto result = cudf::detail::gather(table_view{{values}},
                                     rows,
                                     out_of_bounds_policy::DONT_CHECK,
                                     negative_index_policy::NOT_ALLOWED,
                                     stream,
                                     mr);
  return std::move(result->release()[0]);
}

groupby_helper::column_ptr groupby_helper::grouped_values(column_view const& values,
                                                          cuda::stream_ref stream,
                                                          rmm::device_async_resource_ref mr)
{
  auto result = cudf::detail::gather(table_view{{values}},
                                     grouped_order(stream),
                                     out_of_bounds_policy::DONT_CHECK,
                                     negative_index_policy::NOT_ALLOWED,
                                     stream,
                                     mr);
  return std::move(result->release()[0]);
}

groupby_helper::column_ptr groupby_helper::unordered_grouped_values(
  column_view const& values, cuda::stream_ref stream, rmm::device_async_resource_ref mr)
{
  build_groups(stream);
  auto result = cudf::detail::gather(table_view{{values}},
                                     _groups->grouped_rows,
                                     out_of_bounds_policy::DONT_CHECK,
                                     negative_index_policy::NOT_ALLOWED,
                                     stream,
                                     mr);
  return std::move(result->release()[0]);
}

std::unique_ptr<table> groupby_helper::unique_keys(cuda::stream_ref stream,
                                                   rmm::device_async_resource_ref mr)
{
  std::unique_ptr<hash::grouped_keys> key_groups;
  if (!_groups && _keys_pre_sorted == sorted::NO && _keys.num_rows() != 0) {
    // Empty aggregation requests need only representatives. Keep this metadata local so a
    // later operation can populate the helper with complete grouped rows.
    auto const temp_mr = cudf::get_current_device_resource_ref();
    key_groups         = std::make_unique<hash::grouped_keys>(hash::group_keys(
      _keys, _include_null_keys, false, stream, cudf::memory_resources{temp_mr, temp_mr}));
  } else {
    build_groups(stream);
  }
  auto const& groups = key_groups ? *key_groups : *_groups;
  return cudf::detail::gather(_keys,
                              groups.key_rows,
                              out_of_bounds_policy::DONT_CHECK,
                              negative_index_policy::NOT_ALLOWED,
                              stream,
                              mr);
}

std::unique_ptr<table> groupby_helper::grouped_keys(cuda::stream_ref stream,
                                                    rmm::device_async_resource_ref mr)
{
  return cudf::detail::gather(_keys,
                              grouped_order(stream),
                              out_of_bounds_policy::DONT_CHECK,
                              negative_index_policy::NOT_ALLOWED,
                              stream,
                              mr);
}

}  // namespace cudf::groupby::detail
