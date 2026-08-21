/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "reader_impl_helpers.hpp"
#include "synthetic_column_helpers.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/labeling/label_segments.cuh>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cub/device/device_transform.cuh>
#include <cuda/iterator>
#include <thrust/binary_search.h>
#include <thrust/scan.h>

#include <numeric>
#include <stdexcept>
#include <utility>

namespace cudf::io::parquet::detail {

namespace {

/**
 * @brief Maps each global row index to its corresponding file-local row index
 */
struct map_global_to_local_row_index {
  std::size_t const* global_row_offsets;  ///< Global row offsets for each row group
  std::size_t const* local_row_offsets;   ///< Source-local start row for each row group
  std::size_t num_row_groups;

  __device__ std::size_t operator()(std::size_t row_idx) const noexcept
  {
    auto const row_group_idx =
      cuda::std::distance(
        global_row_offsets,
        thrust::upper_bound(
          thrust::seq, global_row_offsets, global_row_offsets + num_row_groups, row_idx)) -
      1;  // Subtract 1 to get the index of the selected row group
    return row_idx - global_row_offsets[row_group_idx] + local_row_offsets[row_group_idx];
  }
};

}  // namespace

std::unique_ptr<cudf::column> synthesize_row_index_column(
  std::span<row_group_info const> row_groups,
  row_range const& read_info,
  cuda::stream_ref stream,
  cudf::memory_resources mr)
{
  using column_type = size_t;

  if (read_info.num_rows == 0) {
    return cudf::make_empty_column(cudf::data_type{cudf::type_to_id<column_type>()});
  }

  // Allocate column data vector
  auto col_data = rmm::device_uvector<column_type>(read_info.num_rows, stream, mr.get_output_mr());

  // Map global row indices in the current row-range to corresponding source-local row indices
  {
    // Collect global and file-local start rows for each selected row group
    auto host_rg_global_offsets =
      cudf::detail::make_empty_pinned_vector<std::size_t>(row_groups.size(), stream);
    auto host_rg_local_offsets =
      cudf::detail::make_empty_pinned_vector<size_t>(row_groups.size(), stream);
    for (auto const& rg : row_groups) {
      host_rg_global_offsets.push_back(rg.start_row);
      host_rg_local_offsets.push_back(rg.source_start_row);
    }

    // Copy to device
    auto const rg_global_offsets = cudf::detail::make_device_uvector_async(
      host_rg_global_offsets, stream, mr.get_temporary_mr());
    auto const rg_local_offsets =
      cudf::detail::make_device_uvector_async(host_rg_local_offsets, stream, mr.get_temporary_mr());

    // For each output row, binary search its row group and compute the (file-local) row index
    CUDF_CUDA_TRY(cub::DeviceTransform::Transform(
      cuda::counting_iterator<std::size_t>(read_info.skip_rows),
      col_data.begin(),
      read_info.num_rows,
      map_global_to_local_row_index{
        rg_global_offsets.data(), rg_local_offsets.data(), rg_global_offsets.size()},
      stream.get()));
    stream.sync();
  }

  return std::make_unique<cudf::column>(
    std::move(col_data), rmm::device_buffer{0, stream, mr.get_output_mr()}, 0);
}

std::unique_ptr<cudf::column> synthesize_source_index_column(
  std::span<std::size_t const> num_rows_per_source,
  cuda::stream_ref stream,
  cudf::memory_resources mr)
{
  using column_type = cudf::size_type;

  auto const num_sources = num_rows_per_source.size();
  auto const num_rows =
    std::accumulate(num_rows_per_source.begin(), num_rows_per_source.end(), std::size_t{0});

  if (num_rows == 0) {
    return cudf::make_empty_column(cudf::data_type{cudf::type_to_id<column_type>()});
  }

  // Single source
  if (num_sources == 1) {
    auto const scalar = cudf::numeric_scalar<column_type>(0, true, stream, mr.get_temporary_mr());
    return cudf::make_column_from_scalar(scalar, num_rows, stream, mr.get_output_mr());
  }

  // Allocate column data vector
  auto col_data = rmm::device_uvector<column_type>(num_rows, stream, mr.get_output_mr());

  // Label each output row with its source index via segment boundaries.
  {
    // Host per-source row offsets, including the final total row count.
    auto host_row_offsets =
      cudf::detail::make_empty_pinned_vector<cudf::size_type>(num_sources + 1, stream);
    host_row_offsets.resize(num_sources + 1);
    host_row_offsets.front() = cudf::size_type{0};
    std::inclusive_scan(
      num_rows_per_source.begin(), num_rows_per_source.end(), host_row_offsets.begin() + 1);
    auto const row_offsets =
      cudf::detail::make_device_uvector_async(host_row_offsets, stream, mr.get_temporary_mr());
    cudf::detail::label_segments(
      row_offsets.begin(), row_offsets.end(), col_data.begin(), col_data.end(), stream);
    stream.sync();
  }

  return std::make_unique<cudf::column>(
    std::move(col_data), rmm::device_buffer{0, stream, mr.get_output_mr()}, 0);
}

std::unique_ptr<cudf::column> synthesize_row_group_index_column(
  cudf::column_view const& source_indices, cuda::stream_ref stream, cudf::memory_resources mr)
{
  using column_type = cudf::size_type;

  CUDF_EXPECTS(source_indices.type().id() == type_id::INT32,
               "Source index column must have INT32 type",
               std::invalid_argument);
  CUDF_EXPECTS(source_indices.null_count() == 0,
               "Source index column must not contain null values",
               std::invalid_argument);

  if (source_indices.is_empty()) {
    return cudf::make_empty_column(cudf::data_type{cudf::type_to_id<column_type>()});
  }

  auto const output_type = data_type{cudf::type_to_id<column_type>()};
  auto output            = cudf::make_fixed_width_column(
    output_type, source_indices.size(), mask_state::UNALLOCATED, stream, mr.get_output_mr());
  auto output_view = output->mutable_view();
  thrust::exclusive_scan_by_key(rmm::exec_policy_nosync(stream, mr.get_temporary_mr()),
                                source_indices.begin<column_type>(),
                                source_indices.end<column_type>(),
                                cuda::make_constant_iterator(column_type{1}),
                                output_view.begin<column_type>(),
                                column_type{0});
  return output;
}

}  // namespace cudf::io::parquet::detail
