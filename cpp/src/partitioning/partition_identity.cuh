// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/gather.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/offsets_iterator_factory.cuh>
#include <cudf/detail/utilities/buffer_factories.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/strings/detail/gather.cuh>
#include <cudf/strings/detail/utilities.hpp>
#include <cudf/strings/utilities.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/traits.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/iterator>
#include <thrust/transform.h>

namespace cudf::detail::partition_identity {
struct string_source {
  void const* offsets;
  size_type row_offset;
  bool wide;
};
struct string_bounds {
  int64_t first;
  int64_t bytes;
};

template <int BlockSize>
CUDF_KERNEL void read_bounds(string_source const* sources,
                             size_type count,
                             size_type rows,
                             string_bounds* output)
{
  auto const i = static_cast<std::size_t>(blockIdx.x) * BlockSize + threadIdx.x;
  if (i >= static_cast<std::size_t>(count)) return;
  auto const source = sources[i];
  auto const first  = source.wide ? static_cast<int64_t const*>(source.offsets)[source.row_offset]
                                  : static_cast<size_type const*>(source.offsets)[source.row_offset];
  auto const last   = source.wide
                        ? static_cast<int64_t const*>(source.offsets)[source.row_offset + rows]
                        : static_cast<size_type const*>(source.offsets)[source.row_offset + rows];
  output[i]         = {first, last - first};
}

/** Cached host metadata proves the payload is invariant under a permutation. */
inline bool is_uniform(column_view const& input)
{
  bool const flat = is_fixed_width(input.type()) || input.type().id() == type_id::STRING;
  return flat && (input.null_count() == input.size() ||
                  (input.type().id() == type_id::STRING && input.null_count() == 0 &&
                   input.head<char>() == nullptr));
}

inline std::unique_ptr<column> uniform_column(column_view const& input,
                                              cuda::stream_ref stream,
                                              rmm::device_async_resource_ref mr)
{
  auto const rows = input.size();
  if (input.type().id() != type_id::STRING)
    return make_fixed_width_column(input.type(), rows, mask_state::ALL_NULL, stream, mr);
  auto offsets =
    make_numeric_column(data_type{type_id::INT32}, rows + 1, mask_state::UNALLOCATED, stream, mr);
  CUDF_CUDA_TRY(cudaMemsetAsync(offsets->mutable_view().data<size_type>(),
                                0,
                                (static_cast<std::size_t>(rows) + 1) * sizeof(size_type),
                                stream.get()));
  return make_strings_column(
    rows,
    std::move(offsets),
    rmm::device_buffer(0, stream, mr),
    input.null_count(),
    cudf::detail::create_null_mask(
      rows, input.null_count() ? mask_state::ALL_NULL : mask_state::UNALLOCATED, stream, mr));
}

/**
 * Identity materialization for CPU-known one-row/one-partition cases.
 *
 * String byte bounds are read only for output allocation and offset rebasing.
 * They never choose the partition algorithm. Nullable strings and nested values
 * use the normal gather with an implicit identity iterator; no N-entry map.
 */
inline std::unique_ptr<table> copy(table_view const& input,
                                   cuda::stream_ref stream,
                                   rmm::device_async_resource_ref mr)
{
  auto const rows = input.num_rows();
  std::vector<string_source> sources;
  for (auto const& col : input)
    if (col.type().id() == type_id::STRING && col.null_count() == 0) {
      auto const& offsets = col.child(strings_column_view::offsets_column_index);
      bool const wide     = offsets.type().id() == type_id::INT64;
      auto const* data    = wide ? static_cast<void const*>(offsets.data<int64_t>())
                                 : static_cast<void const*>(offsets.data<size_type>());
      sources.push_back({data, col.offset(), wide});
    }
  auto bounds = [&] {
    if (sources.empty()) return std::vector<string_bounds>{};
    auto descriptors =
      cudf::detail::make_device_buffer(sources, stream, cudf::get_current_device_resource_ref());
    rmm::device_uvector<string_bounds> device_bounds(sources.size(), stream);
    read_bounds<256><<<(sources.size() + 255) / 256, 256, 0, stream.get()>>>(
      descriptors.data(), sources.size(), rows, device_bounds.data());
    CUDF_CUDA_TRY(cudaPeekAtLastError());
    return cudf::detail::make_std_vector(device_bounds, stream);
  }();
  auto const threshold = strings::get_offset64_threshold();
  for (auto const& b : bounds)
    CUDF_EXPECTS(b.bytes < threshold || strings::is_large_strings_enabled(),
                 "Size of output exceeds the column size limit",
                 std::overflow_error);

  std::vector<std::unique_ptr<column>> columns;
  columns.reserve(input.num_columns());
  std::size_t string_index = 0;
  auto const indices       = cuda::counting_iterator<size_type>{0};
  for (auto const& col : input) {
    if (is_uniform(col)) {
      if (col.type().id() == type_id::STRING && col.null_count() == 0) { ++string_index; }
      columns.push_back(uniform_column(col, stream, mr));
      continue;
    }
    if (is_fixed_width(col.type())) {
      auto const width = cudf::size_of(col.type());
      rmm::device_buffer data(col.head<char>() + static_cast<std::size_t>(col.offset()) * width,
                              static_cast<std::size_t>(rows) * width,
                              stream,
                              mr);
      auto mask = col.null_count() > 0
                    ? cudf::detail::copy_bitmask(col, stream, mr)
                    : cudf::detail::create_null_mask(0, mask_state::UNALLOCATED, stream, mr);
      columns.push_back(std::make_unique<column>(
        col.type(), rows, std::move(data), std::move(mask), col.null_count()));
    } else if (col.type().id() == type_id::STRING && col.null_count() == 0) {
      auto const b               = bounds[string_index++];
      auto const& source_offsets = col.child(strings_column_view::offsets_column_index);
      auto const type = data_type{b.bytes >= threshold ? type_id::INT64 : type_id::INT32};
      std::unique_ptr<column> offsets;
      if (type == source_offsets.type() && b.first == 0) {
        auto const width = cudf::size_of(type);
        rmm::device_buffer data(
          source_offsets.head<char>() +
            (static_cast<std::size_t>(source_offsets.offset()) + col.offset()) * width,
          (static_cast<std::size_t>(rows) + 1) * width,
          stream,
          mr);
        offsets = std::make_unique<column>(
          type,
          rows + 1,
          std::move(data),
          cudf::detail::create_null_mask(0, mask_state::UNALLOCATED, stream, mr),
          0);
      } else {
        offsets = make_numeric_column(type, rows + 1, mask_state::UNALLOCATED, stream, mr);
        auto src =
          cudf::detail::offsetalator_factory::make_input_iterator(source_offsets, col.offset());
        auto dst =
          cudf::detail::offsetalator_factory::make_output_iterator(offsets->mutable_view());
        thrust::transform(
          rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
          indices,
          indices + rows + 1,
          dst,
          [src, first = b.first] __device__(size_type i) { return src[i] - first; });
      }
      auto const* chars_begin = b.bytes ? col.head<char>() + b.first : nullptr;
      rmm::device_buffer chars(chars_begin, b.bytes, stream, mr);
      columns.push_back(make_strings_column(
        rows,
        std::move(offsets),
        std::move(chars),
        0,
        cudf::detail::create_null_mask(0, mask_state::UNALLOCATED, stream, mr)));
    } else if (col.type().id() == type_id::STRING) {
      auto gathered = strings::detail::gather<false>(
        strings_column_view{col}, indices, indices + rows, stream, cudf::memory_resources{mr});
      gathered->set_null_mask(cudf::detail::copy_bitmask(col, stream, mr), col.null_count());
      columns.push_back(std::move(gathered));
    } else {
      auto gathered = cudf::detail::gather(table_view{{col}},
                                           indices,
                                           indices + rows,
                                           out_of_bounds_policy::DONT_CHECK,
                                           stream,
                                           cudf::memory_resources{mr});
      auto owned    = gathered->release();
      columns.push_back(std::move(owned.front()));
    }
  }
  return std::make_unique<table>(std::move(columns), rows);
}
}  // namespace cudf::detail::partition_identity
