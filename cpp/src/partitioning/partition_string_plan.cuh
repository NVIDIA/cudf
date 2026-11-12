// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "partition_map_plan.cuh"
#include "partition_strings.cuh"

#include <cudf/detail/gather.hpp>
#include <cudf/strings/detail/gather.cuh>

#include <cuda/std/cstring>
#include <cuda/std/memory>

#include <limits>
#include <optional>

namespace cudf::detail::partition_strings_plan {

struct sizes_output {
  partition_string_column const* columns;
  size_type count;
  size_type* scatter;
  __device__ void operator()(size_type source, size_type destination) const
  {
    if (scatter) { scatter[source] = destination; }
    for (size_type c = 0; c < count; ++c) {
      auto const col                  = columns[c];
      col.output_offsets[destination] = col.input_offsets[source + 1] - col.input_offsets[source];
    }
  }
};

CUDF_KERNEL void scatter_ranked_sizes(uint16_t const* ranks,
                                      size_type const* prefixes,
                                      size_type rows,
                                      size_type tiles,
                                      partition_string_column const* columns,
                                      size_type count)
{
  auto const zero = prefixes[blockIdx.x];
  auto const one  = prefixes[tiles + blockIdx.x];
#pragma unroll
  for (int item = 0; item < 4; ++item) {
    auto const source =
      std::size_t{blockIdx.x} * partition_map_plan::tile_rows + item * blockDim.x + threadIdx.x;
    if (source < static_cast<std::size_t>(rows)) {
      auto const packed = ranks[source];
      auto const rank   = packed % partition_map_plan::tile_rows;
      auto const target = (packed < partition_map_plan::tile_rows ? zero : one) + rank;
      sizes_output{columns, count, nullptr}(source, target);
    }
  }
}

/** Copy from packed binary-bin/tile-rank pairs without a full-width row permutation. */
CUDF_KERNEL void copy_ranked(uint16_t const* ranks,
                             size_type const* prefixes,
                             size_type rows,
                             size_type tiles,
                             partition_string_column const* columns)
{
  static_assert(partition_map_plan::tile_rows % 64 == 0);
  auto const col = columns[blockIdx.y];
  for (auto base = std::size_t{blockIdx.x} * 64; base < static_cast<std::size_t>(rows);
       base += std::size_t{gridDim.x} * 64) {
    // Each 64-row batch stays inside one tile. Load both bin bases independently
    // of the packed rank so prefix loads can overlap the rank load.
    auto const tile   = base / partition_map_plan::tile_rows;
    auto const zero   = prefixes[tile];
    auto const one    = prefixes[tiles + tile];
    auto const source = base + threadIdx.x / 4;
    size_type target  = 0;
    if (source < static_cast<std::size_t>(rows)) {
      auto const packed = ranks[source];
      auto const rank   = packed % partition_map_plan::tile_rows;
      target            = (packed < partition_map_plan::tile_rows ? zero : one) + rank;
    }
    copy_group<true>(col, source, target, rows);
  }
}

class output_buffers {
 public:
  std::vector<partition_string_column> columns;
  std::vector<std::unique_ptr<column>> offsets;
  std::vector<rmm::device_buffer> characters;
  cuda::device_buffer<partition_string_column> device_columns;
  rmm::device_buffer scan_storage;
  size_type rows;

  output_buffers(table_view const& input,
                 cuda::stream_ref stream,
                 rmm::device_async_resource_ref mr)
    : device_columns(stream, cudf::get_current_device_resource_ref(), 0, cuda::no_init),
      scan_storage(0, stream),
      rows(input.num_rows())
  {
    for (auto const& col : input) {
      columns.push_back({strings_column_view(col).offsets().data<size_type>() + col.offset(),
                         col.head<char>(),
                         nullptr,
                         nullptr});
    }
    auto const temporary_mr = cudf::get_current_device_resource_ref();
    device_columns          = cudf::detail::make_device_buffer(columns, stream, temporary_mr);
    rmm::device_uvector<size_type> device_sizes(columns.size(), stream);
    partition_string_sizes<<<(columns.size() + 255) / 256, 256, 0, stream.get()>>>(
      device_columns.data(), columns.size(), rows, device_sizes.data());
    CUDF_CUDA_TRY(cudaPeekAtLastError());
    auto const sizes = cudf::detail::make_pinned_vector(device_sizes, stream);
    // Mandatory allocation bounds also enforce the standard output offset-type
    // limit. They do not choose the map algorithm or byte-copy strategy.
    if (*std::max_element(sizes.begin(), sizes.end()) >= strings::get_offset64_threshold()) {
      return;
    }
    for (std::size_t c = 0; c < columns.size(); ++c) {
      offsets.push_back(make_numeric_column(
        data_type{type_id::INT32}, rows + 1, mask_state::UNALLOCATED, stream, mr));
      characters.emplace_back(sizes[c], stream, mr);
      columns[c].output_offsets = offsets.back()->mutable_view().data<size_type>();
      columns[c].output_chars   = static_cast<char*>(characters.back().data());
      CUDF_CUDA_TRY(
        cudaMemsetAsync(columns[c].output_offsets + rows, 0, sizeof(size_type), stream.get()));
    }
    device_columns    = cudf::detail::make_device_buffer(columns, stream, temporary_mr);
    std::size_t bytes = 0;
    CUDF_CUDA_TRY(cub::DeviceScan::ExclusiveSum(nullptr,
                                                bytes,
                                                static_cast<size_type*>(nullptr),
                                                static_cast<size_type*>(nullptr),
                                                rows + 1,
                                                stream.get()));
    scan_storage = rmm::device_buffer(bytes, stream);
  }

  bool admitted() const { return offsets.size() == columns.size(); }

  void scan(cuda::stream_ref stream)
  {
    for (auto const& col : columns) {
      auto bytes = scan_storage.size();
      CUDF_CUDA_TRY(cub::DeviceScan::ExclusiveSum(scan_storage.data(),
                                                  bytes,
                                                  col.output_offsets,
                                                  col.output_offsets,
                                                  rows + 1,
                                                  stream.get()));
    }
  }

  std::unique_ptr<table> release(cuda::stream_ref stream, rmm::device_async_resource_ref mr)
  {
    std::vector<std::unique_ptr<column>> output;
    for (std::size_t c = 0; c < columns.size(); ++c) {
      output.push_back(make_strings_column(
        rows,
        std::move(offsets[c]),
        std::move(characters[c]),
        0,
        cudf::detail::create_null_mask(0, mask_state::UNALLOCATED, stream, mr)));
    }
    return std::make_unique<table>(std::move(output), rows);
  }
};

template <typename MapType>
std::optional<std::pair<std::unique_ptr<table>, std::vector<size_type>>> partition(
  table_view const& input,
  column_view const& map,
  size_type partitions,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  if (partitions < 2 || partitions > 256 || (partitions != 2 && input.num_columns() < 2) ||
      input.num_rows() < 1024 || input.num_columns() == 0 || input.num_columns() > 65'535 ||
      !std::all_of(input.begin(), input.end(), [](auto const& col) {
        return col.type().id() == type_id::STRING && col.null_count() == 0 &&
               strings_column_view(col).offsets().type().id() == type_id::INT32;
      })) {
    return std::nullopt;
  }
  namespace plans = partition_map_plan;
  auto const rows = input.num_rows();
  // Preserve fused materialization for multi-column strings at larger partition
  // counts. With two partitions, small schemas can eliminate one or both maps.
  if (input.num_columns() >= 4 || partitions != 2) {
    rmm::device_uvector<size_type> gather_map(rows, stream);
    auto offsets = plans::run(
      map.begin<MapType>(),
      rows,
      partitions,
      partitions == 2 ? plans::algorithm::binary : plans::algorithm::super2,
      [indices = gather_map.data()] __device__(size_type source, size_type destination) {
        indices[destination] = source;
      },
      false,
      stream);
    if (auto output = gather_partition_strings(
          input, device_span<size_type const>{gather_map.data(), gather_map.size()}, stream, mr)) {
      return std::pair{std::move(output), std::move(offsets)};
    }
    // Preserve the existing permutation when output offsets require the generic path.
    auto output =
      cudf::detail::gather(input,
                           device_span<size_type const>{gather_map.data(), gather_map.size()},
                           out_of_bounds_policy::DONT_CHECK,
                           negative_index_policy::NOT_ALLOWED,
                           stream,
                           mr);
    return std::pair{std::move(output), std::move(offsets)};
  }
  if (input.num_columns() == 1 && rows >= 4'000'000) { return std::nullopt; }
  if (strings::get_offset64_threshold() != std::numeric_limits<size_type>::max()) {
    return std::nullopt;
  }
  output_buffers output(input, stream, mr);
  if (!output.admitted()) { return std::nullopt; }
  std::vector<size_type> offsets;
  constexpr size_type large_input_crossover = 4'000'000;
  if (rows >= large_input_crossover) {
    auto const blocks = (rows - 1) / plans::tile_rows + 1;
    rmm::device_uvector<size_type> prefixes(static_cast<std::size_t>(partitions) * blocks, stream);
    rmm::device_uvector<size_type> boundaries(partitions + 1, stream);
    rmm::device_uvector<uint16_t> ranks(rows, stream);
    plans::count_tiles<1, true, true><<<blocks, 256, 0, stream.get()>>>(
      map.begin<MapType>(), rows, partitions, ranks.data(), prefixes.data());
    plans::scan(prefixes.data(), prefixes.size(), stream);
    plans::get_offsets<<<1, 256, 0, stream.get()>>>(
      prefixes.data(), blocks, partitions, rows, boundaries.data());
    scatter_ranked_sizes<<<blocks, 256, 0, stream.get()>>>(ranks.data(),
                                                           prefixes.data(),
                                                           rows,
                                                           blocks,
                                                           output.device_columns.data(),
                                                           input.num_columns());
    output.scan(stream);
    auto const copy_blocks =
      std::min<std::size_t>(65'535, (static_cast<std::size_t>(rows) + 63) / 64);
    copy_ranked<<<dim3(copy_blocks, input.num_columns()), 256, 0, stream.get()>>>(
      ranks.data(), prefixes.data(), rows, blocks, output.device_columns.data());
    CUDF_CUDA_TRY(cudaPeekAtLastError());
    offsets = cudf::detail::make_std_vector(boundaries, stream);
  } else {
    rmm::device_uvector<size_type> scatter_map(rows, stream);
    offsets = plans::run(
      map.begin<MapType>(),
      rows,
      partitions,
      plans::algorithm::binary,
      sizes_output{output.device_columns.data(), input.num_columns(), scatter_map.data()},
      false,
      stream);
    output.scan(stream);
    auto const blocks = std::min<std::size_t>(65'535, (static_cast<std::size_t>(rows) + 63) / 64);
    if (input.num_columns() == 1) {
      copy_characters<false><<<dim3(blocks, 1), 256, 0, stream.get()>>>(
        output.device_columns.data(), scatter_map.data(), rows);
    } else {
      copy_characters<true><<<dim3(blocks, input.num_columns()), 256, 0, stream.get()>>>(
        output.device_columns.data(), scatter_map.data(), rows);
    }
    CUDF_CUDA_TRY(cudaPeekAtLastError());
  }
  return std::pair{output.release(stream, mr), std::move(offsets)};
}
}  // namespace cudf::detail::partition_strings_plan
