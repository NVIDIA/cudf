// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "partition_map_plan.cuh"
#include "partition_strings.cuh"

#include <cudf/strings/detail/gather.cuh>

#include <cuda/std/cstring>
#include <cuda/std/memory>

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

/** Recompute destinations from retained tile prefixes instead of storing N indices. */
template <typename MapType>
CUDF_KERNEL void copy_recomputed(MapType const* map,
                                 size_type const* prefixes,
                                 size_type rows,
                                 size_type partitions,
                                 partition_string_column const* columns,
                                 size_type count)
{
  namespace plans = partition_map_plan;
  __shared__ int histogram[plans::warps][plans::bins], base[plans::bins];
  if (threadIdx.x < partitions) {
    base[threadIdx.x] = prefixes[std::size_t{threadIdx.x} * gridDim.x + blockIdx.x];
  }
  __syncthreads();
  int ranks[4], keys[4], destinations[4];
  int const start = blockIdx.x * plans::tile_rows;
  plans::local_tile<true>(map, rows, start, partitions, histogram, ranks, keys);
  if (threadIdx.x < partitions) {
    int sum = 0;
    for (int w = 0; w < plans::warps; ++w) {
      auto const value          = histogram[w][threadIdx.x];
      histogram[w][threadIdx.x] = sum;
      sum += value;
    }
  }
  __syncthreads();
  int const lane = threadIdx.x % 32, warp = threadIdx.x / 32;
#pragma unroll
  for (int item = 0; item < 4; ++item) {
    int const row = start + warp * 128 + item * 32 + lane;
    destinations[item] =
      row < rows ? base[keys[item]] + histogram[warp][keys[item]] + ranks[item] : 0;
  }
  for (size_type c = 0; c < count; ++c) {
#pragma unroll
    for (int item = 0; item < 4; ++item) {
      int const row = start + warp * 128 + item * 32 + lane;
#pragma unroll
      for (int batch = 0; batch < 4; ++batch) {
        auto const owner       = lane / 4 + batch * 8;
        auto const source      = __shfl_sync(0xffffffffu, row, owner);
        auto const destination = __shfl_sync(0xffffffffu, destinations[item], owner);
        copy_group(columns[c], source, destination, rows);
      }
    }
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
    return std::nullopt;
  }
  output_buffers output(input, stream, mr);
  if (!output.admitted()) { return std::nullopt; }
  std::vector<size_type> offsets;
  constexpr size_type recompute_crossover = 4'000'000;
  if (rows >= recompute_crossover) {
    auto const blocks = (rows - 1) / plans::tile_rows + 1;
    rmm::device_uvector<size_type> prefixes(static_cast<std::size_t>(partitions) * blocks, stream);
    rmm::device_uvector<size_type> boundaries(partitions + 1, stream);
    plans::count_tiles<1, false><<<blocks, 256, 0, stream.get()>>>(
      map.begin<MapType>(), rows, partitions, nullptr, prefixes.data());
    plans::scan(prefixes.data(), prefixes.size(), stream);
    plans::get_offsets<<<1, 256, 0, stream.get()>>>(
      prefixes.data(), blocks, partitions, rows, boundaries.data());
    plans::scatter_recompute<1><<<blocks, 256, 0, stream.get()>>>(
      map.begin<MapType>(),
      prefixes.data(),
      rows,
      partitions,
      sizes_output{output.device_columns.data(), input.num_columns(), nullptr});
    output.scan(stream);
    copy_recomputed<<<blocks, 256, 0, stream.get()>>>(map.begin<MapType>(),
                                                      prefixes.data(),
                                                      rows,
                                                      partitions,
                                                      output.device_columns.data(),
                                                      input.num_columns());
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
    copy_characters<<<dim3(blocks, input.num_columns()), 256, 0, stream.get()>>>(
      output.device_columns.data(), scatter_map.data(), rows);
    CUDF_CUDA_TRY(cudaPeekAtLastError());
  }
  return std::pair{output.release(stream, mr), std::move(offsets)};
}
}  // namespace cudf::detail::partition_strings_plan
