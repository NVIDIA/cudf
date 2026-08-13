/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/concatenate.hpp>
#include <cudf/contiguous_split.hpp>
#include <cudf/copying.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/span.hpp>

#include <cudf_streaming/partition_utils.hpp>
#include <cudf_streaming/utils.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>

#include <rapidsmpf/cuda_stream.hpp>
#include <rapidsmpf/error.hpp>
#include <rapidsmpf/memory/buffer.hpp>
#include <rapidsmpf/memory/buffer_resource.hpp>
#include <rapidsmpf/memory/memory_type.hpp>
#include <rapidsmpf/memory/packed_data.hpp>
#include <rapidsmpf/nvtx.hpp>
#include <rapidsmpf/shuffler/shuffler.hpp>
#include <rapidsmpf/utils/misc.hpp>

#include <ranges>
#include <utility>

namespace cudf_streaming {

std::pair<std::vector<cudf::table_view>, std::unique_ptr<cudf::table>> partition_and_split(
  cudf::table_view const& table,
  std::vector<cudf::size_type> const& columns_to_hash,
  int num_partitions,
  cudf::hash_id hash_function,
  std::uint32_t seed,
  rmm::cuda_stream_view stream,
  rapidsmpf::BufferResource* br,
  rapidsmpf::AllowOverbooking allow_overbooking)
{
  RAPIDSMPF_MEMORY_PROFILE(br->statistics(), br->device_mr());
  if (table.num_rows() == 0) {
    // Return views of a copy of the empty `table`.
    auto owner = std::make_unique<cudf::table>(table, stream, br->device_mr());
    return {std::vector<cudf::table_view>(rapidsmpf::safe_cast<std::size_t>(num_partitions),
                                          owner->view()),
            std::move(owner)};
  }

  // hash_partition does a deep-copy. Therefore, we need to reserve memory for
  // at least the size of the table.
  auto reservation =
    br->reserve_device_memory_and_spill(estimated_memory_usage(table, stream), allow_overbooking);
  auto [partition_table, offsets] = cudf::hash_partition(
    table, columns_to_hash, num_partitions, hash_function, seed, stream, br->device_mr());
  reservation.clear();

  // Notice, the offset argument for split() and hash_partition() doesn't align.
  // hash_partition() returns the start offset of each partition thus we have to
  // skip the first offset. See: <https://github.com/rapidsai/cudf/issues/4607>.
  auto partition_offsets =
    cudf::host_span<cudf::size_type const>(offsets.data() + 1, offsets.size() - 2);

  // split does not make any copies.
  auto tbl_partitioned = cudf::split(partition_table->view(), partition_offsets, stream);

  return {std::move(tbl_partitioned), std::move(partition_table)};
}

std::unordered_map<rapidsmpf::shuffler::PartID, rapidsmpf::PackedData> partition_and_pack(
  cudf::table_view const& table,
  std::vector<cudf::size_type> const& columns_to_hash,
  int num_partitions,
  cudf::hash_id hash_function,
  std::uint32_t seed,
  rmm::cuda_stream_view stream,
  rapidsmpf::BufferResource* br,
  rapidsmpf::AllowOverbooking allow_overbooking)
{
  RAPIDSMPF_NVTX_FUNC_RANGE();
  RAPIDSMPF_MEMORY_PROFILE(br->statistics(), br->device_mr());
  RAPIDSMPF_EXPECTS(num_partitions > 0, "Need to split to at least one partition");
  if (table.num_rows() == 0) {
    auto splits =
      std::vector<cudf::size_type>(rapidsmpf::safe_cast<std::uint64_t>(num_partitions - 1), 0);
    return split_and_pack(table, splits, stream, br, allow_overbooking);
  }

  // hash_partition does a deep-copy. Therefore, we need to reserve memory for
  // at least the size of the table.
  auto reservation =
    br->reserve_device_memory_and_spill(estimated_memory_usage(table, stream), allow_overbooking);
  auto [reordered, split_points] = cudf::hash_partition(
    table, columns_to_hash, num_partitions, hash_function, seed, stream, br->device_mr());
  reservation.clear();
  std::vector<cudf::size_type> splits(split_points.begin() + 1, split_points.end() - 1);
  return split_and_pack(reordered->view(), splits, stream, br, allow_overbooking);
}

std::unordered_map<rapidsmpf::shuffler::PartID, rapidsmpf::PackedData> split_and_pack(
  cudf::table_view const& table,
  std::vector<cudf::size_type> const& splits,
  rmm::cuda_stream_view stream,
  rapidsmpf::BufferResource* br,
  rapidsmpf::AllowOverbooking allow_overbooking)
{
  RAPIDSMPF_NVTX_FUNC_RANGE();
  RAPIDSMPF_MEMORY_PROFILE(br->statistics(), br->device_mr());
  std::unordered_map<rapidsmpf::shuffler::PartID, rapidsmpf::PackedData> ret;

  // contiguous split does a deep-copy. Therefore, we need to reserve memory for
  // at least the size of the table.
  auto reservation = br->reserve_device_memory_and_spill(
    cudf::packed_size(table, stream, br->device_mr()), allow_overbooking);
  auto packed = cudf::contiguous_split(table, splits, stream, br->device_mr());
  reservation.clear();
  ret.reserve(packed.size());
  for (rapidsmpf::shuffler::PartID i = 0; rapidsmpf::safe_cast<std::size_t>(i) < packed.size();
       i++) {
    auto pack = std::move(packed[i].data);
    ret.emplace(
      i,
      rapidsmpf::PackedData(std::move(pack.metadata), br->move(std::move(pack.gpu_data), stream)));
  }
  return ret;
}

std::unique_ptr<cudf::table> unpack_and_concat(std::vector<rapidsmpf::PackedData>&& partitions,
                                               rmm::cuda_stream_view stream,
                                               rapidsmpf::BufferResource* br,
                                               rapidsmpf::AllowOverbooking allow_overbooking)
{
  RAPIDSMPF_NVTX_FUNC_RANGE();
  RAPIDSMPF_MEMORY_PROFILE(br->statistics(), br->device_mr());

  // Let's find the total size of the partitions and how much of the packed data we
  // need to move to device memory (unspill).
  std::size_t total_size      = 0;
  std::size_t non_device_size = 0;
  for (auto& packed_data : partitions) {
    if (!packed_data.empty()) {
      std::size_t size = packed_data.data->size;
      total_size += size;
      if (packed_data.data->mem_type() != rapidsmpf::MemoryType::DEVICE) {
        non_device_size += size;
      }
    }
  }

  std::vector<cudf::table_view> unpacked;
  std::vector<cudf::packed_columns> references;
  std::vector<rmm::cuda_stream_view> packed_data_streams;
  unpacked.reserve(partitions.size());
  references.reserve(partitions.size());
  packed_data_streams.reserve(partitions.size());

  // Reserve device memory for the unspill AND the cudf::unpack() calls.
  auto reservation =
    br->reserve_device_memory_and_spill(total_size + non_device_size, allow_overbooking);
  for (auto& packed_data : partitions) {
    if (!packed_data.empty()) {
      if (packed_data.data->size > 0) {  // No need to sync empty buffers.
        packed_data_streams.push_back(packed_data.data->stream());
      }
      unpacked.push_back(cudf::unpack(references.emplace_back(
        std::move(packed_data.metadata),
        br->move_to_device_buffer(std::move(packed_data.data), reservation))));
    }
  }
  reservation.clear();

  // We need to synchronize `stream` with the packed_data and update their
  // underlying device buffers to use `stream` going forward. This ensures
  // the packed data are not deallocated before we have a chance to
  // concatenate them on `stream`.
  rapidsmpf::cuda_stream_join(std::views::single(stream), packed_data_streams);
  for (cudf::packed_columns& packed_columns : references) {
    packed_columns.gpu_data->set_stream(stream);
  }

  reservation = br->reserve_device_memory_and_spill(total_size, allow_overbooking);
  return cudf::concatenate(unpacked, stream, br->device_mr());
}

}  // namespace cudf_streaming
