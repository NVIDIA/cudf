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

#include <optional>
#include <ranges>
#include <stdexcept>
#include <utility>

namespace cudf_streaming {

namespace {

/**
 * @brief Verify that a caller-provided reservation covers the next allocation.
 *
 * A reservation is only consumed after the allocation it covers has landed, so
 * without this check an undersized reservation is not detected until the memory
 * has already been allocated.
 *
 * @param reservation The caller's reservation, or `nullptr`.
 * @param size The number of bytes about to be allocated.
 *
 * @throws rapidsmpf::reservation_error if the reservation does not cover @p size.
 */
void check_reservation(rapidsmpf::MemoryReservation const* reservation, std::size_t size)
{
  if (reservation == nullptr) { return; }
  RAPIDSMPF_EXPECTS(reservation->size() >= size,
                    "MemoryReservation(" + rapidsmpf::format_nbytes(reservation->size()) +
                      ") isn't big enough (" + rapidsmpf::format_nbytes(size) + ")",
                    rapidsmpf::reservation_error);
}

/**
 * @brief Reserve device memory unless the caller already provided a reservation.
 *
 * @param reservation The caller's reservation, or `nullptr` to reserve.
 * @param size The number of bytes to cover.
 * @param br Buffer resource to reserve from.
 * @param allow_overbooking Passed to `reserve_device_memory_and_spill()`.
 *
 * @return The new reservation, or `std::nullopt` when the caller provided one.
 */
[[nodiscard]] std::optional<rapidsmpf::MemoryReservation> reserve_unless_provided(
  rapidsmpf::MemoryReservation* reservation,
  std::size_t size,
  rapidsmpf::BufferResource* br,
  rapidsmpf::AllowOverbooking allow_overbooking)
{
  if (reservation != nullptr) { return std::nullopt; }
  return br->reserve_device_memory_and_spill(size, allow_overbooking);
}

/**
 * @brief Consume `size` bytes of the reservation that covered a completed allocation.
 *
 * A reservation is counted on top of the memory already allocated, so it must be
 * consumed once the allocation it covers has landed.
 *
 * @param br Buffer resource holding the reservation.
 * @param reservation The caller's reservation, or `nullptr`.
 * @param own The reservation made by `reserve_unless_provided()`, if any.
 * @param size The number of bytes that were allocated.
 *
 * @throws rapidsmpf::reservation_error if the reservation does not cover @p size.
 */
void consume_reservation(rapidsmpf::BufferResource* br,
                         rapidsmpf::MemoryReservation* reservation,
                         std::optional<rapidsmpf::MemoryReservation>& own,
                         std::size_t size)
{
  if (reservation != nullptr) {
    br->release(*reservation, size);
  } else {
    own.reset();
  }
}

/**
 * @brief Total size of the partitions and how much of it is not in device memory.
 *
 * @param partitions The packed partitions.
 *
 * @return A pair of the total size and the non-device size, both in bytes.
 */
template <typename Range>
[[nodiscard]] std::pair<std::size_t, std::size_t> partition_sizes(Range&& partitions)
{
  std::size_t total_size      = 0;
  std::size_t non_device_size = 0;
  for (auto const& packed_data : partitions) {
    if (!packed_data.empty()) {
      std::size_t const size = packed_data.data->size;
      total_size += size;
      if (packed_data.data->mem_type() != rapidsmpf::MemoryType::DEVICE) {
        non_device_size += size;
      }
    }
  }
  return {total_size, non_device_size};
}

std::unordered_map<rapidsmpf::shuffler::PartID, rapidsmpf::PackedData> split_and_pack_impl(
  cudf::table_view const& table,
  std::vector<cudf::size_type> const& splits,
  rmm::cuda_stream_view stream,
  rapidsmpf::BufferResource* br,
  rapidsmpf::AllowOverbooking allow_overbooking,
  rapidsmpf::MemoryReservation* reservation);

}  // namespace

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
  // skip the first offset. See: <https://github.com/NVIDIA/cudf/issues/4607>.
  auto partition_offsets =
    cudf::host_span<cudf::size_type const>(offsets.data() + 1, offsets.size() - 2);

  // split does not make any copies.
  auto tbl_partitioned = cudf::split(partition_table->view(), partition_offsets, stream);

  return {std::move(tbl_partitioned), std::move(partition_table)};
}

namespace {

std::unordered_map<rapidsmpf::shuffler::PartID, rapidsmpf::PackedData> partition_and_pack_impl(
  cudf::table_view const& table,
  std::vector<cudf::size_type> const& columns_to_hash,
  int num_partitions,
  cudf::hash_id hash_function,
  std::uint32_t seed,
  rmm::cuda_stream_view stream,
  rapidsmpf::BufferResource* br,
  rapidsmpf::AllowOverbooking allow_overbooking,
  rapidsmpf::MemoryReservation* reservation)
{
  RAPIDSMPF_NVTX_FUNC_RANGE();
  RAPIDSMPF_MEMORY_PROFILE(br->statistics(), br->device_mr());
  RAPIDSMPF_EXPECTS(num_partitions > 0, "Need to split to at least one partition");
  if (table.num_rows() == 0) {
    auto splits =
      std::vector<cudf::size_type>(rapidsmpf::safe_cast<std::uint64_t>(num_partitions - 1), 0);
    return split_and_pack_impl(table, splits, stream, br, allow_overbooking, reservation);
  }

  // hash_partition does a deep-copy. Therefore, we need to reserve memory for
  // at least the size of the table. `packed_size()` measures the same bytes as the
  // copy needs, rounded up to the packing alignment, so it is a safe over-estimate.
  auto const reorder_bytes = cudf::packed_size(table, stream, br->device_mr());
  check_reservation(reservation, reorder_bytes);
  auto own_reservation = reserve_unless_provided(reservation, reorder_bytes, br, allow_overbooking);
  auto [reordered, split_points] = cudf::hash_partition(
    table, columns_to_hash, num_partitions, hash_function, seed, stream, br->device_mr());
  consume_reservation(br, reservation, own_reservation, reorder_bytes);
  std::vector<cudf::size_type> splits(split_points.begin() + 1, split_points.end() - 1);
  return split_and_pack_impl(reordered->view(), splits, stream, br, allow_overbooking, reservation);
}

}  // namespace

std::size_t partition_and_pack_cost(cudf::table_view const& table,
                                    rmm::cuda_stream_view stream,
                                    rapidsmpf::BufferResource* br)
{
  // The reorder and the packed partitions are each about one packed table, and an
  // empty table skips the reorder entirely.
  auto const packed_size = cudf::packed_size(table, stream, br->device_mr());
  return table.num_rows() == 0 ? packed_size : 2 * packed_size;
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
  return partition_and_pack_impl(table,
                                 columns_to_hash,
                                 num_partitions,
                                 hash_function,
                                 seed,
                                 stream,
                                 br,
                                 allow_overbooking,
                                 nullptr);
}

std::unordered_map<rapidsmpf::shuffler::PartID, rapidsmpf::PackedData> partition_and_pack(
  cudf::table_view const& table,
  std::vector<cudf::size_type> const& columns_to_hash,
  int num_partitions,
  cudf::hash_id hash_function,
  std::uint32_t seed,
  rmm::cuda_stream_view stream,
  rapidsmpf::BufferResource* br,
  rapidsmpf::MemoryReservation& reservation)
{
  RAPIDSMPF_EXPECTS(reservation.mem_type() == rapidsmpf::MemoryType::DEVICE,
                    "reservation must be for device memory",
                    std::invalid_argument);
  RAPIDSMPF_EXPECTS(reservation.br() == br,
                    "reservation must belong to the given buffer resource",
                    std::invalid_argument);
  return partition_and_pack_impl(table,
                                 columns_to_hash,
                                 num_partitions,
                                 hash_function,
                                 seed,
                                 stream,
                                 br,
                                 rapidsmpf::AllowOverbooking::NO,
                                 &reservation);
}

namespace {

std::unordered_map<rapidsmpf::shuffler::PartID, rapidsmpf::PackedData> split_and_pack_impl(
  cudf::table_view const& table,
  std::vector<cudf::size_type> const& splits,
  rmm::cuda_stream_view stream,
  rapidsmpf::BufferResource* br,
  rapidsmpf::AllowOverbooking allow_overbooking,
  rapidsmpf::MemoryReservation* reservation)
{
  RAPIDSMPF_NVTX_FUNC_RANGE();
  RAPIDSMPF_MEMORY_PROFILE(br->statistics(), br->device_mr());
  std::unordered_map<rapidsmpf::shuffler::PartID, rapidsmpf::PackedData> ret;

  // contiguous split does a deep-copy. Therefore, we need to reserve memory for
  // at least the size of the table.
  auto const packed_bytes = cudf::packed_size(table, stream, br->device_mr());
  check_reservation(reservation, packed_bytes);
  auto own_reservation = reserve_unless_provided(reservation, packed_bytes, br, allow_overbooking);
  auto packed          = cudf::contiguous_split(table, splits, stream, br->device_mr());
  consume_reservation(br, reservation, own_reservation, packed_bytes);
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

}  // namespace

std::size_t split_and_pack_cost(cudf::table_view const& table,
                                rmm::cuda_stream_view stream,
                                rapidsmpf::BufferResource* br)
{
  return cudf::packed_size(table, stream, br->device_mr());
}

std::unordered_map<rapidsmpf::shuffler::PartID, rapidsmpf::PackedData> split_and_pack(
  cudf::table_view const& table,
  std::vector<cudf::size_type> const& splits,
  rmm::cuda_stream_view stream,
  rapidsmpf::BufferResource* br,
  rapidsmpf::AllowOverbooking allow_overbooking)
{
  return split_and_pack_impl(table, splits, stream, br, allow_overbooking, nullptr);
}

std::unordered_map<rapidsmpf::shuffler::PartID, rapidsmpf::PackedData> split_and_pack(
  cudf::table_view const& table,
  std::vector<cudf::size_type> const& splits,
  rmm::cuda_stream_view stream,
  rapidsmpf::BufferResource* br,
  rapidsmpf::MemoryReservation& reservation)
{
  RAPIDSMPF_EXPECTS(reservation.mem_type() == rapidsmpf::MemoryType::DEVICE,
                    "reservation must be for device memory",
                    std::invalid_argument);
  RAPIDSMPF_EXPECTS(reservation.br() == br,
                    "reservation must belong to the given buffer resource",
                    std::invalid_argument);
  return split_and_pack_impl(
    table, splits, stream, br, rapidsmpf::AllowOverbooking::NO, &reservation);
}

namespace {

std::unique_ptr<cudf::table> unpack_and_concat_impl(std::vector<rapidsmpf::PackedData>&& partitions,
                                                    rmm::cuda_stream_view stream,
                                                    rapidsmpf::BufferResource* br,
                                                    rapidsmpf::AllowOverbooking allow_overbooking,
                                                    rapidsmpf::MemoryReservation* reservation)
{
  RAPIDSMPF_NVTX_FUNC_RANGE();
  RAPIDSMPF_MEMORY_PROFILE(br->statistics(), br->device_mr());

  auto const [total_size, non_device_size] = partition_sizes(partitions);

  std::vector<cudf::table_view> unpacked;
  std::vector<cudf::packed_columns> references;
  std::vector<rmm::cuda_stream_view> packed_data_streams;
  unpacked.reserve(partitions.size());
  references.reserve(partitions.size());
  packed_data_streams.reserve(partitions.size());

  // Covers the unspill below and the concatenation at the end.
  check_reservation(reservation, total_size + non_device_size);
  auto own_reservation =
    reserve_unless_provided(reservation, total_size + non_device_size, br, allow_overbooking);
  rapidsmpf::MemoryReservation& res = reservation != nullptr ? *reservation : *own_reservation;

  // `move_to_device_buffer()` consumes `non_device_size` of the reservation, leaving
  // `total_size` for the concatenation.
  for (auto& packed_data : partitions) {
    if (!packed_data.empty()) {
      if (packed_data.data->size > 0) {  // No need to sync empty buffers.
        packed_data_streams.push_back(packed_data.data->stream());
      }
      unpacked.push_back(cudf::unpack(
        references.emplace_back(std::move(packed_data.metadata),
                                br->move_to_device_buffer(std::move(packed_data.data), res))));
    }
  }

  // We need to synchronize `stream` with the packed_data and update their
  // underlying device buffers to use `stream` going forward. This ensures
  // the packed data are not deallocated before we have a chance to
  // concatenate them on `stream`.
  rapidsmpf::cuda_stream_join(std::views::single(stream), packed_data_streams);
  for (cudf::packed_columns& packed_columns : references) {
    packed_columns.gpu_data->set_stream(stream);
  }

  auto ret = cudf::concatenate(unpacked, stream, br->device_mr());
  consume_reservation(br, reservation, own_reservation, total_size);
  return ret;
}

}  // namespace

std::size_t unpack_and_concat_cost(std::vector<rapidsmpf::PackedData> const& partitions)
{
  auto const [total_size, non_device_size] = partition_sizes(partitions);
  return total_size + non_device_size;
}

std::size_t unpack_and_concat_cost(std::vector<rapidsmpf::PackedData const*> const& partitions)
{
  auto const [total_size, non_device_size] = partition_sizes(
    partitions | std::views::transform([](auto const* p) -> rapidsmpf::PackedData const& {
      RAPIDSMPF_EXPECTS(p != nullptr, "partition cannot be NULL", std::invalid_argument);
      // A moved-from `PackedData` keeps its outer object but nulls its members, and
      // `PackedData::empty()` dereferences both.
      RAPIDSMPF_EXPECTS(p->metadata != nullptr && p->data != nullptr,
                        "partition has already been consumed",
                        std::invalid_argument);
      return *p;
    }));
  return total_size + non_device_size;
}

std::unique_ptr<cudf::table> unpack_and_concat(std::vector<rapidsmpf::PackedData>&& partitions,
                                               rmm::cuda_stream_view stream,
                                               rapidsmpf::BufferResource* br,
                                               rapidsmpf::AllowOverbooking allow_overbooking)
{
  return unpack_and_concat_impl(std::move(partitions), stream, br, allow_overbooking, nullptr);
}

std::unique_ptr<cudf::table> unpack_and_concat(std::vector<rapidsmpf::PackedData>&& partitions,
                                               rmm::cuda_stream_view stream,
                                               rapidsmpf::BufferResource* br,
                                               rapidsmpf::MemoryReservation& reservation)
{
  RAPIDSMPF_EXPECTS(reservation.mem_type() == rapidsmpf::MemoryType::DEVICE,
                    "reservation must be for device memory",
                    std::invalid_argument);
  RAPIDSMPF_EXPECTS(reservation.br() == br,
                    "reservation must belong to the given buffer resource",
                    std::invalid_argument);
  return unpack_and_concat_impl(
    std::move(partitions), stream, br, rapidsmpf::AllowOverbooking::NO, &reservation);
}

}  // namespace cudf_streaming
