/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/packed_types.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <coro/task.hpp>
#include <rapidsmpf/memory/content_description.hpp>
#include <rapidsmpf/memory/packed_data.hpp>
#include <rapidsmpf/owning_wrapper.hpp>
#include <rapidsmpf/streaming/core/context.hpp>
#include <rapidsmpf/streaming/core/memory_reserve_or_wait.hpp>
#include <rapidsmpf/streaming/core/message.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>

namespace cudf_streaming {

/**
 * @brief A unit of table data in a streaming pipeline.
 *
 * Represents either an unpacked `cudf::table` or a `PackedData`.
 *
 * table_chunks may be initially unavailable (e.g., if the data is packed or spilled),
 * and can be made available (i.e., materialized to device memory) on demand.
 */
class table_chunk {
 public:
  /**
   * @brief Indicates whether the table_chunk holds an exclusive or shared view
   * of the underlying table data.
   *
   * This boolean enum is used to explicitly express ownership semantics
   * when constructing a table_chunk from a `cudf::table_view`.
   *
   * - `exclusive_view::YES`: The table_chunk has exclusive ownership of
   *   the table's device memory and are considered spillable.
   *
   * - `exclusive_view::NO`: The table_chunk is a non-owning view of data
   *   managed elsewhere. The memory may be shared or externally owned,
   *   and the chunk is therefore not spillable.
   */
  enum class exclusive_view : bool {
    NO,
    YES,
  };

  /**
   * @brief Construct a table_chunk from a device table.
   *
   * @param table Device-resident table.
   * @param stream The CUDA stream on which the table was created.
   */
  table_chunk(std::unique_ptr<cudf::table> table, rmm::cuda_stream_view stream);

  /**
   * @brief Construct a table_chunk from a device table view.
   *
   * The table_chunk does not take ownership of the underlying data; instead, the
   * provided @p owner object is kept alive for the lifetime of the table_chunk.
   * The caller is responsible for ensuring that the underlying device memory
   * referenced by @p table_view remains valid during this period.
   *
   * This constructor is typically used when creating a table_chunk from Python,
   * where @p owner is used to keep the corresponding Python object alive until
   * the table_chunk is destroyed.
   *
   * @param table_view Device-resident table view.
   * @param stream CUDA stream on which the table was created.
   * @param owner Object owning the memory backing @p table_view. This object will be
   * destroyed last when the table_chunk is destroyed or spilled.
   * @param exclusive_view Specifies whether this table_chunk has exclusive ownership
   * semantics over the underlying table data:
   *   - When `exclusive_view::YES`, the following guarantees must hold:
   *       - The @p table_view is the sole representation of the table.
   *       - The @p owner exclusively owns the table memory.
   *     These guarantees allow the table_chunk to be spillable and ensure that
   *     destroying @p owner will correctly free the associated device memory.
   *   - When `exclusive_view::NO`, the chunk is considered a non-owning view and
   *     is therefore not spillable.
   */
  table_chunk(cudf::table_view table_view,
              rmm::cuda_stream_view stream,
              rapidsmpf::OwningWrapper&& owner,
              exclusive_view exclusive_view);

  /**
   * @brief Construct a table_chunk from a packed data blob.
   *
   * The packed data's CUDA stream will be associated the new table chunk.
   *
   * @param packed_data Serialized host/device data with metadata.
   */
  table_chunk(std::unique_ptr<rapidsmpf::PackedData> packed_data);

  ~table_chunk() = default;

  /**
   * @brief Move constructor
   *
   * @note After this call `other.is_available() == false`.
   * @param other The table_chunk to move from.
   */
  table_chunk(table_chunk&& other) noexcept;

  /**
   * @brief Move assignment
   *
   * @note After this call `other.is_available() == false`.
   * @param other The table_chunk to move from.
   * @return Reference to this.
   */
  table_chunk& operator=(table_chunk&& other) noexcept;
  table_chunk(table_chunk const&)            = delete;
  table_chunk& operator=(table_chunk const&) = delete;

  /**
   * @brief Returns the CUDA stream on which this table chunk was created.
   *
   * @return The CUDA stream view.
   */
  [[nodiscard]] rmm::cuda_stream_view stream() const noexcept;

  /**
   * @brief Number of bytes allocated for the data in the specified memory type.
   *
   * @param mem_type The memory type to query.
   * @return Number of bytes allocated.
   */
  [[nodiscard]] std::size_t data_alloc_size(rapidsmpf::MemoryType mem_type) const;

  /**
   * @brief Indicates whether the underlying cudf table data is fully available in
   * device memory.
   *
   * @return `true` if the table is already available; otherwise, `false`.
   */
  [[nodiscard]] bool is_available() const noexcept;

  /**
   * @brief Returns the estimated cost (in bytes) of making the table available.
   *
   * Currently, only device memory cost is tracked.
   *
   * @return The cost in bytes.
   */
  [[nodiscard]] std::size_t make_available_cost() const noexcept;

  /**
   * @brief Moves this table chunk into a new one with its cudf table made available.
   *
   * As part of the move, a copy or unpack may be performed, the associated CUDA
   * stream is used.
   *
   * @param reservation Memory reservation for allocations if needed.
   * @return A new table_chunk with data available on device.
   *
   * @note After this call, the current object is in a moved-from state;
   * only reassignment, movement, or destruction are valid.
   */
  [[nodiscard]] table_chunk make_available(rapidsmpf::MemoryReservation& reservation);

  /**
   * @brief Moves this table chunk into a new one with its cudf table made available.
   *
   * Takes ownership of the memory reservation and consumes it entirely as part
   * of making the data available on device. The full reservation is considered
   * used, even if the actual allocation requires fewer bytes.
   *
   * @param reservation Memory reservation to be consumed for allocations.
   * @return A new table_chunk with data available on device.
   *
   * @note After this call, the current object is in a moved-from state; only
   * reassignment, movement, or destruction are valid.
   */
  [[nodiscard]] table_chunk make_available(rapidsmpf::MemoryReservation&& reservation);

  /**
   * @brief Move this table chunk into a new one with its cudf table made available.
   *
   * This variant of make_available() is a coroutine that may suspend if device
   * memory is not immediately available.
   *
   * @note After this call, the current object is in a moved-from state; only
   * reassignment, movement, or destruction are valid.
   *
   * @param ctx Streaming context used to access the memory reservation mechanism.
   * @param net_memory_delta Estimated change in memory usage after the reservation
   * is granted and all work using the returned `table_chunk` has completed. See
   * `MemoryReserveOrWait::reserve_or_wait` for details.
   * @return A new `table_chunk` that is available on device.
   *
   * @throws std::runtime_error If shutdown occurs before the reservation can be
   * processed.
   * @throws std::overflow_error If no progress is possible within the timeout and
   * overbooking is disabled.
   */
  [[nodiscard]] coro::task<table_chunk> make_available(
    std::shared_ptr<rapidsmpf::streaming::Context> ctx,
    std::int64_t net_memory_delta =
      rapidsmpf::streaming::MemoryReserveOrWait::missing_net_memory_delta);

  /**
   * @brief Returns a view of the underlying table.
   *
   * The table must be available in device memory.
   *
   * @return cudf::table_view representing the table.
   *
   * @throws std::invalid_argument if `is_available() == false`.
   */
  [[nodiscard]] cudf::table_view table_view() const;

  /**
   * @brief Indicates whether this table chunk can be spilled from device to host memory.
   *
   * A table chunk is considered spillable if it owns its underlying memory. This is
   * true when it was created from one of the following:
   *   - A device-owning source such as a `cudf::table`, `cudf::packed_columns`, or
   *     `PackedData`.
   *   - A `cudf::table_view` constructed with `is_exclusive_view == true`, indicating
   *     that the view is the sole representation of the underlying data and that its
   *     owner exclusively manages the table's memory.
   *
   * In contrast, chunks constructed from non-exclusive `cudf::table_view` instances are
   * non-owning views of externally managed memory and therefore not spillable.
   *
   * To spill a table chunk from device to host memory, first call `copy()` to create a
   * host-side copy, then delete or overwrite the original device chunk. If
   * `is_spillable() == true`, destroying the original device chunk will release the
   * associated device memory.
   *
   * @return `true` if the table chunk owns its memory and can be spilled; otherwise
   * `false`.
   */
  [[nodiscard]] bool is_spillable() const;

  /**
   * @brief Create a deep copy of the table chunk.
   *
   * Allocates new memory for all buffers in the table using the specified
   * `reservation`, which determines the target memory type (e.g., host or device).
   * As a consequence, the `is_available()` status may differ in the new copy. For
   * example, copying an available table chunk from device to host memory will result
   * in an unavailable copy.
   *
   * @param reservation Memory reservation used to track and limit allocations.
   * @return A new `table_chunk` instance containing copies of all buffers and metadata.
   *
   * @throws rapidsmpf::reservation_error If the total allocation size exceeds the
   * available reservation.
   */
  [[nodiscard]] table_chunk copy(rapidsmpf::MemoryReservation& reservation) const;

  /**
   * @brief Convert this table chunk to a `PackedData`, avoiding unnecessary copies.
   *
   * If the chunk's data is already in packed form (e.g., it arrived over the network
   * or was constructed from a `PackedData`), the packed data is moved out directly
   * with no copy. Otherwise the table is serialized via `cudf::pack()`.
   *
   * @param br Buffer resource used for the device memory resource when packing
   * is required.
   * @return A unique pointer to the resulting `PackedData`.
   *
   * @throws std::invalid_argument If the data is not already packed and
   * `is_available() == false`.
   *
   * @note After this call, this object is in a moved-from state; only reassignment,
   * movement, or destruction are valid.
   *
   * @note No memory reservation is required. If the data is already in packed form,
   * no allocation occurs. If packing is required, `cudf::pack()` allocates device
   * memory that is not tracked via a reservation.
   */
  [[nodiscard]] std::unique_ptr<rapidsmpf::PackedData> into_packed_data(
    rapidsmpf::BufferResource* br) &&;

  /**
   * @brief Return the shape of the table stored by the table chunk.
   *
   * @return Pair of number of rows and number of columns.
   */
  [[nodiscard]] std::pair<cudf::size_type, cudf::size_type> shape() const noexcept;

 private:
  ///< @brief Optional owning object if the table_chunk was constructed from a
  ///< table_view.
  rapidsmpf::OwningWrapper owner_{};

  // At most, one of the following unique pointers is non-null. If all of them are null,
  // the table_chunk is a non-owning view.
  // TODO: use a variant and drop the unique pointers?
  std::unique_ptr<cudf::table> table_;
  std::unique_ptr<rapidsmpf::PackedData> packed_data_;

  // Has value iff this table_chunk is available.
  std::optional<cudf::table_view> table_view_;

  // Zero initialized data allocation size (one for each memory type).
  std::array<std::size_t, rapidsmpf::MEMORY_TYPES.size()> data_alloc_size_ = {};
  std::size_t make_available_cost_;  // For now, only device memory cost is tracked.

  rmm::cuda_stream_view stream_;
  bool is_spillable_;
};

/**
 * @brief Generate a content description for a `table_chunk`.
 *
 * @param obj The object's content to describe.
 * @return A new content description.
 */
rapidsmpf::ContentDescription get_content_description(table_chunk const& obj);

/**
 * @brief Wrap a `table_chunk` into a `Message`.
 *
 * @param sequence_number Ordering identifier for the message.
 * @param chunk The chunk to wrap into a message.
 * @return A `Message` encapsulating the provided chunk as its payload.
 */
rapidsmpf::streaming::Message to_message(std::uint64_t sequence_number,
                                         std::unique_ptr<table_chunk> chunk);

}  // namespace cudf_streaming
