/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/contiguous_split.hpp>

#include <cudf_streaming/table_chunk.hpp>

#include <rmm/mr/per_device_resource.hpp>

#include <rapidsmpf/cuda_stream.hpp>
#include <rapidsmpf/memory/buffer.hpp>
#include <rapidsmpf/statistics.hpp>
#include <rapidsmpf/stream_ordered_timing.hpp>
#include <rapidsmpf/streaming/core/lineariser.hpp>

#include <cassert>
#include <memory>

namespace cudf_streaming {

table_chunk::table_chunk(std::unique_ptr<cudf::table> table, rmm::cuda_stream_view stream)
  : table_{std::move(table)}, stream_{stream}, is_spillable_{true}
{
  RAPIDSMPF_EXPECTS(table_ != nullptr, "table pointer cannot be null", std::invalid_argument);
  table_view_ = table_->view();
  data_alloc_size_[static_cast<std::size_t>(rapidsmpf::MemoryType::DEVICE)] =
    cudf::packed_size(*table_view_, stream_, rmm::mr::get_current_device_resource_ref());
  make_available_cost_ = 0;
}

table_chunk::table_chunk(cudf::table_view table_view,
                         rmm::cuda_stream_view stream,
                         rapidsmpf::OwningWrapper&& owner,
                         exclusive_view exclusive_view)
  : owner_{std::move(owner)},
    table_view_{table_view},
    stream_{stream},
    is_spillable_{static_cast<bool>(exclusive_view)}
{
  data_alloc_size_[static_cast<std::size_t>(rapidsmpf::MemoryType::DEVICE)] =
    cudf::packed_size(table_view, stream_, rmm::mr::get_current_device_resource_ref());
  make_available_cost_ = 0;
}

table_chunk::table_chunk(std::unique_ptr<rapidsmpf::PackedData> packed_data)
  : packed_data_{std::move(packed_data)}, is_spillable_{true}
{
  RAPIDSMPF_EXPECTS(
    packed_data_ != nullptr, "packed data pointer cannot be null", std::invalid_argument);
  RAPIDSMPF_EXPECTS(!packed_data_->empty(), "packed data cannot be empty", std::invalid_argument);
  // Initialize stream_ here rather than in the member-initializer list to avoid
  // dereferencing packed_data_ before the null check above.
  stream_ = packed_data_->data->stream();
  data_alloc_size_[static_cast<std::size_t>(packed_data_->data->mem_type())] =
    packed_data_->data->size;
  if (packed_data_->data->mem_type() != rapidsmpf::MemoryType::DEVICE) {
    make_available_cost_ = packed_data_->data->size;
  } else {
    // table data is in device memory. We can trivially unpack it and make it
    // available.
    table_view_          = cudf::unpack(packed_data_->metadata->data(),
                               reinterpret_cast<std::uint8_t const*>(packed_data_->data->data()));
    make_available_cost_ = 0;
  }
}

table_chunk::table_chunk(table_chunk&& other) noexcept
  : owner_(std::move(other.owner_)),
    table_(std::move(other.table_)),
    packed_data_(std::move(other.packed_data_)),
    table_view_(std::exchange(other.table_view_, std::nullopt)),
    data_alloc_size_(other.data_alloc_size_),
    make_available_cost_(other.make_available_cost_),
    stream_(other.stream_),
    is_spillable_(other.is_spillable_)
{
}

table_chunk& table_chunk::operator=(table_chunk&& other) noexcept
{
  if (this != &other) {
    owner_               = std::move(other.owner_);
    table_               = std::move(other.table_);
    packed_data_         = std::move(other.packed_data_);
    table_view_          = std::exchange(other.table_view_, std::nullopt);
    data_alloc_size_     = other.data_alloc_size_;
    make_available_cost_ = other.make_available_cost_;
    stream_              = other.stream_;
    is_spillable_        = other.is_spillable_;
  }
  return *this;
}

rmm::cuda_stream_view table_chunk::stream() const noexcept { return stream_; }

std::size_t table_chunk::data_alloc_size(rapidsmpf::MemoryType mem_type) const
{
  return data_alloc_size_.at(static_cast<std::size_t>(mem_type));
}

bool table_chunk::is_available() const noexcept { return table_view_.has_value(); }

std::size_t table_chunk::make_available_cost() const noexcept { return make_available_cost_; }

table_chunk table_chunk::make_available(rapidsmpf::MemoryReservation& reservation)
{
  if (is_available()) { return std::move(*this); }
  // Table chunk is not available. This means that the table data is not in device
  // memory. We need to move the table data to device memory using a device reservation.
  RAPIDSMPF_EXPECTS(reservation.mem_type() == rapidsmpf::MemoryType::DEVICE,
                    "device memory reservation is required");
  RAPIDSMPF_EXPECTS(packed_data_ != nullptr, "packed data pointer cannot be null");
  auto packed_data  = std::move(packed_data_);
  packed_data->data = reservation.br()->move(std::move(packed_data->data), reservation);
  return table_chunk{std::move(packed_data)};
}

table_chunk table_chunk::make_available(rapidsmpf::MemoryReservation&& reservation)
{
  rapidsmpf::MemoryReservation& res = reservation;
  return make_available(res);
}

coro::task<table_chunk> table_chunk::make_available(
  std::shared_ptr<rapidsmpf::streaming::Context> ctx, std::int64_t net_memory_delta)
{
  co_return make_available(co_await reserve_memory(ctx, make_available_cost(), net_memory_delta));
}

cudf::table_view table_chunk::table_view() const
{
  RAPIDSMPF_EXPECTS(is_available(),
                    "the table view is unavailable, please make sure it is "
                    "unspilled and unpacked (see `make_available`).",
                    std::invalid_argument);
  return table_view_.value();
}

bool table_chunk::is_spillable() const { return is_spillable_; }

table_chunk table_chunk::copy(rapidsmpf::MemoryReservation& reservation) const
{
  // This method handles the two possible cases. Note that
  // `!is_available() && packed_data_ == nullptr` is an invalid state, so the
  // remaining valid combinations collapse into:
  //
  // 1. The chunk is available and not yet packed. The table is copied/packed
  //    into the reservation-specified memory type using libcudf:
  //    a. DEVICE       - cudf-copy table_view() into device memory.
  //    b. PINNED_HOST  - cudf::pack table_view() directly into pinned memory.
  //    c. HOST         - cudf::pack table_view() into intermediate device
  //                      memory and then copy to host memory.
  //
  // 2. The chunk data is already packed (packed_data_ != nullptr).
  //    Use buffer_copy() to copy the packed data into the reservation-
  //    specified memory type. The original memory type of the chunk does
  //    not matter.
  rapidsmpf::BufferResource* br = reservation.br();

  // If the table view is available and the table is not packed, we can use libcudf to
  // copy the table in device memory, or pack it to pinned/ host memory. Else, fall
  // through to case 2 (ie. use buffer_copy).
  if (is_available() && packed_data_ == nullptr) {
    switch (reservation.mem_type()) {
      case rapidsmpf::MemoryType::DEVICE:  // Case 1a.
      {
        // Use libcudf to copy the table_view().
        auto const nbytes = data_alloc_size(rapidsmpf::MemoryType::DEVICE);
        auto statistics   = br->statistics();
        rapidsmpf::StreamOrderedTiming timing{stream(), statistics};
        auto table = std::make_unique<cudf::table>(table_view(), stream(), br->device_mr());
        statistics->record_copy(
          rapidsmpf::MemoryType::DEVICE, rapidsmpf::MemoryType::DEVICE, nbytes, std::move(timing));
        // And update the provided `reservation`.
        br->release(reservation, nbytes);
        return table_chunk(std::move(table), stream());
      }
      case rapidsmpf::MemoryType::PINNED_HOST:  // Case 1b.
      {
        rapidsmpf::StreamOrderedTiming timing{stream(), br->statistics()};

        // use cudf pack with pinned mr
        auto packed_pinned = cudf::pack(table_view(), stream(), br->pinned_mr());
        auto nbytes        = packed_pinned.gpu_data->size();

        br->statistics()->record_copy(rapidsmpf::MemoryType::DEVICE,
                                      rapidsmpf::MemoryType::PINNED_HOST,
                                      nbytes,
                                      std::move(timing));
        // update the provided `reservation`
        br->release(reservation, nbytes);
        auto host_buffer = br->move(std::move(packed_pinned.gpu_data), stream());
        return table_chunk(std::make_unique<rapidsmpf::PackedData>(
          std::move(packed_pinned.metadata), std::move(host_buffer)));
      }
      case rapidsmpf::MemoryType::HOST:  // Case 1c.
      {
        // We use libcudf's pack() to serialize `table_view()` into a
        // packed_columns and then we move the packed_columns' gpu_data to a
        // new host buffer.
        // TODO: use `cudf::chunked_pack()` with a bounce buffer. Currently,
        // `cudf::pack()` allocates device memory we haven't reserved.
        auto packed_columns = cudf::pack(table_view(), stream(), br->device_mr());
        auto packed_data    = std::make_unique<rapidsmpf::PackedData>(
          std::move(packed_columns.metadata),
          br->move(std::move(packed_columns.gpu_data), stream()));

        packed_data->data = br->move(std::move(packed_data->data), reservation);
        return table_chunk(std::move(packed_data));
      }
      default: RAPIDSMPF_FAIL("MemoryType: unknown");
    }
  }
  // `!is_available() && packed_data_ == nullptr` is an invalid state, so
  // reaching this point implies `packed_data_ != nullptr`.
  RAPIDSMPF_EXPECTS(packed_data_ != nullptr, "something went wrong");

  // Case 2. The chunk data is already packed (packed_data_ != nullptr). We need
  // to copy the packed data into the reservation-specified memory type.
  auto const nbytes = packed_data_->data->size;
  auto metadata     = std::make_unique<std::vector<std::uint8_t>>(*packed_data_->metadata);
  auto data         = br->make_buffer(nbytes, packed_data_->stream(), reservation);
  rapidsmpf::buffer_copy(br->statistics(), *data, *packed_data_->data, nbytes);
  return table_chunk(std::make_unique<rapidsmpf::PackedData>(std::move(metadata), std::move(data)));
}

std::unique_ptr<rapidsmpf::PackedData> table_chunk::into_packed_data(
  rapidsmpf::BufferResource* br) &&
{
  if (packed_data_) {
    table_view_ = std::nullopt;
    return std::move(packed_data_);
  }
  RAPIDSMPF_EXPECTS(is_available(), "table_chunk must be available; call make_available() first");
  // TODO: use `cudf::chunked_pack()` with a bounce buffer. Currently,
  // `cudf::pack()` allocates device memory we haven't reserved.
  auto packed_columns = cudf::pack(table_view_.value(), stream_, br->device_mr());
  table_view_         = std::nullopt;
  return std::make_unique<rapidsmpf::PackedData>(
    std::move(packed_columns.metadata), br->move(std::move(packed_columns.gpu_data), stream_));
}

std::pair<cudf::size_type, cudf::size_type> table_chunk::shape() const noexcept
{
  if (packed_data_ != nullptr) {
    auto view = cudf::packed_metadata_view(*packed_data_->metadata);
    return {view.num_rows(), view.num_columns()};
  }
  assert(table_view_.has_value() && "shape() called on moved-from table_chunk");
  return {table_view_->num_rows(), table_view_->num_columns()};
}

rapidsmpf::ContentDescription get_content_description(table_chunk const& obj)
{
  rapidsmpf::ContentDescription ret{obj.is_spillable()
                                      ? rapidsmpf::ContentDescription::Spillable::YES
                                      : rapidsmpf::ContentDescription::Spillable::NO};
  for (auto mem_type : rapidsmpf::MEMORY_TYPES) {
    ret.content_size(mem_type) = obj.data_alloc_size(mem_type);
  }
  return ret;
}

rapidsmpf::streaming::Message to_message(std::uint64_t sequence_number,
                                         std::unique_ptr<table_chunk> chunk)
{
  auto cd = get_content_description(*chunk);
  return rapidsmpf::streaming::Message{
    sequence_number,
    std::move(chunk),
    cd,
    [](rapidsmpf::streaming::Message const& msg,
       rapidsmpf::MemoryReservation& reservation) -> rapidsmpf::streaming::Message {
      auto const& self = msg.get<table_chunk>();
      auto chunk       = std::make_unique<table_chunk>(self.copy(reservation));
      auto cd          = get_content_description(*chunk);
      return rapidsmpf::streaming::Message{
        msg.sequence_number(), std::move(chunk), cd, msg.copy_cb()};
    }};
}

}  // namespace cudf_streaming
