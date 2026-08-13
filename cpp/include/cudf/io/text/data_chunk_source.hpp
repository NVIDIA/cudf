/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/utilities/export.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_pool.hpp>
#include <rmm/device_buffer.hpp>

/**
 * @file
 * @brief Interfaces for providing stream-ordered chunks of device data to text-based readers.
 */

namespace CUDF_EXPORT cudf {
namespace io {
namespace text {

/**
 * @addtogroup io_readers
 * @{
 */

/**
 * @brief A contract guaranteeing stream-ordered memory access to the underlying device data.
 *
 * This class guarantees access to the underlying data for the stream on which the data was
 * allocated. Possible implementations may own the device data, or may only have a view over the
 * data. Any work enqueued to the stream on which this data was allocated is guaranteed to be
 * performed prior to the destruction of the underlying data, but otherwise no guarantees are made
 * regarding if or when the underlying data gets destroyed.
 */
class device_data_chunk {
 public:
  virtual ~device_data_chunk() = default;
  /**
   * @pure @brief Returns a pointer to the underlying device data.
   *
   * @return A pointer to the underlying device data
   */
  [[nodiscard]] virtual char const* data() const = 0;
  /**
   * @pure @brief Returns the size of the underlying device data.
   *
   * @return The size of the underlying device data
   */
  [[nodiscard]] virtual std::size_t size() const = 0;
  /**
   * @pure @brief Returns a span over the underlying device data.
   *
   * @return A span over the underlying device data
   */
  virtual operator device_span<char const>() const = 0;
};

/**
 * @brief a reader capable of producing views over device memory.
 *
 * The data chunk reader API encapsulates the idea of statefully traversing and loading a data
 * source. A data source may be a file, a region of device memory, or a region of host memory.
 * Reading data from these data sources efficiently requires different strategies depending on the
 * type of data source, type of compression, capabilities of the host and device, the data's
 * destination. Whole-file decompression should be hidden behind this interface.
 */
class data_chunk_reader {
 public:
  virtual ~data_chunk_reader() = default;
  /**
   * @pure @brief Skips the specified number of bytes in the data source.
   *
   * @param size The number of bytes to skip
   */
  virtual void skip_bytes(std::size_t size) = 0;

  /**
   * @pure @brief Get the next chunk of bytes from the data source
   *
   * Performs any necessary work to read and prepare the underlying data source for consumption as a
   * view over device memory. Common implementations may read from a file, copy data from host
   * memory, allocate temporary memory, perform iterative decompression, or even launch device
   * kernels.
   *
   * @param size number of bytes to read
   * @param stream stream to associate allocations or perform work required to obtain chunk
   * @return a chunk of data up to @p size bytes. May return less than @p size bytes if
   * reader reaches end of underlying data source. Returned data must be accessed in stream order
   * relative to the specified @p stream
   */
  virtual std::unique_ptr<device_data_chunk> get_next_chunk(std::size_t size,
                                                            rmm::cuda_stream_view stream) = 0;
};

/**
 * @brief a data source capable of creating a reader which can produce views of the data source in
 * device memory.
 */
class data_chunk_source {
 public:
  virtual ~data_chunk_source() = default;

  /**
   * @pure @brief Get a reader for the data source.
   *
   * @return `data_chunk_reader` object for the data source
   */
  [[nodiscard]] virtual std::unique_ptr<data_chunk_reader> create_reader() const = 0;
};

/** @} */  // end of group

}  // namespace text
}  // namespace io
}  // namespace CUDF_EXPORT cudf
