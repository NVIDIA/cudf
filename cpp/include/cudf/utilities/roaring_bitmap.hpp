/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/std/cstddef>

#include <memory>
#include <span>

/**
 * @file
 * @brief Roaring bitmap APIs
 */

namespace CUDF_EXPORT cudf {

/**
 * @addtogroup utility_roaring_bitmap
 * @{
 */

/**
 * @brief Enumerates the supported roaring bitmap key types
 */
enum class roaring_bitmap_type : uint8_t {
  BITS_32 = 0,  ///< 32-bit roaring bitmap (keys are uint32)
  BITS_64 = 1   ///< 64-bit roaring bitmap (keys are uint64)
};

/**
 * @brief A type-erased wrapper around cuco's roaring bitmap supporting both 32-bit and 64-bit keys
 *
 * This class holds a span of serialized roaring bitmap data (specified 32-bit or 64-bit) on the
 * host and lazily materializes the corresponding underlying cuco roaring bitmap when
 * `materialize()` is called or implicitly when `contains_async()` is first called.
 *
 * Example usage (64-bit version):
 * @code
 *   auto bitmap = cudf::roaring_bitmap(cudf::roaring_bitmap_type::BITS_64, serialized_bitmap_data);
 *   bitmap.materialize(stream);
 *   auto result = bitmap.contains_async(keys_column, stream, mr);
 * @endcode
 */
class roaring_bitmap {
 public:
  /**
   * @brief Constructs a roaring_bitmap from serialized bitmap data (payload)
   *
   * The serialized bitmap data must remain valid until the underlying cuco roaring bitmap is
   * materialized via `materialize()`.
   *
   * @param type The bitmap key type (BITS_32 or BITS_64)
   * @param serialized_bitmap_data Host span of bytes containing a roaring bitmap serialized in
   * portable format
   *
   * @throw std::invalid_argument if the serialized bitmap data is empty
   */
  explicit roaring_bitmap(roaring_bitmap_type type,
                          std::span<cuda::std::byte const> serialized_bitmap_data);

  /**
   * @brief Destructor for the roaring bitmap class
   */
  ~roaring_bitmap();

  /**
   * @brief Move constructor for the roaring bitmap class
   *
   * @param other Roaring bitmap to move from
   */
  roaring_bitmap(roaring_bitmap&& other) noexcept;

  /**
   * @brief Move assignment operator for the roaring bitmap class
   *
   * @param other Roaring bitmap to move from
   * @return Reference to the moved-from roaring bitmap
   */
  roaring_bitmap& operator=(roaring_bitmap&& other) noexcept;

  roaring_bitmap(roaring_bitmap const&) = delete;

  roaring_bitmap& operator=(roaring_bitmap const&) = delete;

  /**
   * @brief Materialize the underlying cuco roaring bitmap
   *
   * The serialized bitmap data span is cleared after this call.
   *
   * @param stream CUDA stream used for device memory operations and kernel launches
   */
  void materialize(rmm::cuda_stream_view stream = cudf::get_default_stream()) const;

  /**
   * @brief Returns the roaring bitmap type
   *
   * @return Roaring bitmap type
   */
  [[nodiscard]] roaring_bitmap_type type() const;

  /**
   * @brief Checks whether the bitmap contains no keys
   *
   * @return Whether the roaring bitmap contains no keys
   */
  [[nodiscard]] bool empty() const;

  /**
   * @brief Returns the number of keys stored in the bitmap
   *
   * @return Number of keys stored in the bitmap
   */
  [[nodiscard]] cuda::std::size_t size() const;

  /**
   * @brief Returns the size of the serialized bitmap storage in bytes
   *
   * @return Size of the serialized bitmap storage in bytes
   */
  [[nodiscard]] cuda::std::size_t size_bytes() const;

  /**
   * @brief Asynchronously queries the bitmap for membership of each key in the input column.
   *
   * The input column must have dtype UINT32 (for `BITS_32`) or UINT64 (for `BITS_64`).
   *
   * @param keys Key column to query
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource for the output column allocation
   *
   * @return A BOOL8 column indicating positions of the present keys
   *
   * @throw std::invalid_argument if the key column dtype is invalid
   */
  [[nodiscard]] std::unique_ptr<cudf::column> contains_async(
    cudf::column_view const& keys,
    rmm::cuda_stream_view stream      = cudf::get_default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref()) const;

  /**
   * @brief Asynchronously queries the bitmap for membership of each key in a column and stores the
   * result in the output column.
   *
   * The input column must have dtype UINT32 (for `BITS_32`) or UINT64 (for `BITS_64`). The output
   * column must have BOOL8 dtype.
   *
   * @param keys Key column to query
   * @param output Output column to store the result
   * @param stream CUDA stream used for device memory operations and kernel launches
   *
   * @throw std::invalid_argument if the key or output column dtypes are invalid
   */
  void contains_async(cudf::column_view const& keys,
                      cudf::mutable_column_view const& output,
                      rmm::cuda_stream_view stream) const;

 private:
  //! Forward declaration of the opaque wrapper of cuco's roaring bitmap
  class impl;
  roaring_bitmap_type _type;
  std::unique_ptr<impl> _impl;
};

/** @} */  // end of group

}  // namespace CUDF_EXPORT cudf
