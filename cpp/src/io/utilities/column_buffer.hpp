/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file column_buffer.hpp
 * @brief cuDF-IO Column-backing buffer utilities
 */

#pragma once

#include <cudf/column/column_factories.hpp>
#include <cudf/io/types.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>

#include <cuda/std/utility>
#include <cuda/stream>

#include <optional>

namespace cudf {
namespace io {
namespace detail {
/**
 * @brief Creates a `device_buffer` for holding `column` data.
 *
 * @param type The intended data type to populate
 * @param size The number of elements to be represented by the mask
 * @param memset_data Defines whether data should be memset to 0
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned device_buffer
 *
 * @return `rmm::device_buffer` Device buffer allocation
 */
inline rmm::device_buffer create_data(data_type type,
                                      size_type size,
                                      bool memset_data,
                                      cuda::stream_ref stream,
                                      rmm::device_async_resource_ref mr)
{
  std::size_t data_size = size_of(type) * size;

  rmm::device_buffer data(data_size, stream, mr);
  if (memset_data) { CUDF_CUDA_TRY(cudaMemsetAsync(data.data(), 0, data_size, stream.get())); }
  return data;
}

inline rmm::device_buffer create_data(data_type type,
                                      size_type size,
                                      cuda::stream_ref stream,
                                      rmm::device_async_resource_ref mr)
{
  return create_data(type, size, true, stream, mr);
}

using string_index_pair = cuda::std::pair<char const*, size_type>;

// forward declare friend functions
template <typename string_policy>
class column_buffer_base;

/**
 * @brief Creates a column from an existing set of device memory buffers.
 *
 * @throws std::bad_alloc if device memory allocation fails
 *
 * @param buffer Column buffer descriptors
 * @param schema_info Schema information for the column to write optionally.
 * @param schema Optional schema used to control string to binary conversions.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 *
 * @return `std::unique_ptr<cudf::column>` Column from the existing device data
 */
template <class string_policy>
std::unique_ptr<column> make_column(column_buffer_base<string_policy>& buffer,
                                    column_name_info* schema_info,
                                    std::optional<reader_column_schema> const& schema,
                                    cuda::stream_ref stream);

template <typename string_policy>
class column_buffer_base {
 public:
  column_buffer_base() = default;

  // construct without a known size. call create() later to actually allocate memory
  column_buffer_base(data_type _type, bool _is_nullable) : type(_type), is_nullable(_is_nullable) {}

  column_buffer_base(data_type _type,
                     size_type _size,
                     bool _is_nullable,
                     cuda::stream_ref stream,
                     rmm::device_async_resource_ref mr)
    : column_buffer_base(_type, _is_nullable)
  {
  }

  // move constructor
  column_buffer_base(column_buffer_base&& col)            = default;
  column_buffer_base& operator=(column_buffer_base&& col) = default;

  // copy constructor
  column_buffer_base(column_buffer_base const& col)            = delete;
  column_buffer_base& operator=(column_buffer_base const& col) = delete;

  // instantiate a column of known type with a specified size.  Allows deferred creation for
  // preprocessing steps such as in the Parquet reader
  void create(size_type _size,
              bool memset_data,
              cuda::stream_ref stream,
              rmm::device_async_resource_ref mr);

  void create(size_type _size, cuda::stream_ref stream, rmm::device_async_resource_ref mr);

  // like create(), but also takes a `cudf::mask_state` to allow initializing the null mask as
  // something other than `ALL_NULL`
  void create_with_mask(size_type _size,
                        cudf::mask_state null_mask_state,
                        bool memset_data,
                        cuda::stream_ref stream,
                        rmm::device_async_resource_ref mr);

  // Create a new column_buffer that has empty data but with the same basic information as the
  // input column, including same type, nullability, name, and user_data.
  static string_policy empty_like(string_policy const& input);

  void set_null_mask(rmm::device_buffer&& mask) { _null_mask = std::move(mask); }

  template <typename T = uint32_t>
  auto null_mask()
  {
    return static_cast<T*>(_null_mask.data());
  }
  auto null_mask_size() { return _null_mask.size(); }
  auto& null_count() { return _null_count; }

  auto data() { return static_cast<string_policy*>(this)->data_impl(); }
  [[nodiscard]] auto data() const { return static_cast<string_policy const*>(this)->data_impl(); }
  [[nodiscard]] auto data_size() const
  {
    return static_cast<string_policy const*>(this)->data_size_impl();
  }

  std::unique_ptr<column> make_string_column(cuda::stream_ref stream)
  {
    return static_cast<string_policy*>(this)->make_string_column_impl(stream);
  }

 protected:
  rmm::device_buffer _data{};
  rmm::device_buffer _null_mask{};
  size_type _null_count{0};
  rmm::device_async_resource_ref _mr{cudf::get_current_device_resource_ref()};

 public:
  data_type type{type_id::EMPTY};
  bool is_nullable{false};
  size_type size{0};
  uint32_t user_data{0};  // arbitrary user data
  // Materialize a STRING/BYTE_ARRAY-backed buffer as a `list<uint8>` column
  bool string_as_binary{false};
  std::string name;

  std::vector<string_policy> children;

  friend std::unique_ptr<column> make_column<string_policy>(
    column_buffer_base& buffer,
    column_name_info* schema_info,
    std::optional<reader_column_schema> const& schema,
    cuda::stream_ref stream);
};

// column buffer that uses a string_index_pair for strings data, requiring a gather step when
// creating a string column
class gather_column_buffer : public column_buffer_base<gather_column_buffer> {
 public:
  gather_column_buffer() = default;

  // construct without a known size. call create() later to actually allocate memory
  gather_column_buffer(data_type _type, bool _is_nullable)
    : column_buffer_base<gather_column_buffer>(_type, _is_nullable)
  {
  }

  gather_column_buffer(data_type _type,
                       size_type _size,
                       bool _is_nullable,
                       cuda::stream_ref stream,
                       rmm::device_async_resource_ref mr)
    : column_buffer_base<gather_column_buffer>(_type, _size, _is_nullable, stream, mr)
  {
    create(_size, stream, mr);
  }

  void allocate_strings_data(bool memset_data, cuda::stream_ref stream);

  [[nodiscard]] void* data_impl() { return _strings ? _strings->data() : _data.data(); }
  [[nodiscard]] void const* data_impl() const { return _strings ? _strings->data() : _data.data(); }
  [[nodiscard]] size_t data_size_impl() const { return _strings ? _strings->size() : _data.size(); }

  std::unique_ptr<column> make_string_column_impl(cuda::stream_ref stream);

 public:
  std::unique_ptr<rmm::device_uvector<string_index_pair>> _strings;
};

// column buffer that stores string data internally which can be passed directly when
// creating a string column
class inline_column_buffer : public column_buffer_base<inline_column_buffer> {
 public:
  inline_column_buffer() = default;

  // construct without a known size. call create() later to actually allocate memory
  inline_column_buffer(data_type _type, bool _is_nullable)
    : column_buffer_base<inline_column_buffer>(_type, _is_nullable)
  {
  }

  inline_column_buffer(data_type _type,
                       size_type _size,
                       bool _is_nullable,
                       cuda::stream_ref stream,
                       rmm::device_async_resource_ref mr)
    : column_buffer_base<inline_column_buffer>(_type, _size, _is_nullable, stream, mr)
  {
    create(_size, stream, mr);
  }

  void allocate_strings_data(bool memset_data, cuda::stream_ref stream);

  void* data_impl() { return _data.data(); }
  [[nodiscard]] void const* data_impl() const { return _data.data(); }
  [[nodiscard]] size_t data_size_impl() const { return _data.size(); }
  std::unique_ptr<column> make_string_column_impl(cuda::stream_ref stream);

  void create_string_data(size_t num_bytes, bool is_large_strings_col, cuda::stream_ref stream);
  void set_initial_string_offset(size_t offset) { initial_string_offset = offset; }

  void* string_data() { return _string_data.data(); }
  [[nodiscard]] void const* string_data() const { return _string_data.data(); }
  [[nodiscard]] size_t string_size() const { return _string_data.size(); }
  [[nodiscard]] bool is_large_strings_column() const { return _is_large_strings_col; }

 private:
  rmm::device_buffer _string_data{};
  bool _is_large_strings_col{};
  size_t initial_string_offset{0};
};

using column_buffer = gather_column_buffer;

/**
 * @brief Creates an equivalent empty column from an existing set of device memory buffers.
 *
 * This function preserves nested column type information by producing complete/identical
 * column hierarchies.
 *
 * @throws std::bad_alloc if device memory allocation fails
 *
 * @param buffer Column buffer descriptors
 * @param schema_info Schema information for the column to write optionally.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned column's device memory
 *
 * @return `std::unique_ptr<cudf::column>` Column from the existing device data
 */
template <class string_policy>
std::unique_ptr<column> empty_like(column_buffer_base<string_policy>& buffer,
                                   column_name_info* schema_info,
                                   cuda::stream_ref stream,
                                   rmm::device_async_resource_ref mr);

/**
 * @brief Given a column_buffer, produce a formatted name string describing the type.
 *
 * @param buffer The column buffer
 *
 * @return A string describing the type of the buffer suitable for printing
 */
template <class string_policy>
std::string type_to_name(column_buffer_base<string_policy> const& buffer);

}  // namespace detail
}  // namespace io
}  // namespace cudf
