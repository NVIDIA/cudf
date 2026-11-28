/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "arrow_utilities.hpp"

#include <cudf/column/column_view.hpp>
#include <cudf/contiguous_split.hpp>
#include <cudf/detail/contiguous_split.hpp>
#include <cudf/detail/interop.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/interop.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/traits.hpp>

#include <cuda/std/concepts>

#include <nanoarrow/nanoarrow.h>
#include <nanoarrow/nanoarrow.hpp>
#include <nanoarrow/nanoarrow_device.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <map>
#include <ranges>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace cudf::experimental {
namespace {

using cudf::detail::chars_buffer_idx;
using cudf::detail::fixed_width_data_buffer_idx;
using metadata_column = packed_metadata_view::column_view;

/**
 * @brief The regions of a packed table in host memory, by column path and kind.
 */
class packed_regions {
 public:
  /**
   * @brief Read the regions of a packed table from its metadata.
   *
   * @throw std::invalid_argument if a region is compressed
   *
   * @param metadata The packed metadata
   * @param data The packed payload in host memory
   */
  packed_regions(std::span<uint8_t const> metadata, std::span<uint8_t const> data) : _data{data}
  {
    for (auto const& region : get_packed_region_info(metadata)) {
      CUDF_EXPECTS(region.codec == pack_compression::none,
                   "unpack_to_arrow_host does not support compressed packed data",
                   std::invalid_argument);
      auto const& info = region.info();
      _regions.emplace(std::pair{info.column_path, info.kind}, info);
    }
  }

  /**
   * @brief Return the first `bytes` bytes of the region of `kind` of the column at `path`.
   *
   * @throw cudf::logic_error if `bytes` is zero, since a buffer without bytes has no region
   * @throw cudf::logic_error if the metadata has no such region
   * @throw std::out_of_range if the bytes are outside of the region or of the payload
   *
   * @param path Path of the column, as in `pack_region_info::column_path`
   * @param kind Kind of the region
   * @param bytes Number of bytes to read
   * @return Pointer to the bytes in the payload
   */
  [[nodiscard]] uint8_t const* buffer(std::vector<size_type> const& path,
                                      pack_region_kind kind,
                                      std::size_t bytes) const
  {
    CUDF_EXPECTS(bytes > 0, "A buffer without bytes has no region");
    auto const it = _regions.find(std::pair{path, kind});
    CUDF_EXPECTS(it != _regions.end(), "Packed metadata is malformed");
    auto const offset = it->second.uncompressed_offset;
    CUDF_EXPECTS(bytes <= it->second.uncompressed_bytes && offset <= _data.size() &&
                   bytes <= _data.size() - offset,
                 "Packed metadata refers to bytes outside of the packed data",
                 std::out_of_range);
    return _data.data() + offset;
  }

 private:
  std::span<uint8_t const> _data;
  std::map<std::pair<std::vector<size_type>, pack_region_kind>, pack_region_info> _regions;
};

/**
 * @brief Builds Arrow host arrays from the metadata and regions of a packed table.
 *
 * Produces the same arrays as `cudf::to_arrow_host`.
 *
 * Every buffer is copied as is, except booleans, which are converted to bits.
 *
 * When `empty` is true, the column and its children are converted with no rows. This is used for
 * the keys of an empty dictionary.
 */
class packed_to_arrow {
 public:
  /**
   * @brief Construct a converter that reads from `regions`.
   *
   * @param regions The regions of the packed table
   */
  explicit packed_to_arrow(packed_regions const& regions) : _regions{regions} {}

  /**
   * @brief Convert the column `col` at `path` to `out`.
   *
   * Dispatches on the type of `col` to the private converters.
   *
   * @param col Metadata of the column
   * @param path Path of the column
   * @param empty Whether to convert the column without rows
   * @param out The array to fill
   */
  void convert(metadata_column const& col,
               std::vector<size_type> const& path,
               bool empty,
               ArrowArray* out) const
  {
    auto const rows  = empty ? 0 : col.num_rows();
    auto const nulls = empty ? 0 : col.null_count();
    switch (col.type().id()) {
      case type_id::EMPTY: init(out, NANOARROW_TYPE_NA, rows, nulls); break;
      case type_id::BOOL8: bools(path, rows, nulls, out); break;
      case type_id::STRING: strings(col, path, rows, nulls, out); break;
      case type_id::LIST: lists(col, path, rows, nulls, empty, out); break;
      case type_id::STRUCT: structs(col, path, rows, nulls, empty, out); break;
      case type_id::DICTIONARY32: dictionary(col, path, rows, nulls, out); break;
      default: fixed_width(col.type(), path, rows, nulls, out); break;
    }
  }

 private:
  /**
   * @brief Initialize `out` as an Arrow array of `type` with `rows` rows and `nulls` nulls.
   *
   * @param out The array to initialize
   * @param type The Arrow storage type
   * @param rows The number of rows
   * @param nulls The number of nulls
   */
  static void init(ArrowArray* out, ArrowType type, size_type rows, size_type nulls)
  {
    NANOARROW_THROW_NOT_OK(ArrowArrayInitFromType(out, type));
    out->length     = rows;
    out->null_count = nulls;
  }

  /**
   * @brief Copy the first `bytes` bytes of a region into the empty `buffer`.
   *
   * Leaves `buffer` empty if `bytes` is zero.
   *
   * @param buffer The empty buffer to fill
   * @param path Path of the column
   * @param kind Kind of the region
   * @param bytes The number of bytes
   */
  void copy_to(ArrowBuffer* buffer,
               std::vector<size_type> const& path,
               pack_region_kind kind,
               std::size_t bytes) const
  {
    CUDF_EXPECTS(buffer->size_bytes == 0, "copy_to requires an empty buffer");
    if (bytes > 0) {
      NANOARROW_THROW_NOT_OK(
        ArrowBufferAppend(buffer, _regions.buffer(path, kind, bytes), static_cast<int64_t>(bytes)));
    }
  }

  /**
   * @brief Copy the null mask of the column at `path` into the validity bitmap of `out`.
   *
   * Does nothing if the column has no nulls.
   *
   * @param path Path of the column
   * @param rows The number of rows
   * @param nulls The number of nulls
   * @param out The array to fill
   */
  void validity(std::vector<size_type> const& path,
                size_type rows,
                size_type nulls,
                ArrowArray* out) const
  {
    if (nulls == 0) { return; }
    auto* bitmap = ArrowArrayValidityBitmap(out);
    NANOARROW_THROW_NOT_OK(ArrowBitmapResize(bitmap, rows, 0));
    auto const bytes = static_cast<std::size_t>(bitmap->buffer.size_bytes);
    std::memcpy(
      bitmap->buffer.data, _regions.buffer(path, pack_region_kind::validity, bytes), bytes);
  }

  /**
   * @brief Return the last offset of the column at `path`.
   *
   * @tparam T The offset type
   * @param path Path of the column
   * @param rows The number of rows
   * @return The offset at index `rows`
   */
  template <cuda::std::integral T>
  [[nodiscard]] T read_last_offset(std::vector<size_type> const& path, size_type rows) const
  {
    auto const bytes = (static_cast<std::size_t>(rows) + 1) * sizeof(T);
    auto const* src  = _regions.buffer(path, pack_region_kind::offsets, bytes);
    T value;
    std::memcpy(&value, src + bytes - sizeof(T), sizeof(T));
    return value;
  }

  /// Convert the fixed-width column at `path`.
  void fixed_width(data_type type,
                   std::vector<size_type> const& path,
                   size_type rows,
                   size_type nulls,
                   ArrowArray* out) const
  {
    CUDF_EXPECTS(is_fixed_width(type), "Packed metadata is malformed");
    init(out, detail::id_to_arrow_storage_type(type.id()), rows, nulls);
    validity(path, rows, nulls, out);
    auto const bytes = static_cast<std::size_t>(rows) * size_of(type);
    copy_to(
      ArrowArrayBuffer(out, fixed_width_data_buffer_idx), path, pack_region_kind::data, bytes);
  }

  /// Convert the boolean column at `path`, one bit per value.
  void bools(std::vector<size_type> const& path,
             size_type rows,
             size_type nulls,
             ArrowArray* out) const
  {
    init(out, NANOARROW_TYPE_BOOL, rows, nulls);
    validity(path, rows, nulls, out);
    auto const num_rows = static_cast<std::size_t>(rows);
    auto* buffer        = ArrowArrayBuffer(out, fixed_width_data_buffer_idx);
    NANOARROW_THROW_NOT_OK(ArrowBufferResize(buffer, static_cast<int64_t>((num_rows + 7) / 8), 0));
    if (num_rows == 0) { return; }
    std::memset(buffer->data, 0, static_cast<std::size_t>(buffer->size_bytes));
    auto const* src = _regions.buffer(path, pack_region_kind::data, num_rows);
    std::ranges::for_each(std::views::iota(std::size_t{0}, num_rows), [&](std::size_t i) {
      if (src[i] != 0) { ArrowBitSet(buffer->data, static_cast<int64_t>(i)); }
    });
  }

  /// Convert the strings column at `path` to a string or large string array.
  void strings(metadata_column const& col,
               std::vector<size_type> const& path,
               size_type rows,
               size_type nulls,
               ArrowArray* out) const
  {
    auto const large =
      col.num_children() > 0 &&
      col.child(strings_column_view::offsets_column_index).type().id() == type_id::INT64;
    init(out, large ? NANOARROW_TYPE_LARGE_STRING : NANOARROW_TYPE_STRING, rows, nulls);
    auto* offsets_buffer = ArrowArrayBuffer(out, fixed_width_data_buffer_idx);
    if (rows == 0) {
      // a single zero offset by convention, as `cudf::to_arrow_host` writes
      NANOARROW_THROW_NOT_OK(large ? ArrowBufferAppendInt64(offsets_buffer, 0)
                                   : ArrowBufferAppendInt32(offsets_buffer, 0));
      return;
    }
    CUDF_EXPECTS(col.num_children() == 1, "Packed metadata is malformed");
    validity(path, rows, nulls, out);
    auto const offset_size = large ? sizeof(int64_t) : sizeof(int32_t);
    auto const num_offsets = static_cast<std::size_t>(rows) + 1;
    auto const chars_size  = large ? read_last_offset<int64_t>(path, rows)
                                   : static_cast<int64_t>(read_last_offset<int32_t>(path, rows));
    CUDF_EXPECTS(chars_size >= 0, "Packed data has a negative string offset");
    copy_to(offsets_buffer, path, pack_region_kind::offsets, num_offsets * offset_size);
    auto const chars_bytes = static_cast<std::size_t>(chars_size);
    copy_to(ArrowArrayBuffer(out, chars_buffer_idx),
            path,
            pack_region_kind::string_characters,
            chars_bytes);
  }

  /// Convert the lists column at `path` and its child.
  void lists(metadata_column const& col,
             std::vector<size_type> const& path,
             size_type rows,
             size_type nulls,
             bool empty,
             ArrowArray* out) const
  {
    init(out, NANOARROW_TYPE_LIST, rows, nulls);
    NANOARROW_THROW_NOT_OK(ArrowArrayAllocateChildren(out, 1));
    validity(path, rows, nulls, out);
    auto* offsets_buffer = ArrowArrayBuffer(out, fixed_width_data_buffer_idx);
    if (rows == 0) {
      NANOARROW_THROW_NOT_OK(ArrowBufferAppendInt32(offsets_buffer, 0));
    } else {
      auto const bytes = (static_cast<std::size_t>(rows) + 1) * sizeof(int32_t);
      copy_to(offsets_buffer, path, pack_region_kind::offsets, bytes);
    }
    convert_child(col, lists_column_view::child_column_index, path, empty, out->children[0]);
  }

  /// Convert the structs column at `path` and its children.
  void structs(metadata_column const& col,
               std::vector<size_type> const& path,
               size_type rows,
               size_type nulls,
               bool empty,
               ArrowArray* out) const
  {
    init(out, NANOARROW_TYPE_STRUCT, rows, nulls);
    NANOARROW_THROW_NOT_OK(ArrowArrayAllocateChildren(out, col.num_children()));
    validity(path, rows, nulls, out);
    std::ranges::for_each(std::views::iota(0, col.num_children()), [&](size_type i) {
      convert_child(col, i, path, empty, out->children[i]);
    });
  }

  /// Convert the dictionary column at `path`, with the keys as its dictionary.
  void dictionary(metadata_column const& col,
                  std::vector<size_type> const& path,
                  size_type rows,
                  size_type nulls,
                  ArrowArray* out) const
  {
    auto const has_children = col.num_children() == 2;
    auto const indices_type = has_children ? col.child(0).type() : data_type{type_id::INT32};
    init(out, detail::id_to_arrow_type(indices_type.id()), rows, nulls);
    NANOARROW_THROW_NOT_OK(ArrowArrayAllocateDictionary(out));
    validity(path, rows, nulls, out);
    auto const bytes  = static_cast<std::size_t>(rows) * size_of(indices_type);
    auto indices_path = path;
    if (has_children) { indices_path.push_back(0); }
    copy_to(ArrowArrayBuffer(out, fixed_width_data_buffer_idx),
            indices_path,
            pack_region_kind::data,
            bytes);
    if (!has_children) {
      init(out->dictionary, NANOARROW_TYPE_INT64, 0, 0);
      return;
    }
    // Like `cudf::to_arrow_host`, an empty column has no keys
    convert_child(col, 1, path, rows == 0, out->dictionary);
  }

  /// Convert child `i` of the column `col` at `path`.
  void convert_child(metadata_column const& col,
                     size_type i,
                     std::vector<size_type> const& path,
                     bool empty,
                     ArrowArray* out) const
  {
    auto child_path = path;
    child_path.push_back(i);
    convert(col.child(i), child_path, empty, out);
  }

  packed_regions const& _regions;
};

/**
 * @brief Return the properties of `col` that determine its Arrow schema.
 *
 * @param col Metadata of the column
 * @return The properties of `col` and its children
 */
cudf::detail::arrow_schema_column to_arrow_schema_column(metadata_column const& col)
{
  cudf::detail::arrow_schema_column result{
    col.type(),
    cudf::detail::packed_metadata_access::null_mask_offset(col) != -1,
    col.null_count() > 0,
    {}};
  result.children.reserve(col.num_children());
  std::ranges::transform(std::views::iota(0, col.num_children()),
                         std::back_inserter(result.children),
                         [&](size_type i) { return to_arrow_schema_column(col.child(i)); });
  return result;
}

}  // namespace

unique_device_array_t unpack_to_arrow_host(std::span<uint8_t const> metadata,
                                           std::span<uint8_t const> data)
{
  CUDF_FUNC_RANGE();
  auto const regions = packed_regions{metadata, data};
  auto const packed  = packed_metadata_view{metadata};

  nanoarrow::UniqueArray tmp;
  NANOARROW_THROW_NOT_OK(ArrowArrayInitFromType(tmp.get(), NANOARROW_TYPE_STRUCT));
  NANOARROW_THROW_NOT_OK(ArrowArrayAllocateChildren(tmp.get(), packed.num_columns()));
  tmp->length     = packed.num_rows();
  tmp->null_count = 0;

  auto const converter = packed_to_arrow{regions};
  std::ranges::for_each(std::views::iota(0, packed.num_columns()), [&](size_type i) {
    converter.convert(packed.column(i), {i}, false, tmp->children[i]);
  });

  ArrowError error;
  if (ArrowArrayFinishBuildingDefault(tmp.get(), &error) != NANOARROW_OK) {
    CUDF_FAIL(std::string{"Failed to build the Arrow array: "} + error.message);
  }

  unique_device_array_t result(new ArrowDeviceArray{}, [](ArrowDeviceArray* arr) {
    if (arr->array.release != nullptr) { ArrowArrayRelease(&arr->array); }
    delete arr;
  });
  result->device_id   = -1;
  result->device_type = ARROW_DEVICE_CPU;
  result->sync_event  = nullptr;
  ArrowArrayMove(tmp.get(), &result->array);
  return result;
}

unique_schema_t unpack_to_arrow_schema(std::span<uint8_t const> metadata,
                                       std::span<column_metadata const> columns_metadata)
{
  CUDF_FUNC_RANGE();
  auto const packed = packed_metadata_view{metadata};
  std::vector<cudf::detail::arrow_schema_column> columns;
  columns.reserve(packed.num_columns());
  std::ranges::transform(std::views::iota(0, packed.num_columns()),
                         std::back_inserter(columns),
                         [&](size_type i) { return to_arrow_schema_column(packed.column(i)); });
  return cudf::detail::to_arrow_schema(columns, columns_metadata);
}

}  // namespace cudf::experimental
