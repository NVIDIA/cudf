/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/contiguous_split.hpp>
#include <cudf/detail/contiguous_split.hpp>
#include <cudf/detail/nvtx/ranges.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace cudf {
namespace detail {

namespace {

/**
 * @brief The data that is stored as anonymous bytes in the `packed_columns` metadata
 * field.
 *
 * The metadata field of the `packed_columns` struct stores a `serialized_table_header`
 * followed by an array of these entries. This struct is exposed here because it is needed
 * by both contiguous_split, pack and unpack.
 */
struct serialized_column {
  serialized_column() = default;

  serialized_column(data_type _type,
                    size_type _size,
                    size_type _null_count,
                    int64_t _data_offset,
                    int64_t _null_mask_offset,
                    size_type _num_children)
    : type(_type),
      size(_size),
      null_count(_null_count),
      data_offset(_data_offset),
      null_mask_offset(_null_mask_offset),
      num_children(_num_children)

  {
  }

  data_type type;
  size_type size{};
  size_type null_count{};
  int64_t data_offset{};       // offset into contiguous data buffer, or -1 if column data is null
  int64_t null_mask_offset{};  // offset into contiguous data buffer, or -1 if column data is null
  size_type num_children{};
  // Explicitly pad to avoid uninitialized padding bits, allowing `serialized_column` to be bit-wise
  // comparable
  int pad{};
};

/**
 * @brief Table-level metadata stored before the serialized column entries.
 *
 * `num_rows` records the table's row count. For a zero-column table (which has no
 * columns to derive it from) it is the only source of the count. For a table with
 * columns it equals the columns' size and is validated against them on unpack.
 */
struct alignas(8) serialized_table_header {
  serialized_table_header() = default;
  serialized_table_header(size_type _num_columns, size_type _num_rows)
    : num_columns(_num_columns), num_rows(_num_rows)
  {
  }

  int32_t version{packed_metadata_version};
  size_type num_columns{};
  size_type num_rows{};
  int32_t pad{};  // Explicitly pad to avoid uninitialized padding bits
};

// The header is serialized with memcpy, so it must not contain padding bytes
// (which the value constructor would leave uninitialized in the output).
static_assert(std::has_unique_object_representations_v<serialized_table_header>);

/**
 * @brief Read the table header at `ptr`.
 *
 * @param ptr Pointer to the start of the header in the metadata buffer.
 * @param buffer_end One past the end of the metadata buffer. When non-null, the
 *        read is bounds-checked against it; when null the check is skipped.
 * @return The deserialized table header
 */
serialized_table_header read_header(std::uint8_t const* ptr,
                                    std::uint8_t const* buffer_end = nullptr)
{
  if (buffer_end) {
    CUDF_EXPECTS(std::cmp_greater_equal(buffer_end - ptr, sizeof(serialized_table_header)),
                 "packed metadata access is out of bounds");
  }
  serialized_table_header header;
  std::memcpy(&header, ptr, sizeof(serialized_table_header));
  CUDF_EXPECTS(header.version == packed_metadata_version,
               "packed metadata has an unsupported format version");
  CUDF_EXPECTS(header.num_columns >= 0, "packed metadata header has negative column count");
  CUDF_EXPECTS(header.num_rows >= 0, "packed metadata header has negative row count");
  return header;
}

// Read a serialized_column entry at `ptr`, optionally checking that the read
// stays within [ptr, buffer_end).  When buffer_end is nullptr the check is
// skipped (used by the internal unpack path which has its own validation).
serialized_column read_entry(std::uint8_t const* ptr, std::uint8_t const* buffer_end = nullptr)
{
  if (buffer_end) {
    CUDF_EXPECTS(std::cmp_greater_equal(buffer_end - ptr, sizeof(serialized_column)),
                 "packed metadata access is out of bounds");
  }
  serialized_column entry;
  std::memcpy(&entry, ptr, sizeof(serialized_column));
  CUDF_EXPECTS(entry.num_children >= 0, "packed metadata column has negative child count");
  return entry;
}

// Returns the total number of serialized_column entries in the subtree
// rooted at the entry at `ptr` (including that entry itself).
size_type subtree_size(std::uint8_t const* ptr, std::uint8_t const* buffer_end = nullptr)
{
  auto entry          = read_entry(ptr, buffer_end);
  size_type count     = 1;
  size_type remaining = entry.num_children;
  while (remaining > 0) {
    ptr += sizeof(serialized_column);
    entry = read_entry(ptr, buffer_end);
    ++count;
    remaining += entry.num_children - 1;
  }
  return count;
}

// Advance past `n` consecutive subtrees starting at `ptr`, returning
// a pointer to the first byte after the skipped subtrees.
uint8_t const* skip_subtrees(std::uint8_t const* ptr,
                             size_type n,
                             std::uint8_t const* buffer_end = nullptr)
{
  for (size_type i = 0; i < n; ++i) {
    ptr += subtree_size(ptr, buffer_end) * sizeof(serialized_column);
  }
  return ptr;
}

/**
 * @brief Deserialize a single column into a column_view
 *
 * Deserializes a single column (it's children are assumed to be already deserialized)
 * non-recursively.
 *
 * @param serial_column Serialized column information
 * @param children Children for the column
 * @param base_ptr Base pointer for the entire contiguous buffer from which all columns
 * were serialized into
 * @return Fully formed column_view
 */
column_view deserialize_column(serialized_column serial_column,
                               std::vector<column_view> const& children,
                               uint8_t const* base_ptr)
{
  auto const data_ptr =
    serial_column.data_offset != -1 ? base_ptr + serial_column.data_offset : nullptr;
  auto const null_mask_ptr =
    serial_column.null_mask_offset != -1
      ? reinterpret_cast<bitmask_type const*>(base_ptr + serial_column.null_mask_offset)
      : nullptr;

  return column_view(serial_column.type,
                     serial_column.size,
                     data_ptr,
                     null_mask_ptr,
                     serial_column.null_count,
                     0,
                     children);
}

/**
 * @brief Build and add metadata for a column and all of it's children, recursively
 *
 *
 * @param mb metadata_builder instance
 * @param col Column to build metadata for
 * @param base_ptr Base pointer for the entire contiguous buffer from which all columns
 * were serialized into
 * @param data_size Size of the incoming buffer
 */
void build_column_metadata(metadata_builder& mb,
                           column_view const& col,
                           uint8_t const* base_ptr,
                           size_t data_size)
{
  uint8_t const* data_ptr = col.size() == 0 || !col.head<uint8_t>() ? nullptr : col.head<uint8_t>();
  if (data_ptr != nullptr) {
    CUDF_EXPECTS(data_ptr >= base_ptr && data_ptr < base_ptr + data_size,
                 "Encountered column data outside the range of input buffer");
  }
  int64_t const data_offset = data_ptr ? data_ptr - base_ptr : -1;

  uint8_t const* null_mask_ptr = col.size() == 0 || !col.nullable()
                                   ? nullptr
                                   : reinterpret_cast<uint8_t const*>(col.null_mask());
  if (null_mask_ptr != nullptr) {
    CUDF_EXPECTS(null_mask_ptr >= base_ptr && null_mask_ptr < base_ptr + data_size,
                 "Encountered column null mask outside the range of input buffer");
  }
  int64_t const null_mask_offset = null_mask_ptr ? null_mask_ptr - base_ptr : -1;

  // add metadata
  mb.add_column_info_to_meta(
    col.type(), col.size(), col.null_count(), data_offset, null_mask_offset, col.num_children());

  std::for_each(
    col.child_begin(), col.child_end(), [&mb, &base_ptr, &data_size](column_view const& col) {
      build_column_metadata(mb, col, base_ptr, data_size);
    });
}

table_view unpack(uint8_t const* metadata, uint8_t const* gpu_data)
{
  // gpu data can be null if everything is empty but the metadata must always be valid
  CUDF_EXPECTS(metadata != nullptr, "Encountered invalid packed column input");
  uint8_t const* base_ptr = gpu_data;
  auto const header       = read_header(metadata);
  auto const num_columns  = header.num_columns;
  auto const num_rows     = header.num_rows;
  // current_ptr tracks position in the metadata byte buffer
  auto const* current_ptr = metadata + sizeof(serialized_table_header);

  std::function<std::vector<column_view>(size_type)> get_columns;
  get_columns = [&current_ptr, base_ptr, &get_columns](size_t num_columns) {
    std::vector<column_view> cols;
    for (size_t i = 0; i < num_columns; i++) {
      auto serial_column = read_entry(current_ptr);
      current_ptr += sizeof(serialized_column);

      std::vector<column_view> const children = get_columns(serial_column.num_children);

      cols.emplace_back(deserialize_column(serial_column, children, base_ptr));
    }

    return cols;
  };

  auto const cols = get_columns(num_columns);
  if (num_columns == 0) {
    // A zero-column table has no columns to derive its row count from; use the
    // count recorded in the table header.
    return table_view{std::vector<column_view>{}, num_rows};
  }
  // For a table with columns the row count is derived from the columns.
  CUDF_EXPECTS(num_rows == cols.front().size(),
               "packed metadata row count does not match the columns");
  return table_view{cols};
}

}  // anonymous namespace

/**
 * @copydoc cudf::detail::pack
 */
packed_columns pack(cudf::table_view const& input,
                    rmm::cuda_stream_view stream,
                    rmm::device_async_resource_ref mr)
{
  // do a contiguous_split with no splits to get the memory for the table
  // arranged as we want it
  auto contig_split_result = cudf::detail::contiguous_split(input, {}, stream, mr);
  return contig_split_result.empty() ? packed_columns{} : std::move(contig_split_result[0].data);
}

std::vector<uint8_t> pack_metadata(table_view const& table,
                                   uint8_t const* contiguous_buffer,
                                   size_t buffer_size,
                                   metadata_builder& builder)
{
  std::for_each(
    table.begin(), table.end(), [&builder, contiguous_buffer, buffer_size](column_view const& col) {
      build_column_metadata(builder, col, contiguous_buffer, buffer_size);
    });

  return builder.build();
}

class metadata_builder_impl {
 public:
  metadata_builder_impl(size_type const num_root_columns, std::optional<size_type> const num_rows)
    : _num_root_columns(num_root_columns), _num_rows(num_rows)
  {
    // Lower bound: exact for flat tables but nested children add more entries and grow the vector.
    _columns.reserve(num_root_columns);
  }

  void add_column_info_to_meta(data_type const col_type,
                               size_type const col_size,
                               size_type const col_null_count,
                               int64_t const data_offset,
                               int64_t const null_mask_offset,
                               size_type const num_children)
  {
    if (_num_rows.has_value() && _columns.empty()) {
      CUDF_EXPECTS(col_size == _num_rows.value(),
                   "num_rows does not match the size of the table's columns",
                   std::invalid_argument);
    }
    _columns.emplace_back(
      col_type, col_size, col_null_count, data_offset, null_mask_offset, num_children);
  }

  [[nodiscard]] std::vector<uint8_t> build() const
  {
    // The header always records the table row count. Either the first top-level column's
    // size for a table with columns, or the explicit count for a zero-column table.
    auto const num_rows = _columns.empty() ? _num_rows.value_or(0) : _columns.front().size;
    auto const header   = serialized_table_header{_num_root_columns, num_rows};
    auto output         = std::vector<uint8_t>(sizeof(serialized_table_header) +
                                       _columns.size() * sizeof(serialized_column));
    std::memcpy(output.data(), &header, sizeof(serialized_table_header));
    if (!_columns.empty()) {
      std::memcpy(output.data() + sizeof(serialized_table_header),
                  _columns.data(),
                  _columns.size() * sizeof(serialized_column));
    }
    return output;
  }

  void clear() { _columns.clear(); }

 private:
  // Number of top-level columns (excludes nested children) stored in the header.
  size_type const _num_root_columns;
  // Explicit table row count, recorded in the header only for a zero-column table.
  std::optional<size_type> const _num_rows;
  // Serialized column entries, depth-first with each column written before its children.
  std::vector<serialized_column> _columns;
};

metadata_builder::metadata_builder(size_type const num_root_columns,
                                   std::optional<size_type> const num_rows)
  : impl(std::make_unique<metadata_builder_impl>(num_root_columns, num_rows))
{
}

metadata_builder::~metadata_builder() = default;

void metadata_builder::add_column_info_to_meta(data_type const col_type,
                                               size_type const col_size,
                                               size_type const col_null_count,
                                               int64_t const data_offset,
                                               int64_t const null_mask_offset,
                                               size_type const num_children)
{
  impl->add_column_info_to_meta(
    col_type, col_size, col_null_count, data_offset, null_mask_offset, num_children);
}

std::vector<uint8_t> metadata_builder::build() const { return impl->build(); }

void metadata_builder::clear() { impl->clear(); }

}  // namespace detail

packed_metadata_view::column_view::column_view(std::span<uint8_t const> buffer) : _buffer(buffer)
{
  auto const entry = detail::read_entry(_buffer.data(), _buffer.data() + _buffer.size());
  _type            = entry.type;
  _size            = entry.size;
  _null_count      = entry.null_count;
  _num_children    = entry.num_children;
}

data_type packed_metadata_view::column_view::type() const { return _type; }

size_type packed_metadata_view::column_view::num_rows() const { return _size; }

size_type packed_metadata_view::column_view::null_count() const { return _null_count; }

size_type packed_metadata_view::column_view::num_children() const { return _num_children; }

packed_metadata_view::column_view packed_metadata_view::column_view::child(size_type i) const
{
  CUDF_EXPECTS(i >= 0 && i < _num_children, "child index out of range", std::out_of_range);
  auto const* end = _buffer.data() + _buffer.size();
  // Children start immediately after this entry in pre-order layout.
  auto const* child_ptr =
    detail::skip_subtrees(_buffer.data() + sizeof(detail::serialized_column), i, end);
  return packed_metadata_view::column_view{{child_ptr, end}};
}

packed_metadata_view::packed_metadata_view(std::span<uint8_t const> buffer)
{
  CUDF_EXPECTS(!buffer.empty(), "metadata buffer must not be empty");
  CUDF_EXPECTS(buffer.size() >= sizeof(detail::serialized_table_header),
               "metadata buffer too small");
  CUDF_EXPECTS(
    (buffer.size() - sizeof(detail::serialized_table_header)) % sizeof(detail::serialized_column) ==
      0,
    "metadata buffer size is not a valid header plus column entry size");
  auto const* end     = buffer.data() + buffer.size();
  auto const* entries = buffer.data() + sizeof(detail::serialized_table_header);
  auto const header   = detail::read_header(buffer.data(), end);
  _num_columns        = header.num_columns;
  _num_rows           = header.num_rows;
  // Walk the top-level columns once to validate two things: every top-level column's size agrees
  // with the recorded row count and the column tree exactly fills the buffer.
  auto const* ptr = entries;
  for (size_type i = 0; i < _num_columns; ++i) {
    auto const entry = detail::read_entry(ptr, end);
    CUDF_EXPECTS(entry.size == _num_rows, "packed metadata row count does not match the columns");
    ptr = detail::skip_subtrees(ptr, 1, end);
  }
  CUDF_EXPECTS(ptr == end, "packed metadata buffer size does not match the encoded column tree");
  _entries = {entries, end};
}

size_type packed_metadata_view::num_columns() const { return _num_columns; }

size_type packed_metadata_view::num_rows() const { return _num_rows; }

packed_metadata_view::column_view packed_metadata_view::column(size_type i) const
{
  CUDF_EXPECTS(i >= 0 && i < _num_columns, "column index out of range", std::out_of_range);
  auto const* end    = _entries.data() + _entries.size();
  auto const* target = detail::skip_subtrees(_entries.data(), i, end);
  return packed_metadata_view::column_view{{target, end}};
}

/**
 * @copydoc cudf::pack
 */
packed_columns pack(cudf::table_view const& input,
                    rmm::cuda_stream_view stream,
                    rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::pack(input, stream, mr);
}

/**
 * @copydoc cudf::pack_metadata
 */
std::vector<uint8_t> pack_metadata(table_view const& table,
                                   uint8_t const* contiguous_buffer,
                                   size_t buffer_size)
{
  CUDF_FUNC_RANGE();
  // A truly empty table (no columns and no rows) serializes to an empty buffer.
  // A zero-column table with a non-zero row count still emits a metadata buffer
  // whose table header records the row count, so the count round-trips.
  if (table.num_columns() == 0 && table.num_rows() == 0) { return std::vector<uint8_t>{}; }

  // Only a zero-column table records a row count. For tables with columns the row count
  // comes from the columns.
  auto const num_rows =
    table.num_columns() == 0 ? std::optional<size_type>{table.num_rows()} : std::nullopt;
  auto builder = cudf::detail::metadata_builder(table.num_columns(), num_rows);
  return detail::pack_metadata(table, contiguous_buffer, buffer_size, builder);
}

table_view unpack(packed_columns const& input)
{
  CUDF_FUNC_RANGE();
  return input.metadata->size() == 0
           ? table_view{}
           : detail::unpack(input.metadata->data(),
                            reinterpret_cast<uint8_t const*>(input.gpu_data->data()));
}

table_view unpack(uint8_t const* metadata, uint8_t const* gpu_data)
{
  CUDF_FUNC_RANGE();
  return detail::unpack(metadata, gpu_data);
}

}  // namespace cudf
