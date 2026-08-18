/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file writer_impl.cu
 * @brief cuDF-IO ORC writer class implementation
 */

#include "io/comp/compression.hpp"
#include "io/orc/orc_gpu.hpp"
#include "io/statistics/column_statistics.cuh"
#include "writer_impl.hpp"

#include <cudf/detail/iterator.cuh>
#include <cudf/detail/null_mask.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/utilities/batched_memcpy.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/cuda_memcpy.hpp>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/detail/utilities/stream_pool.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/aligned.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>
#include <rmm/mr/polymorphic_allocator.hpp>

#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cuda/functional>
#include <cuda/iterator>
#include <cuda/std/climits>
#include <cuda/std/limits>
#include <cuda/std/optional>
#include <cuda/std/utility>
#include <cuda/stream>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
#include <thrust/for_each.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/tabulate.h>
#include <thrust/transform.h>

#include <algorithm>
#include <cstring>
#include <functional>
#include <numeric>
#include <tuple>
#include <utility>

namespace cudf::io::orc::detail {

// Alignment of every non-empty extent within the encoded and gathered arenas.
constexpr size_t extent_alignment = rmm::CUDA_ALLOCATION_ALIGNMENT;

template <typename T>
[[nodiscard]] CUDF_HOST_DEVICE constexpr int varint_size(T val)
{
  auto len = 1u;
  while (val > 0x7f) {
    val >>= 7;
    ++len;
  }
  return len;
}

struct row_group_index_info {
  int32_t pos       = -1;  // Position
  int32_t blk_pos   = -1;  // Block Position
  int32_t comp_pos  = -1;  // Compressed Position
  int32_t comp_size = -1;  // Compressed size
};

namespace {
/**
 * @brief Translates cuDF compression to ORC compression.
 */
CompressionKind to_orc_compression(compression_type compression)
{
  switch (compression) {
    case compression_type::AUTO:
    case compression_type::SNAPPY: return CompressionKind::SNAPPY;
    case compression_type::ZLIB: return CompressionKind::ZLIB;
    case compression_type::ZSTD: return CompressionKind::ZSTD;
    case compression_type::LZ4: return CompressionKind::LZ4;
    case compression_type::NONE: return CompressionKind::NONE;
    default: CUDF_FAIL("Unsupported compression type");
  }
}

/**
 * @brief Returns the block size for a given compression format.
 */
size_t compression_block_size(compression_type compression)
{
  auto const comp_limit = compress_max_allowed_chunk_size(compression);

  constexpr size_t max_block_size = 256 * 1024;
  return std::min(comp_limit.value_or(max_block_size), max_block_size);
}

/**
 * @brief Translates cuDF dtype to ORC datatype
 */
constexpr TypeKind to_orc_type(cudf::type_id id, bool list_column_as_map)
{
  switch (id) {
    case cudf::type_id::INT8: return TypeKind::BYTE;
    case cudf::type_id::INT16: return TypeKind::SHORT;
    case cudf::type_id::INT32: return TypeKind::INT;
    case cudf::type_id::INT64: return TypeKind::LONG;
    case cudf::type_id::FLOAT32: return TypeKind::FLOAT;
    case cudf::type_id::FLOAT64: return TypeKind::DOUBLE;
    case cudf::type_id::BOOL8: return TypeKind::BOOLEAN;
    case cudf::type_id::TIMESTAMP_DAYS: return TypeKind::DATE;
    case cudf::type_id::TIMESTAMP_SECONDS:
    case cudf::type_id::TIMESTAMP_MICROSECONDS:
    case cudf::type_id::TIMESTAMP_MILLISECONDS:
    case cudf::type_id::TIMESTAMP_NANOSECONDS: return TypeKind::TIMESTAMP;
    case cudf::type_id::STRING: return TypeKind::STRING;
    case cudf::type_id::DECIMAL32:
    case cudf::type_id::DECIMAL64:
    case cudf::type_id::DECIMAL128: return TypeKind::DECIMAL;
    case cudf::type_id::LIST: return list_column_as_map ? TypeKind::MAP : TypeKind::LIST;
    case cudf::type_id::STRUCT: return TypeKind::STRUCT;
    default: return TypeKind::INVALID_TYPE_KIND;
  }
}

/**
 * @brief Translates time unit to nanoscale multiple.
 */
constexpr int32_t to_clockscale(cudf::type_id timestamp_id)
{
  switch (timestamp_id) {
    case cudf::type_id::TIMESTAMP_SECONDS: return 9;
    case cudf::type_id::TIMESTAMP_MILLISECONDS: return 6;
    case cudf::type_id::TIMESTAMP_MICROSECONDS: return 3;
    case cudf::type_id::TIMESTAMP_NANOSECONDS:
    default: return 0;
  }
}

/**
 * @brief Returns the precision of the given decimal type.
 */
constexpr auto orc_precision(cudf::type_id decimal_id)
{
  using namespace numeric;
  switch (decimal_id) {
    case cudf::type_id::DECIMAL32: return cuda::std::numeric_limits<decimal32::rep>::digits10;
    case cudf::type_id::DECIMAL64: return cuda::std::numeric_limits<decimal64::rep>::digits10;
    case cudf::type_id::DECIMAL128: return cuda::std::numeric_limits<decimal128::rep>::digits10;
    default: return 0;
  }
}

}  // namespace

/**
 * @brief Helper class that adds ORC-specific column info
 */
class orc_column_view {
 public:
  /**
   * @brief Constructor that extracts out the string position + length pairs
   * for building dictionaries for string columns
   */
  explicit orc_column_view(uint32_t index,
                           int str_idx,
                           orc_column_view* parent,
                           column_view const& col,
                           column_in_metadata const& metadata)
    : cudf_column{col},
      _index{index},
      _str_idx{str_idx},
      _is_child{parent != nullptr},
      _type_width{cudf::is_fixed_width(col.type()) ? cudf::size_of(col.type()) : 0},
      _type_kind{to_orc_type(col.type().id(), metadata.is_map())},
      _scale{(_type_kind == TypeKind::DECIMAL) ? -col.type().scale()
                                               : to_clockscale(col.type().id())},
      _precision{metadata.is_decimal_precision_set() ? metadata.get_decimal_precision()
                                                     : orc_precision(col.type().id())},
      name{metadata.get_name()}
  {
    if (metadata.is_nullability_defined()) { nullable_from_metadata = metadata.nullable(); }
    if (parent != nullptr) {
      parent->add_child(_index);
      _parent_index = parent->index();
    }

    if (_type_kind == TypeKind::MAP) {
      auto const struct_col = col.child(lists_column_view::child_column_index);
      CUDF_EXPECTS(struct_col.null_count() == 0,
                   "struct column of a MAP column should not have null elements");
      CUDF_EXPECTS(struct_col.num_children() == 2, "MAP column must have two child columns");
    }
  }

  void add_child(uint32_t child_idx) { children.emplace_back(child_idx); }

  auto type() const noexcept { return cudf_column.type(); }
  auto is_string() const noexcept { return cudf_column.type().id() == type_id::STRING; }

  void attach_rowgroup_char_counts(host_span<size_type const> counts)
  {
    rowgroup_char_counts = counts;
  }

  [[nodiscard]] auto rowgroup_char_count(size_type rg_idx) const
  {
    return rowgroup_char_counts[rg_idx];
  }

  [[nodiscard]] auto char_count() const
  {
    return std::accumulate(rowgroup_char_counts.begin(), rowgroup_char_counts.end(), size_type{0});
  }

  [[nodiscard]] auto const& decimal_offsets() const { return d_decimal_offsets; }
  void attach_decimal_offsets(uint32_t* sizes_ptr) { d_decimal_offsets = sizes_ptr; }

  void attach_stripe_dicts(host_span<stripe_dictionary const> host_stripe_dicts,
                           device_span<stripe_dictionary const> dev_stripe_dicts)
  {
    stripe_dicts   = host_stripe_dicts;
    d_stripe_dicts = dev_stripe_dicts;
  }

  [[nodiscard]] auto const& host_stripe_dict(size_t stripe) const
  {
    CUDF_EXPECTS(is_string(), "Stripe dictionary is only present in string columns.");
    return stripe_dicts[stripe];
  }

  [[nodiscard]] auto const& device_stripe_dicts() const noexcept { return d_stripe_dicts; }

  // Index in the table
  [[nodiscard]] uint32_t index() const noexcept { return _index; }
  // Index in the table, including only string columns
  [[nodiscard]] uint32_t str_index() const noexcept { return _str_idx; }
  // Id in the ORC file
  [[nodiscard]] auto id() const noexcept { return _index + 1; }

  [[nodiscard]] auto is_child() const noexcept { return _is_child; }
  auto parent_index() const noexcept { return _parent_index.value(); }
  auto child_begin() const noexcept { return children.cbegin(); }
  auto child_end() const noexcept { return children.cend(); }
  auto num_children() const noexcept { return children.size(); }

  [[nodiscard]] auto type_width() const noexcept { return _type_width; }
  auto size() const noexcept { return cudf_column.size(); }

  auto null_count() const noexcept { return cudf_column.null_count(); }
  auto null_mask() const noexcept { return cudf_column.null_mask(); }
  [[nodiscard]] bool nullable() const noexcept { return null_mask() != nullptr; }
  auto user_defined_nullable() const noexcept { return nullable_from_metadata; }

  [[nodiscard]] auto scale() const noexcept { return _scale; }
  [[nodiscard]] auto precision() const noexcept { return _precision; }

  void set_orc_encoding(ColumnEncodingKind e) noexcept { _encoding_kind = e; }
  [[nodiscard]] auto orc_kind() const noexcept { return _type_kind; }
  [[nodiscard]] auto orc_encoding() const noexcept { return _encoding_kind; }
  [[nodiscard]] std::string_view orc_name() const noexcept { return name; }

 private:
  column_view cudf_column;

  // Identifier within the set of columns
  uint32_t _index = 0;
  // Identifier within the set of string columns
  int _str_idx;
  bool _is_child = false;

  // ORC-related members
  TypeKind _type_kind               = INVALID_TYPE_KIND;
  ColumnEncodingKind _encoding_kind = INVALID_ENCODING_KIND;
  std::string name;

  size_t _type_width = 0;
  int32_t _scale     = 0;
  int32_t _precision = 0;

  host_span<size_type const> rowgroup_char_counts;

  host_span<stripe_dictionary const> stripe_dicts;
  device_span<stripe_dictionary const> d_stripe_dicts;

  // Offsets for encoded decimal elements. Used to enable direct writing of encoded decimal elements
  // into the output stream.
  uint32_t* d_decimal_offsets = nullptr;

  std::optional<bool> nullable_from_metadata;
  std::vector<uint32_t> children;
  std::optional<uint32_t> _parent_index;
};

/**
 * Non-owning view of a cuDF table that includes ORC-related information.
 *
 * Columns hierarchy is flattened and stored in pre-order.
 */
struct orc_table_view {
  std::vector<orc_column_view> columns;
  rmm::device_uvector<orc_column_device_view> d_columns;
  std::vector<uint32_t> string_column_indices;
  rmm::device_uvector<uint32_t> d_string_column_indices;

  [[nodiscard]] auto num_columns() const noexcept { return columns.size(); }
  [[nodiscard]] size_type num_rows() const noexcept
  {
    return columns.empty() ? 0 : columns.front().size();
  }
  [[nodiscard]] auto num_string_columns() const noexcept { return string_column_indices.size(); }

  auto& column(uint32_t idx) { return columns.at(idx); }
  [[nodiscard]] auto const& column(uint32_t idx) const { return columns.at(idx); }

  auto& string_column(uint32_t idx) { return columns.at(string_column_indices.at(idx)); }
  [[nodiscard]] auto const& string_column(uint32_t idx) const
  {
    return columns.at(string_column_indices.at(idx));
  }
};

namespace {
struct string_length_functor {
  __device__ inline size_type operator()(int const i) const
  {
    // we translate from 0 -> num_chunks * 2 because each statistic has a min and max
    // string and we need to calculate lengths for both.
    if (i >= num_chunks * 2) return 0;

    // min strings are even values, max strings are odd values of i
    auto const should_copy_min = i % 2 == 0;
    // index of the chunk
    auto const idx = i / 2;
    auto& str_val  = should_copy_min ? stripe_stat_chunks[idx].min_value.str_val
                                     : stripe_stat_chunks[idx].max_value.str_val;
    auto const str = stripe_stat_merge[idx].stats_dtype == dtype_string;
    return str ? str_val.length : 0;
  }

  int const num_chunks;
  statistics_chunk const* stripe_stat_chunks;
  statistics_merge_group const* stripe_stat_merge;
};

CUDF_KERNEL void copy_string_data(char* string_pool,
                                  size_type* offsets,
                                  statistics_chunk* chunks,
                                  statistics_merge_group const* groups)
{
  auto const idx = blockIdx.x / 2;
  if (groups[idx].stats_dtype == dtype_string) {
    // min strings are even values, max strings are odd values of i
    auto const should_copy_min = blockIdx.x % 2 == 0;
    auto& str_val = should_copy_min ? chunks[idx].min_value.str_val : chunks[idx].max_value.str_val;
    auto dst      = &string_pool[offsets[blockIdx.x]];
    auto src      = str_val.ptr;

    for (thread_index_type i = threadIdx.x; i < str_val.length; i += blockDim.x) {
      dst[i] = src[i];
    }
    if (threadIdx.x == 0) { str_val.ptr = dst; }
  }
}

}  // namespace

intermediate_statistics::intermediate_statistics(orc_table_view const& table,
                                                 cuda::stream_ref stream)
  : stripe_stat_chunks(0, stream)
{
  std::transform(
    table.columns.begin(), table.columns.end(), std::back_inserter(col_types), [](auto const& col) {
      return col.type();
    });
}

void persisted_statistics::persist(uint64_t num_table_rows,
                                   single_write_mode write_mode,
                                   intermediate_statistics&& intermediate_stats,
                                   cuda::stream_ref stream)
{
  col_types = std::move(intermediate_stats.col_types);
  num_rows += num_table_rows;
  if (num_table_rows == 0) { return; }
  stats_dtypes = std::move(intermediate_stats.stats_dtypes);

  if (write_mode == single_write_mode::NO) {
    // persist the strings in the chunks into a string pool and update pointers
    auto const num_chunks = static_cast<int>(intermediate_stats.stripe_stat_chunks.size());
    // min offset and max offset + 1 for total size
    rmm::device_uvector<size_type> offsets((num_chunks * 2) + 1, stream);

    auto iter = cudf::detail::make_counting_transform_iterator(
      0,
      string_length_functor{num_chunks,
                            intermediate_stats.stripe_stat_chunks.data(),
                            intermediate_stats.stripe_stat_merge.device_ptr()});
    thrust::exclusive_scan(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                           iter,
                           iter + offsets.size(),
                           offsets.begin());

    // pull size back to host
    auto const total_string_pool_size = offsets.element(num_chunks * 2, stream);
    if (total_string_pool_size > 0) {
      rmm::device_uvector<char> string_pool(total_string_pool_size, stream);

      // offsets describes where in the string pool each string goes. Going with the simple
      // approach for now, but it is possible something fancier with breaking up each thread into
      // copying x bytes instead of a single string is the better method since we are dealing in
      // min/max strings they almost certainly will not be uniform length.
      copy_string_data<<<num_chunks * 2, 256, 0, stream.get()>>>(
        string_pool.data(),
        offsets.data(),
        intermediate_stats.stripe_stat_chunks.data(),
        intermediate_stats.stripe_stat_merge.device_ptr());
      CUDF_CUDA_TRY(cudaGetLastError());
      string_pools.emplace_back(std::move(string_pool));
    }
  }

  stripe_stat_chunks.emplace_back(std::move(intermediate_stats.stripe_stat_chunks));
  stripe_stat_merge.emplace_back(std::move(intermediate_stats.stripe_stat_merge));
}

namespace {
/**
 * @brief Gathers stripe information.
 *
 * @param columns List of columns
 * @param rowgroup_bounds Ranges of rows in each rowgroup [rowgroup][column]
 * @param max_stripe_size Maximum size of each stripe, both in bytes and in rows
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @return List of stripe descriptors
 */
file_segmentation calculate_segmentation(host_span<orc_column_view const> columns,
                                         hostdevice_2dvector<rowgroup_rows>&& rowgroup_bounds,
                                         stripe_size_limits max_stripe_size,
                                         cuda::stream_ref stream)
{
  // Number of stripes is not known in advance. Only reserve a single element to use pinned memory
  // resource if at all enabled.
  auto infos                    = cudf::detail::make_empty_host_vector<stripe_rowgroups>(1, stream);
  size_type const num_rowgroups = rowgroup_bounds.size().first;
  size_type stripe_start        = 0;
  size_t stripe_bytes           = 0;
  size_type stripe_rows         = 0;
  for (size_type rg_idx = 0; rg_idx < num_rowgroups; ++rg_idx) {
    auto const rowgroup_total_bytes =
      std::accumulate(columns.begin(), columns.end(), 0ul, [&](size_t total_size, auto const& col) {
        auto const rows = rowgroup_bounds[rg_idx][col.index()].size();
        if (col.is_string()) {
          return total_size + rows + col.rowgroup_char_count(rg_idx);
        } else {
          return total_size + col.type_width() * rows;
        }
      });

    auto const rowgroup_rows_max =
      std::max_element(rowgroup_bounds[rg_idx].begin(),
                       rowgroup_bounds[rg_idx].end(),
                       [](auto& l, auto& r) { return l.size() < r.size(); })
        ->size();
    // Check if adding the current rowgroup to the stripe will make the stripe too large or long
    if ((rg_idx > stripe_start) && (stripe_bytes + rowgroup_total_bytes > max_stripe_size.bytes ||
                                    stripe_rows + rowgroup_rows_max > max_stripe_size.rows)) {
      infos.push_back(stripe_rowgroups{static_cast<size_type>(infos.size()),
                                       stripe_start,
                                       static_cast<size_type>(rg_idx - stripe_start)});
      stripe_start = rg_idx;
      stripe_bytes = 0;
      stripe_rows  = 0;
    }

    stripe_bytes += rowgroup_total_bytes;
    stripe_rows += rowgroup_rows_max;
    if (rg_idx + 1 == num_rowgroups) {
      infos.push_back(stripe_rowgroups{static_cast<size_type>(infos.size()),
                                       stripe_start,
                                       static_cast<size_type>(num_rowgroups - stripe_start)});
    }
  }

  return {std::move(rowgroup_bounds), std::move(infos)};
}

/**
 * @brief Returns the maximum size of RLE encoded values of an integer type.
 **/
template <typename T>
size_t max_varint_size()
{
  // varint encodes 7 bits in each byte
  return cudf::util::div_rounding_up_unsafe(sizeof(T) * 8, 7);
}

size_t rle_stream_size(TypeKind kind, size_t count)
{
  using cudf::util::div_rounding_up_unsafe;
  constexpr auto byte_rle_max_len = 128;
  switch (kind) {
    case TypeKind::BOOLEAN:
      return div_rounding_up_unsafe(count, byte_rle_max_len * 8) * (byte_rle_max_len + 1);
    case TypeKind::BYTE:
      return div_rounding_up_unsafe(count, byte_rle_max_len) * (byte_rle_max_len + 1);
    case TypeKind::SHORT:
      return div_rounding_up_unsafe(count, encode_block_size) *
             (encode_block_size * max_varint_size<int16_t>() + 2);
    case TypeKind::FLOAT:
    case TypeKind::INT:
    case TypeKind::DATE:
      return div_rounding_up_unsafe(count, encode_block_size) *
             (encode_block_size * max_varint_size<int32_t>() + 2);
    case TypeKind::LONG:
    case TypeKind::DOUBLE:
      return div_rounding_up_unsafe(count, encode_block_size) *
             (encode_block_size * max_varint_size<int64_t>() + 2);
    default: CUDF_FAIL("Unsupported ORC type for RLE stream size: " + std::to_string(kind));
  }
}

/**
 * @brief Builds up per-column streams.
 *
 * @param[in,out] columns List of columns
 * @param[in] segmentation stripe and rowgroup ranges
 * @param[in] decimal_column_sizes Sizes of encoded decimal columns
 * @param[in] enable_dictionary Whether dictionary encoding is enabled
 * @param[in] compression Compression type to use
 * @param[in] write_mode The write mode (single or chunked)
 * @return List of stream descriptors
 */
orc_streams create_streams(host_span<orc_column_view> columns,
                           file_segmentation const& segmentation,
                           std::map<uint32_t, size_t> const& decimal_column_sizes,
                           bool enable_dictionary,
                           compression_type compression,
                           single_write_mode write_mode)
{
  // 'column 0' row index stream
  std::vector<Stream> streams{{ROW_INDEX, 0}};  // TODO: Separate index and data streams?
  // First n + 1 streams are row index streams
  streams.reserve(columns.size() + 1);
  std::transform(columns.begin(), columns.end(), std::back_inserter(streams), [](auto const& col) {
    return Stream{ROW_INDEX, col.id()};
  });

  std::vector<int32_t> ids(columns.size() * CI_NUM_STREAMS, -1);
  std::vector<TypeKind> types(streams.size(), INVALID_TYPE_KIND);

  for (auto& column : columns) {
    auto const is_nullable = [&]() -> bool {
      if (write_mode == single_write_mode::YES) {
        return column.nullable();
      } else {
        // For chunked write, when not provided nullability, we assume the worst case scenario
        // that all columns are nullable.
        auto const chunked_nullable = column.user_defined_nullable().value_or(true);
        CUDF_EXPECTS(chunked_nullable or column.null_count() == 0,
                     "Mismatch in metadata prescribed nullability and input column. "
                     "Metadata for input column with nulls cannot prescribe nullability = false");
        return chunked_nullable;
      }
    }();

    auto RLE_column_size = [&](TypeKind type_kind) {
      return std::accumulate(
        cuda::counting_iterator<size_t>{0},
        cuda::counting_iterator{segmentation.num_rowgroups()},
        0ul,
        [&](auto data_size, auto rg_idx) {
          return data_size +
                 rle_stream_size(type_kind, segmentation.rowgroups[rg_idx][column.index()].size());
        });
    };

    auto const kind = column.orc_kind();

    auto add_stream =
      [&](stream_index_type index_type, StreamKind kind, TypeKind type_kind, size_t size) {
        auto const max_alignment_padding = compress_required_chunk_alignment(compression) - 1;
        auto const base                  = column.index() * CI_NUM_STREAMS;
        ids[base + index_type]           = streams.size();
        streams.push_back(
          Stream{kind,
                 column.id(),
                 (size == 0) ? 0 : size + max_alignment_padding * segmentation.num_rowgroups()});
        types.push_back(type_kind);
      };

    auto add_RLE_stream = [&](stream_index_type index_type, StreamKind kind, TypeKind type_kind) {
      add_stream(index_type, kind, type_kind, RLE_column_size(type_kind));
    };

    if (is_nullable) { add_RLE_stream(CI_PRESENT, PRESENT, TypeKind::BOOLEAN); }
    switch (kind) {
      case TypeKind::BOOLEAN:
      case TypeKind::BYTE:
        add_RLE_stream(CI_DATA, DATA, kind);
        column.set_orc_encoding(DIRECT);
        break;
      case TypeKind::SHORT:
      case TypeKind::INT:
      case TypeKind::LONG:
      case TypeKind::DATE:
        add_RLE_stream(CI_DATA, DATA, kind);
        column.set_orc_encoding(DIRECT_V2);
        break;
      case TypeKind::FLOAT:
      case TypeKind::DOUBLE:
        // Pass through if no nulls (no RLE encoding for floating point)
        add_stream(CI_DATA, DATA, kind, (column.null_count() != 0) ? RLE_column_size(kind) : 0);
        column.set_orc_encoding(DIRECT);
        break;
      case TypeKind::STRING: {
        bool enable_dict           = enable_dictionary;
        size_t dict_data_size      = 0;
        size_t dict_strings        = 0;
        size_t dict_lengths_div512 = 0;
        for (auto const& stripe : segmentation.stripes) {
          auto const sd = column.host_stripe_dict(stripe.id);
          enable_dict   = (enable_dict && sd.is_enabled);
          if (enable_dict) {
            dict_strings += sd.entry_count;
            dict_lengths_div512 += (sd.entry_count + 0x1ff) >> 9;
            dict_data_size += sd.char_count;
          }
        }

        size_t const direct_data_size = column.char_count();
        if (enable_dict) {
          uint32_t dict_bits = 0;
          for (dict_bits = 1; dict_bits < 32; dict_bits <<= 1) {
            if (dict_strings <= (1ull << dict_bits)) break;
          }
          auto const valid_count = column.size() - column.null_count();
          dict_data_size += (dict_bits * valid_count + 7) >> 3;
        }

        // Decide between direct or dictionary encoding
        if (enable_dict && dict_data_size < direct_data_size) {
          add_RLE_stream(CI_DATA, DATA, TypeKind::INT);
          add_stream(CI_DATA2, LENGTH, TypeKind::INT, dict_lengths_div512 * (512 * 4 + 2));
          add_stream(CI_DICTIONARY, DICTIONARY_DATA, TypeKind::CHAR, std::max(dict_data_size, 1ul));
          column.set_orc_encoding(DICTIONARY_V2);
        } else {
          add_stream(CI_DATA, DATA, TypeKind::CHAR, std::max<size_t>(direct_data_size, 1));
          add_RLE_stream(CI_DATA2, LENGTH, TypeKind::INT);
          column.set_orc_encoding(DIRECT_V2);
        }
        break;
      }
      case TypeKind::TIMESTAMP:
        add_RLE_stream(CI_DATA, DATA, TypeKind::LONG);
        add_RLE_stream(CI_DATA2, SECONDARY, TypeKind::LONG);
        column.set_orc_encoding(DIRECT_V2);
        break;
      case TypeKind::DECIMAL:
        // varint values (NO RLE)
        // data_stream_size = decimal_column_sizes.at(column.index());
        add_stream(CI_DATA, DATA, TypeKind::DECIMAL, decimal_column_sizes.at(column.index()));
        // scale stream TODO: compute exact size since all elems are equal
        add_RLE_stream(CI_DATA2, SECONDARY, TypeKind::INT);
        column.set_orc_encoding(DIRECT_V2);
        break;
      case TypeKind::LIST:
      case TypeKind::MAP:
        // no data stream, only lengths
        add_RLE_stream(CI_DATA2, LENGTH, TypeKind::INT);
        column.set_orc_encoding(DIRECT_V2);
        break;
      case TypeKind::STRUCT:
        // Only has the present stream
        break;
      default: CUDF_FAIL("Unsupported ORC type kind");
    }
  }
  return {std::move(streams), std::move(ids), std::move(types)};
}

std::vector<std::vector<rowgroup_rows>> calculate_aligned_rowgroup_bounds(
  orc_table_view const& orc_table, file_segmentation const& segmentation, cuda::stream_ref stream)
{
  if (segmentation.num_rowgroups() == 0) return {};

  auto d_pd_set_counts_data = rmm::device_uvector<cudf::size_type>(
    orc_table.num_columns() * segmentation.num_rowgroups(), stream);
  auto const d_pd_set_counts =
    device_2dspan<cudf::size_type>{d_pd_set_counts_data, orc_table.num_columns()};
  reduce_pushdown_masks(orc_table.d_columns, segmentation.rowgroups, d_pd_set_counts, stream);

  auto aligned_rgs = hostdevice_2dvector<rowgroup_rows>(
    segmentation.num_rowgroups(), orc_table.num_columns(), stream);
  CUDF_CUDA_TRY(cudf::detail::memcpy_async(aligned_rgs.base_device_ptr(),
                                           segmentation.rowgroups.base_device_ptr(),
                                           aligned_rgs.count() * sizeof(rowgroup_rows),
                                           stream));
  auto const d_stripes = cudf::detail::make_device_uvector_async(
    segmentation.stripes, stream, cudf::get_current_device_resource_ref());

  // One thread per column, per stripe
  thrust::for_each_n(
    rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
    cuda::counting_iterator<size_t>{0},
    orc_table.num_columns() * segmentation.num_stripes(),
    [columns = device_span<orc_column_device_view const>{orc_table.d_columns},
     stripes = device_span<stripe_rowgroups const>{d_stripes},
     d_pd_set_counts,
     out_rowgroups = aligned_rgs.device_view()] __device__(auto& idx) {
      uint32_t const col_idx = idx / stripes.size();
      // No alignment needed for root columns
      if (not columns[col_idx].parent_index.has_value()) return;

      auto const stripe_idx     = idx % stripes.size();
      auto const stripe         = stripes[stripe_idx];
      auto const parent_col_idx = columns[col_idx].parent_index.value();
      auto const parent_column  = columns[parent_col_idx];
      auto const stripe_end     = stripe.first + stripe.size;

      auto seek_last_borrow_rg = [&](auto rg_idx, size_type& bits_to_borrow) {
        auto curr         = rg_idx + 1;
        auto curr_rg_size = [&]() {
          return parent_column.pushdown_mask != nullptr ? d_pd_set_counts[curr][parent_col_idx]
                                                        : out_rowgroups[curr][col_idx].size();
        };
        while (curr < stripe_end and curr_rg_size() <= bits_to_borrow) {
          // All bits from rowgroup borrowed, make the rowgroup empty
          out_rowgroups[curr][col_idx].begin = out_rowgroups[curr][col_idx].end;
          bits_to_borrow -= curr_rg_size();
          ++curr;
        }
        return curr;
      };

      int previously_borrowed = 0;
      for (auto rg_idx = stripe.first; rg_idx + 1 < stripe_end; ++rg_idx) {
        auto& rg = out_rowgroups[rg_idx][col_idx];

        if (parent_column.pushdown_mask == nullptr) {
          // No pushdown mask, all null mask bits will be encoded
          // Align on rowgroup size (can be misaligned for list children)
          if (rg.size() % 8) {
            auto bits_to_borrow           = 8 - rg.size() % 8;
            auto const last_borrow_rg_idx = seek_last_borrow_rg(rg_idx, bits_to_borrow);
            if (last_borrow_rg_idx == stripe_end) {
              // Didn't find enough bits to borrow, move the rowgroup end to the stripe end
              rg.end = out_rowgroups[stripe_end - 1][col_idx].end;
              // Done with this stripe
              break;
            }
            auto& last_borrow_rg = out_rowgroups[last_borrow_rg_idx][col_idx];
            last_borrow_rg.begin += bits_to_borrow;
            rg.end = last_borrow_rg.begin;
            // Skip the rowgroups we emptied in the loop
            rg_idx = last_borrow_rg_idx - 1;
          }
        } else {
          // pushdown mask present; null mask bits w/ set pushdown mask bits will be encoded
          // Use the number of set bits in pushdown mask as size
          auto bits_to_borrow = [&]() {
            auto const parent_valid_count = d_pd_set_counts[rg_idx][parent_col_idx];
            if (parent_valid_count < previously_borrowed) {
              // Borrow to make an empty rowgroup
              return previously_borrowed - parent_valid_count;
            }
            auto const misalignment = (parent_valid_count - previously_borrowed) % 8;
            return (8 - misalignment) % 8;
          }();

          if (bits_to_borrow == 0) {
            // Didn't borrow any bits for this rowgroup
            previously_borrowed = 0;
            continue;
          }

          // Find rowgroup in which we finish the search for missing bits
          auto const last_borrow_rg_idx = seek_last_borrow_rg(rg_idx, bits_to_borrow);
          if (last_borrow_rg_idx == stripe_end) {
            // Didn't find enough bits to borrow, move the rowgroup end to the stripe end
            rg.end = out_rowgroups[stripe_end - 1][col_idx].end;
            // Done with this stripe
            break;
          }

          auto& last_borrow_rg = out_rowgroups[last_borrow_rg_idx][col_idx];
          // First row that does not need to be borrowed
          auto borrow_end = last_borrow_rg.begin;

          // Adjust the number of bits to borrow in the next iteration
          previously_borrowed = bits_to_borrow;

          // Find word in which we finish the search for missing bits (guaranteed to be available)
          while (bits_to_borrow != 0) {
            auto const mask = cudf::detail::get_mask_offset_word(
              parent_column.pushdown_mask, 0, borrow_end, borrow_end + 32);
            auto const valid_in_word = __popc(mask);

            if (valid_in_word > bits_to_borrow) break;
            bits_to_borrow -= valid_in_word;
            borrow_end += 32;
          }

          // Find the last of the missing bits (guaranteed to be available)
          while (bits_to_borrow != 0) {
            if (bit_is_set(parent_column.pushdown_mask, borrow_end)) { --bits_to_borrow; };
            ++borrow_end;
          }

          last_borrow_rg.begin = borrow_end;
          rg.end               = borrow_end;
          // Skip the rowgroups we emptied in the loop
          rg_idx = last_borrow_rg_idx - 1;
        }
      }
    });

  aligned_rgs.device_to_host(stream);

  std::vector<std::vector<rowgroup_rows>> h_aligned_rgs;
  h_aligned_rgs.reserve(segmentation.num_rowgroups());
  std::transform(cuda::counting_iterator<size_t>{0},
                 cuda::counting_iterator{segmentation.num_rowgroups()},
                 std::back_inserter(h_aligned_rgs),
                 [&](auto idx) -> std::vector<rowgroup_rows> {
                   return {aligned_rgs[idx].begin(), aligned_rgs[idx].end()};
                 });

  return h_aligned_rgs;
}

struct segmented_valid_cnt_input {
  bitmask_type const* mask;
  std::vector<size_type> indices;
};

/**
 * @brief Storage of one (stripe, stream) pair within an encoded arena.
 */
struct extent_info {
  size_t size{0};            ///< upper bound on what the encoder writes
  size_t offset{0};          ///< byte offset within the arena
  bool has_slack{false};     ///< whether `size` is a strict upper bound rather than exact
  bool is_transient{false};  ///< placed in `encoded_data::transient_buffer`
};

/**
 * @brief Encodes the columns' data into the ORC stream layout.
 *
 * @param[in] orc_table Table to be written, with ORC-related information
 * @param[in] dec_chunk_sizes Sizes of encoded decimal elements
 * @param[in] segmentation stripe and rowgroup ranges
 * @param[in] streams List of stream descriptors
 * @param[in] uncomp_block_align Required alignment of the codec's chunks
 * @param[in] stream CUDA stream used for device memory operations and kernel launches
 * @return The encoded data, along with a [stripe][strm_id] description of every extent, flattened
 * with `streams.size()` elements per row
 */
std::pair<encoded_data, std::vector<extent_info>> encode_columns(
  orc_table_view const& orc_table,
  encoder_decimal_info&& dec_chunk_sizes,
  file_segmentation const& segmentation,
  orc_streams const& streams,
  uint32_t uncomp_block_align,
  cuda::stream_ref stream)
{
  CUDF_EXPECTS(uncomp_block_align > 0 and extent_alignment % uncomp_block_align == 0,
               "Internal ORC writer error: extent alignment is not a multiple of the codec's chunk "
               "alignment");

  auto const num_columns = orc_table.num_columns();
  hostdevice_2dvector<encoder_chunk> chunks(num_columns, segmentation.num_rowgroups(), stream);

  auto const aligned_rowgroups = calculate_aligned_rowgroup_bounds(orc_table, segmentation, stream);

  // Initialize column chunks' descriptions
  std::map<size_type, segmented_valid_cnt_input> validity_check_inputs;

  for (auto const& column : orc_table.columns) {
    for (auto const& stripe : segmentation.stripes) {
      for (auto rg_idx_it = stripe.cbegin(); rg_idx_it < stripe.cend(); ++rg_idx_it) {
        auto const rg_idx      = *rg_idx_it;
        auto& ck               = chunks[column.index()][rg_idx];
        ck.start_row           = segmentation.rowgroups[rg_idx][column.index()].begin;
        ck.num_rows            = segmentation.rowgroups[rg_idx][column.index()].size();
        ck.null_mask_start_row = aligned_rowgroups[rg_idx][column.index()].begin;
        ck.null_mask_num_rows  = aligned_rowgroups[rg_idx][column.index()].size();
        ck.encoding_kind       = column.orc_encoding();
        ck.type_kind           = column.orc_kind();
        auto const is_str_dict =
          ck.type_kind == TypeKind::STRING and ck.encoding_kind == DICTIONARY_V2;
        ck.dict_index = is_str_dict ? column.host_stripe_dict(stripe.id).index.data() : nullptr;
        ck.dict_data_order =
          is_str_dict ? column.host_stripe_dict(stripe.id).data_order.data() : nullptr;
        ck.dtype_len = (ck.type_kind == TypeKind::STRING) ? 1 : column.type_width();
        ck.scale     = column.scale();
        ck.decimal_offsets =
          (ck.type_kind == TypeKind::DECIMAL) ? column.decimal_offsets() : nullptr;
      }
    }
  }
  chunks.host_to_device_async(stream);
  // TODO (future): pass columns separately from chunks (to skip this step)
  // and remove info from chunks that is common for the entire column
  thrust::for_each_n(
    rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
    cuda::counting_iterator<size_t>{0},
    chunks.count(),
    [chunks = chunks.device_view(),
     cols = device_span<orc_column_device_view const>{orc_table.d_columns}] __device__(auto& idx) {
      auto const col_idx             = idx / chunks.size().second;
      auto const rg_idx              = idx % chunks.size().second;
      chunks[col_idx][rg_idx].column = &cols[col_idx];
    });

  auto validity_check_indices = [&](size_t col_idx) {
    std::vector<size_type> indices;
    for (auto const& stripe : segmentation.stripes) {
      for (auto rg_idx_it = stripe.cbegin(); rg_idx_it < stripe.cend() - 1; ++rg_idx_it) {
#if defined(__GNUC__) && (__GNUC__ >= 14)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdangling-reference"
#endif
        auto const& chunk = chunks[col_idx][*rg_idx_it];
#if defined(__GNUC__) && (__GNUC__ >= 14)
#pragma GCC diagnostic pop
#endif
        indices.push_back(chunk.start_row);
        indices.push_back(chunk.start_row + chunk.num_rows);
      }
    }
    return indices;
  };
  for (auto const& column : orc_table.columns) {
    if (column.orc_kind() == TypeKind::BOOLEAN && column.nullable()) {
      validity_check_inputs[column.index()] = {column.null_mask(),
                                               validity_check_indices(column.index())};
    }
  }
  for (auto& cnt_in : validity_check_inputs) {
    auto const valid_counts =
      cudf::detail::segmented_valid_count(cnt_in.second.mask, cnt_in.second.indices, stream);
    CUDF_EXPECTS(
      std::none_of(valid_counts.cbegin(),
                   valid_counts.cend(),
                   [](auto valid_count) { return valid_count % 8; }),
      "There's currently a bug in encoding boolean columns. Suggested workaround is to convert "
      "to int8 type."
      " Please see https://github.com/NVIDIA/cudf/issues/6763 for more information.");
  }

  hostdevice_2dvector<encoder_chunk_streams> chunk_streams(
    num_columns, segmentation.num_rowgroups(), stream);

  // Compute per-rowgroup stream lengths and per-(stripe, strm_id) extent sizes, along with
  // whether an extent size may exceed what the encoder ends up writing.
  auto const num_streams = streams.size();
  std::vector<extent_info> extent_storage(segmentation.num_stripes() * num_streams);
  auto const extents = host_2dspan<extent_info>{extent_storage, num_streams};
  for (auto const& stripe : segmentation.stripes) {
    for (size_t col_idx = 0; col_idx < num_columns; col_idx++) {
      for (int strm_type = 0; strm_type < CI_NUM_STREAMS; ++strm_type) {
        auto const& column = orc_table.column(col_idx);
        auto col_streams   = chunk_streams[col_idx];
        auto const strm_id = streams.id(col_idx * CI_NUM_STREAMS + strm_type);

        std::for_each(stripe.cbegin(), stripe.cend(), [&](auto rg_idx) {
          col_streams[rg_idx].ids[strm_type]     = strm_id;
          col_streams[rg_idx].lengths[strm_type] = 0;
        });

        if (strm_id < 0) { continue; }

        size_t stripe_size = 0;
        // Alignment padding leaves a gap after every rowgroup's chunk, so it is slack in itself.
        bool has_slack = uncomp_block_align > 1;
        std::for_each(stripe.cbegin(), stripe.cend(), [&](auto rg_idx) {
#if defined(__GNUC__) && (__GNUC__ >= 14)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdangling-reference"
#endif
          auto const& ck = chunks[col_idx][rg_idx];
#if defined(__GNUC__) && (__GNUC__ >= 14)
#pragma GCC diagnostic pop
#endif
          auto& strm = col_streams[rg_idx];

          if ((strm_type == CI_DICTIONARY) ||
              (strm_type == CI_DATA2 && ck.encoding_kind == DICTIONARY_V2)) {
            if (rg_idx == *stripe.cbegin()) {
              auto const stripe_dict = column.host_stripe_dict(stripe.id);
              strm.lengths[strm_type] =
                (strm_type == CI_DICTIONARY)
                  ? stripe_dict.char_count
                  : (((stripe_dict.entry_count + 0x1ff) >> 9) * (512 * 4 + 2));
              // Only the size of RLE-encoded lengths is an estimate
              has_slack |= (strm_type != CI_DICTIONARY);
            } else {
              strm.lengths[strm_type] = 0;
            }
          } else if (strm_type == CI_DATA && ck.type_kind == TypeKind::STRING &&
                     ck.encoding_kind == DIRECT_V2) {
            auto const char_count   = column.rowgroup_char_count(rg_idx);
            strm.lengths[strm_type] = std::max(char_count, 1);
            // The `max` reserves a byte the encoder does not write
            has_slack |= (char_count == 0);
          } else if (strm_type == CI_DATA && streams[strm_id].length == 0 &&
                     (ck.type_kind == DOUBLE || ck.type_kind == FLOAT)) {
            // Pass-through. The encoder reports this length back unchanged, so it is exact.
            strm.lengths[strm_type] = ck.num_rows * ck.dtype_len;
          } else if (ck.type_kind == DECIMAL && strm_type == CI_DATA) {
            strm.lengths[strm_type] = dec_chunk_sizes.rg_sizes.at(col_idx)[rg_idx];
          } else {
            strm.lengths[strm_type] = rle_stream_size(streams.type(strm_id), ck.num_rows);
            has_slack               = true;
          }
          // Allow extra space for alignment
          stripe_size += strm.lengths[strm_type] + uncomp_block_align - 1;
        });

        auto& extent     = extents[stripe.id][strm_id];
        extent.size      = stripe_size;
        extent.has_slack = has_slack;
      }
    }
  }

  // Extents that `gather_stripes` is certain to compact go into `transient_buffer`, so that arena
  // can be freed as soon as gathering completes.
  size_t persistent_arena_size = 0;
  size_t transient_arena_size  = 0;
  for (size_t s = 0; s < segmentation.num_stripes(); ++s) {
    for (size_t strm_id = 0; strm_id < num_streams; ++strm_id) {
      auto& extent = extents[s][strm_id];
      if (extent.size == 0) { continue; }
      extent.is_transient = segmentation.stripes[s].size > 1 and extent.has_slack;
      auto& arena_size    = extent.is_transient ? transient_arena_size : persistent_arena_size;
      arena_size          = util::round_up_unsafe<size_t>(arena_size, extent_alignment);
      extent.offset       = arena_size;
      arena_size += extent.size;
    }
  }

  rmm::device_uvector<uint8_t> persistent_buffer(persistent_arena_size, stream);
  rmm::device_uvector<uint8_t> transient_buffer(transient_arena_size, stream);

  std::vector<std::vector<device_span<uint8_t>>> encoded_views(
    segmentation.num_stripes(), std::vector<device_span<uint8_t>>(num_streams));
  for (size_t s = 0; s < segmentation.num_stripes(); ++s) {
    for (size_t strm_id = 0; strm_id < num_streams; ++strm_id) {
      auto const extent = extents[s][strm_id];
      // Zero-size extents keep a null pointer, matching the empty device_uvector they replaced.
      if (extent.size == 0) { continue; }
      auto& arena               = extent.is_transient ? transient_buffer : persistent_buffer;
      encoded_views[s][strm_id] = device_span<uint8_t>{arena.data() + extent.offset, extent.size};
    }
  }

  // Point each rowgroup's chunk at its place within the extent, rounded up to the codec alignment.
  for (auto const& stripe : segmentation.stripes) {
    for (size_t col_idx = 0; col_idx < num_columns; col_idx++) {
      for (int strm_type = 0; strm_type < CI_NUM_STREAMS; ++strm_type) {
        auto col_streams   = chunk_streams[col_idx];
        auto const strm_id = streams.id(col_idx * CI_NUM_STREAMS + strm_type);

        for (auto rg_idx_it = stripe.cbegin(); rg_idx_it < stripe.cend(); ++rg_idx_it) {
          auto const rg_idx = *rg_idx_it;
#if defined(__GNUC__) && (__GNUC__ >= 14)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdangling-reference"
#endif
          auto const& ck = chunks[col_idx][rg_idx];
#if defined(__GNUC__) && (__GNUC__ >= 14)
#pragma GCC diagnostic pop
#endif
          auto& strm = col_streams[rg_idx];

          if (strm_id < 0 or (strm_type == CI_DATA && streams[strm_id].length == 0 &&
                              (ck.type_kind == DOUBLE || ck.type_kind == FLOAT))) {
            strm.data_ptrs[strm_type] = nullptr;
          } else {
            if ((strm_type == CI_DICTIONARY) ||
                (strm_type == CI_DATA2 && ck.encoding_kind == DICTIONARY_V2)) {
              strm.data_ptrs[strm_type] = encoded_views[stripe.id][strm_id].data();
            } else {
              strm.data_ptrs[strm_type] = (rg_idx_it == stripe.cbegin())
                                            ? encoded_views[stripe.id][strm_id].data()
                                            : (col_streams[rg_idx - 1].data_ptrs[strm_type] +
                                               col_streams[rg_idx - 1].lengths[strm_type]);
            }
          }
          auto const misalignment =
            reinterpret_cast<intptr_t>(strm.data_ptrs[strm_type]) % uncomp_block_align;
          if (misalignment != 0) {
            strm.data_ptrs[strm_type] += (uncomp_block_align - misalignment);
          }
        }
      }
    }
  }

  chunk_streams.host_to_device_async(stream);

  if (orc_table.num_rows() > 0) {
    if (orc_table.num_string_columns() != 0) {
      auto d_stripe_dict = orc_table.string_column(0).device_stripe_dicts();
      encode_stripe_dictionaries(d_stripe_dict.data(),
                                 orc_table.d_columns,
                                 chunks,
                                 orc_table.num_string_columns(),
                                 segmentation.num_stripes(),
                                 chunk_streams,
                                 stream);
    }

    encode_orc_column_data(chunks, chunk_streams, stream);
  }
  chunk_streams.device_to_host(stream);

  return {encoded_data{std::move(persistent_buffer),
                       std::move(transient_buffer),
                       rmm::device_uvector<uint8_t>{0, stream},  // filled by gather_stripes
                       std::move(encoded_views),
                       std::move(chunk_streams)},
          std::move(extent_storage)};
}

// TODO: remove StripeInformation from this function and return strm_desc instead
/**
 * @brief Returns stripe information after compacting columns' individual data
 * chunks into contiguous data streams.
 *
 * @param[in] num_index_streams Total number of index streams
 * @param[in] segmentation stripe and rowgroup ranges
 * @param[in] extents Extent descriptions [stripe][data_stream]; extents marked transient are held
 * in `enc_data->transient_buffer` and must all be gathered
 * @param[in,out] enc_data ORC per-chunk streams of encoded data
 * @param[in,out] strm_desc List of stream descriptors [stripe][data_stream]
 * @param[in] stream CUDA stream used for device memory operations and kernel launches
 * @return The stripes' information
 */
std::vector<StripeInformation> gather_stripes(size_t num_index_streams,
                                              file_segmentation const& segmentation,
                                              host_2dspan<extent_info const> extents,
                                              encoded_data* enc_data,
                                              hostdevice_2dvector<stripe_stream>* strm_desc,
                                              cuda::stream_ref stream)
{
  if (segmentation.num_stripes() == 0) { return {}; }

  auto const num_streams_in_data = enc_data->data[0].size();

  // Compaction destination of one (stripe, stream) pair within the gather arena.
  struct gather_extent {
    size_t size{0};    // what the encoder actually wrote
    size_t offset{0};  // byte offset within the arena
    bool gathered{false};
    device_span<uint8_t> view{};
  };
  std::vector<gather_extent> gather_storage(segmentation.num_stripes() * num_streams_in_data);
  auto const gather_extents = host_2dspan<gather_extent>{gather_storage, num_streams_in_data};

  // Compute per-(stripe, stream) actual sizes and decide which need a gathered copy.
  for (auto const& stripe : segmentation.stripes) {
    for (size_t col_idx = 0; col_idx < enc_data->streams.size().first; col_idx++) {
      auto const& col_streams = (enc_data->streams)[col_idx];
      for (int k = 0; k < CI_INDEX; k++) {
        auto const stream_id = col_streams[0].ids[k];
        if (stream_id == -1) { continue; }

        auto const actual_stripe_size =
          std::accumulate(col_streams.begin() + stripe.first,
                          col_streams.begin() + stripe.first + stripe.size,
                          0ul,
                          [&](auto const& sum, auto const& strm) { return sum + strm.lengths[k]; });

        auto const allocated_stripe_size = enc_data->data[stripe.id][stream_id].size();
        CUDF_EXPECTS(allocated_stripe_size >= actual_stripe_size,
                     "Internal ORC writer error: insufficient allocation size for encoded data");

        // Compact when the chunks are not already contiguous, i.e. when the encoder wrote less
        // than the extent was sized for. Extents in `transient_buffer` are compacted regardless,
        // so that arena can be released below.
        bool const gathered = (stripe.size > 1 and (extents[stripe.id][stream_id].is_transient or
                                                    allocated_stripe_size > actual_stripe_size));

        auto& extent    = gather_extents[stripe.id][stream_id];
        extent.size     = actual_stripe_size;
        extent.gathered = gathered;
      }
    }
  }

  // Lay out gather destinations in a single arena, with the same alignment as the encoded arenas.
  size_t gather_total = 0;
  for (size_t s = 0; s < segmentation.num_stripes(); ++s) {
    for (size_t strm_id = 0; strm_id < num_streams_in_data; ++strm_id) {
      auto& extent = gather_extents[s][strm_id];
      if (!extent.gathered) { continue; }
      gather_total  = util::round_up_unsafe<size_t>(gather_total, extent_alignment);
      extent.offset = gather_total;
      gather_total += extent.size;
    }
  }
  rmm::device_uvector<uint8_t> gather_buffer(std::max<size_t>(gather_total, 1), stream);

  // Build strm_desc entries and record gather destination spans.
  std::vector<StripeInformation> stripes(segmentation.num_stripes());
  for (auto const& stripe : segmentation.stripes) {
    for (size_t col_idx = 0; col_idx < enc_data->streams.size().first; col_idx++) {
      auto const& col_streams = (enc_data->streams)[col_idx];
      for (int k = 0; k < CI_INDEX; k++) {
        auto const stream_id = col_streams[0].ids[k];
        if (stream_id == -1) { continue; }

        auto& extent     = gather_extents[stripe.id][stream_id];
        uint8_t* dst_ptr = nullptr;
        if (extent.gathered) {
          // Non-null even when the extent is empty, unlike the empty device_uvector this replaced.
          // `init_batched_memcpy_kernel` repoints the per-rowgroup data_ptrs at this arena, which
          // is what lets `transient_buffer` be released without leaving them dangling.
          dst_ptr     = gather_buffer.data() + extent.offset;
          extent.view = device_span<uint8_t>{dst_ptr, extent.size};
        }

        auto* ss           = &(*strm_desc)[stripe.id][stream_id - num_index_streams];
        ss->data_ptr       = dst_ptr;  // null when not gathered; init_batched_memcpy_kernel skips
        ss->stream_size    = extent.size;
        ss->first_chunk_id = stripe.first;
        ss->num_chunks     = stripe.size;
        ss->column_id      = col_idx;
        ss->stream_type    = k;
      }
    }

    stripes[stripe.id].numberOfRows =
      stripe.size == 0 ? 0
                       : segmentation.rowgroups[stripe.first + stripe.size - 1][0].end -
                           segmentation.rowgroups[stripe.first][0].begin;
  }

  strm_desc->host_to_device_async(stream);
  // TODO: use cub::DeviceMemcpy::Batched
  compact_orc_data_streams(*strm_desc, enc_data->streams, stream);
  strm_desc->device_to_host_async(stream);
  enc_data->streams.device_to_host(stream);

  // Replace data views for gathered (stripe, stream) with the gathered-arena
  // spans, so consumers that read enc_data->data observe the post-gather state.
  for (size_t stripe_id = 0; stripe_id < enc_data->data.size(); ++stripe_id) {
    for (size_t stream_id = 0; stream_id < num_streams_in_data; ++stream_id) {
      auto const extent = gather_extents[stripe_id][stream_id];
      if (extent.gathered) { enc_data->data[stripe_id][stream_id] = extent.view; }
    }
  }

  // Hold the gathered arena for lifetime management, and release the arena it copied from.
  enc_data->gathered_buffer  = std::move(gather_buffer);
  enc_data->transient_buffer = rmm::device_uvector<uint8_t>{0, stream};

  return stripes;
}

void set_stat_desc_leaf_cols(device_span<orc_column_device_view const> columns,
                             device_span<stats_column_desc> stat_desc,
                             cuda::stream_ref stream)
{
  thrust::for_each(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                   cuda::counting_iterator<size_t>{0},
                   cuda::counting_iterator{stat_desc.size()},
                   [=] __device__(auto idx) { stat_desc[idx].leaf_column = &columns[idx]; });
}

cudf::detail::hostdevice_vector<uint8_t> allocate_and_encode_blobs(
  cudf::detail::hostdevice_vector<statistics_merge_group>& stats_merge_groups,
  device_span<statistics_chunk const> stat_chunks,
  int num_stat_blobs,
  cuda::stream_ref stream)
{
  // figure out the buffer size needed for protobuf format
  orc_init_statistics_buffersize(
    stats_merge_groups.device_ptr(), stat_chunks.data(), num_stat_blobs, stream);

  // get stats_merge_groups[num_stat_blobs - 1] via a host pinned bounce buffer
  auto const max_blobs = [&]() {
    auto max_blobs_element = cudf::detail::make_pinned_vector<statistics_merge_group>(
      cudf::device_span<statistics_merge_group>{stats_merge_groups.device_ptr(num_stat_blobs - 1),
                                                1},
      stream);
    return max_blobs_element.front();
  }();

  cudf::detail::hostdevice_vector<uint8_t> blobs(max_blobs.start_chunk + max_blobs.num_chunks,
                                                 stream);
  orc_encode_statistics(blobs.device_ptr(),
                        stats_merge_groups.device_ptr(),
                        stat_chunks.data(),
                        num_stat_blobs,
                        stream);
  stats_merge_groups.device_to_host_async(stream);
  blobs.device_to_host(stream);
  return blobs;
}

[[nodiscard]] statistics_dtype kind_to_stats_type(TypeKind kind)
{
  switch (kind) {
    case TypeKind::BOOLEAN: return dtype_bool;
    case TypeKind::BYTE: return dtype_int8;
    case TypeKind::SHORT: return dtype_int16;
    case TypeKind::INT: return dtype_int32;
    case TypeKind::LONG: return dtype_int64;
    case TypeKind::FLOAT: return dtype_float32;
    case TypeKind::DOUBLE: return dtype_float64;
    case TypeKind::STRING: return dtype_string;
    case TypeKind::DATE: return dtype_int32;
    case TypeKind::TIMESTAMP: return dtype_timestamp64;
    case TypeKind::DECIMAL: return dtype_decimal64;
    default: return dtype_none;
  }
}

/**
 * @brief Returns column statistics in an intermediate format.
 *
 * @param stats_freq Frequency of statistics to be included in the output file
 * @param orc_table Table information to be written
 * @param segmentation stripe and rowgroup ranges
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @return The statistic information
 */
intermediate_statistics gather_statistic_blobs(statistics_freq const stats_freq,
                                               orc_table_view const& orc_table,
                                               file_segmentation const& segmentation,
                                               cuda::stream_ref stream)
{
  auto const num_rowgroup_blobs     = segmentation.rowgroups.count();
  auto const num_stripe_blobs       = segmentation.num_stripes() * orc_table.num_columns();
  auto const are_statistics_enabled = stats_freq != statistics_freq::STATISTICS_NONE;
  if (not are_statistics_enabled or num_rowgroup_blobs + num_stripe_blobs == 0) {
    return intermediate_statistics{orc_table, stream};
  }

  cudf::detail::hostdevice_vector<stats_column_desc> stat_desc(orc_table.num_columns(), stream);
  cudf::detail::hostdevice_vector<statistics_merge_group> rowgroup_merge(num_rowgroup_blobs,
                                                                         stream);
  cudf::detail::hostdevice_vector<statistics_merge_group> stripe_merge(num_stripe_blobs, stream);
  std::vector<statistics_dtype> col_stats_dtypes;
  std::vector<data_type> col_types;
  auto rowgroup_stat_merge = rowgroup_merge.host_ptr();
  auto stripe_stat_merge   = stripe_merge.host_ptr();

  for (auto const& column : orc_table.columns) {
    stats_column_desc* desc = &stat_desc[column.index()];
    desc->stats_dtype       = kind_to_stats_type(column.orc_kind());
    desc->num_rows          = column.size();
    desc->num_values        = column.size();
    if (desc->stats_dtype == dtype_timestamp64) {
      // Timestamp statistics are in milliseconds
      switch (column.scale()) {
        case 9: desc->ts_scale = 1000; break;
        case 6: desc->ts_scale = 0; break;
        case 3: desc->ts_scale = -1000; break;
        case 0: desc->ts_scale = -1000000; break;
        default: desc->ts_scale = 0; break;
      }
    } else {
      desc->ts_scale = 0;
    }
    col_stats_dtypes.push_back(desc->stats_dtype);
    col_types.push_back(column.type());
    for (auto const& stripe : segmentation.stripes) {
      auto& grp       = stripe_stat_merge[column.index() * segmentation.num_stripes() + stripe.id];
      grp.col_dtype   = column.type();
      grp.stats_dtype = desc->stats_dtype;
      grp.start_chunk =
        static_cast<uint32_t>(column.index() * segmentation.num_rowgroups() + stripe.first);
      grp.num_chunks = stripe.size;
      for (auto rg_idx_it = stripe.cbegin(); rg_idx_it != stripe.cend(); ++rg_idx_it) {
        auto& rg_grp =
          rowgroup_stat_merge[column.index() * segmentation.num_rowgroups() + *rg_idx_it];
        rg_grp.col_dtype   = column.type();
        rg_grp.stats_dtype = desc->stats_dtype;
        rg_grp.start_chunk = *rg_idx_it;
        rg_grp.num_chunks  = 1;
      }
    }
  }
  stat_desc.host_to_device_async(stream);
  rowgroup_merge.host_to_device_async(stream);
  stripe_merge.host_to_device_async(stream);
  set_stat_desc_leaf_cols(orc_table.d_columns, stat_desc, stream);

  // The rowgroup stat chunks are written out in each stripe. The stripe and file-level chunks are
  // written in the footer. To prevent persisting the rowgroup stat chunks across multiple write
  // calls in a chunked write situation, these allocations are split up so stripe data can persist
  // until the footer is written and rowgroup data can be freed after being written to the stripe.
  rmm::device_uvector<statistics_chunk> rowgroup_chunks(num_rowgroup_blobs, stream);
  rmm::device_uvector<statistics_chunk> stripe_chunks(num_stripe_blobs, stream);
  auto rowgroup_stat_chunks = rowgroup_chunks.data();
  auto stripe_stat_chunks   = stripe_chunks.data();

  rmm::device_uvector<statistics_group> rowgroup_groups(num_rowgroup_blobs, stream);
  orc_init_statistics_groups(
    rowgroup_groups.data(), stat_desc.device_ptr(), segmentation.rowgroups, stream);

  detail::calculate_group_statistics<detail::io_file_format::ORC>(
    rowgroup_chunks.data(), rowgroup_groups.data(), num_rowgroup_blobs, stream);

  detail::merge_group_statistics<detail::io_file_format::ORC>(
    stripe_stat_chunks, rowgroup_stat_chunks, stripe_merge.device_ptr(), num_stripe_blobs, stream);

  // With chunked writes, the orc table can be deallocated between write calls.
  // This forces our hand to encode row groups and stripes only in this stage and further
  // we have to persist any data from the table that we need later. The
  // minimum and maximum string inside the `str_val` structure inside `statistics_val` in
  // `statistic_chunk` that are copies of the largest and smallest strings in the row group,
  // or stripe need to be persisted between write calls. We write rowgroup data with each
  // stripe and then save each stripe's stats until the end where we merge those all together
  // to get the file-level stats.

  // Skip rowgroup blobs when encoding, if chosen granularity is coarser than "ROW_GROUP".
  auto const is_granularity_rowgroup = stats_freq == ORC_STATISTICS_ROW_GROUP;
  // we have to encode the row groups now IF they are being written out
  auto rowgroup_blobs = [&]() -> std::vector<col_stats_blob> {
    if (not is_granularity_rowgroup) { return {}; }

    cudf::detail::hostdevice_vector<uint8_t> blobs =
      allocate_and_encode_blobs(rowgroup_merge, rowgroup_chunks, num_rowgroup_blobs, stream);

    std::vector<col_stats_blob> rowgroup_blobs(num_rowgroup_blobs);
    for (size_t i = 0; i < num_rowgroup_blobs; i++) {
      auto const stat_begin = blobs.host_ptr(rowgroup_merge[i].start_chunk);
      auto const stat_end   = stat_begin + rowgroup_merge[i].num_chunks;
      rowgroup_blobs[i].assign(stat_begin, stat_end);
    }
    return rowgroup_blobs;
  }();

  return {std::move(rowgroup_blobs),
          std::move(stripe_chunks),
          std::move(stripe_merge),
          std::move(col_stats_dtypes),
          std::move(col_types)};
}

/**
 * @brief Returns column statistics encoded in ORC protobuf format stored in the footer.
 *
 * @param footer ORC footer containing stripe and type information
 * @param per_chunk_stats Persisted per-chunk statistics from `gather_statistic_blobs`
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @return The encoded statistic blobs
 */
encoded_footer_statistics finish_statistic_blobs(Footer const& footer,
                                                 persisted_statistics& per_chunk_stats,
                                                 cuda::stream_ref stream)
{
  auto stripe_size_iter = cuda::transform_iterator(per_chunk_stats.stripe_stat_merge.begin(),
                                                   [](auto const& s) { return s.size(); });

  auto const num_columns = footer.types.size() - 1;
  auto const num_stripes = footer.stripes.size();

  auto const num_stripe_blobs =
    thrust::reduce(stripe_size_iter, stripe_size_iter + per_chunk_stats.stripe_stat_merge.size());
  auto const num_file_blobs = num_columns;
  auto const num_blobs      = static_cast<int>(num_stripe_blobs + num_file_blobs);

  if (num_stripe_blobs == 0) {
    if (num_file_blobs == 0) { return {}; }

    // Create empty file stats and merge groups
    auto h_stat_chunks = cudf::detail::make_host_vector<statistics_chunk>(num_file_blobs, stream);
    cudf::detail::hostdevice_vector<statistics_merge_group> stats_merge(num_file_blobs, stream);
    // Fill in stats_merge and stat_chunks on the host
    for (auto i = 0u; i < num_file_blobs; ++i) {
      stats_merge[i].col_dtype   = per_chunk_stats.col_types[i];
      stats_merge[i].stats_dtype = kind_to_stats_type(footer.types[i + 1].kind);
      // Write the sum for empty columns, equal to zero
      h_stat_chunks[i].has_sum = true;
    }
    //  Copy to device
    auto const d_stat_chunks = cudf::detail::make_device_uvector_async<statistics_chunk>(
      h_stat_chunks, stream, cudf::get_current_device_resource_ref());
    stats_merge.host_to_device_async(stream);

    // Encode and return
    cudf::detail::hostdevice_vector<uint8_t> hd_file_blobs =
      allocate_and_encode_blobs(stats_merge, d_stat_chunks, num_file_blobs, stream);

    // Copy blobs to host (actual size)
    std::vector<col_stats_blob> file_blobs(num_file_blobs);
    for (auto i = 0u; i < num_file_blobs; i++) {
      auto const stat_begin = hd_file_blobs.host_ptr(stats_merge[i].start_chunk);
      auto const stat_end   = stat_begin + stats_merge[i].num_chunks;
      file_blobs[i].assign(stat_begin, stat_end);
    }

    return {{}, std::move(file_blobs)};
  }

  // merge the stripe persisted data and add file data
  rmm::device_uvector<statistics_chunk> stat_chunks(num_blobs, stream);
  cudf::detail::hostdevice_vector<statistics_merge_group> stats_merge(num_blobs, stream);

  // we need to merge the stat arrays from the persisted data.
  // this needs to be done carefully because each array can contain
  // a different number of stripes and stripes from each column must be
  // located next to each other. We know the total number of stripes and
  // we know the size of each array. The number of stripes per column in a chunk array can
  // be calculated by dividing the number of chunks by the number of columns.
  // That many chunks need to be copied at a time to the proper destination.
  size_t num_entries_seen        = 0;
  auto const num_buffers_to_copy = per_chunk_stats.stripe_stat_chunks.size() * num_columns * 2;
  auto h_srcs = cudf::detail::make_empty_host_vector<void*>(num_buffers_to_copy, stream);
  auto h_dsts = cudf::detail::make_empty_host_vector<void*>(num_buffers_to_copy, stream);
  auto h_lens = cudf::detail::make_empty_host_vector<size_t>(num_buffers_to_copy, stream);

  for (size_t i = 0; i < per_chunk_stats.stripe_stat_chunks.size(); ++i) {
    auto const stripes_per_col = per_chunk_stats.stripe_stat_chunks[i].size() / num_columns;

    for (size_t col = 0; col < num_columns; ++col) {
      h_srcs.push_back(per_chunk_stats.stripe_stat_chunks[i].data() + col * stripes_per_col);
      h_dsts.push_back(stat_chunks.data() + (num_stripes * col) + num_entries_seen);
      h_lens.push_back(stripes_per_col * sizeof(statistics_chunk));

      h_srcs.push_back(per_chunk_stats.stripe_stat_merge[i].device_ptr() + col * stripes_per_col);
      h_dsts.push_back(stats_merge.device_ptr() + (num_stripes * col) + num_entries_seen);
      h_lens.push_back(stripes_per_col * sizeof(statistics_merge_group));
    }
    num_entries_seen += stripes_per_col;
  }

  auto const& mr    = cudf::get_current_device_resource_ref();
  auto const d_srcs = cudf::detail::make_device_uvector_async(h_srcs, stream, mr);
  auto const d_dsts = cudf::detail::make_device_uvector_async(h_dsts, stream, mr);
  auto const d_lens = cudf::detail::make_device_uvector_async(h_lens, stream, mr);
  cudf::detail::batched_memcpy_async(
    d_srcs.begin(), d_dsts.begin(), d_lens.begin(), d_srcs.size(), stream);

  auto file_stats_merge =
    cudf::detail::make_host_vector<statistics_merge_group>(num_file_blobs, stream);
  for (auto i = 0u; i < num_file_blobs; ++i) {
    auto col_stats         = &file_stats_merge[i];
    col_stats->col_dtype   = per_chunk_stats.col_types[i];
    col_stats->stats_dtype = per_chunk_stats.stats_dtypes[i];
    col_stats->start_chunk = static_cast<uint32_t>(i * num_stripes);
    col_stats->num_chunks  = static_cast<uint32_t>(num_stripes);
  }

  auto d_file_stats_merge = stats_merge.device_ptr(num_stripe_blobs);
  cudf::detail::cuda_memcpy_async<statistics_merge_group>(
    device_span<statistics_merge_group>{stats_merge.device_ptr(num_stripe_blobs), num_file_blobs},
    file_stats_merge,
    stream);

  auto file_stat_chunks = stat_chunks.data() + num_stripe_blobs;
  detail::merge_group_statistics<detail::io_file_format::ORC>(
    file_stat_chunks, stat_chunks.data(), d_file_stats_merge, num_file_blobs, stream);

  cudf::detail::hostdevice_vector<uint8_t> blobs =
    allocate_and_encode_blobs(stats_merge, stat_chunks, num_blobs, stream);

  auto stripe_stat_merge = stats_merge.host_ptr();

  std::vector<col_stats_blob> stripe_blobs(num_stripe_blobs);
  for (size_t i = 0; i < num_stripe_blobs; i++) {
    auto const stat_begin = blobs.host_ptr(stripe_stat_merge[i].start_chunk);
    auto const stat_end   = stat_begin + stripe_stat_merge[i].num_chunks;
    stripe_blobs[i].assign(stat_begin, stat_end);
  }

  std::vector<col_stats_blob> file_blobs(num_file_blobs);
  auto file_stat_merge = stats_merge.host_ptr(num_stripe_blobs);
  for (auto i = 0u; i < num_file_blobs; i++) {
    auto const stat_begin = blobs.host_ptr(file_stat_merge[i].start_chunk);
    auto const stat_end   = stat_begin + file_stat_merge[i].num_chunks;
    file_blobs[i].assign(stat_begin, stat_end);
  }

  return {std::move(stripe_blobs), std::move(file_blobs)};
}

/**
 * @brief Writes the specified column's row index stream.
 *
 * @param[in] stripe_id Stripe's identifier
 * @param[in] stream_id Stream identifier (column id + 1)
 * @param[in] columns List of columns
 * @param[in] segmentation stripe and rowgroup ranges
 * @param[in] enc_streams List of encoder chunk streams [column][rowgroup]
 * @param[in] strm_desc List of stream descriptors
 * @param[in] comp_res Output status for compressed streams
 * @param[in] rg_stats row group level statistics
 * @param[in,out] stripe Stream's parent stripe
 * @param[in,out] streams List of all streams
 * @param[in] compression The compression format
 * @param[in] compression_blocksize The block size used for compression
 * @param[in] out_sink Sink for writing data
 */
void write_index_stream(int32_t stripe_id,
                        int32_t stream_id,
                        host_span<orc_column_view const> columns,
                        file_segmentation const& segmentation,
                        host_2dspan<encoder_chunk_streams const> enc_streams,
                        host_2dspan<stripe_stream const> strm_desc,
                        host_span<codec_exec_result const> comp_res,
                        host_span<col_stats_blob const> rg_stats,
                        StripeInformation* stripe,
                        orc_streams* streams,
                        compression_type compression,
                        size_t compression_blocksize,
                        std::unique_ptr<data_sink> const& out_sink)
{
  row_group_index_info present;
  row_group_index_info data;
  row_group_index_info data2;
  auto const column_id = stream_id - 1;

  auto find_record = [=, &strm_desc](encoder_chunk_streams const& stream, stream_index_type type) {
    row_group_index_info record;
    if (stream.ids[type] > 0) {
      record.pos = 0;
      if (compression != compression_type::NONE) {
#if defined(__GNUC__) && (__GNUC__ >= 14)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdangling-reference"
#endif
        auto const& ss = strm_desc[stripe_id][stream.ids[type] - (columns.size() + 1)];
#if defined(__GNUC__) && (__GNUC__ >= 14)
#pragma GCC diagnostic pop
#endif
        record.blk_pos   = ss.first_block;
        record.comp_pos  = 0;
        record.comp_size = ss.stream_size;
      }
    }
    return record;
  };
  auto scan_record = [=, &comp_res](encoder_chunk_streams const& stream,
                                    stream_index_type type,
                                    row_group_index_info& record) {
    if (record.pos >= 0) {
      record.pos += stream.lengths[type];
      while ((record.pos >= 0) && (record.blk_pos >= 0) &&
             (static_cast<size_t>(record.pos) >= compression_blocksize) &&
             (record.comp_pos + block_header_size + comp_res[record.blk_pos].bytes_written <
              static_cast<size_t>(record.comp_size))) {
        record.pos -= compression_blocksize;
        record.comp_pos += block_header_size + comp_res[record.blk_pos].bytes_written;
        record.blk_pos += 1;
      }
    }
  };

  auto kind = TypeKind::STRUCT;
  // TBD: Not sure we need an empty index stream for column 0
  if (stream_id != 0) {
#if defined(__GNUC__) && (__GNUC__ >= 14)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdangling-reference"
#endif
    auto const& strm = enc_streams[column_id][0];
#if defined(__GNUC__) && (__GNUC__ >= 14)
#pragma GCC diagnostic pop
#endif
    present = find_record(strm, CI_PRESENT);
    data    = find_record(strm, CI_DATA);
    data2   = find_record(strm, CI_DATA2);

    // Change string dictionary to int from index point of view
    kind = columns[column_id].orc_kind();
    if (kind == TypeKind::STRING && columns[column_id].orc_encoding() == DICTIONARY_V2) {
      kind = TypeKind::INT;
    }
  }

  protobuf_writer pbw((compression != compression_type::NONE) ? 3 : 0);

  // Add row index entries
  auto const& rowgroups_range = segmentation.stripes[stripe_id];
  std::for_each(rowgroups_range.cbegin(), rowgroups_range.cend(), [&](auto rowgroup) {
    pbw.put_row_index_entry(present.comp_pos,
                            present.pos,
                            data.comp_pos,
                            data.pos,
                            data2.comp_pos,
                            data2.pos,
                            kind,
                            (rg_stats.empty() or stream_id == 0)
                              ? nullptr
                              : (&rg_stats[column_id * segmentation.num_rowgroups() + rowgroup]));

    if (stream_id != 0) {
#if defined(__GNUC__) && (__GNUC__ >= 14)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdangling-reference"
#endif
      const auto& strm = enc_streams[column_id][rowgroup];
#if defined(__GNUC__) && (__GNUC__ >= 14)
#pragma GCC diagnostic pop
#endif
      scan_record(strm, CI_PRESENT, present);
      scan_record(strm, CI_DATA, data);
      scan_record(strm, CI_DATA2, data2);
    }
  });

  (*streams)[stream_id].length = pbw.size();
  if (compression != compression_type::NONE) {
    uint32_t uncomp_ix_len = (uint32_t)((*streams)[stream_id].length - 3) * 2 + 1;
    pbw.buffer()[0]        = static_cast<uint8_t>(uncomp_ix_len >> 0);
    pbw.buffer()[1]        = static_cast<uint8_t>(uncomp_ix_len >> 8);
    pbw.buffer()[2]        = static_cast<uint8_t>(uncomp_ix_len >> 16);
  }
  out_sink->host_write(pbw.data(), pbw.size());
  stripe->indexLength += pbw.size();
}

/**
 * @brief Write the specified column's data streams
 *
 * @param[in] strm_desc Stream's descriptor
 * @param[in] enc_stream Chunk's streams
 * @param[in] compressed_data Compressed stream data
 * @param[in,out] bounce_buffer Pinned memory bounce buffer for D2H data transfer
 * @param[in,out] stripe Stream's parent stripe
 * @param[in,out] streams List of all streams
 * @param[in] compression The compression format
 * @param[in] out_sink Sink for writing data
 * @param[in] stream CUDA stream used for device memory operations and kernel launches
 * @return An std::future that should be synchronized to ensure the writing is complete
 */
std::future<void> write_data_stream(stripe_stream const& strm_desc,
                                    encoder_chunk_streams const& enc_stream,
                                    uint8_t const* compressed_data,
                                    host_span<uint8_t> bounce_buffer,
                                    StripeInformation* stripe,
                                    orc_streams* streams,
                                    compression_type compression,
                                    std::unique_ptr<data_sink> const& out_sink,
                                    cuda::stream_ref stream)
{
  auto const length                                        = strm_desc.stream_size;
  (*streams)[enc_stream.ids[strm_desc.stream_type]].length = length;
  if (length == 0) {
    return std::async(std::launch::deferred, [] {});
  }

  auto const* stream_in = (compression == compression_type::NONE)
                            ? enc_stream.data_ptrs[strm_desc.stream_type]
                            : (compressed_data + strm_desc.bfr_offset);

  auto write_task = [&]() {
    if (out_sink->is_device_write_preferred(length)) {
      return out_sink->device_write_async(stream_in, length, stream);
    } else {
      cudf::detail::cuda_memcpy(
        bounce_buffer.subspan(0, length), device_span<uint8_t const>{stream_in, length}, stream);

      out_sink->host_write(bounce_buffer.data(), length);
      return std::async(std::launch::deferred, [] {});
    }
  }();
  stripe->dataLength += length;
  return write_task;
}

/**
 * @brief Insert 3-byte uncompressed block headers in a byte vector
 *
 * @param compression The compression kind
 * @param compression_blocksize The block size used for compression
 * @param v The destitation byte vector to write, which must include initial 3-byte header
 */
void add_uncompressed_block_headers(compression_type compression,
                                    size_t compression_blocksize,
                                    std::vector<uint8_t>& v)
{
  if (compression != compression_type::NONE) {
    size_t uncomp_len = v.size() - 3, pos = 0, block_len;
    while (uncomp_len > compression_blocksize) {
      block_len  = compression_blocksize * 2 + 1;
      v[pos + 0] = static_cast<uint8_t>(block_len >> 0);
      v[pos + 1] = static_cast<uint8_t>(block_len >> 8);
      v[pos + 2] = static_cast<uint8_t>(block_len >> 16);
      pos += 3 + compression_blocksize;
      v.insert(v.begin() + pos, 3, 0);
      uncomp_len -= compression_blocksize;
    }
    block_len  = uncomp_len * 2 + 1;
    v[pos + 0] = static_cast<uint8_t>(block_len >> 0);
    v[pos + 1] = static_cast<uint8_t>(block_len >> 8);
    v[pos + 2] = static_cast<uint8_t>(block_len >> 16);
  }
}

void pushdown_lists_null_mask(orc_column_view const& col,
                              device_span<orc_column_device_view> d_columns,
                              bitmask_type const* parent_pd_mask,
                              device_span<bitmask_type> out_mask,
                              cuda::stream_ref stream)
{
  // Set all bits - correct unless there's a mismatch between offsets and null mask
  CUDF_CUDA_TRY(cudaMemsetAsync(static_cast<void*>(out_mask.data()),
                                255,
                                out_mask.size() * sizeof(bitmask_type),
                                stream.get()));

  // Reset bits where a null list element has rows in the child column
  thrust::for_each_n(
    rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
    cuda::counting_iterator<uint32_t>{0},
    col.size(),
    [d_columns, col_idx = col.index(), parent_pd_mask, out_mask] __device__(auto& idx) {
      auto const d_col        = d_columns[col_idx];
      auto const is_row_valid = d_col.is_valid(idx) and bit_value_or(parent_pd_mask, idx, true);
      if (not is_row_valid) {
        auto offsets                = d_col.child(lists_column_view::offsets_column_index);
        auto const child_rows_begin = offsets.element<int32_t>(idx + d_col.offset());
        auto const child_rows_end   = offsets.element<int32_t>(idx + 1 + d_col.offset());
        for (auto child_row = child_rows_begin; child_row < child_rows_end; ++child_row)
          clear_bit(out_mask.data(), child_row);
      }
    });
}

/**
 * @brief All pushdown masks in a table.
 *
 * Pushdown masks are applied to child column(s). Only bits of the child column null mask that
 * correspond to set pushdown mask bits are encoded into the output file. Similarly, rows where
 * pushdown mask is 0 are treated as invalid and not included in the output.
 */
struct pushdown_null_masks {
  // Owning vector for masks in device memory
  std::vector<rmm::device_uvector<bitmask_type>> data;
  // Pointers to pushdown masks in device memory. Can be same for multiple columns.
  cudf::detail::host_vector<bitmask_type const*> masks;
};

pushdown_null_masks init_pushdown_null_masks(orc_table_view& orc_table, cuda::stream_ref stream)
{
  auto mask_ptrs =
    cudf::detail::make_empty_host_vector<bitmask_type const*>(orc_table.num_columns(), stream);
  std::vector<rmm::device_uvector<bitmask_type>> pd_masks;
  for (auto const& col : orc_table.columns) {
    // Leaf columns don't need pushdown masks
    if (col.num_children() == 0) {
      mask_ptrs.push_back({nullptr});
      continue;
    }
    auto const parent_pd_mask = col.is_child() ? mask_ptrs[col.parent_index()] : nullptr;
    auto const null_mask      = col.null_mask();

    if (null_mask == nullptr and parent_pd_mask == nullptr) {
      mask_ptrs.push_back({nullptr});
      continue;
    }
    if (col.orc_kind() == STRUCT) {
      if (null_mask != nullptr and parent_pd_mask == nullptr) {
        // Reuse own null mask
        mask_ptrs.push_back(null_mask);
      } else if (null_mask == nullptr and parent_pd_mask != nullptr) {
        // Reuse parent's pushdown mask
        mask_ptrs.push_back(parent_pd_mask);
      } else {
        // Both are nullable, allocate new pushdown mask
        pd_masks.emplace_back(num_bitmask_words(col.size()), stream);
        mask_ptrs.push_back({pd_masks.back().data()});

        thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                          null_mask,
                          null_mask + pd_masks.back().size(),
                          parent_pd_mask,
                          pd_masks.back().data(),
                          cuda::std::bit_and<bitmask_type>());
      }
    }
    if (col.orc_kind() == LIST or col.orc_kind() == MAP) {
      // Need a new pushdown mask unless both the parent and current column are not nullable
      auto const child_col = orc_table.column(col.child_begin()[0]);
      // pushdown mask applies to child column(s); use the child column size
      pd_masks.emplace_back(num_bitmask_words(child_col.size()), stream);
      mask_ptrs.push_back({pd_masks.back().data()});
      pushdown_lists_null_mask(col, orc_table.d_columns, parent_pd_mask, pd_masks.back(), stream);
    }
  }

  // Attach null masks to device column views (async)
  auto const d_mask_ptrs = cudf::detail::make_device_uvector_async(
    mask_ptrs, stream, cudf::get_current_device_resource_ref());
  thrust::for_each_n(
    rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
    cuda::counting_iterator<size_t>{0},
    orc_table.num_columns(),
    [cols = device_span<orc_column_device_view>{orc_table.d_columns},
     ptrs = device_span<bitmask_type const* const>{d_mask_ptrs}] __device__(auto& idx) {
      cols[idx].pushdown_mask = ptrs[idx];
    });

  return {std::move(pd_masks), std::move(mask_ptrs)};
}

template <typename T>
struct device_stack {
  __device__ device_stack(T* stack_storage, int capacity) : stack(stack_storage), capacity(capacity)
  {
  }
  __device__ void push(T const& val)
  {
    cudf_assert(size < capacity and "Stack overflow");
    stack[size++] = val;
  }
  __device__ T pop()
  {
    cudf_assert(size > 0 and "Stack underflow");
    return stack[--size];
  }
  __device__ bool empty() { return size == 0; }

 private:
  T* stack;
  int capacity;
  int size{0};
};

orc_table_view make_orc_table_view(table_view const& table,
                                   table_device_view const& d_table,
                                   table_input_metadata const& table_meta,
                                   cuda::stream_ref stream)
{
  std::vector<orc_column_view> orc_columns;
  std::vector<uint32_t> str_col_indexes;

  std::function<void(column_view const&, orc_column_view*, column_in_metadata const&)>
    append_orc_column =
      [&](column_view const& col, orc_column_view* parent_col, column_in_metadata const& col_meta) {
        int const str_idx =
          (col.type().id() == type_id::STRING) ? static_cast<int>(str_col_indexes.size()) : -1;

        auto const new_col_idx = orc_columns.size();
        orc_columns.emplace_back(new_col_idx, str_idx, parent_col, col, col_meta);
        if (orc_columns[new_col_idx].is_string()) { str_col_indexes.push_back(new_col_idx); }

        auto const kind = orc_columns[new_col_idx].orc_kind();
        if (kind == TypeKind::LIST) {
          append_orc_column(col.child(lists_column_view::child_column_index),
                            &orc_columns[new_col_idx],
                            col_meta.child(lists_column_view::child_column_index));
        } else if (kind == TypeKind::STRUCT) {
          for (auto child_idx = 0; child_idx != col.num_children(); ++child_idx) {
            append_orc_column(
              col.child(child_idx), &orc_columns[new_col_idx], col_meta.child(child_idx));
          }
        } else if (kind == TypeKind::MAP) {
          // MAP: skip to the list child - include grandchildren columns instead of children
          auto const real_parent_col   = col.child(lists_column_view::child_column_index);
          auto const& real_parent_meta = col_meta.child(lists_column_view::child_column_index);
          CUDF_EXPECTS(real_parent_meta.num_children() == 2,
                       "Map struct column should have exactly two children");
          // process MAP key
          append_orc_column(
            real_parent_col.child(0), &orc_columns[new_col_idx], real_parent_meta.child(0));
          // process MAP value
          append_orc_column(
            real_parent_col.child(1), &orc_columns[new_col_idx], real_parent_meta.child(1));
        }
      };

  for (auto col_idx = 0; col_idx < table.num_columns(); ++col_idx) {
    append_orc_column(table.column(col_idx), nullptr, table_meta.column_metadata[col_idx]);
  }

  auto type_kinds = cudf::detail::make_empty_host_vector<TypeKind>(orc_columns.size(), stream);
  std::transform(
    orc_columns.cbegin(), orc_columns.cend(), std::back_inserter(type_kinds), [](auto& orc_column) {
      return orc_column.orc_kind();
    });
  auto const d_type_kinds = cudf::detail::make_device_uvector_async(
    type_kinds, stream, cudf::get_current_device_resource_ref());

  rmm::device_uvector<orc_column_device_view> d_orc_columns(orc_columns.size(), stream);
  using stack_value_type =
    cuda::std::pair<column_device_view const*, cuda::std::optional<uint32_t>>;
  rmm::device_uvector<stack_value_type> stack_storage(orc_columns.size(), stream);

  // pre-order append ORC device columns
  cudf::detail::device_single_thread(
    [d_orc_cols         = device_span<orc_column_device_view>{d_orc_columns},
     d_type_kinds       = device_span<TypeKind const>{d_type_kinds},
     d_table            = d_table,
     stack_storage      = stack_storage.data(),
     stack_storage_size = stack_storage.size()] __device__() {
      device_stack stack(stack_storage, stack_storage_size);

      thrust::for_each(
        thrust::seq,
        cuda::std::make_reverse_iterator(d_table.end()),
        cuda::std::make_reverse_iterator(d_table.begin()),
        [&stack](column_device_view const& c) { stack.push({&c, cuda::std::nullopt}); });

      uint32_t idx = 0;
      while (not stack.empty()) {
        auto [col, parent] = stack.pop();
        d_orc_cols[idx]    = orc_column_device_view{*col, parent};

        if (d_type_kinds[idx] == TypeKind::MAP) {
          // Skip to the list child - do not include the child column, just grandchildren columns
          col = &col->children()[lists_column_view::child_column_index];
        }

        if (col->type().id() == type_id::LIST) {
          stack.push({&col->children()[lists_column_view::child_column_index], idx});
        } else if (col->type().id() == type_id::STRUCT) {
          thrust::for_each(thrust::seq,
                           cuda::std::make_reverse_iterator(col->children().end()),
                           cuda::std::make_reverse_iterator(col->children().begin()),
                           [&stack, idx](column_device_view const& c) { stack.push({&c, idx}); });
        }
        ++idx;
      }
    },
    stream);

  return {std::move(orc_columns),
          std::move(d_orc_columns),
          str_col_indexes,
          cudf::detail::make_device_uvector(
            str_col_indexes, stream, cudf::get_current_device_resource_ref())};
}

hostdevice_2dvector<rowgroup_rows> calculate_rowgroup_bounds(orc_table_view const& orc_table,
                                                             size_type rowgroup_size,
                                                             cuda::stream_ref stream)
{
  auto const num_rowgroups =
    cudf::util::div_rounding_up_unsafe<size_t, size_t>(orc_table.num_rows(), rowgroup_size);

  hostdevice_2dvector<rowgroup_rows> rowgroup_bounds(
    num_rowgroups, orc_table.num_columns(), stream);
  thrust::for_each_n(
    rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
    cuda::counting_iterator<size_t>{0},
    num_rowgroups,
    [cols      = device_span<orc_column_device_view const>{orc_table.d_columns},
     rg_bounds = rowgroup_bounds.device_view(),
     rowgroup_size] __device__(auto rg_idx) mutable {
      thrust::transform(
        thrust::seq, cols.begin(), cols.end(), rg_bounds[rg_idx].begin(), [&](auto const& col) {
          // Root column
          if (!col.parent_index.has_value()) {
            size_type const rows_begin = rg_idx * rowgroup_size;
            auto const rows_end        = static_cast<size_type>(cuda::std::min(
              static_cast<int64_t>(rows_begin) + rowgroup_size, static_cast<int64_t>(col.size())));
            return rowgroup_rows{rows_begin, rows_end};
          } else {
            // Child column
            auto const parent_index           = *col.parent_index;
            orc_column_device_view parent_col = cols[parent_index];
            auto const parent_rg              = rg_bounds[rg_idx][parent_index];
            if (parent_col.type().id() != type_id::LIST) {
              auto const offset_diff = parent_col.offset() - col.offset();
              return rowgroup_rows{parent_rg.begin + offset_diff, parent_rg.end + offset_diff};
            }

            auto offsets = parent_col.child(lists_column_view::offsets_column_index);
            auto const rows_begin =
              offsets.element<int32_t>(parent_rg.begin + parent_col.offset()) - col.offset();
            auto const rows_end =
              offsets.element<int32_t>(parent_rg.end + parent_col.offset()) - col.offset();

            return rowgroup_rows{rows_begin, rows_end};
          }
        });
    });
  rowgroup_bounds.device_to_host(stream);

  return rowgroup_bounds;
}

// returns host vector of per-rowgroup sizes
encoder_decimal_info decimal_chunk_sizes(orc_table_view& orc_table,
                                         file_segmentation const& segmentation,
                                         cuda::stream_ref stream)
{
  std::map<uint32_t, rmm::device_uvector<uint32_t>> elem_sizes;
  // Compute per-element offsets (within each row group) on the device
  for (auto& orc_col : orc_table.columns) {
    if (orc_col.orc_kind() == DECIMAL) {
      auto& current_sizes =
        elem_sizes.insert({orc_col.index(), rmm::device_uvector<uint32_t>(orc_col.size(), stream)})
          .first->second;
      thrust::tabulate(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                       current_sizes.begin(),
                       current_sizes.end(),
                       [d_cols  = device_span<orc_column_device_view const>{orc_table.d_columns},
                        col_idx = orc_col.index()] __device__(auto idx) {
                         auto const& col          = d_cols[col_idx];
                         auto const pushdown_mask = [&]() -> cudf::bitmask_type const* {
                           auto const parent_index = d_cols[col_idx].parent_index;
                           if (!parent_index.has_value()) return nullptr;
                           return d_cols[parent_index.value()].pushdown_mask;
                         }();

                         if (col.is_null(idx) or not bit_value_or(pushdown_mask, idx, true))
                           return 0;

                         __int128_t const element =
                           col.type().id() == type_id::DECIMAL32   ? col.element<int32_t>(idx)
                           : col.type().id() == type_id::DECIMAL64 ? col.element<int64_t>(idx)
                                                                   : col.element<__int128_t>(idx);

                         __int128_t const sign      = (element < 0) ? 1 : 0;
                         __uint128_t zigzaged_value = ((element ^ -sign) * 2) + sign;

                         return varint_size(zigzaged_value);
                       });

      orc_col.attach_decimal_offsets(current_sizes.data());
    }
  }
  if (elem_sizes.empty()) return {};

  // Compute element offsets within each row group
  decimal_sizes_to_offsets(segmentation.rowgroups, elem_sizes, stream);

  // Gather the row group sizes and copy to host
  auto d_tmp_rowgroup_sizes = rmm::device_uvector<uint32_t>(segmentation.num_rowgroups(), stream);
  std::map<uint32_t, cudf::detail::host_vector<uint32_t>> rg_sizes;
  for (auto const& [col_idx, esizes] : elem_sizes) {
    // Copy last elem in each row group - equal to row group size
    thrust::tabulate(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                     d_tmp_rowgroup_sizes.begin(),
                     d_tmp_rowgroup_sizes.end(),
                     [src       = esizes.data(),
                      col_idx   = col_idx,
                      rg_bounds = segmentation.rowgroups.device_view()] __device__(auto idx) {
                       return src[rg_bounds[idx][col_idx].end - 1];
                     });

    rg_sizes.emplace(col_idx, cudf::detail::make_host_vector_async(d_tmp_rowgroup_sizes, stream));
  }

  return {std::move(elem_sizes), std::move(rg_sizes)};
}

std::map<uint32_t, size_t> decimal_column_sizes(
  std::map<uint32_t, cudf::detail::host_vector<uint32_t>> const& chunk_sizes)
{
  std::map<uint32_t, size_t> column_sizes;
  std::transform(chunk_sizes.cbegin(),
                 chunk_sizes.cend(),
                 std::inserter(column_sizes, column_sizes.end()),
                 [](auto const& chunk_size) -> std::pair<uint32_t, size_t> {
                   return {
                     chunk_size.first,
                     std::accumulate(chunk_size.second.cbegin(), chunk_size.second.cend(), 0lu)};
                 });
  return column_sizes;
}

std::unique_ptr<table_input_metadata> make_table_meta(table_view const& input)
{
  auto table_meta = std::make_unique<table_input_metadata>(input);

  // Fill unnamed columns' names in table_meta
  std::function<void(column_in_metadata&, std::string)> add_default_name =
    [&](column_in_metadata& col_meta, std::string default_name) {
      if (col_meta.get_name().empty()) { col_meta.set_name(default_name); }
      for (size_type i = 0; i < col_meta.num_children(); ++i) {
        add_default_name(col_meta.child(i), std::to_string(i));
      }
    };
  for (size_t i = 0; i < table_meta->column_metadata.size(); ++i) {
    add_default_name(table_meta->column_metadata[i], "_col" + std::to_string(i));
  }

  return table_meta;
}

// Computes the number of characters in each rowgroup for each string column and attaches the
// results to the corresponding orc_column_view. The owning host vector is returned.
auto set_rowgroup_char_counts(orc_table_view& orc_table,
                              device_2dspan<rowgroup_rows const> rowgroup_bounds,
                              cuda::stream_ref stream)
{
  auto const num_rowgroups = rowgroup_bounds.size().first;
  auto const num_str_cols  = orc_table.num_string_columns();

  auto counts         = rmm::device_uvector<size_type>(num_str_cols * num_rowgroups, stream);
  auto counts_2d_view = device_2dspan<size_type>(counts, num_rowgroups);
  rowgroup_char_counts(counts_2d_view,
                       orc_table.d_columns,
                       rowgroup_bounds,
                       orc_table.d_string_column_indices,
                       stream);

  auto const h_counts = cudf::detail::make_host_vector(counts, stream);

  for (auto col_idx : orc_table.string_column_indices) {
    auto& str_column = orc_table.column(col_idx);
    str_column.attach_rowgroup_char_counts(
      {h_counts.data() + str_column.str_index() * num_rowgroups, num_rowgroups});
  }

  return h_counts;
}

// Holds the stripe dictionary descriptors and dictionary buffers.
struct stripe_dictionaries {
  hostdevice_2dvector<stripe_dictionary> views;            // descriptors [string_column][stripe]
  std::vector<rmm::device_uvector<uint32_t>> data_owner;   // dictionary data owner, per stripe
  std::vector<rmm::device_uvector<uint32_t>> index_owner;  // dictionary index owner, per stripe
  std::vector<rmm::device_uvector<uint32_t>> order_owner;  // dictionary order owner, per stripe

  // Should be called after encoding is complete to deallocate the dictionary buffers.
  void on_encode_complete(cuda::stream_ref stream)
  {
    data_owner.clear();
    index_owner.clear();
    order_owner.clear();

    for (auto& sd : views.host_view().flat_view()) {
      sd.data       = {};
      sd.index      = {};
      sd.data_order = {};
    }
    views.host_to_device_async(stream);
  }
};

/**
 * @brief Compares two rows in a strings column
 */
struct string_rows_less {
  device_span<orc_column_device_view> cols;
  uint32_t col_idx;
  __device__ bool operator()(size_type lhs_idx, size_type rhs_idx) const
  {
    auto const& col = cols[col_idx];
    return col.element<string_view>(lhs_idx) < col.element<string_view>(rhs_idx);
  }
};

// Build stripe dictionaries for string columns
stripe_dictionaries build_dictionaries(orc_table_view& orc_table,
                                       file_segmentation const& segmentation,
                                       bool sort_dictionaries,
                                       cuda::stream_ref stream)
{
  // Variable to keep track of the current total map storage size
  size_t total_map_storage_size = 0;
  std::vector<std::vector<size_t>> hash_maps_storage_offsets(
    orc_table.string_column_indices.size());
  for (auto col_idx : orc_table.string_column_indices) {
    auto& str_column = orc_table.column(col_idx);
    for (auto const& stripe : segmentation.stripes) {
      auto const stripe_num_rows =
        stripe.size == 0 ? 0
                         : segmentation.rowgroups[stripe.first + stripe.size - 1][col_idx].end -
                             segmentation.rowgroups[stripe.first][col_idx].begin;
      hash_maps_storage_offsets[str_column.str_index()].emplace_back(total_map_storage_size);
      total_map_storage_size += stripe_num_rows * occupancy_factor;
    }
    hash_maps_storage_offsets[str_column.str_index()].emplace_back(total_map_storage_size);
  }

  hostdevice_2dvector<stripe_dictionary> stripe_dicts(
    orc_table.num_string_columns(), segmentation.num_stripes(), stream);
  if (stripe_dicts.count() == 0) return {std::move(stripe_dicts), {}, {}};

  // Create a single bulk storage to use for all sub-dictionaries
  auto map_storage = std::make_unique<storage_type>(
    total_map_storage_size, rmm::mr::polymorphic_allocator<char>{}, stream.get());

  // Initialize stripe dictionaries
  for (auto col_idx : orc_table.string_column_indices) {
    auto& str_column       = orc_table.column(col_idx);
    auto const str_col_idx = str_column.str_index();
    str_column.attach_stripe_dicts(stripe_dicts[str_col_idx],
                                   stripe_dicts.device_view()[str_col_idx]);
    for (auto const& stripe : segmentation.stripes) {
      auto const stripe_idx = stripe.id;
      auto& sd              = stripe_dicts[str_col_idx][stripe_idx];

      sd.map_slots      = {map_storage->data() + hash_maps_storage_offsets[str_col_idx][stripe_idx],
                           hash_maps_storage_offsets[str_col_idx][stripe_idx + 1] -
                             hash_maps_storage_offsets[str_col_idx][stripe_idx]};
      sd.column_idx     = col_idx;
      sd.start_row      = segmentation.rowgroups[stripe.first][col_idx].begin;
      sd.start_rowgroup = stripe.first;
      sd.num_rows =
        segmentation.rowgroups[stripe.first + stripe.size - 1][col_idx].end - sd.start_row;

      sd.entry_count = 0;
      sd.char_count  = 0;
    }
  }
  stripe_dicts.host_to_device_async(stream);

  map_storage->initialize_async({KEY_SENTINEL, VALUE_SENTINEL}, {stream.get()});
  populate_dictionary_hash_maps(stripe_dicts, orc_table.d_columns, stream);
  // Copy the entry counts and char counts from the device to the host
  stripe_dicts.device_to_host(stream);

  // Data owners; can be cleared after encode
  std::vector<rmm::device_uvector<uint32_t>> dict_data_owner;
  std::vector<rmm::device_uvector<uint32_t>> dict_index_owner;
  std::vector<rmm::device_uvector<uint32_t>> dict_order_owner;
  // Make decision about which stripes to encode with dictionary encoding
  for (auto col_idx : orc_table.string_column_indices) {
    auto& str_column = orc_table.column(col_idx);
    bool col_use_dictionary{false};
    for (auto const& stripe : segmentation.stripes) {
      auto const stripe_idx        = stripe.id;
      auto const str_col_idx       = str_column.str_index();
      auto& sd                     = stripe_dicts[str_col_idx][stripe_idx];
      auto const direct_char_count = std::accumulate(
        cuda::counting_iterator{stripe.first},
        cuda::counting_iterator{stripe.first + stripe.size},
        0,
        [&](auto total, auto const& rg) { return total + str_column.rowgroup_char_count(rg); });
      // Enable dictionary encoding if the dictionary size is smaller than the direct encode size
      // The estimate excludes the LENGTH stream size, which is present in both cases
      sd.is_enabled = [&]() {
        auto const dict_index_size = varint_size(sd.entry_count);
        return sd.char_count + dict_index_size * sd.entry_count < direct_char_count;
      }();
      if (sd.is_enabled) {
        dict_data_owner.emplace_back(sd.entry_count, stream);
        sd.data            = dict_data_owner.back();
        col_use_dictionary = true;
      } else {
        // Clear hash map storage as dictionary encoding is not used for this stripe
        sd.map_slots = {};
      }
    }
    // If any stripe uses dictionary encoding, allocate index storage for the whole column
    if (col_use_dictionary) {
      dict_index_owner.emplace_back(str_column.size(), stream);
      for (auto& sd : stripe_dicts[str_column.str_index()]) {
        sd.index = dict_index_owner.back();
      }
    }
  }
  stripe_dicts.host_to_device_async(stream);

  collect_map_entries(stripe_dicts, stream);
  get_dictionary_indices(stripe_dicts, orc_table.d_columns, stream);

  // synchronize to ensure the copy is complete before we clear `map_slots`
  stream.sync();

  // deallocate hash map storage, unused after this point
  map_storage.reset();

  // Clear map slots and attach order buffers
  auto dictionaries_flat = stripe_dicts.host_view().flat_view();
  for (auto& sd : dictionaries_flat) {
    if (not sd.is_enabled) { continue; }

    sd.map_slots = {};
    if (sort_dictionaries) {
      dict_order_owner.emplace_back(sd.entry_count, stream);
      sd.data_order = dict_order_owner.back();
    } else {
      sd.data_order = {};
    }
  }
  stripe_dicts.host_to_device_async(stream);

  // Sort stripe dictionaries alphabetically
  if (sort_dictionaries) {
    auto streams = cudf::detail::fork_streams(stream, std::min<size_t>(dict_order_owner.size(), 8));
    auto stream_idx = 0;
    for (auto& sd : dictionaries_flat) {
      if (not sd.is_enabled) { continue; }

      auto const& current_stream = streams[stream_idx];

      // Sort the dictionary data and create a mapping from the sorted order to the original
      thrust::sequence(
        rmm::exec_policy_nosync(current_stream, cudf::get_current_device_resource_ref()),
        sd.data_order.begin(),
        sd.data_order.end());
      thrust::sort_by_key(
        rmm::exec_policy_nosync(current_stream, cudf::get_current_device_resource_ref()),
        sd.data.begin(),
        sd.data.end(),
        sd.data_order.begin(),
        string_rows_less{orc_table.d_columns, sd.column_idx});

      // Create the inverse permutation - i.e. the mapping from the original order to the sorted
      auto order_copy = cudf::detail::make_device_uvector_async<uint32_t>(
        sd.data_order, current_stream, cudf::get_current_device_resource_ref());
      thrust::scatter(
        rmm::exec_policy_nosync(current_stream, cudf::get_current_device_resource_ref()),
        cuda::counting_iterator<uint32_t>{0},
        cuda::counting_iterator{static_cast<uint32_t>(sd.data_order.size())},
        order_copy.begin(),
        sd.data_order.begin());

      stream_idx = (stream_idx + 1) % streams.size();
    }

    cudf::detail::join_streams(streams, stream);
  }

  return {std::move(stripe_dicts),
          std::move(dict_data_owner),
          std::move(dict_index_owner),
          std::move(dict_order_owner)};
}

struct stripe_stream_size_less {
  __device__ bool operator()(stripe_stream const& lhs, stripe_stream const& rhs) const
  {
    return lhs.stream_size < rhs.stream_size;
  }
};

[[nodiscard]] uint32_t find_largest_stream_size(device_2dspan<stripe_stream const> ss,
                                                cuda::stream_ref stream)
{
  auto const longest_stream =
    thrust::max_element(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                        ss.data(),
                        ss.data() + ss.count(),
                        stripe_stream_size_less{});

  auto const h_longest_stream =
    cudf::detail::make_host_vector(device_span<stripe_stream const>{longest_stream, 1}, stream);
  return h_longest_stream[0].stream_size;
}

/**
 * @brief Perform the processing steps needed to convert the input table into the output ORC data
 * for writing, such as compression and ORC encoding.
 *
 * @param input The input table
 * @param table_meta The table metadata
 * @param max_stripe_size Maximum size of stripes in the output file
 * @param row_index_stride The row index stride
 * @param enable_dictionary Whether dictionary is enabled
 * @param sort_dictionaries Whether to sort the dictionaries
 * @param compression The compression format
 * @param compression_blocksize The block size used for compression
 * @param stats_freq Column statistics granularity type for parquet/orc writers
 * @param collect_compression_stats Flag to indicate if compression statistics should be collected
 * @param write_mode Flag to indicate if there is only a single table write
 * @param out_sink Sink for writing data
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @return A tuple of the intermediate results containing the processed data
 */
auto convert_table_to_orc_data(table_view const& input,
                               table_input_metadata const& table_meta,
                               stripe_size_limits max_stripe_size,
                               size_type row_index_stride,
                               bool enable_dictionary,
                               bool sort_dictionaries,
                               compression_type compression,
                               size_t compression_blocksize,
                               statistics_freq stats_freq,
                               bool collect_compression_stats,
                               single_write_mode write_mode,
                               data_sink const& out_sink,
                               cuda::stream_ref stream)
{
  auto const input_tview = table_device_view::create(input, stream);

  auto orc_table = make_orc_table_view(input, *input_tview, table_meta, stream);

  // This is unused but it holds memory buffers for later access thus needs to be kept alive.
  [[maybe_unused]] auto const pd_masks = init_pushdown_null_masks(orc_table, stream);

  auto rowgroup_bounds = calculate_rowgroup_bounds(orc_table, row_index_stride, stream);

  [[maybe_unused]] auto const rg_char_counts_data =
    set_rowgroup_char_counts(orc_table, rowgroup_bounds, stream);

  // Decide stripe boundaries based on rowgroups and char counts
  auto segmentation =
    calculate_segmentation(orc_table.columns, std::move(rowgroup_bounds), max_stripe_size, stream);

  auto stripe_dicts    = build_dictionaries(orc_table, segmentation, sort_dictionaries, stream);
  auto dec_chunk_sizes = decimal_chunk_sizes(orc_table, segmentation, stream);

  auto const block_align = compress_required_chunk_alignment(compression);

  auto streams = create_streams(orc_table.columns,
                                segmentation,
                                decimal_column_sizes(dec_chunk_sizes.rg_sizes),
                                enable_dictionary,
                                compression,
                                write_mode);

  auto [enc_data, extents] = encode_columns(
    orc_table, std::move(dec_chunk_sizes), segmentation, streams, block_align, stream);

  stripe_dicts.on_encode_complete(stream);

  auto const num_rows = input.num_rows();

  // Assemble individual disparate column chunks into contiguous data streams
  size_type const num_index_streams = (orc_table.num_columns() + 1);
  auto const num_data_streams       = streams.size() - num_index_streams;
  hostdevice_2dvector<stripe_stream> strm_descs(
    segmentation.num_stripes(), num_data_streams, stream);
  auto stripes = gather_stripes(num_index_streams,
                                segmentation,
                                host_2dspan<extent_info const>{extents, streams.size()},
                                &enc_data,
                                &strm_descs,
                                stream);

  if (num_rows == 0) {
    return std::tuple{std::move(enc_data),
                      std::move(segmentation),
                      std::move(orc_table),
                      rmm::device_uvector<uint8_t>{0, stream},               // compressed_data
                      cudf::detail::hostdevice_vector<codec_exec_result>{},  // comp_results
                      std::move(strm_descs),
                      intermediate_statistics{orc_table, stream},
                      std::optional<writer_compression_statistics>{},
                      std::move(streams),
                      std::move(stripes),
                      std::move(stripe_dicts.views),
                      cudf::detail::make_pinned_vector_async<uint8_t>(0, stream)};
  }

  // Allocate intermediate output stream buffer
  size_t compressed_bfr_size   = 0;
  size_t num_compressed_blocks = 0;

  auto const largest_stream_size = find_largest_stream_size(strm_descs, stream);
  auto const max_compressed_block_size =
    max_compressed_size(compression, std::min<size_t>(largest_stream_size, compression_blocksize));
  auto const padded_max_compressed_block_size =
    util::round_up_unsafe<size_t>(max_compressed_block_size, block_align);
  auto const padded_block_header_size =
    util::round_up_unsafe<size_t>(block_header_size, block_align);

  for (auto& ss : strm_descs.host_view().flat_view()) {
    size_t stream_size = ss.stream_size;
    if (compression != compression_type::NONE) {
      ss.first_block = num_compressed_blocks;
      ss.bfr_offset  = compressed_bfr_size;

      auto num_blocks =
        std::max<uint32_t>((stream_size + compression_blocksize - 1) / compression_blocksize, 1);
      stream_size += num_blocks * block_header_size;
      num_compressed_blocks += num_blocks;
      compressed_bfr_size +=
        (padded_block_header_size + padded_max_compressed_block_size) * num_blocks;
    }
  }

  // Compress the data streams
  rmm::device_uvector<uint8_t> compressed_data(compressed_bfr_size, stream);
  cudf::detail::hostdevice_vector<codec_exec_result> comp_results(num_compressed_blocks, stream);
  std::optional<writer_compression_statistics> compression_stats;
  thrust::fill(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
               comp_results.d_begin(),
               comp_results.d_end(),
               codec_exec_result{0, codec_status::FAILURE});
  if (compression != compression_type::NONE) {
    strm_descs.host_to_device_async(stream);
    compression_stats = compress_orc_data_streams(compressed_data,
                                                  num_compressed_blocks,
                                                  compression,
                                                  compression_blocksize,
                                                  max_compressed_block_size,
                                                  block_align,
                                                  collect_compression_stats,
                                                  strm_descs,
                                                  enc_data.streams,
                                                  comp_results,
                                                  stream);

    // deallocate encoded data as it is not needed anymore. Frees the arenas that outlived the
    // gather and clears the spans that referenced them.
    enc_data.persistent_buffer = rmm::device_uvector<uint8_t>{0, stream};
    enc_data.gathered_buffer   = rmm::device_uvector<uint8_t>{0, stream};
    enc_data.data.clear();

    strm_descs.device_to_host_async(stream);
    comp_results.device_to_host(stream);
  }

  auto const max_out_stream_size = [&]() {
    uint32_t max_stream_size = 0;
    for (auto const& ss : strm_descs.host_view().flat_view()) {
      if (!out_sink.is_device_write_preferred(ss.stream_size)) {
        max_stream_size = std::max(max_stream_size, ss.stream_size);
      }
    }
    return max_stream_size;
  }();

  auto bounce_buffer = cudf::detail::make_pinned_vector_async<uint8_t>(max_out_stream_size, stream);

  auto intermediate_stats = gather_statistic_blobs(stats_freq, orc_table, segmentation, stream);

  return std::tuple{std::move(enc_data),
                    std::move(segmentation),
                    std::move(orc_table),
                    std::move(compressed_data),
                    std::move(comp_results),
                    std::move(strm_descs),
                    std::move(intermediate_stats),
                    std::move(compression_stats),
                    std::move(streams),
                    std::move(stripes),
                    std::move(stripe_dicts.views),
                    std::move(bounce_buffer)};
}

}  // namespace

writer::impl::impl(std::unique_ptr<data_sink> sink,
                   orc_writer_options const& options,
                   single_write_mode mode,
                   cuda::stream_ref stream)
  : _stream(stream),
    _max_stripe_size{options.get_stripe_size_bytes(), options.get_stripe_size_rows()},
    _row_index_stride{options.get_row_index_stride()},
    _compression{options.get_compression()},
    _compression_blocksize(compression_block_size(_compression)),
    _compression_statistics(options.get_compression_statistics()),
    _stats_freq(options.get_statistics_freq()),
    _sort_dictionaries{options.get_enable_dictionary_sort()},
    _single_write_mode(mode),
    _kv_meta(options.get_key_value_metadata()),
    _out_sink(std::move(sink))
{
  if (options.get_metadata()) {
    _table_meta = std::make_unique<table_input_metadata>(*options.get_metadata());
  }
  CUDF_EXPECTS(is_supported_write_orc(_compression),
               "Compression type not supported for ORC writer");
}

writer::impl::impl(std::unique_ptr<data_sink> sink,
                   chunked_orc_writer_options const& options,
                   single_write_mode mode,
                   cuda::stream_ref stream)
  : _stream(stream),
    _max_stripe_size{options.get_stripe_size_bytes(), options.get_stripe_size_rows()},
    _row_index_stride{options.get_row_index_stride()},
    _compression{options.get_compression()},
    _compression_blocksize(compression_block_size(_compression)),
    _compression_statistics(options.get_compression_statistics()),
    _stats_freq(options.get_statistics_freq()),
    _sort_dictionaries{options.get_enable_dictionary_sort()},
    _single_write_mode(mode),
    _kv_meta(options.get_key_value_metadata()),
    _out_sink(std::move(sink))
{
  if (options.get_metadata()) {
    _table_meta = std::make_unique<table_input_metadata>(*options.get_metadata());
  }
  CUDF_EXPECTS(is_supported_write_orc(_compression),
               "Compression type not supported for ORC writer");
}

writer::impl::~impl() { close(); }

void writer::impl::write(table_view const& input)
{
  CUDF_EXPECTS(_state != writer_state::CLOSED, "Data has already been flushed to out and closed");

  if (not _table_meta) { _table_meta = make_table_meta(input); }

  // All kinds of memory allocation and data compressions/encoding are performed here.
  // If any error occurs, such as out-of-memory exception, the internal state of the current writer
  // is still intact.
  // Note that `out_sink_` is intentionally passed by const reference to prevent accidentally
  // writing anything to it.
  [[maybe_unused]] auto [enc_data,
                         segmentation,
                         orc_table,
                         compressed_data,
                         comp_results,
                         strm_descs,
                         intermediate_stats,
                         compression_stats,
                         streams,
                         stripes,
                         stripe_dicts, /* unused, but its data will be accessed via pointer later */
                         bounce_buffer] =
    convert_table_to_orc_data(input,
                              *_table_meta,
                              _max_stripe_size,
                              _row_index_stride,
                              _enable_dictionary,
                              _sort_dictionaries,
                              _compression,
                              _compression_blocksize,
                              _stats_freq,
                              _compression_statistics != nullptr,
                              _single_write_mode,
                              *_out_sink,
                              _stream);

  if (_state == writer_state::NO_DATA_WRITTEN) {
    // Write the ORC file header if this is the first write
    _out_sink->host_write(MAGIC, std::strlen(MAGIC));
  }

  // Compression/encoding were all successful. Now write the intermediate results.
  write_orc_data_to_sink(enc_data,
                         segmentation,
                         orc_table,
                         compressed_data,
                         comp_results,
                         strm_descs,
                         intermediate_stats.rowgroup_blobs,
                         streams,
                         stripes,
                         bounce_buffer);

  // Update data into the footer. This needs to be called even when num_rows==0.
  add_table_to_footer_data(orc_table, stripes);

  // Update file-level and compression statistics
  update_statistics(orc_table.num_rows(), std::move(intermediate_stats), compression_stats);

  _state = writer_state::DATA_WRITTEN;
}

void writer::impl::update_statistics(
  size_type num_rows,
  intermediate_statistics&& intermediate_stats,
  std::optional<writer_compression_statistics> const& compression_stats)
{
  _persisted_stripe_statistics.persist(
    num_rows, _single_write_mode, std::move(intermediate_stats), _stream);

  if (compression_stats.has_value() and _compression_statistics != nullptr) {
    *_compression_statistics += compression_stats.value();
  }
}

void writer::impl::write_orc_data_to_sink(encoded_data const& enc_data,
                                          file_segmentation const& segmentation,
                                          orc_table_view const& orc_table,
                                          device_span<uint8_t const> compressed_data,
                                          host_span<codec_exec_result const> comp_results,
                                          host_2dspan<stripe_stream const> strm_descs,
                                          host_span<col_stats_blob const> rg_stats,
                                          orc_streams& streams,
                                          host_span<StripeInformation> stripes,
                                          host_span<uint8_t> bounce_buffer)
{
  if (orc_table.num_rows() == 0) { return; }

  // Write stripes
  std::vector<std::future<void>> write_tasks;
  for (size_t stripe_id = 0; stripe_id < stripes.size(); ++stripe_id) {
    auto& stripe = stripes[stripe_id];

    stripe.offset = _out_sink->bytes_written();

    // Column (skippable) index streams appear at the start of the stripe
    size_type const num_index_streams = (orc_table.num_columns() + 1);
    for (size_type stream_id = 0; stream_id < num_index_streams; ++stream_id) {
      write_index_stream(stripe_id,
                         stream_id,
                         orc_table.columns,
                         segmentation,
                         enc_data.streams,
                         strm_descs,
                         comp_results,
                         rg_stats,
                         &stripe,
                         &streams,
                         _compression,
                         _compression_blocksize,
                         _out_sink);
    }

    // Column data consisting one or more separate streams
    for (auto const& strm_desc : strm_descs[stripe_id]) {
      write_tasks.push_back(write_data_stream(
        strm_desc,
        enc_data.streams[strm_desc.column_id][segmentation.stripes[stripe_id].first],
        compressed_data.data(),
        bounce_buffer,
        &stripe,
        &streams,
        _compression,
        _out_sink,
        _stream));
    }

    // Write stripefooter consisting of stream information
    StripeFooter sf;
    sf.streams = streams;
    sf.columns.resize(orc_table.num_columns() + 1);
    sf.columns[0].kind = DIRECT;
    for (size_t i = 1; i < sf.columns.size(); ++i) {
      sf.columns[i].kind = orc_table.column(i - 1).orc_encoding();
      sf.columns[i].dictionarySize =
        (sf.columns[i].kind == DICTIONARY_V2)
          ? orc_table.column(i - 1).host_stripe_dict(stripe_id).entry_count
          : 0;
      if (orc_table.column(i - 1).orc_kind() == TIMESTAMP) { sf.writerTimezone = "UTC"; }
    }

    protobuf_writer pbw;
    pbw.write(sf);
    if (_compression == compression_type::NONE) {
      _out_sink->host_write(pbw.data(), pbw.size());
      stripe.footerLength = pbw.size();
    } else {
      std::size_t bytes_written  = 0;
      std::size_t written_sf_len = 0;
      while (written_sf_len < pbw.size()) {
        auto const block_size = std::min(_compression_blocksize, pbw.size() - written_sf_len);
        auto const header_val = block_size * 2 + 1;  // 1 means uncompressed
        CUDF_EXPECTS(header_val >> 24 == 0, "Block length exceeds maximum size");
        std::array const header{static_cast<uint8_t>(header_val >> 0),
                                static_cast<uint8_t>(header_val >> 8),
                                static_cast<uint8_t>(header_val >> 16)};

        _out_sink->host_write(header.data(), header.size());
        _out_sink->host_write(pbw.data() + written_sf_len, block_size);
        written_sf_len += block_size;
        bytes_written += header.size() + block_size;
      }
      stripe.footerLength = bytes_written;
    }
  }
  for (auto& task : write_tasks) {
    task.get();
  }
}

void writer::impl::add_table_to_footer_data(orc_table_view const& orc_table,
                                            std::vector<StripeInformation>& stripes)
{
  if (_footer.headerLength == 0) {
    // First call
    _footer.headerLength   = std::strlen(MAGIC);
    _footer.writer         = cudf_writer_code;
    _footer.rowIndexStride = _row_index_stride;
    _footer.types.resize(1 + orc_table.num_columns());
    _footer.types[0].kind = STRUCT;
    for (auto const& column : orc_table.columns) {
      if (!column.is_child()) {
        _footer.types[0].subtypes.emplace_back(column.id());
        _footer.types[0].fieldNames.emplace_back(column.orc_name());
      }
    }
    for (auto const& column : orc_table.columns) {
      auto& schema_type = _footer.types[column.id()];
      schema_type.kind  = column.orc_kind();
      if (column.orc_kind() == DECIMAL) {
        schema_type.scale     = static_cast<uint32_t>(column.scale());
        schema_type.precision = column.precision();
      }
      std::transform(column.child_begin(),
                     column.child_end(),
                     std::back_inserter(schema_type.subtypes),
                     [&](auto const& child_idx) { return orc_table.column(child_idx).id(); });
      if (column.orc_kind() == STRUCT) {
        std::transform(column.child_begin(),
                       column.child_end(),
                       std::back_inserter(schema_type.fieldNames),
                       [&](auto const& child_idx) {
                         return std::string{orc_table.column(child_idx).orc_name()};
                       });
      }
    }
  } else {
    // verify the user isn't passing mismatched tables
    CUDF_EXPECTS(_footer.types.size() == 1 + orc_table.num_columns(),
                 "Mismatch in table structure between multiple calls to write");
    CUDF_EXPECTS(
      std::all_of(orc_table.columns.cbegin(),
                  orc_table.columns.cend(),
                  [&](auto const& col) { return _footer.types[col.id()].kind == col.orc_kind(); }),
      "Mismatch in column types between multiple calls to write");
  }
  _footer.stripes.insert(_footer.stripes.end(),
                         std::make_move_iterator(stripes.begin()),
                         std::make_move_iterator(stripes.end()));
  _footer.numberOfRows += orc_table.num_rows();
}

void writer::impl::close()
{
  if (_state != writer_state::DATA_WRITTEN) {
    // writer is either closed or no data has been written
    _state = writer_state::CLOSED;
    return;
  }
  PostScript ps;

  if (_stats_freq != statistics_freq::STATISTICS_NONE) {
    // Write column statistics
    auto statistics = finish_statistic_blobs(_footer, _persisted_stripe_statistics, _stream);

    // File-level statistics
    {
      _footer.statistics.reserve(_footer.types.size());
      protobuf_writer pbw;

      // Root column: number of rows
      pbw.put_uint(encode_field_number<size_type>(1));
      pbw.put_uint(_persisted_stripe_statistics.num_rows);
      // Root column: has nulls
      pbw.put_uint(encode_field_number<size_type>(10));
      pbw.put_uint(0);
      _footer.statistics.emplace_back(pbw.release());

      // Add file stats, stored after stripe stats in `column_stats`
      _footer.statistics.insert(_footer.statistics.end(),
                                std::make_move_iterator(statistics.file_level.begin()),
                                std::make_move_iterator(statistics.file_level.end()));
    }

    // Stripe-level statistics
    if (_stats_freq == statistics_freq::STATISTICS_ROWGROUP or
        _stats_freq == statistics_freq::STATISTICS_PAGE) {
      _orc_meta.stripeStats.resize(_footer.stripes.size());
      for (size_t stripe_id = 0; stripe_id < _footer.stripes.size(); stripe_id++) {
        _orc_meta.stripeStats[stripe_id].colStats.resize(_footer.types.size());
        protobuf_writer pbw;

        // Root column: number of rows
        pbw.put_uint(encode_field_number<size_type>(1));
        pbw.put_uint(_footer.stripes[stripe_id].numberOfRows);
        // Root column: has nulls
        pbw.put_uint(encode_field_number<size_type>(10));
        pbw.put_uint(0);
        _orc_meta.stripeStats[stripe_id].colStats[0] = pbw.release();

        for (size_t col_idx = 0; col_idx < _footer.types.size() - 1; col_idx++) {
          size_t idx = _footer.stripes.size() * col_idx + stripe_id;
          _orc_meta.stripeStats[stripe_id].colStats[1 + col_idx] =
            std::move(statistics.stripe_level[idx]);
        }
      }
    }
  }

  _persisted_stripe_statistics.clear();

  _footer.contentLength = _out_sink->bytes_written();
  std::transform(
    _kv_meta.begin(), _kv_meta.end(), std::back_inserter(_footer.metadata), [&](auto const& udata) {
      return UserMetadataItem{udata.first, udata.second};
    });

  // Write statistics metadata
  if (not _orc_meta.stripeStats.empty()) {
    protobuf_writer pbw((_compression != compression_type::NONE) ? 3 : 0);
    pbw.write(_orc_meta);
    add_uncompressed_block_headers(_compression, _compression_blocksize, pbw.buffer());
    ps.metadataLength = pbw.size();
    _out_sink->host_write(pbw.data(), pbw.size());
  } else {
    ps.metadataLength = 0;
  }
  protobuf_writer pbw((_compression != compression_type::NONE) ? 3 : 0);
  pbw.write(_footer);
  add_uncompressed_block_headers(_compression, _compression_blocksize, pbw.buffer());

  // Write postscript metadata
  ps.footerLength         = pbw.size();
  ps.compression          = to_orc_compression(_compression);
  ps.compressionBlockSize = _compression_blocksize;
  ps.version              = {0, 12};  // Hive 0.12
  ps.writerVersion        = cudf_writer_version;
  ps.magic                = MAGIC;

  auto const ps_length = static_cast<uint8_t>(pbw.write(ps));
  pbw.put_byte(ps_length);
  _out_sink->host_write(pbw.data(), pbw.size());
  _out_sink->flush();

  _state = writer_state::CLOSED;
}

// Forward to implementation
writer::writer(std::unique_ptr<data_sink> sink,
               orc_writer_options const& options,
               single_write_mode mode,
               cuda::stream_ref stream)
  : _impl(std::make_unique<impl>(std::move(sink), options, mode, stream))
{
}

// Forward to implementation
writer::writer(std::unique_ptr<data_sink> sink,
               chunked_orc_writer_options const& options,
               single_write_mode mode,
               cuda::stream_ref stream)
  : _impl(std::make_unique<impl>(std::move(sink), options, mode, stream))
{
}

// Destructor within this translation unit
writer::~writer() = default;

// Forward to implementation
void writer::write(table_view const& table) { _impl->write(table); }

// Forward to implementation
void writer::close() { _impl->close(); }

}  // namespace cudf::io::orc::detail
