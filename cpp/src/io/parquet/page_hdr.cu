/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cuda/std/__utility/cmp.h"
#include "error.hpp"
#include "io/utilities/block_utils.cuh"
#include "parquet_gpu.hpp"

#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/detail/utilities/integer_utils.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <cooperative_groups.h>
#include <cuda/iterator>
#include <cuda/std/cstring>
#include <cuda/std/iterator>
#include <cuda/std/tuple>
#include <thrust/binary_search.h>

namespace cudf::io::parquet::detail {

namespace {

auto constexpr decode_page_headers_block_size     = 4 * cudf::detail::warp_size;
auto constexpr count_page_headers_block_size      = 4 * cudf::detail::warp_size;
auto constexpr build_string_dict_index_block_size = 4 * cudf::detail::warp_size;

namespace cg = cooperative_groups;

/**
 * @brief Minimal thrift implementation for parsing page headers
 *
 * See: https://github.com/apache/thrift/blob/master/doc/specs/thrift-compact-protocol.md
 */
struct byte_stream_s {
  uint8_t const* cur{};
  uint8_t const* end{};
  uint8_t const* base{};
  // Parsed symbols
  PageType page_type{};
  PageInfo page{};
  ColumnChunkDesc ck{};
};

/**
 * @brief Get current byte from the byte stream
 *
 * @param bs Byte stream
 *
 * @return Current byte pointed to by the byte stream
 */
inline __device__ unsigned int getb(byte_stream_s* bs)
{
  return (bs->cur < bs->end) ? *bs->cur++ : 0;
}

inline __device__ void skip_bytes(byte_stream_s* bs, size_t bytecnt)
{
  bytecnt = min(bytecnt, (size_t)(bs->end - bs->cur));
  bs->cur += bytecnt;
}

/**
 * @brief Decode unsigned integer from a byte stream using VarInt encoding
 *
 * Concatenate least significant 7 bits of each byte to form a 32 bit
 * integer. Most significant bit of each byte indicates if more bytes
 * are to be used to form the number.
 *
 * @param bs Byte stream
 *
 * @return Decoded 32 bit integer
 */
__device__ uint32_t get_u32(byte_stream_s* bs)
{
  uint32_t v = 0, l = 0, c;
  do {
    c = getb(bs);
    v |= (c & 0x7f) << l;
    l += 7;
  } while (c & 0x80);
  return v;
}

/**
 * @brief Decode signed integer from a byte stream using zigzag encoding
 *
 * The number n encountered in a byte stream translates to
 * -1^(n%2) * ceil(n/2), with the exception of 0 which remains the same.
 * i.e. 0, 1, 2, 3, 4, 5 etc convert to 0, -1, 1, -2, 2 respectively.
 *
 * @param bs Byte stream
 *
 * @return Decoded 32 bit integer
 */
inline __device__ int32_t get_i32(byte_stream_s* bs)
{
  uint32_t u = get_u32(bs);
  return (int32_t)((u >> 1u) ^ -(int32_t)(u & 1));
}

/**
 * @brief Skip a struct field in the byte stream
 *
 * @param bs Byte stream
 * @param field_type Field type
 */
__device__ void skip_struct_field(byte_stream_s* bs, int field_type)
{
  uint32_t struct_depth = 0;
  uint32_t rep_cnt      = 0;

  do {
    if (rep_cnt != 0) {
      rep_cnt--;
    } else if (struct_depth != 0) {
      unsigned int c;
      do {
        c = getb(bs);
        if (!c) --struct_depth;
      } while (!c && struct_depth);
      if (!struct_depth) break;
      field_type = c & 0xf;
      if (!(c & 0xf0)) get_i32(bs);
    }
    switch (static_cast<FieldType>(field_type)) {
      case FieldType::BOOLEAN_TRUE:
      case FieldType::BOOLEAN_FALSE: break;
      case FieldType::I16:
      case FieldType::I32:
      case FieldType::I64: get_u32(bs); break;
      case FieldType::I8: skip_bytes(bs, 1); break;
      case FieldType::DOUBLE: skip_bytes(bs, 8); break;
      case FieldType::BINARY: skip_bytes(bs, get_u32(bs)); break;
      case FieldType::LIST:
      case FieldType::SET: {  // NOTE: skipping a list of lists is not handled
        auto const c = getb(bs);
        uint32_t n   = c >> 4;
        if (n == 0xf) { n = get_u32(bs); }
        field_type = c & 0xf;
        if (static_cast<FieldType>(field_type) == FieldType::STRUCT) {
          struct_depth += n;
        } else {
          rep_cnt = n;
        }
      } break;
      case FieldType::STRUCT: struct_depth++; break;
    }
  } while (rep_cnt || struct_depth);
}

/**
 * @brief Check if the column chunk has nesting
 *
 * @param chunk Column chunk
 *
 * @return True if the column chunk has nesting
 */
__device__ inline bool is_nested(ColumnChunkDesc const& chunk)
{
  return chunk.max_nesting_depth > 1;
}

/**
 * @brief Check if the column chunk is a list type
 *
 * @param chunk Column chunk
 *
 * @return True if the column chunk is a list type
 */
__device__ inline bool is_list(ColumnChunkDesc const& chunk)
{
  return chunk.max_level[level_type::REPETITION] > 0;
}

/**
 * @brief Check if the column chunk is a byte array type
 *
 * @param chunk Column chunk
 *
 * @return True if the column chunk is a byte array type
 */
__device__ inline bool is_byte_array(ColumnChunkDesc const& chunk)
{
  return chunk.physical_type == Type::BYTE_ARRAY;
}

/**
 * @brief Check if the column chunk is a boolean type
 *
 * @param chunk Column chunk
 *
 * @return True if the column chunk is a boolean type
 */
__device__ inline bool is_boolean(ColumnChunkDesc const& chunk)
{
  return chunk.physical_type == Type::BOOLEAN;
}

/**
 * @brief Determine which decode kernel to run for the given page.
 *
 * @param page The page to decode
 * @param chunk Column chunk the page belongs to
 * @return `kernel_mask_bits` value for the given page
 */
__device__ decode_kernel_mask kernel_mask_for_page(PageInfo const& page,
                                                   ColumnChunkDesc const& chunk)
{
  if (page.flags & PAGEINFO_FLAGS_DICTIONARY) { return decode_kernel_mask::NONE; }

  if (page.encoding == Encoding::DELTA_BINARY_PACKED) {
    return decode_kernel_mask::DELTA_BINARY;
  } else if (page.encoding == Encoding::DELTA_BYTE_ARRAY) {
    return decode_kernel_mask::DELTA_BYTE_ARRAY;
  } else if (page.encoding == Encoding::DELTA_LENGTH_BYTE_ARRAY) {
    return decode_kernel_mask::DELTA_LENGTH_BA;
  } else if (is_boolean(chunk)) {
    return is_list(chunk)     ? decode_kernel_mask::BOOLEAN_LIST
           : is_nested(chunk) ? decode_kernel_mask::BOOLEAN_NESTED
                              : decode_kernel_mask::BOOLEAN;
  }

  if (is_string_col(chunk)) {
    // check for string before byte_stream_split so FLBA will go to the right kernel
    if (page.encoding == Encoding::PLAIN) {
      return is_list(chunk)     ? decode_kernel_mask::STRING_LIST
             : is_nested(chunk) ? decode_kernel_mask::STRING_NESTED
                                : decode_kernel_mask::STRING;
    } else if (is_dictionary_encoding(page.encoding)) {
      return is_list(chunk)     ? decode_kernel_mask::STRING_DICT_LIST
             : is_nested(chunk) ? decode_kernel_mask::STRING_DICT_NESTED
                                : decode_kernel_mask::STRING_DICT;
    } else if (page.encoding == Encoding::BYTE_STREAM_SPLIT) {
      return is_list(chunk)     ? decode_kernel_mask::STRING_STREAM_SPLIT_LIST
             : is_nested(chunk) ? decode_kernel_mask::STRING_STREAM_SPLIT_NESTED
                                : decode_kernel_mask::STRING_STREAM_SPLIT;
    }
  }

  if (!is_byte_array(chunk)) {
    if (page.encoding == Encoding::PLAIN) {
      return is_list(chunk)     ? decode_kernel_mask::FIXED_WIDTH_NO_DICT_LIST
             : is_nested(chunk) ? decode_kernel_mask::FIXED_WIDTH_NO_DICT_NESTED
                                : decode_kernel_mask::FIXED_WIDTH_NO_DICT;
    } else if (is_dictionary_encoding(page.encoding)) {
      return is_list(chunk)     ? decode_kernel_mask::FIXED_WIDTH_DICT_LIST
             : is_nested(chunk) ? decode_kernel_mask::FIXED_WIDTH_DICT_NESTED
                                : decode_kernel_mask::FIXED_WIDTH_DICT;
    } else if (page.encoding == Encoding::BYTE_STREAM_SPLIT) {
      return is_list(chunk)     ? decode_kernel_mask::BYTE_STREAM_SPLIT_FIXED_WIDTH_LIST
             : is_nested(chunk) ? decode_kernel_mask::BYTE_STREAM_SPLIT_FIXED_WIDTH_NESTED
                                : decode_kernel_mask::BYTE_STREAM_SPLIT_FIXED_WIDTH_FLAT;
    }
  }

  if (page.encoding == Encoding::BYTE_STREAM_SPLIT) {
    return decode_kernel_mask::BYTE_STREAM_SPLIT;
  }

  // non-string, non-delta, non-split_stream
  return decode_kernel_mask::GENERAL;
}

/**
 * @brief Functor to set value to bool read from byte stream
 *
 * @return True if field type is not bool
 */
struct ParquetFieldBool {
  int field;
  bool& val;

  __device__ ParquetFieldBool(int f, bool& v) : field(f), val(v) {}

  inline __device__ bool operator()(byte_stream_s* bs, int field_type)
  {
    val = static_cast<FieldType>(field_type) == FieldType::BOOLEAN_TRUE;
    return not(static_cast<FieldType>(field_type) == FieldType::BOOLEAN_TRUE or
               static_cast<FieldType>(field_type) == FieldType::BOOLEAN_FALSE);
  }
};

/**
 * @brief Functor to set value to 32 bit integer read from byte stream
 *
 * @return True if field type is not int32
 */
struct ParquetFieldInt32 {
  int field;
  int32_t& val;

  __device__ ParquetFieldInt32(int f, int32_t& v) : field(f), val(v) {}

  inline __device__ bool operator()(byte_stream_s* bs, int field_type)
  {
    val = get_i32(bs);
    return (static_cast<FieldType>(field_type) != FieldType::I32);
  }
};

/**
 * @brief Functor to set value to enum read from byte stream
 *
 * @return True if field type is not int32
 */
template <typename Enum>
struct ParquetFieldEnum {
  int field;
  Enum& val;

  __device__ ParquetFieldEnum(int f, Enum& v) : field(f), val(v) {}

  inline __device__ bool operator()(byte_stream_s* bs, int field_type)
  {
    val = static_cast<Enum>(get_i32(bs));
    return (static_cast<FieldType>(field_type) != FieldType::I32);
  }
};

/**
 * @brief Functor to run operator on byte stream
 *
 * @return True if field type is not struct type or if the calling operator
 * fails
 */
template <typename Operator>
struct ParquetFieldStruct {
  int field;
  Operator op;

  __device__ ParquetFieldStruct(int f) : field(f) {}

  inline __device__ bool operator()(byte_stream_s* bs, int field_type)
  {
    return ((static_cast<FieldType>(field_type) != FieldType::STRUCT) || !op(bs));
  }
};

/**
 * @brief Functor to run an operator
 *
 * The purpose of this functor is to replace a switch case. If the field in
 * the argument is equal to the field specified in any element of the tuple
 * of operators then it is run with the byte stream and field type arguments.
 *
 * If the field does not match any of the functors then skip_struct_field is
 * called over the byte stream.
 *
 * @return Return value of the selected operator or false if no operator
 * matched the field value
 */
template <int index>
struct FunctionSwitchImpl {
  template <typename... Operator>
  static inline __device__ bool run(byte_stream_s* bs,
                                    int field_type,
                                    int const& field,
                                    cuda::std::tuple<Operator...>& ops)
  {
    if (field == cuda::std::get<index>(ops).field) {
      return cuda::std::get<index>(ops)(bs, field_type);
    } else {
      return FunctionSwitchImpl<index - 1>::run(bs, field_type, field, ops);
    }
  }
};

template <>
struct FunctionSwitchImpl<0> {
  template <typename... Operator>
  static inline __device__ bool run(byte_stream_s* bs,
                                    int field_type,
                                    int const& field,
                                    cuda::std::tuple<Operator...>& ops)
  {
    if (field == cuda::std::get<0>(ops).field) {
      return cuda::std::get<0>(ops)(bs, field_type);
    } else {
      skip_struct_field(bs, field_type);
      return false;
    }
  }
};

/**
 * @brief Function to parse page header based on the tuple of functors provided
 *
 * Bytes are read from the byte stream and the field delta and field type are
 * matched up against user supplied reading functors. If they match then the
 * corresponding values are written to references pointed to by the functors.
 *
 * @return Returns false if an unexpected field is encountered while reading
 * byte stream. Otherwise true is returned.
 */
template <typename... Operator>
inline __device__ bool parse_header(cuda::std::tuple<Operator...>& op, byte_stream_s* bs)
{
  constexpr int index = cuda::std::tuple_size<cuda::std::tuple<Operator...>>::value - 1;
  int field           = 0;
  while (true) {
    auto const current_byte = getb(bs);
    if (!current_byte) break;
    int const field_delta = current_byte >> 4;
    int const field_type  = current_byte & 0xf;
    field                 = field_delta ? field + field_delta : get_i32(bs);
    bool exit_function    = FunctionSwitchImpl<index>::run(bs, field_type, field, op);
    if (exit_function) { return false; }
  }
  return true;
}

/**
 * @brief Functor to parse v1 data page header
 *
 * @param bs Byte stream
 *
 * @return True if the data page header is parsed successfully
 */
struct parse_data_page_header_fn {
  __device__ bool operator()(byte_stream_s* bs)
  {
    auto op =
      cuda::std::make_tuple(ParquetFieldInt32(1, bs->page.num_input_values),
                            ParquetFieldEnum<Encoding>(2, bs->page.encoding),
                            ParquetFieldEnum<Encoding>(3, bs->page.definition_level_encoding),
                            ParquetFieldEnum<Encoding>(4, bs->page.repetition_level_encoding));
    return parse_header(op, bs);
  }
};

/**
 * @brief Functor to parse dictionary page header
 *
 * @param bs Byte stream
 *
 * @return True if the dictionary page header is parsed successfully
 */
struct parse_dictionary_page_header_fn {
  __device__ bool operator()(byte_stream_s* bs)
  {
    auto op = cuda::std::make_tuple(ParquetFieldInt32(1, bs->page.num_input_values),
                                    ParquetFieldEnum<Encoding>(2, bs->page.encoding));
    return parse_header(op, bs);
  }
};

/**
 * @brief Functor to parse V2 data page header
 *
 * @param bs Byte stream
 *
 * @return True if the data page header V2 is parsed successfully
 */
struct parse_data_page_header_v2_fn {
  __device__ bool operator()(byte_stream_s* bs)
  {
    auto op =
      cuda::std::make_tuple(ParquetFieldInt32(1, bs->page.num_input_values),
                            ParquetFieldInt32(2, bs->page.num_nulls),
                            ParquetFieldInt32(3, bs->page.num_rows),
                            ParquetFieldEnum<Encoding>(4, bs->page.encoding),
                            ParquetFieldInt32(5, bs->page.lvl_bytes[level_type::DEFINITION]),
                            ParquetFieldInt32(6, bs->page.lvl_bytes[level_type::REPETITION]),
                            ParquetFieldBool(7, bs->page.is_compressed));
    return parse_header(op, bs);
  }
};

/**
 * @brief Functor to parse page header from byte stream
 *
 * @param bs Byte stream
 *
 * @return True if the page header is parsed successfully
 */
struct parse_page_header_fn {
  __device__ bool operator()(byte_stream_s* bs)
  {
    auto op = cuda::std::make_tuple(ParquetFieldEnum<PageType>(1, bs->page_type),
                                    ParquetFieldInt32(2, bs->page.uncompressed_page_size),
                                    ParquetFieldInt32(3, bs->page.compressed_page_size),
                                    ParquetFieldStruct<parse_data_page_header_fn>(5),
                                    ParquetFieldStruct<parse_dictionary_page_header_fn>(7),
                                    ParquetFieldStruct<parse_data_page_header_v2_fn>(8));
    return parse_header(op, bs);
  }
};

/**
 * @brief Parse and validate a page header from a byte stream
 *
 * @param bs Byte stream
 * @return True if the page header is parsed and has a valid compressed size
 */
inline __device__ bool parse_valid_page_header(byte_stream_s* bs)
{
  return parse_page_header_fn{}(bs) and
         (bs->page.compressed_page_size > 0 or
          (bs->page.compressed_page_size == 0 and bs->page_type == PageType::DICTIONARY_PAGE));
}

/**
 * @brief Zero out page header info
 *
 * @param bs Byte stream
 */
void __forceinline__ __device__ zero_out_page_header_info(byte_stream_s* bs)
{
  cuda::std::memset(&bs->page, 0, sizeof(bs->page));
  // Apply non-zero defaults
  bs->page.skipped_values = -1;
  bs->page.is_compressed  = true;
  bs->page.kernel_mask    = decode_kernel_mask::NONE;
}

/**
 * @brief Kernel for outputting page headers from the specified column chunks
 *
 * @param[in] chunks Device span of column chunks
 * @param[out] chunk_pages List of chunk-sorted page info (headers)
 * @param[out] error_code Pointer to the error code for kernel failures
 */
CUDF_KERNEL
void __launch_bounds__(decode_page_headers_block_size)
  decode_page_headers_kernel(device_span<ColumnChunkDesc const> chunks,
                             device_span<chunk_page_info> chunk_pages,
                             kernel_error::pointer error_code)
{
  auto constexpr num_warps_per_block = decode_page_headers_block_size / cudf::detail::warp_size;

  auto const block = cg::this_thread_block();
  auto const warp  = cg::tiled_partition<cudf::detail::warp_size>(block);

  auto const warp_id = warp.meta_group_rank();
  auto const lane_id = warp.thread_rank();
  auto const chunk_idx =
    static_cast<cudf::size_type>((cg::this_grid().block_rank() * num_warps_per_block) + warp_id);
  auto const num_chunks = static_cast<cudf::size_type>(chunks.size());

  if (chunk_idx >= num_chunks) { return; }

  __shared__ byte_stream_s bs_g[num_warps_per_block];
  __shared__ kernel_error::value_type error[num_warps_per_block];

  auto const bs = &bs_g[warp_id];

  cg::invoke_one(warp, [&]() {
    bs->ck         = chunks[chunk_idx];
    error[warp_id] = 0;
  });
  warp.sync();

  cg::invoke_one(warp, [&]() {
    bs->base = bs->cur = bs->ck.compressed_data;
    bs->end            = bs->base + bs->ck.compressed_size;
    // Clear page header info before writing known fields
    zero_out_page_header_info(bs);
    bs->page.chunk_idx      = chunk_idx;
    bs->page.src_col_schema = bs->ck.src_col_schema;
  });
  size_t const num_values        = bs->ck.num_values;
  size_t values_found            = 0;
  uint32_t data_page_count       = 0;
  uint32_t dictionary_page_count = 0;
  auto* page_info                = chunk_pages[chunk_idx].pages;
  auto const max_num_pages       = bs->ck.num_data_pages + bs->ck.num_dict_pages;
  auto const num_dict_pages      = bs->ck.num_dict_pages;
  warp.sync();

  while (values_found < num_values and bs->cur < bs->end) {
    int index_out = -1;
    // Let all threads read before `parse_page_header_fn{}`
    warp.sync();

    // Must be lane 0 here as `shuffle(values_found)` assumes lane 0 is the root
    if (lane_id == 0) {
      // this computation is only valid for flat schemas. for nested schemas,
      // they will be recomputed in the preprocess step by examining repetition and
      // definition levels
      bs->page.chunk_row += bs->page.num_rows;
      bs->page.num_rows      = 0;
      bs->page.flags         = 0;
      bs->page.str_bytes     = 0;
      bs->page.str_bytes_all = 0;
      // zero out V2 info
      bs->page.is_compressed                     = true;
      bs->page.num_nulls                         = 0;
      bs->page.lvl_bytes[level_type::DEFINITION] = 0;
      bs->page.lvl_bytes[level_type::REPETITION] = 0;
      if (parse_valid_page_header(bs)) {
        if (not is_supported_encoding(bs->page.encoding)) {
          error[warp_id] |=
            static_cast<kernel_error::value_type>(decode_error::UNSUPPORTED_ENCODING);
        }
        switch (bs->page_type) {
          case PageType::DATA_PAGE:
            index_out = num_dict_pages + data_page_count;
            data_page_count++;
            // this computation is only valid for flat schemas. for nested schemas,
            // they will be recomputed in the preprocess step by examining repetition and
            // definition levels
            bs->page.num_rows = bs->page.num_input_values;
            values_found += bs->page.num_input_values;
            break;
          case PageType::DATA_PAGE_V2:
            index_out = num_dict_pages + data_page_count;
            data_page_count++;
            bs->page.flags |= PAGEINFO_FLAGS_V2;
            values_found += bs->page.num_input_values;
            // V2 only uses RLE, so it was removed from the header
            bs->page.definition_level_encoding = Encoding::RLE;
            bs->page.repetition_level_encoding = Encoding::RLE;
            break;
          case PageType::DICTIONARY_PAGE:
            index_out = dictionary_page_count;
            dictionary_page_count++;
            bs->page.flags |= PAGEINFO_FLAGS_DICTIONARY;
            break;
          default:
            error[warp_id] |=
              static_cast<kernel_error::value_type>(decode_error::INVALID_PAGE_TYPE);
            bs->cur = bs->end;
            break;
        }
        bs->page.page_data = const_cast<uint8_t*>(bs->cur);
        bs->cur += bs->page.compressed_page_size;
        if (bs->cur > bs->end) {
          error[warp_id] |=
            static_cast<kernel_error::value_type>(decode_error::DATA_STREAM_OVERRUN);
        }
        bs->page.kernel_mask = kernel_mask_for_page(bs->page, bs->ck);
      } else {
        error[warp_id] |= static_cast<kernel_error::value_type>(decode_error::INVALID_PAGE_HEADER);
        bs->cur = bs->end;
      }
      if (index_out >= 0 and index_out < max_num_pages) { page_info[index_out] = bs->page; }
    }
    values_found = shuffle(values_found);
    warp.sync();
  }
  cg::invoke_one(warp, [&]() {
    if (error[warp_id] != 0) { set_error(error[warp_id], error_code); }
  });
}

/**
 * @brief Kernel for counting the number of page headers from the specified column chunks
 *
 * @param[in] chunks Device span of column chunks
 * @param[out] error_code Pointer to the error code for kernel failures
 */
CUDF_KERNEL void __launch_bounds__(count_page_headers_block_size)
  count_page_headers_kernel(cudf::device_span<ColumnChunkDesc> chunks,
                            kernel_error::pointer error_code)
{
  auto constexpr num_warps_per_block = decode_page_headers_block_size / cudf::detail::warp_size;

  auto const block = cg::this_thread_block();
  auto const warp  = cg::tiled_partition<cudf::detail::warp_size>(block);

  auto const warp_id = warp.meta_group_rank();
  auto const lane_id = warp.thread_rank();
  auto const chunk_idx =
    static_cast<cudf::size_type>((cg::this_grid().block_rank() * num_warps_per_block) + warp_id);
  auto const num_chunks = static_cast<cudf::size_type>(chunks.size());

  if (chunk_idx >= num_chunks) { return; }

  __shared__ byte_stream_s bs_g[num_warps_per_block];
  __shared__ kernel_error::value_type error[num_warps_per_block];

  auto const bs = &bs_g[warp_id];

  cg::invoke_one(warp, [&] {
    bs->ck         = chunks[chunk_idx];
    error[warp_id] = 0;
    bs->base = bs->cur = bs->ck.compressed_data;
    bs->end            = bs->base + bs->ck.compressed_size;
  });
  warp.sync();
  size_t const num_values        = bs->ck.num_values;
  size_t values_found            = 0;
  uint32_t data_page_count       = 0;
  uint32_t dictionary_page_count = 0;
  warp.sync();
  while (values_found < num_values and bs->cur < bs->end) {
    // Let all threads read before `parse_page_header_fn{}`
    warp.sync();
    // Must be lane 0 here as `shuffle(values_found)` assumes lane 0 is the root
    if (lane_id == 0) {
      if (parse_valid_page_header(bs)) {
        if (not is_supported_encoding(bs->page.encoding)) {
          error[warp_id] |=
            static_cast<kernel_error::value_type>(decode_error::UNSUPPORTED_ENCODING);
        }
        switch (bs->page_type) {
          case PageType::DATA_PAGE:
            data_page_count++;
            values_found += bs->page.num_input_values;
            break;
          case PageType::DATA_PAGE_V2:
            data_page_count++;
            values_found += bs->page.num_input_values;
            break;
          case PageType::DICTIONARY_PAGE: dictionary_page_count++; break;
          default:
            error[warp_id] |=
              static_cast<kernel_error::value_type>(decode_error::INVALID_PAGE_TYPE);
            bs->cur = bs->end;
            break;
        }
        bs->cur += bs->page.compressed_page_size;
        if (bs->cur > bs->end) {
          error[warp_id] |=
            static_cast<kernel_error::value_type>(decode_error::DATA_STREAM_OVERRUN);
        }
      } else {
        error[warp_id] |= static_cast<kernel_error::value_type>(decode_error::INVALID_PAGE_HEADER);
        bs->cur = bs->end;
      }
    }
    values_found = shuffle(values_found);
    warp.sync();
  }
  cg::invoke_one(warp, [&] {
    chunks[chunk_idx].num_data_pages = data_page_count;
    chunks[chunk_idx].num_dict_pages = dictionary_page_count;
    if (error[warp_id] != 0) { set_error(error[warp_id], error_code); }
  });
}

/**
 * @brief Functor to decode specified page headers from corresponding page data spans
 */
struct decode_from_page_data_fn {
  cudf::device_span<ColumnChunkDesc const> colchunks;
  cudf::device_span<PageInfo> pages;
  cudf::device_span<cudf::device_span<uint8_t const> const> page_data;
  cudf::device_span<size_type const> chunk_page_offsets;
  kernel_error::pointer error_code;

  __device__ void operator()(size_type page_idx) const noexcept
  {
    auto const num_chunks = static_cast<cudf::size_type>(colchunks.size());

    // Binary search the column chunk index for this page
    auto const chunk_idx = static_cast<cudf::size_type>(
      cuda::std::distance(
        chunk_page_offsets.begin(),
        thrust::upper_bound(
          thrust::seq, chunk_page_offsets.begin(), chunk_page_offsets.end(), page_idx)) -
      1);

    // Check if the chunk index is valid.
    if (chunk_idx < 0 or chunk_idx >= num_chunks) {
      set_error(static_cast<kernel_error::value_type>(decode_error::DATA_STREAM_OVERRUN),
                error_code);
      return;
    }

    auto const page_span = page_data[page_idx];

    byte_stream_s bs{};
    bs.ck   = colchunks[chunk_idx];
    bs.base = bs.cur = page_span.data();
    bs.end           = page_span.data() + page_span.size();

    // Clear the logical page descriptor.
    zero_out_page_header_info(&bs);

    bs.page.chunk_idx      = chunk_idx;
    bs.page.src_col_schema = bs.ck.src_col_schema;
    // bs.page.chunk_row not computed here and will be filled in later by
    // `fill_in_page_info()`.

    // Return if empty page span (pruned page)
    if (page_span.empty()) {
      pages[page_idx] = bs.page;
      return;
    }

    // Parsed page must be valid and not empty
    if (not parse_valid_page_header(&bs)) {
      set_error(static_cast<kernel_error::value_type>(decode_error::INVALID_PAGE_HEADER),
                error_code);
      return;
    }
    if (not is_supported_encoding(bs.page.encoding)) {
      set_error(static_cast<kernel_error::value_type>(decode_error::UNSUPPORTED_ENCODING),
                error_code);
      return;
    }
    switch (bs.page_type) {
      case PageType::DATA_PAGE:
        // this computation is only valid for flat schemas. for nested schemas,
        // they will be recomputed in the preprocess step by examining repetition and
        // definition levels
        bs.page.num_rows = bs.page.num_input_values;
        break;
      case PageType::DATA_PAGE_V2:
        bs.page.flags |= PAGEINFO_FLAGS_V2;
        // V2 only uses RLE, so it was removed from the header
        bs.page.definition_level_encoding = Encoding::RLE;
        bs.page.repetition_level_encoding = Encoding::RLE;
        break;
      case PageType::DICTIONARY_PAGE: bs.page.flags |= PAGEINFO_FLAGS_DICTIONARY; break;
      default:
        set_error(static_cast<kernel_error::value_type>(decode_error::INVALID_PAGE_TYPE),
                  error_code);
        return;
    }

    // Ensure we read the entire page.
    if (cuda::std::cmp_not_equal(bs.end - bs.cur, bs.page.compressed_page_size)) {
      set_error(static_cast<kernel_error::value_type>(decode_error::DATA_STREAM_OVERRUN),
                error_code);
      return;
    }

    bs.page.page_data   = const_cast<uint8_t*>(bs.cur);
    bs.page.kernel_mask = kernel_mask_for_page(bs.page, bs.ck);

    // Copy over the page info from byte stream
    pages[page_idx] = bs.page;
  }
};

/**
 * @brief Kernel for building dictionary index for the specified column chunks
 *
 * This function builds an index to point to each dictionary entry
 * (string format is 4-byte little-endian string length followed by character
 * data). The index is a 32-bit integer which contains the offset of each string
 * relative to the beginning of the dictionary page data.
 *
 * @param[in] chunks List of column chunks
 * @param[in] num_chunks Number of column chunks
 */
CUDF_KERNEL void __launch_bounds__(build_string_dict_index_block_size)
  build_string_dictionary_index_kernel(ColumnChunkDesc* chunks,
                                       int32_t num_chunks,
                                       kernel_error::pointer error_code)
{
  auto constexpr num_warps_per_block = build_string_dict_index_block_size / cudf::detail::warp_size;
  __shared__ ColumnChunkDesc chunk_g[num_warps_per_block];

  auto const block   = cg::this_thread_block();
  auto const warp    = cg::tiled_partition<cudf::detail::warp_size>(block);
  auto const warp_id = warp.meta_group_rank();
  int const chunk    = (cg::this_grid().block_rank() * num_warps_per_block) + warp_id;

  if (chunk >= num_chunks) { return; }

  ColumnChunkDesc* const ck = &chunk_g[warp_id];
  cg::invoke_one(warp, [&] { *ck = chunks[chunk]; });
  warp.sync();

  cg::invoke_one(warp, [&] {
    if (ck->num_dict_pages > 0 && ck->str_dict_index) {
      // Data type to describe a string
      string_index_pair* dict_index = ck->str_dict_index;
      uint8_t const* dict           = ck->dict_page->page_data;
      int const dict_size           = ck->dict_page->uncompressed_page_size;
      int32_t const num_entries     = ck->dict_page->num_input_values;
      if (num_entries < 0 or dict_size < 0) {
        set_error(static_cast<kernel_error::value_type>(decode_error::INVALID_DICT_WIDTH),
                  error_code);
        return;
      }
      int pos = 0, cur = 0;
      for (int i = 0; i < num_entries; i++) {
        int len = 0;
        if (ck->physical_type == Type::FIXED_LEN_BYTE_ARRAY) {
          if (cur + ck->type_length <= dict_size) {
            len = ck->type_length;
            pos = cur;
            cur += len;
          } else {
            cur = dict_size;
          }
        } else {
          if (cur + 4 <= dict_size) {
            len =
              dict[cur + 0] | (dict[cur + 1] << 8) | (dict[cur + 2] << 16) | (dict[cur + 3] << 24);
            if (len >= 0 && cur + 4 + len <= dict_size) {
              pos = cur + 4;
              cur = pos + len;
            } else {
              cur = dict_size;
            }
          }
        }
        // TODO: Could store 8 entries in shared mem, then do a single warp-wide store
        dict_index[i].first  = reinterpret_cast<char const*>(dict + pos);
        dict_index[i].second = len;
      }
    }
  });
}

}  // namespace

void count_page_headers(cudf::detail::hostdevice_span<ColumnChunkDesc> chunks,
                        kernel_error::pointer error_code,
                        rmm::cuda_stream_view stream)
{
  static_assert(count_page_headers_block_size % cudf::detail::warp_size == 0,
                "Block size for decode page headers kernel must be a multiple of warp size");

  auto constexpr num_warps_per_block = count_page_headers_block_size / cudf::detail::warp_size;
  auto const num_blocks              = cudf::util::div_rounding_up_unsafe<cudf::size_type>(
    chunks.size(), num_warps_per_block);  // 1 warp per chunk

  dim3 dim_block(count_page_headers_block_size, 1);
  dim3 dim_grid(num_blocks, 1);

  count_page_headers_kernel<<<dim_grid, dim_block, 0, stream.value()>>>(chunks, error_code);
  CUDF_CUDA_TRY(cudaGetLastError());
}

void decode_page_headers(cudf::device_span<ColumnChunkDesc const> chunks,
                         cudf::device_span<chunk_page_info> chunk_pages,
                         kernel_error::pointer error_code,
                         rmm::cuda_stream_view stream)
{
  static_assert(decode_page_headers_block_size % cudf::detail::warp_size == 0,
                "Block size for decode page headers kernel must be a multiple of warp size");
  CUDF_EXPECTS(chunk_pages.size() == chunks.size(),
               "Chunk page info must contain one entry per chunk");

  auto const num_chunks              = static_cast<cudf::size_type>(chunks.size());
  auto constexpr num_warps_per_block = decode_page_headers_block_size / cudf::detail::warp_size;
  auto const num_blocks =
    cudf::util::div_rounding_up_unsafe(num_chunks, num_warps_per_block);  // 1 warp per chunk

  dim3 dim_block(decode_page_headers_block_size, 1);
  dim3 dim_grid(num_blocks, 1);

  decode_page_headers_kernel<<<dim_grid, dim_block, 0, stream.value()>>>(
    chunks, chunk_pages, error_code);
  CUDF_CUDA_TRY(cudaGetLastError());
}

void decode_page_headers_from_page_data(
  cudf::device_span<ColumnChunkDesc const> chunks,
  cudf::device_span<PageInfo> pages,
  cudf::device_span<cudf::device_span<uint8_t const> const> page_data,
  cudf::device_span<size_type const> chunk_page_offsets,
  kernel_error::pointer error_code,
  rmm::cuda_stream_view stream)
{
  CUDF_EXPECTS(chunk_page_offsets.size() == chunks.size() + 1,
               "Chunk page offsets must cover all chunks");
  thrust::for_each(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                   cuda::counting_iterator<cudf::size_type>{0},
                   cuda::counting_iterator{static_cast<cudf::size_type>(pages.size())},
                   decode_from_page_data_fn{.colchunks          = chunks,
                                            .pages              = pages,
                                            .page_data          = page_data,
                                            .chunk_page_offsets = chunk_page_offsets,
                                            .error_code         = error_code});
}

void build_string_dictionary_index(ColumnChunkDesc* chunks,
                                   int32_t num_chunks,
                                   kernel_error::pointer error_code,
                                   rmm::cuda_stream_view stream)
{
  static_assert(
    build_string_dict_index_block_size % cudf::detail::warp_size == 0,
    "Block size for build string dictionary index kernel must be a multiple of warp size");
  auto constexpr num_warps_per_block = build_string_dict_index_block_size / cudf::detail::warp_size;
  auto const num_blocks =
    cudf::util::div_rounding_up_unsafe(num_chunks, num_warps_per_block);  // 1 warp per chunk

  dim3 dim_block(build_string_dict_index_block_size, 1);
  dim3 dim_grid(num_blocks, 1);

  build_string_dictionary_index_kernel<<<dim_grid, dim_block, 0, stream.value()>>>(
    chunks, num_chunks, error_code);
  CUDF_CUDA_TRY(cudaGetLastError());
}

}  // namespace cudf::io::parquet::detail
