/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "hybrid_scan_helpers.hpp"
#include "hybrid_scan_impl.hpp"
#include "io/parquet/expression_transform_helpers.hpp"
#include "io/parquet/parquet_gpu.hpp"
#include "io/parquet/timestamp_utils.cuh"
#include "io/utilities/block_utils.cuh"

#include <cudf/ast/detail/operators.hpp>
#include <cudf/ast/expressions.hpp>
#include <cudf/detail/cuco_helpers.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/hashing/detail/default_hash.cuh>
#include <cudf/utilities/span.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_checks.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/exec_policy.hpp>
#include <rmm/mr/polymorphic_allocator.hpp>

#include <cuco/extent.cuh>
#include <cuco/static_set.cuh>
#include <cuda/iterator>

#include <optional>

namespace cudf::io::parquet::experimental::detail {

using parquet::detail::chunk_page_info;
using parquet::detail::ColumnChunkDesc;
using parquet::detail::decode_error;
using parquet::detail::PageInfo;

namespace {

namespace cg = cooperative_groups;

/// Supported fixed width types for row group pruning using dictionaries
template <typename T>
auto constexpr is_supported_fixed_width_type =
  not cuda::std::is_same_v<T, bool> and not cudf::is_compound<T>();

/// All supported types for row group pruning using dictionaries
template <typename T>
auto constexpr is_supported_dictionary_type =
  is_supported_fixed_width_type<T> or cuda::std::is_same_v<T, cudf::string_view>;

/// Concept for supported fixed width types for row group pruning using dictionaries
template <typename T>
concept SupportedFixedWidthType = is_supported_fixed_width_type<T>;

/// Concept for all supported types for row group pruning using dictionaries
template <typename T>
concept SupportedDictionaryType = is_supported_dictionary_type<T>;

/// Size of INT96 physical type
auto constexpr INT96_SIZE = 12;
/// Decode kernel block size. Must be a multiple of warp_size
auto constexpr DECODE_BLOCK_SIZE = 4 * cudf::detail::warp_size;
/// Maximum query block size. Must be a multiple of warp_size
auto constexpr MAX_QUERY_BLOCK_SIZE = 8 * cudf::detail::warp_size;
/// Maximum number of literals to evaluate while decoding column dictionaries
auto constexpr MAX_INLINE_LITERALS = 2;

// cuCollections static set parameters
using key_type  = cudf::size_type;  ///< Using column indices (size_type) as set keys
using slot_type = key_type;         ///< Hash set slot type is the same as the key type
auto constexpr EMPTY_KEY_SENTINEL = key_type{-1};  ///< We will never encounter a -1 row index
auto constexpr SET_CG_SIZE        = 1;  ///< Cooperative group size for cuco::static_set_ref
auto constexpr BUCKET_SIZE        = 1;  ///< Number of concurrent slots handled by each thread
auto constexpr OCCUPANCY_FACTOR = 0.7;  ///< cuCollections suggests targeting a 70% occupancy factor

/// Hash function for cuco::static_set_ref
template <SupportedDictionaryType T>
using hasher_type = cudf::hashing::detail::MurmurHash3_x86_32<T>;

/// cuco::static_set_ref storage type
using storage_type     = cuco::bucket_storage<slot_type,
                                              BUCKET_SIZE,
                                              cuco::extent<std::size_t>,
                                              rmm::mr::polymorphic_allocator<char>>;
using storage_ref_type = typename storage_type::ref_type;

/**
 * @brief Hash functor for inserting values into a `cuco::static_set`
 *
 * @tparam T Supported underlying data type of the cudf column
 */
template <SupportedDictionaryType T>
struct insert_hash_functor {
  cudf::device_span<T const> const decoded_data;
  uint32_t const seed{DEFAULT_HASH_SEED};
  constexpr __device__ __forceinline__ typename hasher_type<T>::result_type operator()(
    key_type key) const noexcept
  {
    return hasher_type<T>{seed}(decoded_data[key]);
  }
};

/**
 * @brief Hash functor for querying values from a `cuco::static_set`
 *
 * @tparam T Supported underlying data type of the cudf column
 */
template <SupportedDictionaryType T>
struct query_hash_functor {
  uint32_t const seed{DEFAULT_HASH_SEED};
  constexpr __device__ __forceinline__ typename hasher_type<T>::result_type operator()(
    T const& value) const noexcept
  {
    return hasher_type<T>{seed}(value);
  }
};

/**
 * @brief Equality functor for inserting values into a `cuco::static_set`
 *
 * @tparam T Supported underlying data type of the cudf column
 */
template <SupportedDictionaryType T>
struct insert_equality_functor {
  cudf::device_span<T const> const decoded_data;
  constexpr __device__ __forceinline__ bool operator()(key_type lhs_key,
                                                       key_type rhs_key) const noexcept
  {
    return decoded_data[lhs_key] == decoded_data[rhs_key];
  }
};

/**
 * @brief Equality functor for querying values from a `cuco::static_set`
 *
 * @tparam T Supported underlying data type of the cudf column
 */
template <SupportedDictionaryType T>
struct query_equality_functor {
  cudf::device_span<T const> const decoded_data;
  constexpr __device__ __forceinline__ bool operator()(T const& value, key_type key) const noexcept
  {
    return value == decoded_data[key];
  }
};

/**
 * @brief Helper function to check if there is a data stream overrun
 *
 * @param offset Offset into the data stream
 * @param length Length of the data to read
 * @param page_data_size Size of the page data

 * @return Boolean indicating if there is a data stream overrun
 */
__device__ __forceinline__ bool is_stream_overrun(size_type offset,
                                                  size_type length,
                                                  size_type page_data_size)
{
  return offset + length > page_data_size;
}

/**
 * @brief Helper function to set error
 *
 * @param error Pointer to the kernel error code
 * @param error_value Error value to set
 */
__device__ __forceinline__ void set_error(kernel_error::pointer error, decode_error error_value)
{
  cuda::atomic_ref<kernel_error::value_type, cuda::thread_scope_block> ref{*error};
  ref.fetch_or(static_cast<kernel_error::value_type>(error_value), cuda::std::memory_order_relaxed);
}

/**
 * @brief Helper function to check if an error has been set
 *
 * @param error Pointer to the kernel error code
 * @return Boolean indicating if an error has been set
 */
__device__ __forceinline__ bool is_error_set(kernel_error::pointer error)
{
  return cuda::atomic_ref<kernel_error::value_type, cuda::thread_scope_block>{*error}.load(
           cuda::std::memory_order_relaxed) != 0;
}

/**
 * @brief Helper function to get the length of INT32 physical type
 *
 * @param logical_type Logical type of the column
 * @return Length of the INT32 physical type
 */
__device__ __forceinline__ int32_t get_int32_type_len(LogicalType const& logical_type)
{
  // Note: This function has been extracted from the snippet at:
  // https://github.com/rapidsai/cudf/blob/c89c83c00c729a86c56570693b627f31408bc2c9/cpp/src/io/parquet/page_decode.cuh#L1278-L1287

  // Check for smaller bitwidths
  if (logical_type.type == LogicalType::INTEGER) {
    return logical_type.bit_width() / 8;
  } else if (logical_type.is_time_millis()) {
    // cudf outputs as INT64
    return 8;
  }

  return sizeof(uint32_t);
}

/**
 * @brief Helper function to decode an INT96 value
 *
 * @param int96_ptr Pointer to the INT96 value to decode
 * @param clock_rate Clock rate of the column
 * @param timestamp64 Pointer to the timestamp64 value to store the decoded value
 */
__device__ __forceinline__ void decode_int96timestamp(uint8_t const* int96_ptr,
                                                      int32_t clock_rate,
                                                      int64_t* timestamp64)
{
  // Note: This function has been modified from the original at
  // https://github.com/rapidsai/cudf/blob/c89c83c00c729a86c56570693b627f31408bc2c9/cpp/src/io/parquet/page_data.cuh#L133-L198

  int64_t nanos = cudf::io::unaligned_load<uint64_t>(int96_ptr);
  int64_t days  = cudf::io::unaligned_load<uint32_t>(int96_ptr + sizeof(int64_t));

  // Convert from Julian day at noon to UTC seconds
  cudf::duration_D duration_days{
    days - 2440588};  // TBD: Should be noon instead of midnight, but this matches pyarrow

  using cuda::std::chrono::duration_cast;

  *timestamp64 = [&]() {
    switch (clock_rate) {
      case 1:  // seconds
        return duration_cast<cudf::duration_s>(duration_days).count() +
               duration_cast<cudf::duration_s>(cudf::duration_ns{nanos}).count();
      case 1'000:  // milliseconds
        return duration_cast<cudf::duration_ms>(duration_days).count() +
               duration_cast<cudf::duration_ms>(cudf::duration_ns{nanos}).count();
      case 1'000'000:  // microseconds
        return duration_cast<cudf::duration_us>(duration_days).count() +
               duration_cast<cudf::duration_us>(cudf::duration_ns{nanos}).count();
      case 1'000'000'000:  // nanoseconds
      default: return duration_cast<cudf::duration_ns>(duration_days).count() + nanos;
    }
  }();
}

/**
 * @brief Query `cuco::static_set`s to evaluate (many) input (in)equality predicates
 *
 * @tparam Supported underlying data type of the cudf column
 * @param decoded_data Span of storage for decoded values from all dictionaries
 * @param results Span of device vector start pointers to store query results, one per predicate
 * @param scalars Pointer to scalar device views, one per predicate
 * @param operators Pointer to corresponding (in)equality operators, one per predicate
 * @param set_storage Pointer to the start of the bulk cuco hash set slots
 * @param set_offsets Pointer to offsets into the bulk set storage for each dictionary
 * @param value_offsets Pointer to offsets into running sum of values in each dictionary
 * @param total_row_groups Total number of row groups
 * @param physical_type Parquet physical type of the column
 */
template <SupportedDictionaryType T>
CUDF_KERNEL void query_dictionaries(cudf::device_span<T> decoded_data,
                                    cudf::device_span<bool*> results,
                                    ast::generic_scalar_device_view const* scalars,
                                    ast::ast_operator const* operators,
                                    slot_type* set_storage,
                                    cudf::size_type const* set_offsets,
                                    cudf::size_type const* value_offsets,
                                    cudf::size_type total_row_groups,
                                    parquet::Type physical_type)
{
  // Each thread block (cg) evaluates one scalar against all column chunk dictionaries (cuco hash
  // sets) of this column
  auto const group = cg::this_thread_block();

  // Scalar to evaluate
  auto const scalar_idx = cg::this_grid().block_rank();
  auto const scalar     = scalars[scalar_idx];

  // Result vector for this scalar
  auto result = results[scalar_idx];

  using equality_fn_type    = query_equality_functor<T>;
  using hash_fn_type        = query_hash_functor<T>;
  using probing_scheme_type = cuco::linear_probing<SET_CG_SIZE, hash_fn_type>;

  // Evaluate the scalar against all cuco hash sets of this column
  for (auto set_idx = group.thread_rank(); set_idx < total_row_groups; set_idx += group.size()) {
    // If the set is empty (no dictionary page data), then skip the dictionary page filter
    if (set_offsets[set_idx + 1] - set_offsets[set_idx] == 0) {
      result[set_idx] = operators[scalar_idx] == ast::ast_operator::EQUAL;
      continue;
    }

    // Set storage reference for the current cuco hash set
    storage_ref_type const storage_ref{set_offsets[set_idx + 1] - set_offsets[set_idx],
                                       set_storage + set_offsets[set_idx]};

    // Create a view of the hash set
    auto hash_set_ref = cuco::static_set_ref{cuco::empty_key{EMPTY_KEY_SENTINEL},
                                             equality_fn_type{decoded_data},
                                             probing_scheme_type{hash_fn_type{}},
                                             cuco::thread_scope_block,
                                             storage_ref};
    auto set_find_ref = hash_set_ref.rebind_operators(cuco::contains);

    // Number of values in this hash set
    auto const num_set_values = value_offsets[set_idx + 1] - value_offsets[set_idx];
    // Literal value to find in this hash set
    auto const literal_value = scalar.value<T>();

    // If the operator is NOT_EQUAL, then we mark the result as true (to be pruned) if and only if
    // the literal is present in the hash set and it's the only value in the hash set
    if (operators[scalar_idx] == ast::ast_operator::NOT_EQUAL) {
      result[set_idx] = num_set_values == 1 and set_find_ref.contains(literal_value);
    }
    // Otherwise, mark the result as true (keep) / false (prune) based on if the literal is present
    // in the hash set
    else {
      result[set_idx] = set_find_ref.contains(literal_value);
    }
  }
}

/**
 * @brief Decode a fixed width value from a page data buffer
 *
 * @tparam Supported underlying (fixed width) data type of the cudf column
 * @param page Dictionary page header information
 * @param chunk Column chunk descriptor
 * @param value_idx Index of the value to decode from page data buffer
 * @param physical_type Parquet physical type of the column
 * @param error Pointer to the kernel error code
 * @return Decoded value
 */
template <SupportedFixedWidthType T>
__device__ T decode_fixed_width_value(PageInfo const& page,
                                      ColumnChunkDesc const& chunk,
                                      int32_t value_idx,
                                      parquet::Type physical_type,
                                      kernel_error::pointer error)
{
  // Page data pointer
  auto const& page_data = page.page_data;

  // Calculate the timestamp scale if this chunk has a timestamp logical type
  auto const timestamp_scale =
    parquet::detail::calc_timestamp_scale(chunk.logical_type, chunk.ts_clock_rate);

  // FLBA length (0 if not FLBA type)
  auto const flba_length = chunk.type_length;

  // Placeholder for the decoded value
  auto decoded_value = T{};

  // Check for decimal types
  auto const is_decimal =
    chunk.logical_type.has_value() and chunk.logical_type.value().type == LogicalType::DECIMAL;
  if (is_decimal and not cudf::is_fixed_point<T>()) {
    set_error(error, decode_error::INVALID_DATA_TYPE);
    return {};
  }

  // Decode the value based on the physical type
  switch (physical_type) {
    case parquet::Type::INT96: {
      // Check if we have a stream overrun
      if (is_stream_overrun(value_idx * INT96_SIZE, INT96_SIZE, page.uncompressed_page_size)) {
        set_error(error, decode_error::DATA_STREAM_OVERRUN);
        return {};
      }
      // Check if the flba length is valid
      if (flba_length != INT96_SIZE or not cuda::std::is_same_v<T, int64_t>) {
        set_error(error, decode_error::INVALID_DATA_TYPE);
        return {};
      }

      // Decode the int96 value from the page data
      decode_int96timestamp(page_data + (value_idx * flba_length),
                            chunk.ts_clock_rate,
                            reinterpret_cast<int64_t*>(&decoded_value));
      break;
    }

    case parquet::Type::FIXED_LEN_BYTE_ARRAY: {
      // Check if we have a stream overrun
      if (is_stream_overrun(value_idx * flba_length, flba_length, page.uncompressed_page_size)) {
        set_error(error, decode_error::DATA_STREAM_OVERRUN);
        return {};
      }
      // Check if the flba length is smaller than or equal to the underlying data type size
      if (flba_length <= 0 or flba_length > sizeof(T)) {
        set_error(error, decode_error::INVALID_DATA_TYPE);
        return {};
      }
      // Decode the flba values as string view
      auto const flba_value = cudf::string_view{
        reinterpret_cast<char const*>(page_data) + value_idx * flba_length, flba_length};
      // Copy the flba value including decimal128 (__int128) from the page data
      cuda::std::memcpy(&decoded_value, flba_value.data(), flba_length);

      // Handle signed integral types
      if constexpr (cudf::is_integral<T>() and cudf::is_signed<T>()) {
        // Shift the unscaled value up and back down to correctly represent negative numbers.
        if (flba_length < sizeof(T)) {
          decoded_value <<= (sizeof(T) - flba_length) * 8;
          decoded_value >>= (sizeof(T) - flba_length) * 8;
        }
      }
      break;
    }
    case parquet::Type::INT32: {
      // Check if we are overruning the data stream
      if (is_stream_overrun(
            value_idx * sizeof(int32_t), sizeof(int32_t), page.uncompressed_page_size)) {
        set_error(error, decode_error::DATA_STREAM_OVERRUN);
        return {};
      }

      // Calculate the bitwidth of the int32 encoded value
      auto const int32_type_len = chunk.logical_type.has_value()
                                    ? get_int32_type_len(chunk.logical_type.value())
                                    : sizeof(uint32_t);
      // Check if we are reading INT32 TIME_MILLIS into 64-bit DURATION_MILLISECONDS
      if (int32_type_len == sizeof(int64_t) and not cudf::is_duration<T>) {
        set_error(error, decode_error::INVALID_DATA_TYPE);
        return {};
      }

      // Handle timestamps
      if constexpr (cudf::is_timestamp<T>()) {
        auto const timestamp =
          cudf::io::unaligned_load<uint32_t>(page_data + (value_idx * sizeof(uint32_t)));
        if (timestamp_scale != 0) {
          decoded_value = T{typename T::duration(static_cast<typename T::rep>(timestamp))};
        } else {
          decoded_value = T{static_cast<typename T::duration>(timestamp)};
        }
      }
      // Handle durations
      else if constexpr (cudf::is_duration<T>()) {
        // Note: This function has been extracted from the snippet at:
        // https://github.com/rapidsai/cudf/blob/594d26768ce86b9c2f389e851ae1afb77032c879/cpp/src/io/parquet/decode_fixed.cu#L159-L163

        // Reading INT32 TIME_MILLIS into 64-bit DURATION_MILLISECONDS
        // TIME_MILLIS is the only duration type stored as int32:
        // https://github.com/apache/parquet-format/blob/master/LogicalTypes.md#deprecated-time-convertedtype

        // Must be either int32 or int64 duration
        if (sizeof(typename T::rep) != sizeof(int32_t) and
            sizeof(typename T::rep) != sizeof(int64_t)) {
          set_error(error, decode_error::INVALID_DATA_TYPE);
          return {};
        }

        auto const duration =
          cudf::io::unaligned_load<uint32_t>(page_data + (value_idx * sizeof(uint32_t)));
        decoded_value = T{static_cast<typename T::rep>(duration)};
      }
      // Handle other int32 encoded values including smaller bitwidths and decimal32
      else {
        // Reading smaller bitwidth values
        if (sizeof(T) > sizeof(int32_t) or sizeof(T) != int32_type_len) {
          set_error(error, decode_error::INVALID_DATA_TYPE);
          return {};
        }
        decoded_value = static_cast<T>(
          cudf::io::unaligned_load<uint32_t>(page_data + (value_idx * sizeof(uint32_t))));
      }
      break;
    }
    case parquet::Type::INT64: {
      // Check if we are overruning the data stream
      if (is_stream_overrun(
            value_idx * sizeof(int64_t), sizeof(int64_t), page.uncompressed_page_size)) {
        set_error(error, decode_error::DATA_STREAM_OVERRUN);
        return {};
      }
      // Handle timestamps
      if constexpr (cudf::is_timestamp<T>()) {
        int64_t const timestamp =
          cudf::io::unaligned_load<uint64_t>(page_data + (value_idx * sizeof(int64_t)));
        if (timestamp_scale != 0) {
          decoded_value = T{typename T::duration(static_cast<typename T::rep>(
            parquet::detail::apply_ts_scale(timestamp, timestamp_scale)))};
        } else {
          decoded_value = T{typename T::duration(static_cast<typename T::rep>(timestamp))};
        }
      }
      // Handle durations and other int64 encoded values including decimal64
      else {
        decoded_value = static_cast<T>(
          cudf::io::unaligned_load<uint64_t>(page_data + (value_idx * sizeof(uint64_t))));
      }
      break;
    }
    case parquet::Type::FLOAT: [[fallthrough]];
    case parquet::Type::DOUBLE: {
      // Check if we are overruning the data stream
      if (is_stream_overrun(value_idx * sizeof(T), sizeof(T), page.uncompressed_page_size)) {
        set_error(error, decode_error::DATA_STREAM_OVERRUN);
        return {};
      }
      cuda::std::memcpy(&decoded_value, page_data + (value_idx * sizeof(T)), sizeof(T));
      break;
    }
    default: {
      // Parquet physical type is not fixed width so set the error code and break early
      set_error(error, decode_error::INVALID_DATA_TYPE);
      return {};
    }
  }

  return decoded_value;
}

/**
 * @brief Decode a string value from a page data buffer
 *
 * @param page_data Pointer to the page data buffer
 * @param page_data_size Page data buffer size
 * @param buffer_offset Current offset into the page data buffer
 * @param error Pointer to the kernel error code
 * @return Decoded string value
 */
__device__ cudf::string_view decode_string_value(uint8_t const* page_data,
                                                 int32_t page_data_size,
                                                 int32_t& buffer_offset,
                                                 kernel_error::pointer error)
{
  if (is_stream_overrun(buffer_offset, sizeof(int32_t), page_data_size)) {
    set_error(error, decode_error::DATA_STREAM_OVERRUN);
    return {};
  }

  // Decode string length
  auto const string_length = static_cast<int32_t>(*(page_data + buffer_offset));
  buffer_offset += sizeof(int32_t);

  // Check if we have a stream overrun
  if (is_stream_overrun(buffer_offset, string_length, page_data_size)) {
    set_error(error, decode_error::DATA_STREAM_OVERRUN);
    return {};
  }

  // Decode cudf::string_view value
  auto const decoded_value =
    cudf::string_view{reinterpret_cast<char const*>(page_data + buffer_offset),
                      static_cast<cudf::size_type>(string_length)};

  // Update the buffer offset
  buffer_offset += string_length;

  return decoded_value;
}

/**
 * @brief Decode string column chunk dictionaries and build `cuco::static_set`s from them,
 * one hash set per dictionary
 *
 * @param pages Column chunk dictionary page headers
 * @param decoded_data Span of storage for decoded values from all dictionaries
 * @param set_storage Pointer to the start of the bulk cuco hash set slots
 * @param set_offsets Pointer to offsets into the bulk set storage for each dictionary
 * @param value_offsets Pointer to offsets into running sum of values in each dictionary
 * @param num_dictionary_columns Total number of columns with dictionaries
 * @param dictionary_col_idx Index of the current dictionary column
 * @param error Pointer to the kernel error code
 */
CUDF_KERNEL void __launch_bounds__(DECODE_BLOCK_SIZE)
  build_string_dictionaries(PageInfo const* pages,
                            cudf::device_span<cudf::string_view> decoded_data,
                            slot_type* set_storage,
                            cudf::size_type const* set_offsets,
                            cudf::size_type const* value_offsets,
                            cudf::size_type total_row_groups,
                            cudf::size_type num_dictionary_columns,
                            cudf::size_type dictionary_col_idx,
                            kernel_error::pointer error)
{
  // Single thread from each warp decodes one dictionary page and inserts into the corresponding
  // cuco hash set
  auto const warp = cg::tiled_partition<cudf::detail::warp_size>(cg::this_thread_block());

  // Index of the current column chunk of the column
  auto const row_group_idx = cudf::detail::grid_1d::global_thread_id() / cudf::detail::warp_size;

  // Return early if the row group index is out of bounds
  if (row_group_idx >= total_row_groups) { return; }

  // Global index of the current column chunk (dictionary page)
  auto const chunk_idx      = dictionary_col_idx + (row_group_idx * num_dictionary_columns);
  auto const& page          = pages[chunk_idx];
  auto const& page_data     = page.page_data;
  auto const page_data_size = page.uncompressed_page_size;

  // If empty buffer or no input values, then return early
  if (page.num_input_values == 0 or page_data_size == 0) { return; }

  using equality_fn_type    = insert_equality_functor<cudf::string_view>;
  using hash_fn_type        = insert_hash_functor<cudf::string_view>;
  using probing_scheme_type = cuco::linear_probing<SET_CG_SIZE, hash_fn_type>;

  // Set storage reference for the current cuco hash set
  storage_ref_type const storage_ref{set_offsets[row_group_idx + 1] - set_offsets[row_group_idx],
                                     set_storage + set_offsets[row_group_idx]};

  // Create a view of the hash set
  auto hash_set_ref   = cuco::static_set_ref{cuco::empty_key<key_type>{EMPTY_KEY_SENTINEL},
                                           equality_fn_type{decoded_data},
                                           probing_scheme_type{hash_fn_type{decoded_data}},
                                           cuco::thread_scope_thread,
                                           storage_ref};
  auto set_insert_ref = hash_set_ref.rebind_operators(cuco::insert);

  // Offset into the running sum of values to be decoded from all column chunks of this column
  auto const value_offset = value_offsets[row_group_idx];

  // Decode values from the current dictionary page with single warp thread
  if (warp.thread_rank() == 0) {
    auto buffer_offset      = int32_t{0};
    auto num_values_decoded = key_type{0};

    // Decode values from the current dictionary page
    while (buffer_offset < page_data_size) {
      // Check if we have a stream overrun
      if (num_values_decoded > page.num_input_values) {
        set_error(error, decode_error::DATA_STREAM_OVERRUN);
        break;
      }

      // Decode cudf::string_view value
      decoded_data[value_offset + num_values_decoded] =
        decode_string_value(page_data, page_data_size, buffer_offset, error);

      // Break if an error has been set within `decode_string_value`
      if (is_error_set(error)) { break; }

      // Insert the key (decoded value's global index) into the cuco hash set
      set_insert_ref.insert(static_cast<key_type>(value_offset + num_values_decoded));

      // Update the number of values decoded
      num_values_decoded++;
    }
  }
}

/**
 * @brief Decode fixed width column chunk dictionaries and build `cuco::static_set`s from them,
 * one hash set per dictionary
 *
 * @tparam Supported underlying (fixed width) data type of the cudf column
 * @param pages Column chunk dictionary page headers
 * @param chunks Column chunk descriptors
 * @param decoded_data Span of storage for decoded values from all dictionaries
 * @param set_storage Pointer to the start of the bulk cuco hash set slots
 * @param set_offsets Pointer to offsets into the bulk set storage for each dictionary
 * @param value_offsets Pointer to offsets into running sum of values in each dictionary
 * @param physical_type Parquet physical type of the column
 * @param num_dictionary_columns Total number of columns with dictionaries
 * @param dictionary_col_idx Index of the current dictionary column
 * @param error Pointer to the kernel error code
 */
template <SupportedFixedWidthType T>
CUDF_KERNEL void __launch_bounds__(DECODE_BLOCK_SIZE)
  build_fixed_width_dictionaries(PageInfo const* pages,
                                 ColumnChunkDesc const* chunks,
                                 cudf::device_span<T> decoded_data,
                                 slot_type* set_storage,
                                 cudf::size_type const* set_offsets,
                                 cudf::size_type const* value_offsets,
                                 parquet::Type physical_type,
                                 cudf::size_type num_dictionary_columns,
                                 cudf::size_type dictionary_col_idx,
                                 kernel_error::pointer error)
{
  // Each thread block (cg) decodes values from one dictionary page and inserts into the
  // corresponding cuco hash set
  auto const group = cg::this_thread_block();

  // Index of the current column chunk of the column
  auto const row_group_idx = cg::this_grid().block_rank();

  // Global index of the current column chunk (dictionary page)
  auto const chunk_idx = dictionary_col_idx + (row_group_idx * num_dictionary_columns);
  auto const& page     = pages[chunk_idx];
  auto const& chunk    = chunks[chunk_idx];

  // If empty buffer or no input values, then return early
  if (page.num_input_values == 0 or page.uncompressed_page_size == 0) { return; }

  // Offset into the running sum of values to be decoded from all column chunks of this column
  auto const value_offset = value_offsets[row_group_idx];

  // Set storage reference for the current cuco hash set
  storage_ref_type const storage_ref{set_offsets[row_group_idx + 1] - set_offsets[row_group_idx],
                                     set_storage + set_offsets[row_group_idx]};

  using equality_fn_type    = insert_equality_functor<T>;
  using hash_fn_type        = insert_hash_functor<T>;
  using probing_scheme_type = cuco::linear_probing<SET_CG_SIZE, hash_fn_type>;

  // Create a view of the hash set
  auto hash_set_ref   = cuco::static_set_ref{cuco::empty_key{EMPTY_KEY_SENTINEL},
                                           equality_fn_type{decoded_data},
                                           probing_scheme_type{hash_fn_type{decoded_data}},
                                           cuco::thread_scope_block,
                                           storage_ref};
  auto set_insert_ref = hash_set_ref.rebind_operators(cuco::insert);

  // Decode values from the current dictionary page
  for (auto value_idx = group.thread_rank(); value_idx < page.num_input_values;
       value_idx += group.num_threads()) {
    // Key (decoded value's global index) to insert into the cuco hash set
    auto const insert_key = static_cast<key_type>(value_offset + value_idx);

    // Decode the value from the page data
    decoded_data[insert_key] =
      decode_fixed_width_value<T>(page, chunk, value_idx, physical_type, error);

    // Return early if an error has been set
    if (is_error_set(error)) { return; }

    // Insert the key (decoded value's global index) into the cuco hash set
    set_insert_ref.insert(insert_key);
  }
}

/**
 * @brief Decode string column chunk dictionaries and evaluate (few) input predicates against
 * decoded values
 *
 * @param pages Column chunk dictionary page headers
 * @param results Span of device vector start pointers to store query results, one per predicate
 * @param scalars Span of scalar device views, one per predicate
 * @param operators Span of corresponding (in)equality operators, one per predicate
 * @param total_num_scalars Total number of predicates
 * @param total_row_groups Total number of row groups
 * @param num_dictionary_columns Total number of columns with dictionaries
 * @param dictionary_col_idx Index of the current dictionary column
 * @param error Pointer to the kernel error code
 */
CUDF_KERNEL void __launch_bounds__(DECODE_BLOCK_SIZE)
  evaluate_few_string_literals(PageInfo const* pages,
                               cudf::device_span<bool*> results,
                               ast::generic_scalar_device_view const* scalars,
                               ast::ast_operator const* operators,
                               cudf::size_type total_num_scalars,
                               cudf::size_type total_row_groups,
                               cudf::size_type num_dictionary_columns,
                               cudf::size_type dictionary_col_idx,
                               kernel_error::pointer error)
{
  // Single thread from each warp decodes values from one dictionary page and evaluates all input
  // predicates against them
  auto const warp = cg::tiled_partition<cudf::detail::warp_size>(cg::this_thread_block());

  // Index of the current column chunk of the column
  auto const row_group_idx = cudf::detail::grid_1d::global_thread_id() / cudf::detail::warp_size;

  // Return early if the row group index is out of bounds
  if (row_group_idx >= total_row_groups) { return; }

  // Global index of the current column chunk (dictionary page)
  auto const chunk_idx      = dictionary_col_idx + (row_group_idx * num_dictionary_columns);
  auto const& page          = pages[chunk_idx];
  auto const& page_data     = page.page_data;
  auto const page_data_size = page.uncompressed_page_size;

  // If the page buffer is empty or has no values to decode, set results and continue
  if (page.num_input_values == 0 or page_data_size == 0) {
    if (warp.thread_rank() == 0 and row_group_idx < total_row_groups) {
      for (auto i = 0; i < total_num_scalars; ++i) {
        // Set the result to true (keep) if the operator is EQUAL, otherwise set it to false (do
        // not prune)
        results[i][row_group_idx] = operators[i] == ast::ast_operator::EQUAL;
      }
    }
    return;
  }

  // Decode values from the current dictionary page with single warp thread
  if (warp.thread_rank() == 0) {
    // Initialize results for all predicates to false
    for (auto i = 0; i < total_num_scalars; ++i) {
      results[i][row_group_idx] = false;
    }

    // Initialize the buffer offset and the number of values decoded
    auto buffer_offset      = int32_t{0};
    auto num_values_decoded = cudf::size_type{0};

    // Decode values from the current dictionary page
    while (buffer_offset < page_data_size) {
      // Check if we have a stream overrun
      if (num_values_decoded > page.num_input_values) {
        set_error(error, decode_error::DATA_STREAM_OVERRUN);
        return;
      }

      // Decode cudf::string_view value
      auto const decoded_value =
        decode_string_value(page_data, page_data_size, buffer_offset, error);

      // Break if an error has been set within `decode_string_value`
      if (is_error_set(error)) { break; }

      // Update the number of values decoded
      num_values_decoded++;

      // Evaluate all input predicates against the decoded string value
      for (auto scalar_idx = 0; scalar_idx < total_num_scalars; ++scalar_idx) {
        // Check if the literal value matches the decoded value
        if (decoded_value == scalars[scalar_idx].value<cudf::string_view>()) {
          // If the operator is NOT_EQUAL, set the result to true (row group to be pruned) if and
          // only if this is the only value in the dictionary page (all values are unique).
          // Otherwise set it to true (row group to be kept)
          results[scalar_idx][row_group_idx] =
            operators[scalar_idx] == ast::ast_operator::EQUAL or page.num_input_values == 1;
        }
      }

      // If operator is EQUAL, check if all literal values have been found and break early.
      // Otherwise, check if we have more than one values in the dictionary (will never evaluate to
      // true so we can easily break) or if the decoded value was a match with the literal value.
      if (thrust::all_of(thrust::seq,
                         cuda::counting_iterator<cudf::size_type>{0},
                         cuda::counting_iterator{total_num_scalars},
                         [&](auto scalar_idx) {
                           return operators[scalar_idx] == ast::ast_operator::EQUAL
                                    ? results[scalar_idx][row_group_idx]
                                    : page.num_input_values > 1 or
                                        results[scalar_idx][row_group_idx];
                         })) {
        break;
      }
    }
  }
}

/**
 * @brief Decode fixed width column chunk dictionaries and evaluate (few) input predicates against
 * decoded values
 *
 * @tparam Supported underlying (fixed width) data type of the cudf column
 * @param pages Column chunk dictionary page headers
 * @param chunks Column chunk descriptors
 * @param results Span of device vector start pointers to store query results, one per predicate
 * @param scalars Span of scalar device views, one per predicate
 * @param operators Span of corresponding (in)equality operators, one per predicate
 * @param physical_type Parquet physical type of the column
 * @param total_num_scalars Total number of predicates
 * @param num_dictionary_columns Total number of columns with dictionaries
 * @param dictionary_col_idx Index of the current dictionary column
 * @param error Pointer to the kernel error code
 */
template <SupportedFixedWidthType T>
CUDF_KERNEL void __launch_bounds__(DECODE_BLOCK_SIZE)
  evaluate_few_fixed_width_literals(PageInfo const* pages,
                                    ColumnChunkDesc const* chunks,
                                    cudf::device_span<bool*> results,
                                    ast::generic_scalar_device_view const* scalars,
                                    ast::ast_operator const* operators,
                                    parquet::Type physical_type,
                                    cudf::size_type total_num_scalars,
                                    cudf::size_type num_dictionary_columns,
                                    cudf::size_type dictionary_col_idx,
                                    kernel_error::pointer error)
{
  // Each thread block decodes values from one dictionary page and evaluates all input predicates
  // against them
  auto const group = cg::this_thread_block();

  // Index of the current column chunk of the column
  auto const row_group_idx = cg::this_grid().block_rank();

  // Global index of the current column chunk (dictionary page)
  auto const chunk_idx = dictionary_col_idx + (row_group_idx * num_dictionary_columns);
  auto const& page     = pages[chunk_idx];
  auto const& chunk    = chunks[chunk_idx];

  // If the page buffer is empty or has no values to decode, then return early
  if (page.num_input_values == 0 or page.uncompressed_page_size == 0) {
    for (auto i = group.thread_rank(); i < total_num_scalars; i += group.num_threads()) {
      // Set the result to true (keep) if the operator is EQUAL, otherwise set it to false (do not
      // prune)
      results[i][row_group_idx] = operators[i] == ast::ast_operator::EQUAL;
    }
    return;
  }

  // Initialize results for all predicates to false
  for (auto i = group.thread_rank(); i < total_num_scalars; i += group.num_threads()) {
    results[i][row_group_idx] = false;
  }

  // Decode values from the current dictionary page with the current thread block
  for (auto value_idx = group.thread_rank(); value_idx < page.num_input_values;
       value_idx += group.num_threads()) {
    // Decode the value from the page data
    auto decoded_value = decode_fixed_width_value<T>(page, chunk, value_idx, physical_type, error);

    // Return early if an error has been set
    if (is_error_set(error)) { return; }

    // Evaluate all input predicates against the decoded value
    for (auto scalar_idx = 0; scalar_idx < total_num_scalars; ++scalar_idx) {
      // Check if the literal value is equal to the decoded value
      if (decoded_value == scalars[scalar_idx].value<T>()) {
        // If the operator is EQUAL, set the result to true (row group to be kept). Otherwise, set
        // the result to true (row group to be pruned) if and only if this is the only value in the
        // dictionary page (all values are unique)
        results[scalar_idx][row_group_idx] =
          operators[scalar_idx] == ast::ast_operator::EQUAL or page.num_input_values == 1;
      }
    }

    // If the operator is EQUAL, check if all literal values have been found and break early.
    // Otherwise, check if we have more than one values in the dictionary (will never evaluate to
    // true so we can easily break) or if the decoded value was a match with the literal value.
    if (thrust::all_of(thrust::seq,
                       cuda::counting_iterator<cudf::size_type>{0},
                       cuda::counting_iterator{total_num_scalars},
                       [&](auto scalar_idx) {
                         return operators[scalar_idx] == ast::ast_operator::EQUAL
                                  ? results[scalar_idx][row_group_idx]
                                  : page.num_input_values > 1 or results[scalar_idx][row_group_idx];
                       })) {
      return;
    }
  }
}

/**
 * @brief Computes dictionary membership results (for each column chunk) into a vector of BOOL8
 * columns, one per (in)equality predicate.
 */
struct dictionary_caster {
  cudf::detail::hostdevice_span<parquet::detail::ColumnChunkDesc const> chunks;
  cudf::detail::hostdevice_span<parquet::detail::PageInfo const> pages;
  std::size_t total_row_groups;
  parquet::Type physical_type;
  cudf::size_type type_length;
  cudf::size_type num_dictionary_columns;
  cudf::size_type dictionary_col_idx;

  /**
   * @brief Build BOOL8 columns from dictionary membership results device buffers
   *
   * @param results_buffers Vector of dictionary membership results device buffers
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource used to allocate the returned device memory
   *
   * @return A vector of BOOL8 columns
   */
  [[nodiscard]] std::vector<std::unique_ptr<cudf::column>> build_columns(
    cudf::host_span<rmm::device_buffer> results_buffers,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr)
  {
    auto columns = std::vector<std::unique_ptr<cudf::column>>{};
    columns.reserve(results_buffers.size());
    std::transform(results_buffers.begin(),
                   results_buffers.end(),
                   std::back_inserter(columns),
                   [&](auto& result_buffer) {
                     return std::make_unique<cudf::column>(
                       cudf::data_type{cudf::type_id::BOOL8},
                       static_cast<cudf::size_type>(total_row_groups),
                       std::move(result_buffer),
                       rmm::device_buffer{0, stream, mr},
                       0);
                   });
    return columns;
  }

  /**
   * @brief Build `cuco::static_set`s from decoded column chunk dictionaries and evaluate (many)
   * input predicates using the built hash sets.
   *
   * @tparam Supported underlying data type of the cudf column
   * @param literals List of literals
   * @param operators List of operators
   * @param stream CUDA stream to use for the kernel launches
   * @param mr Device memory resource used to allocate the returned columns' device memory
   *
   * @return A vector of BOOL8 columns containing dictionary membership results, one per predicate
   */
  template <SupportedDictionaryType T>
  std::vector<std::unique_ptr<cudf::column>> evaluate_many_literals(
    cudf::host_span<ast::literal* const> literals,
    cudf::host_span<ast::ast_operator const> operators,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr)
  {
    // Host vectors to store the running number of hash set slots and decoded values for all
    // dictionaries
    auto host_set_offsets =
      cudf::detail::make_empty_host_vector<cudf::size_type>(total_row_groups + 1, stream);
    auto host_value_offsets =
      cudf::detail::make_empty_host_vector<cudf::size_type>(total_row_groups + 1, stream);
    host_set_offsets.push_back(0);
    host_value_offsets.push_back(0);

    // Define the probing scheme type with either (insert or query) hash functor to use it in
    // `cuco::make_valid_extent`
    using hash_fn_type        = insert_hash_functor<T>;
    using probing_scheme_type = cuco::linear_probing<SET_CG_SIZE, hash_fn_type>;

    // Compute the running number of hash set slots and decoded values for all dictionaries
    std::for_each(
      cuda::counting_iterator<std::size_t>{0},
      cuda::counting_iterator{total_row_groups},
      [&](auto row_group_idx) {
        auto const chunk_idx        = dictionary_col_idx + (row_group_idx * num_dictionary_columns);
        auto const num_input_values = pages[chunk_idx].num_input_values;
        // Update the running number of values in this dictionary
        host_value_offsets.push_back(host_value_offsets.back() + num_input_values);
        // Compute the number of hash set slots needed to target the desired occupancy factor
        host_set_offsets.push_back(
          host_set_offsets.back() +
          static_cast<cudf::size_type>(cuco::make_valid_extent<probing_scheme_type, storage_type>(
            num_input_values, OCCUPANCY_FACTOR)));
      });

    // Get the default device resource ref for temporary memory allocations
    auto const default_mr = cudf::get_current_device_resource_ref();

    // Device vectors to store the running number of hash set slots and decoded values for all
    // dictionaries
    auto const set_offsets =
      cudf::detail::make_device_uvector_async(host_set_offsets, stream, default_mr);
    auto const value_offsets =
      cudf::detail::make_device_uvector_async(host_value_offsets, stream, default_mr);

    auto const total_set_storage_size = static_cast<std::size_t>(host_set_offsets.back());
    auto const total_num_values       = static_cast<std::size_t>(host_value_offsets.back());
    auto const total_num_literals     = static_cast<cudf::size_type>(literals.size());

    // Create a single bulk storage used by all cuco hash sets
    auto set_storage =
      storage_type{total_set_storage_size, rmm::mr::polymorphic_allocator<char>{}, stream.value()};

    // Initialize storage with the empty key sentinel
    set_storage.initialize_async(EMPTY_KEY_SENTINEL, {stream.value()});

    // Device vector to store the decoded values for all dictionaries
    rmm::device_uvector<T> decoded_data{total_num_values, stream, default_mr};
    kernel_error error_code(stream);

    // Host vector of scalar device views from all literals
    std::vector<ast::generic_scalar_device_view> host_scalars;
    host_scalars.reserve(total_num_literals);
    std::for_each(literals.begin(), literals.end(), [&](auto const& literal) {
      host_scalars.push_back(literal->get_value());
    });
    // Device vector of all scalars device views
    auto const scalars = cudf::detail::make_device_uvector_async(host_scalars, stream, default_mr);

    // Device vector of all operators
    auto const d_operators = cudf::detail::make_device_uvector_async(operators, stream, default_mr);

    // Device buffers to store the dictionary membership results for all predicates
    std::vector<rmm::device_buffer> results_buffers(total_num_literals);
    // Host vector of pointers to the result buffers
    auto host_results_ptrs = cudf::detail::make_host_vector<bool*>(total_num_literals, stream);
    std::for_each(cuda::counting_iterator<cudf::size_type>{0},
                  cuda::counting_iterator{total_num_literals},
                  [&](auto i) {
                    // Allocate the results buffer using the user-provided memory resource (output
                    // memory)
                    results_buffers[i]   = rmm::device_buffer(total_row_groups, stream, mr);
                    host_results_ptrs[i] = static_cast<bool*>(results_buffers[i].data());
                  });
    if constexpr (not cuda::std::is_same_v<T, cudf::string_view>) {
      // Decode fixed width dictionaries and insert them to cuco hash sets, one dictionary per
      // thread block
      build_fixed_width_dictionaries<T>
        <<<total_row_groups, DECODE_BLOCK_SIZE, 0, stream.value()>>>(pages.device_begin(),
                                                                     chunks.device_begin(),
                                                                     decoded_data,
                                                                     set_storage.data(),
                                                                     set_offsets.data(),
                                                                     value_offsets.data(),
                                                                     physical_type,
                                                                     num_dictionary_columns,
                                                                     dictionary_col_idx,
                                                                     error_code.data());
      CUDF_CUDA_TRY(cudaGetLastError());

    } else {
      // Check if the decode block size is a multiple of the warp size
      static_assert(DECODE_BLOCK_SIZE % cudf::detail::warp_size == 0,
                    "DECODE_BLOCK_SIZE must be a multiple of warp_size");
      // Check if the physical type is a string
      CUDF_EXPECTS(physical_type == parquet::Type::BYTE_ARRAY, "Unsupported physical type");

      // Number of warps per thread block
      std::size_t const warps_per_block = DECODE_BLOCK_SIZE / cudf::detail::warp_size;
      // Number of thread blocks to get at least `total_row_groups` warps
      auto const num_blocks =
        cudf::util::div_rounding_up_safe<std::size_t>(total_row_groups, warps_per_block);

      // Decode string dictionaries and insert them to cuco hash sets, one dictionary per
      // warp
      build_string_dictionaries<<<num_blocks, DECODE_BLOCK_SIZE, 0, stream.value()>>>(
        pages.device_begin(),
        decoded_data,
        set_storage.data(),
        set_offsets.data(),
        value_offsets.data(),
        total_row_groups,
        num_dictionary_columns,
        dictionary_col_idx,
        error_code.data());
      CUDF_CUDA_TRY(cudaGetLastError());
    }

    // Check if there are any errors in data decoding
    if (auto const error = error_code.value_sync(stream); error != 0) {
      CUDF_FAIL("Dictionary decode failed with code(s) " + kernel_error::to_string(error));
    }

    static_assert(MAX_QUERY_BLOCK_SIZE % cudf::detail::warp_size == 0,
                  "MAX_QUERY_BLOCK_SIZE must be a multiple of warp_size");

    // Compute an optimal thread block size for the query kernel so that we can query all cuco
    // hash sets of this column per thread block
    // Cap the query block size at MAX_QUERY_BLOCK_SIZE
    auto query_block_size = [&]() {
      // Check if we have less than warp_size row groups. If so, use warp_size
      auto query_block_size = std::max<cudf::size_type>(cudf::detail::warp_size, total_row_groups);
      // Round up to the next power of 2 if block size is not already a multiple of the warp size
      if (query_block_size % cudf::detail::warp_size != 0) {
        query_block_size = cudf::size_type{1} << (32 - cuda::std::countl_zero(
                                                         static_cast<uint32_t>(query_block_size)));
      }
      return std::min<cudf::size_type>(query_block_size, MAX_QUERY_BLOCK_SIZE);
    }();

    // Device vector of pointers to the result buffers
    auto results_ptrs =
      cudf::detail::make_device_uvector_async(host_results_ptrs, stream, default_mr);

    // Query one predicate against all cuco hash sets of this column using a thread block
    query_dictionaries<T>
      <<<total_num_literals, query_block_size, 0, stream.value()>>>(decoded_data,
                                                                    results_ptrs,
                                                                    scalars.data(),
                                                                    d_operators.data(),
                                                                    set_storage.data(),
                                                                    set_offsets.data(),
                                                                    value_offsets.data(),
                                                                    total_row_groups,
                                                                    physical_type);
    CUDF_CUDA_TRY(cudaGetLastError());

    // Build the BOOL8 columns from the results buffers
    return build_columns(results_buffers, stream, mr);
  }

  /**
   * @brief Decode column chunk dictionaries and evaluate (few) input predicates while decoding
   *
   * @tparam Supported underlying data type of the cudf column
   * @param literals List of literals
   * @param operators List of operators
   * @param stream CUDA stream to use for the kernel launches
   * @param mr Device memory resource used to allocate the returned columns' device memory
   *
   * @return A vector of BOOL8 columns containing dictionary membership results, one per predicate
   */
  template <SupportedDictionaryType T>
  std::vector<std::unique_ptr<cudf::column>> evaluate_few_literals(
    cudf::host_span<ast::literal* const> literals,
    cudf::host_span<ast::ast_operator const> operators,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr)
  {
    // Get the total number of scalars and literals
    auto const total_num_literals = static_cast<cudf::size_type>(literals.size());

    // Get the default device resource ref for temporary memory allocations
    auto const default_mr = cudf::get_current_device_resource_ref();

    // Host vector of scalar device views from all literals
    std::vector<ast::generic_scalar_device_view> host_scalars;
    host_scalars.reserve(total_num_literals);
    std::for_each(literals.begin(), literals.end(), [&](auto const& literal) {
      host_scalars.push_back(literal->get_value());
    });
    // Device vector of all scalars device views
    auto const scalars = cudf::detail::make_device_uvector_async(host_scalars, stream, default_mr);

    // Device vector of all operators
    auto const d_operators = cudf::detail::make_device_uvector_async(operators, stream, default_mr);

    // Device buffers to store the dictionary membership results for all predicates
    std::vector<rmm::device_buffer> results_buffers(total_num_literals);
    // Host vector of pointers to the result buffers
    auto host_results_ptrs = cudf::detail::make_host_vector<bool*>(total_num_literals, stream);
    std::for_each(cuda::counting_iterator<cudf::size_type>{0},
                  cuda::counting_iterator{total_num_literals},
                  [&](auto i) {
                    // Allocate the results buffer using the user-provided memory resource (output
                    // memory)
                    results_buffers[i]   = rmm::device_buffer(total_row_groups, stream, mr);
                    host_results_ptrs[i] = static_cast<bool*>(results_buffers[i].data());
                  });

    // Device vector of pointers to the result buffers
    auto results_ptrs =
      cudf::detail::make_device_uvector_async(host_results_ptrs, stream, default_mr);

    // Error code for the dictionary decode kernel
    kernel_error error_code(stream);

    if constexpr (not cuda::std::is_same_v<T, cudf::string_view>) {
      // Decode fixed width dictionaries and evaluate literals against them, one dictionary per
      // thread block
      evaluate_few_fixed_width_literals<T>
        <<<total_row_groups, DECODE_BLOCK_SIZE, 0, stream.value()>>>(pages.device_begin(),
                                                                     chunks.device_begin(),
                                                                     results_ptrs,
                                                                     scalars.data(),
                                                                     d_operators.data(),
                                                                     physical_type,
                                                                     total_num_literals,
                                                                     num_dictionary_columns,
                                                                     dictionary_col_idx,
                                                                     error_code.data());
      CUDF_CUDA_TRY(cudaGetLastError());
    } else {
      static_assert(DECODE_BLOCK_SIZE % cudf::detail::warp_size == 0,
                    "decoder block size must be a multiple of warp_size");
      CUDF_EXPECTS(physical_type == parquet::Type::BYTE_ARRAY, "Unsupported physical type");

      // Number of warps per thread block
      std::size_t const warps_per_block = DECODE_BLOCK_SIZE / cudf::detail::warp_size;
      // Number of thread blocks to get at least `total_row_groups` warps
      auto const num_blocks =
        cudf::util::div_rounding_up_safe<std::size_t>(total_row_groups, warps_per_block);

      // Decode string dictionaries and evaluate all literals against them, one dictionary per
      // warp
      evaluate_few_string_literals<<<num_blocks, DECODE_BLOCK_SIZE, 0, stream.value()>>>(
        pages.device_begin(),
        results_ptrs,
        scalars.data(),
        d_operators.data(),
        total_num_literals,
        total_row_groups,
        num_dictionary_columns,
        dictionary_col_idx,
        error_code.data());
      CUDF_CUDA_TRY(cudaGetLastError());
    }

    // Check if there are any errors in data decoding
    if (auto const error = error_code.value_sync(stream); error != 0) {
      CUDF_FAIL("Dictionary decode failed with code(s) " + kernel_error::to_string(error));
    }

    // Build the BOOL8 columns from the results buffers
    return build_columns(results_buffers, stream, mr);
  }

  template <typename T>
  std::vector<std::unique_ptr<cudf::column>> operator()(
    cudf::data_type dtype,
    cudf::host_span<ast::literal* const> literals,
    cudf::host_span<ast::ast_operator const> operators,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr)
  {
    // Boolean, List, Struct, Dictionary types are not supported
    if constexpr (not is_supported_dictionary_type<T>) {
      CUDF_FAIL("Dictionaries do not support boolean or compound types");
    } else {
      // Make sure all literals have the same type as the predicate column
      std::for_each(literals.begin(), literals.end(), [&](auto const& literal) {
        // Check if the literal has the same type as the predicate column
        CUDF_EXPECTS(
          dtype == literal->get_data_type() and
            cudf::have_same_types(
              cudf::column_view{dtype, 0, {}, {}, 0, 0, {}},
              cudf::scalar_type_t<T>(T{}, false, stream, cudf::get_current_device_resource_ref())),
          "Mismatched predicate column and literal types");
      });

      // If there are only a few literals, just evaluate expression while decoding dictionary data
      if (literals.size() <= MAX_INLINE_LITERALS) {
        return evaluate_few_literals<T>(literals, operators, stream, mr);
      } else {
        // Else, decode dictionaries to `cudf::static_set`s and evaluate all expressions
        return evaluate_many_literals<T>(literals, operators, stream, mr);
      }
    }
  }
};

/**
 * @brief Converts AST expression to dictionary membership (DictionaryAST) expression.
 * This is used in row group filtering based on equality predicate.
 */
class dictionary_expression_converter : public equality_literals_collector {
 public:
  dictionary_expression_converter(ast::expression const& expr,
                                  cudf::host_span<cudf::data_type const> output_dtypes,
                                  cudf::host_span<std::vector<ast::literal*> const> literals,
                                  rmm::cuda_stream_view stream)
    : _literals{literals},
      _always_true_scalar{std::make_unique<cudf::numeric_scalar<bool>>(true, true, stream)},
      _always_true{std::make_unique<ast::literal>(*_always_true_scalar)}
  {
    // Set the output data types
    _output_dtypes = output_dtypes;

    // Compute and store columns literals offsets
    _col_literals_offsets.reserve(static_cast<cudf::size_type>(_output_dtypes.size()) + 1);
    _col_literals_offsets.emplace_back(0);

    // Set column literal count offsets
    std::transform(literals.begin(),
                   literals.end(),
                   std::back_inserter(_col_literals_offsets),
                   [&](auto const& col_literal_map) {
                     return _col_literals_offsets.back() +
                            static_cast<cudf::size_type>(col_literal_map.size());
                   });

    // Add this visitor
    expr.accept(*this);
  }

  /**
   * @brief Delete equality literals getter as it's not needed in the derived class
   */
  [[nodiscard]] std::vector<std::vector<ast::literal*>> get_equality_literals() && = delete;

  // Bring all overloads of `visit` from equality_predicate_collector into scope
  using equality_literals_collector::visit;

  /**
   * @copydoc ast::detail::expression_transformer::visit(ast::operation const& )
   */
  std::reference_wrapper<ast::expression const> visit(ast::operation const& expr) override
  {
    using cudf::ast::ast_operator;
    using parquet::detail::extract_binary_operands;
    using parquet::detail::operand_kind;

    auto const input_op       = expr.get_operator();
    auto const operator_arity = cudf::ast::detail::ast_operator_arity(input_op);

    // Unary operation
    if (operator_arity == 1) {
      auto visit_operands_fn = [this](auto const& operands) {
        return this->visit_operands(operands);
      };
      return parquet::detail::apply_unary_membership_transform(
        expr, _dictionary_expr, *_always_true, *this, visit_operands_fn);
    }

    // Binary operation
    auto const [op, lhs_kind, rhs_kind, col_ref, literal] = extract_binary_operands(expr);

    // Push expressions for `col op lit` or `lit op col` forms
    if (lhs_kind == operand_kind::COLUMN_REF and rhs_kind == operand_kind::LITERAL) {
      col_ref->accept(*this);

      if (op == ast_operator::EQUAL or op == ast_operator::NOT_EQUAL) {
        auto const col_idx            = col_ref->get_column_index();
        auto const& equality_literals = _literals[col_idx];
        auto col_literal_offset       = _col_literals_offsets[col_idx];
        auto const literal_iter =
          std::find(equality_literals.cbegin(), equality_literals.cend(), literal);
        CUDF_EXPECTS(literal_iter != equality_literals.end(),
                     "Dictionary expression converter encountered an unexpected literal");
        col_literal_offset += std::distance(equality_literals.cbegin(), literal_iter);

        auto const& value = _dictionary_expr.push(ast::column_reference{col_literal_offset});

        if (op == ast_operator::NOT_EQUAL) {
          // For NOT_EQUAL operator, simply evaluate boolean is_false(value) expression as
          // NOT(value). The value indicates if the row group should be pruned (if the literal is
          // present in the hash set and it's the only value in the hash set)
          _dictionary_expr.push(ast::operation{ast_operator::NOT, value});
        } else {
          // For EQUAL operator, evaluate boolean is_true(value) expression as IDENTITY(value)
          // The value indicates if the row group should be kept (if the literal is present in the
          // hash set)
          _dictionary_expr.push(ast::operation{ast_operator::IDENTITY, value});
        }
      }  // For all other expressions, push the `_always_true` expression
      else {
        _dictionary_expr.push(ast::operation{ast_operator::IDENTITY, *_always_true});
        return *_always_true;
      }
    }  // Visit operands and push expression for `expr op expr` form
    else if (lhs_kind == operand_kind::EXPRESSION and rhs_kind == operand_kind::EXPRESSION) {
      auto new_operands = visit_operands(expr.get_operands());
      _dictionary_expr.push(ast::operation{op, new_operands.front(), new_operands.back()});
    }  // Push _always_true for `col op col`, `expr op col`, `expr op lit` forms
    else {
      _dictionary_expr.push(ast::operation{ast_operator::IDENTITY, *_always_true});
      return *_always_true;
    }
    return _dictionary_expr.back();
  }

  /**
   * @brief Returns the AST to apply on dictionary membership.
   *
   * @return AST operation expression
   */
  [[nodiscard]] std::reference_wrapper<ast::expression const> get_dictionary_expr() const
  {
    return _dictionary_expr.back();
  }

 private:
  std::vector<cudf::size_type> _col_literals_offsets;
  cudf::host_span<std::vector<ast::literal*> const> _literals;
  ast::tree _dictionary_expr;
  std::unique_ptr<cudf::numeric_scalar<bool>> _always_true_scalar;
  std::unique_ptr<ast::literal> _always_true;
};

}  // namespace

std::optional<std::vector<std::vector<cudf::size_type>>>
aggregate_reader_metadata::apply_dictionary_filter(
  cudf::detail::hostdevice_span<parquet::detail::ColumnChunkDesc const> chunks,
  cudf::detail::hostdevice_span<parquet::detail::PageInfo const> pages,
  std::span<std::vector<size_type> const> input_row_group_indices,
  std::span<std::vector<ast::literal*> const> literals,
  std::span<std::vector<ast::ast_operator> const> operators,
  std::size_t total_row_groups,
  std::span<data_type const> output_dtypes,
  std::span<int const> dictionary_col_schemas,
  std::reference_wrapper<ast::expression const> filter,
  rmm::cuda_stream_view stream) const
{
  // Number of input table columns
  auto const num_input_columns = static_cast<cudf::size_type>(output_dtypes.size());
  // Number of columns with dictionaries
  auto const num_dictionary_columns = static_cast<cudf::size_type>(dictionary_col_schemas.size());
  // Get parquet types for the predicate columns
  auto const parquet_types = get_parquet_types(
    cudf::host_span<std::vector<size_type> const>{input_row_group_indices.data(),
                                                  input_row_group_indices.size()},
    cudf::host_span<int const>{dictionary_col_schemas.data(), dictionary_col_schemas.size()});

  // Convert dictionary membership for (in)equality predicate columns to a table
  // containing a column for each `col[i] == literal` or `col[i] != literal` predicate
  // to be evaluated. The table contains #sources * #column_chunks_per_src rows.
  std::vector<std::unique_ptr<cudf::column>> dictionary_membership_columns;

  // Memory resource to allocate dictionary membership columns with
  auto const mr = cudf::get_current_device_resource_ref();

  // Dictionary column index currently being processed
  cudf::size_type dictionary_col_idx = 0;

  // For each input column
  std::for_each(
    cuda::counting_iterator<cudf::size_type>{0},
    cuda::counting_iterator{num_input_columns},
    [&](auto input_col_idx) {
      auto const& dtype = output_dtypes[input_col_idx];

      // Skip if no equality literals for this column
      if (literals[input_col_idx].empty()) { return; }

      // Skip if non-comparable (compound) type except string
      if (cudf::is_compound(dtype) and dtype.id() != cudf::type_id::STRING) { return; }

      // Create a dictionary membership caster struct for the current column
      dictionary_caster const dictionary_col{chunks,
                                             pages,
                                             total_row_groups,
                                             parquet_types[dictionary_col_idx],
                                             chunks[dictionary_col_idx].type_length,
                                             num_dictionary_columns,
                                             dictionary_col_idx};

      // Process all predicates associated with the current column and build a BOOL8 column per
      // predicate
      auto dict_columns = cudf::type_dispatcher<dispatch_storage_type>(dtype,
                                                                       dictionary_col,
                                                                       dtype,
                                                                       literals[input_col_idx],
                                                                       operators[input_col_idx],
                                                                       stream,
                                                                       mr);

      // Add the built columns to the vector of columns
      dictionary_membership_columns.insert(dictionary_membership_columns.end(),
                                           std::make_move_iterator(dict_columns.begin()),
                                           std::make_move_iterator(dict_columns.end()));

      // Increment the dictionary column index
      dictionary_col_idx++;
    });

  // Create a table from columns
  auto const dictionary_membership_table = cudf::table(std::move(dictionary_membership_columns));

  // Convert AST to DictionaryAST expression with reference to dictionary membership
  // in above `dictionary_membership_table`
  dictionary_expression_converter dictionary_expr{
    filter.get(),
    cudf::host_span<cudf::data_type const>{output_dtypes.data(), output_dtypes.size()},
    cudf::host_span<std::vector<ast::literal*> const>{literals.data(), literals.size()},
    stream};

  // Filter dictionary membership table with the DictionaryAST expression and collect
  // filtered row group indices
  return parquet::detail::collect_filtered_row_group_indices(
    dictionary_membership_table,
    dictionary_expr.get_dictionary_expr(),
    cudf::host_span<std::vector<size_type> const>{input_row_group_indices.data(),
                                                  input_row_group_indices.size()},
    stream);
}

dictionary_literals_collector::dictionary_literals_collector(
  ast::expression const& expr, std::span<cudf::data_type const> output_dtypes)
{
  _output_dtypes =
    cudf::host_span<cudf::data_type const>{output_dtypes.data(), output_dtypes.size()};
  auto const num_input_columns = static_cast<cudf::size_type>(_output_dtypes.size());
  _literals.resize(num_input_columns);
  _operators.resize(num_input_columns);
  expr.accept(*this);
}

std::reference_wrapper<ast::expression const> dictionary_literals_collector::visit(
  ast::operation const& expr)
{
  using cudf::ast::ast_operator;
  using parquet::detail::extract_binary_operands;
  using parquet::detail::extract_unary_operand;
  using parquet::detail::operand_kind;

  auto const input_op       = expr.get_operator();
  auto const operator_arity = cudf::ast::detail::ast_operator_arity(input_op);

  if (operator_arity == 1) {
    auto const [kind, col_ref] = extract_unary_operand(expr);

    if (kind == operand_kind::COLUMN_REF) {
      col_ref->accept(*this);
    } else {
      std::ignore = visit_operands(expr.get_operands());
    }
    return expr;
  }

  // Binary operation
  auto const [op, lhs_kind, rhs_kind, col_ref, literal] = extract_binary_operands(expr);

  // Push expressions for `col op lit` or `lit op col` forms
  if (lhs_kind == operand_kind::COLUMN_REF and rhs_kind == operand_kind::LITERAL) {
    col_ref->accept(*this);
    if (op == ast_operator::EQUAL or op == ast_operator::NOT_EQUAL) {
      auto const col_idx = col_ref->get_column_index();
      _literals[col_idx].emplace_back(const_cast<ast::literal*>(literal));
      _operators[col_idx].emplace_back(op);
    }
  }  // For all other forms, visit operands to collect any nested literals
  else {
    std::ignore = visit_operands(expr.get_operands());
  }
  return expr;
}

std::pair<std::vector<std::vector<ast::literal*>>, std::vector<std::vector<ast::ast_operator>>>
dictionary_literals_collector::get_literals_and_operators() &&
{
  return {std::move(_literals), std::move(_operators)};
}

}  // namespace cudf::io::parquet::experimental::detail
