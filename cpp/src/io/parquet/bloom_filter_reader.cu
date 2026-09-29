/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "compact_protocol_reader.hpp"
#include "expression_transform_helpers.hpp"
#include "io/utilities/time_utils.hpp"
#include "reader_impl_helpers.hpp"
#include "timestamp_utils.cuh"

#include <cudf/ast/detail/operators.hpp>
#include <cudf/ast/expressions.hpp>
#include <cudf/detail/cuco_helpers.hpp>
#include <cudf/detail/transform.hpp>
#include <cudf/hashing/detail/xxhash_64.cuh>
#include <cudf/io/parquet_io_utils.hpp>
#include <cudf/io/parquet_schema.hpp>
#include <cudf/logger.hpp>
#include <cudf/reduction/bloom_filter.cuh>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_checks.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/exec_policy.hpp>

#include <cuco/bloom_filter_ref.cuh>
#include <cuda/iterator>
#include <cuda/std/chrono>
#include <cuda/stream>
#include <thrust/tabulate.h>
#include <thrust/uninitialized_fill.h>

#include <functional>
#include <future>
#include <numeric>
#include <optional>
#include <ranges>
#include <span>
#include <utility>

namespace cudf::io::parquet::detail {
namespace {

/**
 * @brief Policy describing the Apache Arrow Block-Split Bloom Filter, hashing keys with cudf's
 * `XXHash_64` (so that `cudf::string_view` and other cudf types are hashed by content, matching the
 * Apache Parquet/Arrow bloom filter specification).
 *
 * Uses cuco's `bloom_filter_policy` with the Apache Arrow layout: 256-bit blocks (8 x
 * `uint32_t`), 8 fingerprint bits per key, fully horizontal add (Theta=8) and fully vertical
 * contains (Phi=8). This layout is bit-compatible with Apache Arrow, as verified by cuCollections
 * `tests/bloom_filter/arrow_compat_test.cu`.
 *
 * @tparam Key The type of the values to generate a fingerprint for.
 */
template <class Key>
using arrow_filter_policy =
  cudf::arrow_bloom_filter_policy<Key, cudf::hashing::detail::XXHash_64<Key>>;

/*
 * Parquet bloom filters hash the plain encoding of the physical type, which can differ from the
 * cudf type the column is read as. The probes below encode the literal as the physical type:
 * INT8/INT16/UINT8/UINT16 and TIME(MILLIS) are stored as INT32, decimals as INT32, INT64 or
 * big-endian FIXED_LEN_BYTE_ARRAY, and timestamps may be INT96.
 */

/**
 * @brief Queries a bloom filter for a string literal.
 */
struct string_probe {
  ast::generic_scalar_device_view literal;

  template <typename BloomFilter>
  __device__ bool operator()(BloomFilter const& filter) const
  {
    return filter.contains(literal.value<cudf::string_view>());
  }
};

/**
 * @brief Queries a bloom filter for a floating-point literal.
 *
 * +0.0 and -0.0 compare equal but hash differently, so a zero literal probes both.
 *
 * @tparam T Floating-point type of the literal and the column
 */
template <typename T>
struct floating_point_probe {
  ast::generic_scalar_device_view literal;

  template <typename BloomFilter>
  __device__ bool operator()(BloomFilter const& filter) const
  {
    auto const value = literal.value<T>();
    if (value == T{0}) { return filter.contains(T{0}) or filter.contains(-T{0}); }
    return filter.contains(value);
  }
};

/**
 * @brief Queries a bloom filter for an integer literal, cast to the INT32 or INT64 physical type.
 *
 * @tparam Rep Integer the literal is stored as
 * @tparam Key INT32 or INT64 physical type
 */
template <typename Rep, typename Key>
struct integer_probe {
  ast::generic_scalar_device_view literal;

  template <typename BloomFilter>
  __device__ bool operator()(BloomFilter const& filter) const
  {
    return filter.contains(static_cast<Key>(literal.value<Rep>()));
  }
};

/**
 * @brief Queries a bloom filter for a nanosecond timestamp literal stored as INT96.
 *
 * INT96 is the nanoseconds since midnight (8 bytes) followed by the Julian day (4 bytes).
 */
struct int96_probe {
  ast::generic_scalar_device_view literal;

  template <typename BloomFilter>
  __device__ bool operator()(BloomFilter const& filter) const
  {
    using namespace cuda::std::chrono;
    auto const nanos            = nanoseconds{literal.value<int64_t>()};
    auto const days_since_epoch = floor<days>(nanos);
    int64_t const time_of_day   = (nanos - days_since_epoch).count();
    uint32_t const julian_day   = days_since_epoch.count() + 2'440'588;  // 1970-01-01
    char bytes[12];
    memcpy(bytes, &time_of_day, sizeof(time_of_day));
    memcpy(bytes + sizeof(time_of_day), &julian_day, sizeof(julian_day));
    return filter.contains(cudf::string_view{bytes, 12});
  }
};

/**
 * @brief Queries a bloom filter for a decimal literal stored as FIXED_LEN_BYTE_ARRAY.
 *
 * The value is big-endian two's complement, sign-extended to the column's `type_length`.
 *
 * @tparam Rep Integer the decimal literal is stored as
 */
template <typename Rep>
struct fixed_len_decimal_probe {
  ast::generic_scalar_device_view literal;
  int32_t type_length;

  template <typename BloomFilter>
  __device__ bool operator()(BloomFilter const& filter) const
  {
    auto const value = static_cast<__int128_t>(literal.value<Rep>());
    char bytes[sizeof(__int128_t)];
    for (int32_t i = 0; i < type_length; ++i) {
      bytes[type_length - 1 - i] = static_cast<char>(value >> (8 * i));
    }
    return filter.contains(cudf::string_view{bytes, type_length});
  }
};

/**
 * @brief Integer that a column value of type `T` is stored as
 */
template <typename T>
struct integer_rep {
  using type = T;
};

template <typename T>
  requires(cudf::is_chrono<T>())
struct integer_rep<T> {
  using type = typename T::rep;
};

// Types whose literal is queried as a string, a floating-point value, or an integer. Decimals are
// dispatched as their integer storage, and timestamps and durations are queried as their rep.
template <typename T>
constexpr bool is_string_probe = cuda::std::is_same_v<T, cudf::string_view>;
template <typename T>
constexpr bool is_floating_point_probe = cuda::std::is_floating_point_v<T>;
template <typename T>
constexpr bool is_integer_probe =
  cudf::is_integral_not_bool<T>() or cuda::std::is_same_v<T, __int128_t> or cudf::is_chrono<T>();

/**
 * @brief Converts bloom filter membership results (for each column chunk) to a device column.
 *
 */
struct bloom_filter_caster {
  cudf::device_span<cudf::device_span<cuda::std::byte const> const> bloom_filter_spans;
  host_span<Type const> parquet_types;
  host_span<int32_t const> parquet_type_lengths;
  std::size_t total_row_groups;
  std::size_t num_equality_columns;

  /**
   * @brief Queries the bloom filter of each row group with `probe`
   *
   * @tparam Key Type hashed into the bloom filter for the physical type
   * @tparam Probe Device functor that queries a bloom filter for the literal
   */
  template <typename Key, typename Probe>
  std::unique_ptr<cudf::column> query_bloom_filter(cudf::size_type equality_col_idx,
                                                   Probe probe,
                                                   cuda::stream_ref stream,
                                                   cudf::memory_resources mr) const
  {
    using policy_type       = arrow_filter_policy<Key>;
    using bloom_filter_type = cuco::
      bloom_filter_ref<Key, cuco::extent<std::size_t>, cuco::thread_scope_thread, policy_type>;
    using filter_block_type = typename bloom_filter_type::filter_block_type;
    using word_type         = typename policy_type::word_type;

    // Filter properties
    auto constexpr bytes_per_block = sizeof(word_type) * policy_type::words_per_block;

    rmm::device_buffer results{total_row_groups, stream, mr.get_output_mr()};
    cudf::device_span<bool> results_span{static_cast<bool*>(results.data()), total_row_groups};

    // Query literal in bloom filters from each column chunk (row group).
    thrust::tabulate(rmm::exec_policy_nosync(stream, mr.get_temporary_mr()),
                     results_span.begin(),
                     results_span.end(),
                     [filter_span          = bloom_filter_spans.data(),
                      probe                = probe,
                      col_idx              = equality_col_idx,
                      num_equality_columns = num_equality_columns] __device__(auto row_group_idx) {
                       // Filter bitset buffer index
                       auto const filter_idx  = col_idx + (num_equality_columns * row_group_idx);
                       auto const filter_size = filter_span[filter_idx].size();

                       // If no bloom filter, then fill in `true` as membership cannot be determined
                       if (filter_size == 0) { return true; }

                       // Number of filter blocks
                       auto const num_filter_blocks = filter_size / bytes_per_block;

                       // Create a bloom filter view. `const_cast` is needed because bloom filter
                       // view expects a mutable view.
                       bloom_filter_type filter{
                         reinterpret_cast<filter_block_type*>(
                           const_cast<cuda::std::byte*>(filter_span[filter_idx].data())),
                         num_filter_blocks,
                         {},   // Thread scope as the same literal is being searched across
                               // different bitsets per thread
                         {}};  // Arrow policy with cudf::hashing::detail::XXHash_64 seeded
                               // with 0 for Arrow compatibility

                       // Query the bloom filter and store results
                       return probe(filter);
                     });

    return std::make_unique<cudf::column>(
      cudf::data_type{cudf::type_id::BOOL8},
      static_cast<cudf::size_type>(total_row_groups),
      std::move(results),
      cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED, stream, mr.get_output_mr()),
      0);
  }

  // Membership is unknown, i.e. `true`, in every row group for a literal that cannot be queried
  [[nodiscard]] std::unique_ptr<cudf::column> unknown_membership(cuda::stream_ref stream,
                                                                 cudf::memory_resources mr) const
  {
    rmm::device_buffer results{total_row_groups, stream, mr.get_output_mr()};
    thrust::uninitialized_fill(rmm::exec_policy_nosync(stream, mr.get_temporary_mr()),
                               static_cast<bool*>(results.data()),
                               static_cast<bool*>(results.data()) + total_row_groups,
                               true);
    return std::make_unique<cudf::column>(
      cudf::data_type{cudf::type_id::BOOL8},
      static_cast<cudf::size_type>(total_row_groups),
      std::move(results),
      cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED, stream, mr.get_output_mr()),
      0);
  }

  // Strings are hashed as their bytes, for both BYTE_ARRAY and FIXED_LEN_BYTE_ARRAY columns
  template <typename T>
    requires(is_string_probe<T>)
  std::unique_ptr<cudf::column> operator()(cudf::size_type equality_col_idx,
                                           cudf::data_type,
                                           ast::literal const* const literal,
                                           cuda::stream_ref stream,
                                           cudf::memory_resources mr) const
  {
    return query_bloom_filter<cudf::string_view>(
      equality_col_idx, string_probe{literal->get_value()}, stream, mr);
  }

  template <typename T>
    requires(is_floating_point_probe<T>)
  std::unique_ptr<cudf::column> operator()(cudf::size_type equality_col_idx,
                                           cudf::data_type,
                                           ast::literal const* const literal,
                                           cuda::stream_ref stream,
                                           cudf::memory_resources mr) const
  {
    return query_bloom_filter<T>(
      equality_col_idx, floating_point_probe<T>{literal->get_value()}, stream, mr);
  }

  template <typename T>
    requires(is_integer_probe<T>)
  std::unique_ptr<cudf::column> operator()(cudf::size_type equality_col_idx,
                                           cudf::data_type dtype,
                                           ast::literal const* const literal,
                                           cuda::stream_ref stream,
                                           cudf::memory_resources mr) const
  {
    using rep_type = typename integer_rep<T>::type;

    // A decimal literal with a different scale would have to be rescaled first
    if (dtype.scale() != literal->get_data_type().scale()) {
      return unknown_membership(stream, mr);
    }

    auto const physical_type = parquet_types[equality_col_idx];
    auto const d_literal     = literal->get_value();
    // INT32 also stores 8 and 16-bit integers, and TIME(MILLIS) which is read as a 64-bit duration
    if constexpr (sizeof(rep_type) <= sizeof(int32_t) or cudf::is_duration<T>()) {
      if (physical_type == Type::INT32) {
        return query_bloom_filter<int32_t>(
          equality_col_idx, integer_probe<rep_type, int32_t>{d_literal}, stream, mr);
      }
    }
    if constexpr (sizeof(rep_type) == sizeof(int64_t)) {
      if (physical_type == Type::INT64) {
        return query_bloom_filter<int64_t>(
          equality_col_idx, integer_probe<rep_type, int64_t>{d_literal}, stream, mr);
      }
    }
    // INT96 values have nanosecond precision, so a coarser literal cannot be encoded as one
    if constexpr (cuda::std::is_same_v<T, cudf::timestamp_ns>) {
      if (physical_type == Type::INT96) {
        return query_bloom_filter<cudf::string_view>(
          equality_col_idx, int96_probe{d_literal}, stream, mr);
      }
    }
    // Decimals dispatch as `int32_t`, `int64_t` or `__int128_t`
    if constexpr (cuda::std::is_same_v<T, int32_t> or cuda::std::is_same_v<T, int64_t> or
                  cuda::std::is_same_v<T, __int128_t>) {
      auto const type_length = parquet_type_lengths[equality_col_idx];
      if (cudf::is_fixed_point(dtype) and physical_type == Type::FIXED_LEN_BYTE_ARRAY and
          type_length <= static_cast<int32_t>(sizeof(__int128_t))) {
        return query_bloom_filter<cudf::string_view>(
          equality_col_idx, fixed_len_decimal_probe<T>{d_literal, type_length}, stream, mr);
      }
    }
    // Decimals stored as BYTE_ARRAY and INT96 read as a coarser timestamp cannot be queried
    return unknown_membership(stream, mr);
  }

  // Booleans cannot be queried
  template <typename T>
    requires(not is_string_probe<T> and not is_floating_point_probe<T> and not is_integer_probe<T>)
  std::unique_ptr<cudf::column> operator()(cudf::size_type,
                                           cudf::data_type,
                                           ast::literal const* const,
                                           cuda::stream_ref stream,
                                           cudf::memory_resources mr) const
  {
    return unknown_membership(stream, mr);
  }
};

/**
 * @brief Converts AST expression to bloom filter membership (BloomfilterAST) expression.
 * This is used in row group filtering based on equality predicate.
 */
class bloom_filter_expression_converter final : public parquet_expression_simplifier {
 public:
  bloom_filter_expression_converter(ast::expression const& expr,
                                    std::span<cudf::data_type const> output_dtypes,
                                    std::span<std::vector<ast::literal*> const> equality_literals)
    : parquet_expression_simplifier{output_dtypes}, _equality_literals{equality_literals}
  {
    // Compute and store columns literals offsets
    _col_literals_offsets.reserve(static_cast<cudf::size_type>(_output_dtypes.size()) + 1);
    _col_literals_offsets.emplace_back(0);

    std::transform(equality_literals.begin(),
                   equality_literals.end(),
                   std::back_inserter(_col_literals_offsets),
                   [&](auto const& col_literal_map) {
                     return _col_literals_offsets.back() +
                            static_cast<cudf::size_type>(col_literal_map.size());
                   });

    _bloom_filter_expr = simplify_expr(expr);
  }

  /**
   * @brief Returns the AST to apply on bloom filter membership
   *
   * @return The membership expression, or std::nullopt if no row group can be pruned
   */
  [[nodiscard]] simplified_expression_opt get_bloom_filter_expr() const
  {
    return _bloom_filter_expr;
  }

 protected:
  /**
   * @copydoc parquet_expression_simplifier::simplify_comparison
   *
   * A bloom filter answers only "might this value be present", so equality is the one comparison
   * it can evaluate. Every other node relaxes via the base class defaults, including `NOT`, whose
   * membership answer cannot be complemented: `¬(some row is 5)` means "no row is 5", not "some
   * row is not 5".
   */
  [[nodiscard]] simplified_expression_opt simplify_comparison(ast::ast_operator op,
                                                              ast::column_reference const& col_ref,
                                                              ast::literal const& literal) override
  {
    using cudf::ast::ast_operator;

    if (op != ast_operator::EQUAL) { return std::nullopt; }

    auto const col_idx            = col_ref.get_column_index();
    auto const& equality_literals = _equality_literals[col_idx];

    // Skip bloom filter probing for timestamp columns with empty vector of literals due to
    // a timestamp scale mismatch — the literal can never match the native values.
    if (cudf::is_timestamp(_output_dtypes[col_idx]) and equality_literals.empty()) {
      return std::nullopt;
    }

    auto const literal_iter =
      std::find(equality_literals.cbegin(), equality_literals.cend(), &literal);
    CUDF_EXPECTS(literal_iter != equality_literals.end(),
                 "Bloom filter expression converter encountered an unexpected literal");

    auto const col_literal_offset =
      _col_literals_offsets[col_idx] +
      static_cast<cudf::size_type>(std::distance(equality_literals.cbegin(), literal_iter));
    auto const& value = _tree.push(ast::column_reference{col_literal_offset});
    return _tree.push(ast::operation{ast_operator::IDENTITY, value});
  }

 private:
  std::vector<cudf::size_type> _col_literals_offsets;
  std::span<std::vector<ast::literal*> const> _equality_literals;
  simplified_expression_opt _bloom_filter_expr;
};

}  // namespace

std::optional<std::pair<int64_t, std::size_t>> parse_bloom_filter_header(
  host_span<uint8_t const> bytes)
{
  using policy_type              = arrow_filter_policy<cuda::std::byte>;
  using word_type                = typename policy_type::word_type;
  auto constexpr bytes_per_block = sizeof(word_type) * policy_type::words_per_block;

  // Deserialize the bloom filter header from the front of the buffer
  BloomFilterHeader header;
  CompactProtocolReader cp{bytes.data(), bytes.size()};
  cp.read(&header);

  // Check if the bloom filter header is valid
  auto const is_header_valid =
    (header.num_bytes % bytes_per_block) == 0 and
    header.compression.compression == BloomFilterCompression::UNCOMPRESSED and
    header.algorithm.algorithm == BloomFilterAlgorithm::SPLIT_BLOCK and
    header.hash.hash == BloomFilterHash::XXHASH;
  if (not is_header_valid) { return std::nullopt; }

  return std::pair{static_cast<int64_t>(cp.bytecount()),
                   static_cast<std::size_t>(header.num_bytes)};
}

std::pair<std::vector<rmm::device_buffer>, std::vector<cudf::device_span<cuda::std::byte const>>>
aggregate_reader_metadata::read_bloom_filters(
  host_span<std::unique_ptr<datasource> const> sources,
  host_span<std::vector<size_type> const> row_group_indices,
  host_span<int const> column_schemas,
  size_type total_row_groups,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr) const
{
  // Descriptors for all the chunks that make up the selected columns
  auto const num_input_columns = column_schemas.size();
  auto const num_chunks        = total_row_groups * num_input_columns;

  // Flag to check if we have at least one valid bloom filter offset
  auto have_bloom_filters = false;
  // Speculatively read when a bloom filter's length is absent, enough to cover the header (and
  // often the whole bitset).
  auto constexpr speculative_read_size = int64_t{256};
  // Build complete bloom filter byte ranges (header + bitset) for every column chunk
  std::vector<std::vector<cudf::io::text::byte_range_info>> bloom_filter_byte_ranges_per_source(
    row_group_indices.size());
  // For all data sources
  std::for_each(
    cuda::counting_iterator<std::size_t>{0},
    cuda::counting_iterator{row_group_indices.size()},
    [&](auto const src_index) {
      auto const& rg_indices = row_group_indices[src_index];
      auto& source_ranges    = bloom_filter_byte_ranges_per_source[src_index];
      auto const source_size = static_cast<int64_t>(sources[src_index]->size());
      source_ranges.reserve(rg_indices.size() * num_input_columns);
      // For all row groups in the source
      std::for_each(rg_indices.cbegin(), rg_indices.cend(), [&](auto const rg_index) {
        // For all column chunks in the row group
        std::for_each(column_schemas.begin(), column_schemas.end(), [&](auto const schema_idx) {
          auto const& col_meta = get_column_metadata(rg_index, src_index, schema_idx);
          if (col_meta.bloom_filter_offset.has_value()) {
            have_bloom_filters = true;
            auto const offset  = col_meta.bloom_filter_offset.value();
            CUDF_EXPECTS(offset >= 0 and offset < source_size,
                         "Bloom filter offset is out of datasource bounds");
            // Length absent: speculatively read enough to recover the header, clamped at EOF
            auto const length = col_meta.bloom_filter_length.has_value()
                                  ? static_cast<int64_t>(col_meta.bloom_filter_length.value())
                                  : std::min(speculative_read_size, source_size - offset);
            CUDF_EXPECTS(length >= 0 and offset + length <= source_size,
                         "Bloom filter length is out of datasource bounds");
            source_ranges.push_back({offset, length});
          } else {
            source_ranges.push_back({0, 0});
          }
        });
      });
    });

  // Exit early if we don't have any bloom filters
  if (not have_bloom_filters) { return {}; }

  // Fetch the header-stripped, 32-byte-aligned bloom filter bitsets to device
  std::vector<std::reference_wrapper<datasource>> datasource_refs;
  datasource_refs.reserve(sources.size());
  std::transform(
    sources.begin(), sources.end(), std::back_inserter(datasource_refs), [](auto const& source) {
      return std::ref(*source);
    });

  auto [bloom_filter_buffers, bitset_spans_per_source] =
    fetch_bloom_filters_to_device(datasource_refs,
                                  bloom_filter_byte_ranges_per_source,
                                  io_submission_policy::INTERLEAVE,
                                  stream,
                                  mr);

  // Flatten the per-source bitset spans into per-chunk order
  std::vector<cudf::device_span<cuda::std::byte const>> bloom_filter_data;
  bloom_filter_data.reserve(num_chunks);
  auto flat_bitset_spans = bitset_spans_per_source | std::views::join;
  std::transform(flat_bitset_spans.begin(),
                 flat_bitset_spans.end(),
                 std::back_inserter(bloom_filter_data),
                 [](auto const& span) { return cuda::std::as_bytes(span); });

  return {std::move(bloom_filter_buffers), std::move(bloom_filter_data)};
}

std::optional<std::vector<std::vector<size_type>>> aggregate_reader_metadata::apply_bloom_filters(
  cudf::host_span<cudf::device_span<cuda::std::byte const> const> bloom_filter_data,
  host_span<std::vector<size_type> const> input_row_group_indices,
  host_span<std::vector<ast::literal*> const> literals,
  size_type total_row_groups,
  host_span<data_type const> output_dtypes,
  host_span<cudf::size_type const> bloom_filter_col_schemas,
  std::reference_wrapper<ast::expression const> filter,
  cuda::stream_ref stream) const
{
  // Convert AST to BloomfilterAST expression with reference to bloom filter membership
  // in above `bloom_filter_membership_table`
  bloom_filter_expression_converter bloom_filter_expr_converter{
    filter.get(),
    std::span{output_dtypes.data(), output_dtypes.size()},
    std::span{literals.data(), literals.size()}};

  // Return early if bloom filters cannot prune any row groups using the filter
  auto const bloom_filter_expr = bloom_filter_expr_converter.get_bloom_filter_expr();
  if (not bloom_filter_expr.has_value()) { return std::nullopt; }

  // Number of input table columns
  auto const num_input_columns = static_cast<cudf::size_type>(output_dtypes.size());

  // Get parquet types for the predicate columns
  auto const parquet_types = get_parquet_types(input_row_group_indices, bloom_filter_col_schemas);

  // Byte lengths of the FIXED_LEN_BYTE_ARRAY predicate columns
  std::vector<int32_t> parquet_type_lengths(bloom_filter_col_schemas.size());
  std::transform(bloom_filter_col_schemas.begin(),
                 bloom_filter_col_schemas.end(),
                 parquet_type_lengths.begin(),
                 [&](auto const schema_idx) { return get_schema(schema_idx).type_length; });

  // The membership table is only used within this function
  auto const mr = cudf::memory_resources{cudf::get_current_device_resource_ref()};

  // Copy bloom filter bitset spans to device
  auto const device_bloom_filter_data =
    cudf::detail::make_device_uvector_async(bloom_filter_data, stream, mr.get_temporary_mr());

  // Create a bloom filter query table caster
  bloom_filter_caster const bloom_filter_col{device_bloom_filter_data,
                                             parquet_types,
                                             parquet_type_lengths,
                                             static_cast<std::size_t>(total_row_groups),
                                             bloom_filter_col_schemas.size()};

  // Converts bloom filter membership for equality predicate columns to a table
  // containing a column for each `col[i] == literal` predicate to be evaluated.
  // The table contains #sources * #column_chunks_per_src rows.
  std::vector<std::unique_ptr<cudf::column>> bloom_filter_membership_columns;
  std::size_t equality_col_idx = 0;
  std::for_each(
    cuda::counting_iterator<std::size_t>{0},
    cuda::counting_iterator{output_dtypes.size()},
    [&](auto input_col_idx) {
      auto const& dtype = output_dtypes[input_col_idx];

      // Skip if no equality literals for this column
      if (literals[input_col_idx].empty()) { return; }

      // Skip if non-comparable (compound) type except string
      if (cudf::is_compound(dtype) and dtype.id() != cudf::type_id::STRING) { return; }

      // Add a column for all literals associated with an equality column
      for (auto const& literal : literals[input_col_idx]) {
        CUDF_EXPECTS(dtype.id() == literal->get_data_type().id(),
                     "Mismatched predicate column and literal types");
        bloom_filter_membership_columns.emplace_back(cudf::type_dispatcher<dispatch_storage_type>(
          dtype, bloom_filter_col, equality_col_idx, dtype, literal, stream, mr));
      }
      equality_col_idx++;
    });

  // Create a table from columns
  auto bloom_filter_membership_table = cudf::table(std::move(bloom_filter_membership_columns));

  // Filter bloom filter membership table with the BloomfilterAST expression and collect
  // filtered row group indices
  return collect_filtered_row_group_indices(
    bloom_filter_membership_table, bloom_filter_expr.value(), input_row_group_indices, stream);
}

equality_literals_collector::equality_literals_collector(
  ast::expression const& expr,
  cudf::host_span<cudf::data_type const> output_dtypes,
  cudf::host_span<cudf::size_type const> output_column_schemas,
  cudf::host_span<SchemaElement const> schema_tree)
  : _output_dtypes{output_dtypes},
    _output_column_schemas{output_column_schemas},
    _schema_tree{schema_tree}
{
  CUDF_EXPECTS(
    _output_column_schemas.empty() or _output_column_schemas.size() == _output_dtypes.size(),
    "output_column_schemas must have the same size as output_dtypes when provided");
  _literals.resize(static_cast<size_type>(_output_dtypes.size()));
  expr.accept(*this);
}

std::reference_wrapper<ast::expression const> equality_literals_collector::visit(
  ast::literal const& expr)
{
  return expr;
}

std::reference_wrapper<ast::expression const> equality_literals_collector::visit(
  ast::column_reference const& expr)
{
  CUDF_EXPECTS(expr.get_table_source() == ast::table_reference::LEFT,
               "DictionaryAST and BloomfilterAST support only left table");
  CUDF_EXPECTS(expr.get_column_index() < static_cast<cudf::size_type>(_output_dtypes.size()),
               "Column index cannot be more than number of columns in the table");
  return expr;
}

std::reference_wrapper<ast::expression const> equality_literals_collector::visit(
  ast::column_name_reference const& expr)
{
  CUDF_FAIL("Column name reference is not supported in DictionaryAST and BloomfilterAST");
}

std::reference_wrapper<ast::expression const> equality_literals_collector::visit(
  ast::operation const& expr)
{
  using cudf::ast::ast_operator;

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

  if (lhs_kind == operand_kind::COLUMN_REF and rhs_kind == operand_kind::LITERAL) {
    col_ref->accept(*this);
    auto const col_idx = col_ref->get_column_index();
    // Do not collect literals for timestamp columns whose output precision differs from
    // the column's native precision as the literal would never match the native values.
    if (not _output_column_schemas.empty() and cudf::is_timestamp(_output_dtypes[col_idx])) {
      auto const schema_idx = _output_column_schemas[col_idx];
      auto const& schema    = _schema_tree[schema_idx];
      auto const clockrate  = cudf::io::detail::to_clockrate(_output_dtypes[col_idx].id());
      if (schema.logical_type.has_value() and
          calc_timestamp_scale(schema.logical_type, clockrate) != 0) {
        return expr;
      }
    }
    if (op == ast_operator::EQUAL) {
      _literals[col_idx].emplace_back(const_cast<ast::literal*>(literal));
    }
  } else {
    // For all other forms, visit operands to collect any nested literals
    std::ignore = visit_operands(expr.get_operands());
  }
  return expr;
}

std::vector<std::vector<ast::literal*>> equality_literals_collector::get_literals() &&
{
  return std::move(_literals);
}

}  // namespace cudf::io::parquet::detail
