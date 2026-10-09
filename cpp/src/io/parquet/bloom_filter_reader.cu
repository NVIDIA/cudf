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

#include <rmm/exec_policy.hpp>

#include <cuco/bloom_filter_ref.cuh>
#include <cuda/buffer>
#include <cuda/iterator>
#include <cuda/std/bit>
#include <cuda/std/chrono>
#include <cuda/stream>
#include <thrust/transform.h>
#include <thrust/uninitialized_fill.h>

#include <algorithm>
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

/**
 * @brief Type hashed into the bloom filter. INT32, INT64, FLOAT and DOUBLE values are hashed as
 * their physical type, while INT96, BYTE_ARRAY and FIXED_LEN_BYTE_ARRAY values are hashed as
 * cudf::string_view.
 */
template <Type physical_type>
constexpr auto bloom_filter_key_type()
{
  if constexpr (physical_type == Type::INT32) {
    return cuda::std::type_identity<int32_t>{};
  } else if constexpr (physical_type == Type::INT64) {
    return cuda::std::type_identity<int64_t>{};
  } else if constexpr (physical_type == Type::FLOAT) {
    return cuda::std::type_identity<float>{};
  } else if constexpr (physical_type == Type::DOUBLE) {
    return cuda::std::type_identity<double>{};
  } else {
    return cuda::std::type_identity<cudf::string_view>{};
  }
}

template <Type physical_type>
using bloom_filter_key = typename decltype(bloom_filter_key_type<physical_type>())::type;

/**
 * @brief Probes a bloom filter for the literal encoded as `physical_type`
 *
 * @tparam Rep Type the literal is stored as
 * @tparam physical_type Parquet physical type of the column
 * @tparam BloomFilter Type of the bloom filter view
 *
 * @param filter Bloom filter view of a column chunk
 * @param literal Literal to probe for
 * @param type_length Byte length of a FIXED_LEN_BYTE_ARRAY column
 * @return Whether the literal may be present in the column chunk
 */
template <typename Rep, Type physical_type, typename BloomFilter>
__device__ bool probe_key(BloomFilter const& filter,
                          ast::generic_scalar_device_view const& literal,
                          int32_t type_length)
{
  // +0.0 and -0.0 compare equal but hash differently, so a zero literal probes both
  if constexpr (physical_type == Type::FLOAT or physical_type == Type::DOUBLE) {
    auto const value = literal.value<Rep>();
    if (value == Rep{0}) { return filter.contains(Rep{0}) or filter.contains(-Rep{0}); }
    return filter.contains(value);
  }

  // INT96 is the nanoseconds since midnight (8 bytes) followed by the Julian day (4 bytes)
  else if constexpr (physical_type == Type::INT96) {
    using namespace cuda::std::chrono;
    auto const nanos            = nanoseconds{literal.value<Rep>()};
    auto const days_since_epoch = floor<days>(nanos);
    int64_t const time_of_day   = (nanos - days_since_epoch).count();
    uint32_t const julian_day   = days_since_epoch.count() + julian_day_unix_epoch;
    char bytes[12];
    cuda::std::memcpy(bytes, &time_of_day, sizeof(time_of_day));
    cuda::std::memcpy(bytes + sizeof(time_of_day), &julian_day, sizeof(julian_day));
    return filter.contains(cudf::string_view{bytes, 12});
  }

  // Decimals are big-endian two's complement, sign-extended to the column's `type_length`,
  // so the value is the trailing `type_length` bytes of the byte-swapped 128-bit integer
  else if constexpr (physical_type == Type::FIXED_LEN_BYTE_ARRAY) {
    auto const big_endian = cuda::std::byteswap(static_cast<__int128_t>(literal.value<Rep>()));
    auto const bytes      = reinterpret_cast<char const*>(&big_endian);
    return filter.contains(
      cudf::string_view{bytes + sizeof(__int128_t) - type_length, type_length});
  }

  // INT32, INT64 and BYTE_ARRAY
  else {
    return filter.contains(static_cast<typename BloomFilter::key_type>(literal.value<Rep>()));
  }
}

/**
 * @brief Converts bloom filter membership results (for each column chunk) to a device column.
 *
 */
struct bloom_filter_caster {
  cudf::device_span<cudf::device_span<cuda::std::byte const> const> bloom_filter_spans;
  std::span<Type const> parquet_types;
  std::span<int32_t const> parquet_type_lengths;
  std::size_t total_row_groups;
  std::size_t num_equality_columns;

  /**
   * @brief Queries the bloom filter of each row group for the literal encoded as `physical_type`
   *
   * @tparam Rep Type the literal is stored as
   * @tparam physical_type Parquet physical type of the column
   */
  template <typename Rep, Type physical_type>
  std::unique_ptr<cudf::column> query_bloom_filter(cudf::size_type equality_col_idx,
                                                   ast::generic_scalar_device_view literal,
                                                   cuda::stream_ref stream,
                                                   cudf::memory_resources mr) const
  {
    using key_type = bloom_filter_key<physical_type>;

    using policy_type       = arrow_filter_policy<key_type>;
    using bloom_filter_type = cuco::
      bloom_filter_ref<key_type, cuco::extent<std::size_t>, cuco::thread_scope_thread, policy_type>;
    using filter_block_type = typename bloom_filter_type::filter_block_type;
    using word_type         = typename policy_type::word_type;

    auto results = rmm::device_uvector<bool>{total_row_groups, stream, mr.get_output_mr()};

    // Filter properties
    auto constexpr bytes_per_block = sizeof(word_type) * policy_type::words_per_block;

    // Query literal in bloom filters from each column chunk (row group).
    thrust::transform(rmm::exec_policy_nosync(stream, mr.get_temporary_mr()),
                      cuda::counting_iterator<std::size_t>{0},
                      cuda::counting_iterator{total_row_groups},
                      results.begin(),
                      [filter_span          = bloom_filter_spans.data(),
                       literal              = literal,
                       type_length          = parquet_type_lengths[equality_col_idx],
                       col_idx              = equality_col_idx,
                       num_equality_columns = num_equality_columns] __device__(auto row_group_idx) {
                        // Filter bitset buffer index
                        auto const filter_idx  = col_idx + (num_equality_columns * row_group_idx);
                        auto const filter_size = filter_span[filter_idx].size();

                        // If no bloom filter, then fill in `true` as membership cannot be
                        // determined
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
                          {}};  // Arrow policy with XXHash_64 seeded
                                // with 0 for Arrow compatibility

                        return probe_key<Rep, physical_type>(filter, literal, type_length);
                      });

    return std::make_unique<cudf::column>(
      std::move(results),
      cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED, stream, mr.get_output_mr()),
      0);
  }

  // Booleans and compound types are not supported
  template <typename T>
  std::unique_ptr<cudf::column> operator()(cudf::size_type,
                                           cudf::data_type,
                                           ast::literal const* const,
                                           cuda::stream_ref,
                                           cudf::memory_resources) const
  {
    CUDF_UNREACHABLE("Bloom filters cannot be queried for boolean or compound types");
  }

  // BYTE_ARRAYS are probed as their bytes
  template <typename T>
    requires(cuda::std::is_same_v<T, cudf::string_view>)
  std::unique_ptr<cudf::column> operator()(cudf::size_type equality_col_idx,
                                           cudf::data_type,
                                           ast::literal const* const literal,
                                           cuda::stream_ref stream,
                                           cudf::memory_resources mr) const
  {
    return query_bloom_filter<T, Type::BYTE_ARRAY>(
      equality_col_idx, literal->get_value(), stream, mr);
  }

  // Floating point types are probed as their physical type
  template <typename T>
    requires(cudf::is_floating_point<T>())
  std::unique_ptr<cudf::column> operator()(cudf::size_type equality_col_idx,
                                           cudf::data_type,
                                           ast::literal const* const literal,
                                           cuda::stream_ref stream,
                                           cudf::memory_resources mr) const
  {
    constexpr auto physical_type = cuda::std::is_same_v<T, float> ? Type::FLOAT : Type::DOUBLE;
    return query_bloom_filter<T, physical_type>(equality_col_idx, literal->get_value(), stream, mr);
  }

  // Integers, decimal storage types and chrono types are probed as their physical type
  template <typename T>
    requires(cudf::is_integral_not_bool<T>() or cudf::is_chrono<T>())
  std::unique_ptr<cudf::column> operator()(cudf::size_type equality_col_idx,
                                           cudf::data_type dtype,
                                           ast::literal const* const literal,
                                           cuda::stream_ref stream,
                                           cudf::memory_resources mr) const
  {
    using rep_type = typename cuda::std::
      conditional_t<cudf::is_chrono<T>(), T, cuda::std::chrono::duration<T>>::rep;

    auto const physical_type = parquet_types[equality_col_idx];
    auto const d_literal     = literal->get_value();

    // INT32 also stores 8 and 16-bit integers, and TIME(MILLIS) which is read as a 64-bit duration
    if constexpr (sizeof(rep_type) <= sizeof(int32_t) or cudf::is_duration<T>()) {
      if (physical_type == Type::INT32) {
        return query_bloom_filter<rep_type, Type::INT32>(equality_col_idx, d_literal, stream, mr);
      }
    }
    // INT64 values are stored as 64-bit integers
    if constexpr (sizeof(rep_type) == sizeof(int64_t)) {
      if (physical_type == Type::INT64) {
        return query_bloom_filter<rep_type, Type::INT64>(equality_col_idx, d_literal, stream, mr);
      }
    }
    // INT96 values have nanosecond precision, so a coarser literal cannot be encoded as one
    if constexpr (cuda::std::is_same_v<T, cudf::timestamp_ns>) {
      if (physical_type == Type::INT96) {
        return query_bloom_filter<rep_type, Type::INT96>(equality_col_idx, d_literal, stream, mr);
      }
    }
    // Decimals are probed as their storage type
    if constexpr (cuda::std::is_same_v<T, cudf::device_storage_type_t<numeric::decimal32>> or
                  cuda::std::is_same_v<T, cudf::device_storage_type_t<numeric::decimal64>> or
                  cuda::std::is_same_v<T, cudf::device_storage_type_t<numeric::decimal128>>) {
      if (cudf::is_fixed_point(dtype) and physical_type == Type::FIXED_LEN_BYTE_ARRAY) {
        auto const type_len = parquet_type_lengths[equality_col_idx];
        CUDF_EXPECTS(type_len > 0 and cuda::std::cmp_less_equal(
                                        type_len, static_cast<int32_t>(sizeof(__int128_t))),
                     "Invalid type length for decimal type",
                     std::invalid_argument);
        return query_bloom_filter<T, Type::FIXED_LEN_BYTE_ARRAY>(
          equality_col_idx, d_literal, stream, mr);
      }
    }

    // Decimals stored as BYTE_ARRAY and INT96 read as a coarser timestamp cannot be queried
    auto true_scalar = cudf::numeric_scalar<bool>(true, true, stream, mr.get_temporary_mr());
    return cudf::make_column_from_scalar(true_scalar, total_row_groups, stream, mr.get_output_mr());
  }
};

/**
 * @brief Whether a bloom filter can be queried for a col equal literal predicate
 *
 * @throws cudf::logic_error if the literal's type differs from the column's
 *
 * @param column_type Output type of the column
 * @param literal Literal compared against the column
 * @return Whether the bloom filter can be queried
 */
[[nodiscard]] bool is_bloom_filterable(cudf::data_type column_type, ast::literal const& literal)
{
  CUDF_EXPECTS(column_type.id() == literal.get_data_type().id(),
               "Mismatched predicate column and literal types");
  // Booleans and non-comparable compound types cannot be queried
  if (column_type.id() == cudf::type_id::BOOL8 or
      (cudf::is_compound(column_type) and column_type.id() != cudf::type_id::STRING)) {
    return false;
  }
  // A decimal literal with a different scale cannot be queried
  return not cudf::is_fixed_point(column_type) or
         column_type.scale() == literal.get_data_type().scale();
}

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

    auto const literal_iter =
      std::find(equality_literals.cbegin(), equality_literals.cend(), &literal);

    // Skip bloom filter probing for literals not collected by the equality literals collector
    if (literal_iter == equality_literals.cend()) { return std::nullopt; }

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

std::pair<std::vector<cuda::device_buffer<uint8_t>>,
          std::vector<cudf::device_span<cuda::std::byte const>>>
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
  std::ranges::transform(bloom_filter_col_schemas,
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

      // Add a column for all literals associated with an equality column
      for (auto const& literal : literals[input_col_idx]) {
        // Non-bloom-filterable literals are not collected by `equality_literals_collector`
        CUDF_EXPECTS(is_bloom_filterable(dtype, *literal),
                     "Bloom filters cannot be queried for the predicate column and literal");
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
  std::span<cudf::data_type const> output_dtypes,
  std::span<cudf::size_type const> output_column_schemas,
  std::span<SchemaElement const> schema_tree)
  : parquet_expression_simplifier{output_dtypes},
    _output_column_schemas{output_column_schemas},
    _schema_tree{schema_tree}
{
  CUDF_EXPECTS(
    _output_column_schemas.empty() or _output_column_schemas.size() == output_dtypes.size(),
    "output_column_schemas must have the same size as output_dtypes when provided");
  _literals.resize(static_cast<size_type>(output_dtypes.size()));
}

equality_literals_collector::equality_literals_collector(
  ast::expression const& expr,
  std::span<cudf::data_type const> output_dtypes,
  std::span<cudf::size_type const> output_column_schemas,
  std::span<SchemaElement const> schema_tree)
  : equality_literals_collector{output_dtypes, output_column_schemas, schema_tree}
{
  collect(expr);
}

void equality_literals_collector::collect(ast::expression const& expr)
{
  _can_filter = simplify_expr(expr).has_value();
}

bool equality_literals_collector::can_filter() const { return _can_filter; }

simplified_expression_opt equality_literals_collector::simplify_comparison(
  ast::ast_operator op, ast::column_reference const& col_ref, ast::literal const& literal)
{
  if (op != ast::ast_operator::EQUAL) { return std::nullopt; }

  auto const col_idx = col_ref.get_column_index();

  // Do not collect literals for timestamp columns whose output precision differs from the column's
  // native precision as the literal would never match the native values.
  if (not _output_column_schemas.empty() and cudf::is_timestamp(_output_dtypes[col_idx])) {
    auto const schema_idx = _output_column_schemas[col_idx];
    auto const& schema    = _schema_tree[schema_idx];
    auto const clockrate  = cudf::io::detail::to_clockrate(_output_dtypes[col_idx].id());
    if (schema.logical_type.has_value() and
        calc_timestamp_scale(schema.logical_type, clockrate) != 0) {
      return std::nullopt;
    }
  }

  // Do not collect non-bloom-filterable literals
  if (not is_bloom_filterable(_output_dtypes[col_idx], literal)) { return std::nullopt; }

  _literals[col_idx].emplace_back(const_cast<ast::literal*>(&literal));
  return placeholder_expr();
}

std::vector<std::vector<ast::literal*>> equality_literals_collector::get_literals() &&
{
  return std::move(_literals);
}

}  // namespace cudf::io::parquet::detail
