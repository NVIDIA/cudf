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
#include <cudf/utilities/span.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_checks.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/exec_policy.hpp>

#include <cuco/bloom_filter_ref.cuh>
#include <cuda/iterator>
#include <thrust/tabulate.h>

#include <functional>
#include <future>
#include <numeric>
#include <optional>
#include <ranges>
#include <utility>

namespace cudf::io::parquet::detail {
namespace {

/**
 * @brief Policy describing the Apache Arrow Block-Split Bloom Filter, hashing keys with cudf's
 * `XXHash_64` (so that `cudf::string_view` and other cudf types are hashed by content, matching the
 * Apache Parquet/Arrow bloom filter specification).
 *
 * Uses cuco's `parametric_filter_policy` with the Apache Arrow layout: 256-bit blocks (8 x
 * `uint32_t`), 8 fingerprint bits per key, fully horizontal add (Theta=8) and fully vertical
 * contains (Phi=8). This layout is bit-compatible with Apache Arrow, as verified by cuCollections
 * `tests/bloom_filter/arrow_compat_test.cu`.
 *
 * @tparam Key The type of the values to generate a fingerprint for.
 */
template <class Key>
using arrow_filter_policy = cudf::arrow_filter_policy<cudf::hashing::detail::XXHash_64<Key>>;

/**
 * @brief Converts bloom filter membership results (for each column chunk) to a device column.
 *
 */
struct bloom_filter_caster {
  cudf::device_span<cudf::device_span<cuda::std::byte const> const> bloom_filter_spans;
  host_span<Type const> parquet_types;
  std::size_t total_row_groups;
  std::size_t num_equality_columns;

  enum class is_int96_timestamp : bool { YES, NO };

  template <typename T, is_int96_timestamp IS_INT96_TIMESTAMP = is_int96_timestamp::NO>
  std::unique_ptr<cudf::column> query_bloom_filter(cudf::size_type equality_col_idx,
                                                   cudf::data_type dtype,
                                                   ast::literal const* const literal,
                                                   rmm::cuda_stream_view stream) const
    requires(not std::is_same_v<T, bool> and
             not(cudf::is_compound<T>() and not std::is_same_v<T, string_view>))
  {
    using key_type          = T;
    using policy_type       = arrow_filter_policy<key_type>;
    using bloom_filter_type = cuco::
      bloom_filter_ref<key_type, cuco::extent<std::size_t>, cuco::thread_scope_thread, policy_type>;
    using filter_block_type = typename bloom_filter_type::filter_block_type;
    using word_type         = typename policy_type::word_type;

    // Check if the literal has the same type as the predicate column
    CUDF_EXPECTS(
      dtype == literal->get_data_type() and
        cudf::have_same_types(
          cudf::column_view{dtype, 0, {}, {}, 0, 0, {}},
          cudf::scalar_type_t<T>(T{}, false, stream, cudf::get_current_device_resource_ref())),
      "Mismatched predicate column and literal types");

    // Filter properties
    auto constexpr bytes_per_block = sizeof(word_type) * policy_type::words_per_block;

    rmm::device_buffer results{total_row_groups, stream, cudf::get_current_device_resource_ref()};
    cudf::device_span<bool> results_span{static_cast<bool*>(results.data()), total_row_groups};

    // Query literal in bloom filters from each column chunk (row group).
    thrust::tabulate(
      rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
      results_span.begin(),
      results_span.end(),
      [filter_span          = bloom_filter_spans.data(),
       d_scalar             = literal->get_value(),
       col_idx              = equality_col_idx,
       num_equality_columns = num_equality_columns] __device__(auto row_group_idx) {
        // Filter bitset buffer index
        auto const filter_idx  = col_idx + (num_equality_columns * row_group_idx);
        auto const filter_size = filter_span[filter_idx].size();

        // If no bloom filter, then fill in `true` as membership cannot be determined
        if (filter_size == 0) { return true; }

        // Number of filter blocks
        auto const num_filter_blocks = filter_size / bytes_per_block;

        // Create a bloom filter view. `const_cast` is needed because bloom filter view expects a
        // mutable view.
        bloom_filter_type filter{reinterpret_cast<filter_block_type*>(
                                   const_cast<cuda::std::byte*>(filter_span[filter_idx].data())),
                                 num_filter_blocks,
                                 {},   // Thread scope as the same literal is being searched across
                                       // different bitsets per thread
                                 {}};  // Arrow policy with cudf::hashing::detail::XXHash_64 seeded
                                       // with 0 for Arrow compatibility

        // If int96_timestamp type, convert literal to string_view and query bloom
        // filter
        if constexpr (cuda::std::is_same_v<T, cudf::string_view> and
                      IS_INT96_TIMESTAMP == is_int96_timestamp::YES) {
          auto const int128_key = static_cast<__int128_t>(d_scalar.value<int64_t>());
          cudf::string_view probe_key{reinterpret_cast<char const*>(&int128_key), 12};
          return filter.contains(probe_key);
        } else {
          // Query the bloom filter and store results
          return filter.contains(d_scalar.value<T>());
        }
      });

    return std::make_unique<cudf::column>(
      cudf::data_type{cudf::type_id::BOOL8},
      static_cast<cudf::size_type>(total_row_groups),
      std::move(results),
      rmm::device_buffer{0, stream, cudf::get_current_device_resource_ref()},
      0);
  }

  // Creates device columns from bloom filter membership
  template <typename T>
  std::unique_ptr<cudf::column> operator()(cudf::size_type equality_col_idx,
                                           cudf::data_type dtype,
                                           ast::literal const* const literal,
                                           rmm::cuda_stream_view stream) const
  {
    // Boolean, List, Struct, Dictionary types are not supported
    if constexpr (std::is_same_v<T, bool> or
                  (cudf::is_compound<T>() and not std::is_same_v<T, string_view>)) {
      CUDF_FAIL("Bloom filters do not support boolean or compound types");
    } else {
      // For INT96 timestamps, use cudf::string_view type and set is_int96_timestamp to YES
      if constexpr (cudf::is_timestamp<T>()) {
        if (parquet_types[equality_col_idx] == Type::INT96) {
          // For INT96 timestamps, use cudf::string_view type and set is_int96_timestamp to YES
          return query_bloom_filter<cudf::string_view, is_int96_timestamp::YES>(
            equality_col_idx, dtype, literal, stream);
        }
      }

      // For all other cases
      return query_bloom_filter<T>(equality_col_idx, dtype, literal, stream);
    }
  }
};

/**
 * @brief Converts AST expression to bloom filter membership (BloomfilterAST) expression.
 * This is used in row group filtering based on equality predicate.
 */
class bloom_filter_expression_converter : public equality_literals_collector {
 public:
  bloom_filter_expression_converter(
    ast::expression const& expr,
    cudf::host_span<cudf::data_type const> output_dtypes,
    cudf::host_span<std::vector<ast::literal*> const> equality_literals,
    rmm::cuda_stream_view stream)
    : _equality_literals{equality_literals},
      _always_true_scalar{std::make_unique<cudf::numeric_scalar<bool>>(true, true, stream)},
      _always_true{std::make_unique<ast::literal>(*_always_true_scalar)}
  {
    // Set the output data types
    _output_dtypes = output_dtypes;

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

    auto const input_op       = expr.get_operator();
    auto const operator_arity = cudf::ast::detail::ast_operator_arity(input_op);

    // Unary operation
    if (operator_arity == 1) {
      auto visit_operands_fn = [this](auto const& operands) {
        return this->visit_operands(operands);
      };
      return parquet::detail::apply_unary_membership_transform(
        expr, _bloom_filter_expr, *_always_true, *this, visit_operands_fn);
    }

    // Binary operation
    auto const [op, lhs_kind, rhs_kind, col_ref, literal] = extract_binary_operands(expr);

    // Push expressions for `col op lit` or `lit op col` forms
    if (lhs_kind == operand_kind::COLUMN_REF and rhs_kind == operand_kind::LITERAL) {
      col_ref->accept(*this);

      if (op == ast_operator::EQUAL) {
        auto const col_idx            = col_ref->get_column_index();
        auto const& equality_literals = _equality_literals[col_idx];
        auto col_literal_offset       = _col_literals_offsets[col_idx];
        // Skip bloom filter probing for timestamp columns with empty vector of literals due to
        // a timestamp scale mismatch — the literal can never match the native values.
        if (cudf::is_timestamp(_output_dtypes[col_idx]) and equality_literals.empty()) {
          return *_always_true;
        }

        auto const literal_iter =
          std::find(equality_literals.cbegin(), equality_literals.cend(), literal);
        CUDF_EXPECTS(literal_iter != equality_literals.end(),
                     "Bloom filter expression converter encountered an unexpected literal");

        col_literal_offset += std::distance(equality_literals.cbegin(), literal_iter);
        auto const& value = _bloom_filter_expr.push(ast::column_reference{col_literal_offset});
        _bloom_filter_expr.push(ast::operation{ast_operator::IDENTITY, value});
      } else {
        _bloom_filter_expr.push(ast::operation{ast_operator::IDENTITY, *_always_true});
        return *_always_true;
      }
    }  // Visit operands and push expression for `expr op expr` form
    else if (lhs_kind == operand_kind::EXPRESSION and rhs_kind == operand_kind::EXPRESSION) {
      auto new_operands = visit_operands(expr.get_operands());
      _bloom_filter_expr.push(ast::operation{op, new_operands.front(), new_operands.back()});
    }  // Push _always_true for `col op col`, `expr op col`, `expr op lit` forms
    else {
      _bloom_filter_expr.push(ast::operation{ast_operator::IDENTITY, *_always_true});
      return *_always_true;
    }

    return _bloom_filter_expr.back();
  }

  /**
   * @brief Returns the AST to apply on bloom filter membership.
   *
   * @return AST operation expression
   */
  [[nodiscard]] std::reference_wrapper<ast::expression const> get_bloom_filter_expr() const
  {
    return _bloom_filter_expr.back();
  }

 private:
  std::vector<cudf::size_type> _col_literals_offsets;
  cudf::host_span<std::vector<ast::literal*> const> _equality_literals;
  ast::tree _bloom_filter_expr;
  std::unique_ptr<cudf::numeric_scalar<bool>> _always_true_scalar;
  std::unique_ptr<ast::literal> _always_true;
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
  rmm::cuda_stream_view stream,
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
    fetch_bloom_filters_to_device(datasource_refs, bloom_filter_byte_ranges_per_source, stream, mr);

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
  rmm::cuda_stream_view stream) const
{
  // Number of input table columns
  auto const num_input_columns = static_cast<cudf::size_type>(output_dtypes.size());

  // Get parquet types for the predicate columns
  auto const parquet_types = get_parquet_types(input_row_group_indices, bloom_filter_col_schemas);

  // Copy bloom filter bitset spans to device
  auto const device_bloom_filter_data = cudf::detail::make_device_uvector_async(
    bloom_filter_data, stream, cudf::get_current_device_resource_ref());

  // Create a bloom filter query table caster
  bloom_filter_caster const bloom_filter_col{device_bloom_filter_data,
                                             parquet_types,
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
        bloom_filter_membership_columns.emplace_back(cudf::type_dispatcher<dispatch_storage_type>(
          dtype, bloom_filter_col, equality_col_idx, dtype, literal, stream));
      }
      equality_col_idx++;
    });

  // Create a table from columns
  auto bloom_filter_membership_table = cudf::table(std::move(bloom_filter_membership_columns));

  // Convert AST to BloomfilterAST expression with reference to bloom filter membership
  // in above `bloom_filter_membership_table`
  bloom_filter_expression_converter bloom_filter_expr{
    filter.get(), output_dtypes, {literals}, stream};

  // Filter bloom filter membership table with the BloomfilterAST expression and collect
  // filtered row group indices
  return collect_filtered_row_group_indices(bloom_filter_membership_table,
                                            bloom_filter_expr.get_bloom_filter_expr(),
                                            input_row_group_indices,
                                            stream);
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

std::vector<std::reference_wrapper<ast::expression const>>
equality_literals_collector::visit_operands(
  cudf::host_span<std::reference_wrapper<ast::expression const> const> operands)
{
  std::vector<std::reference_wrapper<ast::expression const>> transformed_operands;
  for (auto const& operand : operands) {
    auto const new_operand = operand.get().accept(*this);
    transformed_operands.push_back(new_operand);
  }
  return transformed_operands;
}

}  // namespace cudf::io::parquet::detail
