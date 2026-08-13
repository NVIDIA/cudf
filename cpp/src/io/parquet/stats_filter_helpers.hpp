/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "timestamp_utils.cuh"

#include <cudf/ast/detail/expression_transformer.hpp>
#include <cudf/ast/expressions.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/io/parquet_schema.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>
#include <cudf/utilities/traits.hpp>

#include <bit>
#include <numeric>
#include <span>
#include <string_view>
#include <type_traits>

namespace cudf::io::parquet::detail {

/// Initial capacity for the chars host vector in host_column
constexpr size_t initial_chars_capacity = 1024;

/**
 * @brief Base utilities for converting and casting stats values
 *
 * Derived classes handle row group or page-level statistics as needed.
 *
 */
class stats_caster_base {
 protected:
  template <typename T>
  static inline T decode_fixed_width_decimal(uint8_t const* stats_val, size_t stats_size)
    requires(cudf::is_integral<T>() and !cudf::is_boolean<T>())
  {
    // Decimal statistics with physical BYTE_ARRAY or FIXED_LEN_BYTE_ARRAY are stored as
    // signed two's-complement values using big-endian byte order. The physical width may be
    // smaller than the selected cudf storage width, so sign extend while decoding

    CUDF_EXPECTS(cudf::is_signed<T>(),
                 "FLBA/BYTE_ARRAY decimals must have signed representation types");
    CUDF_EXPECTS(stats_size > 0, "Parquet reader encountered an empty decimal statistics vector");
    CUDF_EXPECTS(stats_size <= sizeof(T),
                 "Parquet reader encountered a statistics vector larger than the type's size");

    // Use std::type_identity to defer and avoid instantiating std::make_unsigned<__int128_t>::type
    // which is not a standard integer type
    // NOLINTNEXTLINE(modernize-type-traits)
    using UnsignedT    = std::conditional_t<std::is_same_v<T, __int128_t>,
                                            std::type_identity<unsigned __int128>,
                                            std::make_unsigned<T>>::type;
    auto const payload = std::span{stats_val, stats_size};
    auto value         = std::accumulate(
      payload.begin(), payload.end(), UnsignedT{0}, [](UnsignedT acc, uint8_t byte) {
        return static_cast<UnsignedT>((acc << CHAR_BIT) | static_cast<UnsignedT>(byte));
      });

    // Check the sign of the first byte to determine if the value is negative
    auto const is_negative_value = std::bit_cast<int8_t>(stats_val[0]) < 0;
    // Sign-extension if negative value and the payload is smaller than the storage type
    if (stats_size < sizeof(T) and is_negative_value) {
      value = static_cast<UnsignedT>(value | (~UnsignedT{0} << (stats_size * CHAR_BIT)));
    }
    return std::bit_cast<T>(value);
  }

  template <typename T>
  static inline T decode_fixed_width_value(uint8_t const* stats_val, size_t stats_size)
    requires((cudf::is_integral<T>() and !cudf::is_boolean<T>()) or cudf::is_chrono<T>())
  {
    CUDF_EXPECTS(stats_size == sizeof(T),
                 "Parquet reader encountered a mismatch in size of statistics vector and the cudf "
                 "storage type");
    auto value = T{};
    std::memcpy(&value, stats_val, std::min(stats_size, sizeof(T)));
    return value;
  }

  template <typename ToType, typename FromType>
  static inline ToType target_type(FromType value, int32_t ts_scale = 0)
  {
    if constexpr (cudf::is_timestamp<ToType>()) {
      if (ts_scale != 0) { value = apply_ts_scale(value, ts_scale); }
      return static_cast<ToType>(
        typename ToType::duration{static_cast<typename ToType::rep>(value)});
    } else if constexpr (std::is_same_v<ToType, string_view>) {
      return ToType{nullptr, 0};
    } else {
      return static_cast<ToType>(value);
    }
  }

  // uses storage type as T
  template <typename T>
  static inline T convert(uint8_t const* stats_val,
                          size_t stats_size,
                          Type const type,
                          int32_t ts_scale = 0)
    requires(cudf::is_dictionary<T>() or cudf::is_nested<T>())
  {
    CUDF_FAIL("Unsupported type for stats casting");
  }

  template <typename T>
  static inline T convert(uint8_t const* stats_val,
                          size_t stats_size,
                          Type const type,
                          int32_t ts_scale = 0)
    requires(cudf::is_boolean<T>())
  {
    CUDF_EXPECTS(type == Type::BOOLEAN, "Invalid type and stats combination");
    return stats_caster_base::target_type<T>(*reinterpret_cast<bool const*>(stats_val));
  }

  // integral but not boolean, and fixed_point, and chrono.
  template <typename T>
  static inline T convert(uint8_t const* stats_val,
                          size_t stats_size,
                          Type const type,
                          int32_t ts_scale = 0)
    requires((cudf::is_integral<T>() and !cudf::is_boolean<T>()) or cudf::is_fixed_point<T>() or
             cudf::is_chrono<T>())
  {
    switch (type) {
      case Type::INT32:
        return stats_caster_base::target_type<T>(
          decode_fixed_width_value<int32_t>(stats_val, stats_size), ts_scale);
      case Type::INT64:
        return stats_caster_base::target_type<T>(
          decode_fixed_width_value<int64_t>(stats_val, stats_size), ts_scale);
      case Type::INT96:  // Deprecated in parquet specification
        return stats_caster_base::target_type<T>(
          static_cast<__int128_t>(decode_fixed_width_value<int64_t>(stats_val, stats_size)) << 32 |
            decode_fixed_width_value<int32_t>(stats_val + sizeof(int64_t), stats_size),
          ts_scale);
      case Type::BYTE_ARRAY: [[fallthrough]];
      case Type::FIXED_LEN_BYTE_ARRAY:
        // Handle chronos, decimals and UUIDs here
        if constexpr (cudf::is_chrono<T>()) {
          // Chrono
          return stats_caster_base::target_type<T>(
            decode_fixed_width_value<typename T::rep>(stats_val, stats_size), ts_scale);
        } else {
          // Decimals (32, 64, 128), 256 bit not yet supported
          if constexpr (cudf::is_fixed_point<T>()) {
            // Using T::rep as raw stats are always decoded as the underlying storage type
            // before conversion
            return stats_caster_base::target_type<T>(
              decode_fixed_width_decimal<typename T::rep>(stats_val, stats_size), ts_scale);
          } else {
            return stats_caster_base::target_type<T>(
              decode_fixed_width_decimal<T>(stats_val, stats_size), ts_scale);
          }
        }
        // TODO(mh): add support for `UUID` (big-endian but no sign extension) here
      default:
        // unsupported type
        CUDF_FAIL("Invalid type and stats combination");
    }
  }

  template <typename T>
  static inline T convert(uint8_t const* stats_val,
                          size_t stats_size,
                          Type const type,
                          int32_t ts_scale = 0)
    requires(cudf::is_floating_point<T>())
  {
    switch (type) {
      case Type::FLOAT:
        return stats_caster_base::target_type<T>(*reinterpret_cast<float const*>(stats_val));
      case Type::DOUBLE:
        return stats_caster_base::target_type<T>(*reinterpret_cast<double const*>(stats_val));
      default: CUDF_FAIL("Invalid type and stats combination");
    }
  }

  template <typename T>
  static inline T convert(uint8_t const* stats_val,
                          size_t stats_size,
                          Type const type,
                          int32_t ts_scale = 0)
    requires(std::is_same_v<T, string_view>)
  {
    switch (type) {
      case Type::BYTE_ARRAY: [[fallthrough]];
      case Type::FIXED_LEN_BYTE_ARRAY:
        return string_view(reinterpret_cast<char const*>(stats_val), stats_size);
      default: CUDF_FAIL("Invalid type and stats combination");
    }
  }

  /**
   * @brief Local struct to hold host columns during stats based filtering
   *
   * @tparam T Type of the column
   */
  template <typename T>
  struct host_column {
    // using thrust::host_vector because std::vector<bool> uses bitmap instead of byte per bool.
    cudf::detail::host_vector<T> val;
    cudf::detail::host_vector<char> chars;
    std::vector<bitmask_type> null_mask;
    cudf::size_type null_count = 0;

    host_column(size_type total_row_groups, rmm::cuda_stream_view stream)
      : val{cudf::detail::make_host_vector<T>(total_row_groups, stream)},
        chars{cudf::detail::make_empty_host_vector<char>(initial_chars_capacity, stream)},
        null_mask(cudf::util::div_rounding_up_safe<cudf::size_type>(
                    cudf::bitmask_allocation_size_bytes(total_row_groups), sizeof(bitmask_type)),
                  ~bitmask_type{0})
    {
    }

    void inline set_index(size_type index,
                          std::optional<std::vector<uint8_t>> const& binary_value,
                          Type const type,
                          int32_t ts_scale = 0)
    {
      if (binary_value.has_value()) {
        // For strings, also insert the characters
        if constexpr (std::is_same_v<T, string_view>) {
          // Create a temporary string view from the binary value
          std::string_view input_value{reinterpret_cast<char const*>(binary_value.value().data()),
                                       binary_value.value().size()};
          // Current offset into the chars vector
          auto const current_chars_offset = chars.size();
          chars.insert(chars.end(), input_value.begin(), input_value.end());
          // Convert to the target type
          val[index] = stats_caster_base::convert<T>(
            reinterpret_cast<uint8_t const*>(chars.data()) + current_chars_offset,
            binary_value.value().size(),
            type);
        } else {
          val[index] = stats_caster_base::convert<T>(
            binary_value.value().data(), binary_value.value().size(), type, ts_scale);
        }
      } else {
        clear_bit_unsafe(null_mask.data(), index);
        null_count++;
      }
    }

    static inline std::tuple<rmm::device_uvector<char>,
                             rmm::device_uvector<size_type>,
                             rmm::device_uvector<size_type>>
    make_strings_children(cudf::host_span<cudf::string_view const> host_strings,
                          cudf::host_span<char const> host_chars,
                          rmm::cuda_stream_view stream,
                          rmm::device_async_resource_ref mr)
    {
      auto offsets =
        cudf::detail::make_empty_host_vector<cudf::size_type>(host_strings.size() + 1, stream);
      auto sizes =
        cudf::detail::make_empty_host_vector<cudf::size_type>(host_strings.size(), stream);
      offsets.push_back(0);
      for (auto const& str : host_strings) {
        offsets.push_back(offsets.back() + str.size_bytes());
        sizes.push_back(str.size_bytes());
      }
      auto d_chars   = cudf::detail::make_device_uvector_async(host_chars, stream, mr);
      auto d_offsets = cudf::detail::make_device_uvector_async(offsets, stream, mr);
      auto d_sizes   = cudf::detail::make_device_uvector_async(sizes, stream, mr);
      stream.synchronize();  // ensures the vectors are not destroyed before the copy is completed
      return {std::move(d_chars), std::move(d_offsets), std::move(d_sizes)};
    }

    [[nodiscard]] std::unique_ptr<column> inline to_device(cudf::data_type dtype,
                                                           rmm::cuda_stream_view stream,
                                                           rmm::device_async_resource_ref mr) const
    {
      if constexpr (std::is_same_v<T, string_view>) {
        auto [d_chars, d_offsets, _] = make_strings_children(val, chars, stream, mr);
        auto null_mask_buffer        = rmm::device_buffer{
          null_mask.data(), cudf::bitmask_allocation_size_bytes(val.size()), stream, mr};
        stream.synchronize();
        return cudf::make_strings_column(
          val.size(),
          std::make_unique<column>(std::move(d_offsets), rmm::device_buffer{0, stream, mr}, 0),
          d_chars.release(),
          null_count,
          std::move(null_mask_buffer));
      }
      auto data             = cudf::detail::make_device_uvector_async(val, stream, mr);
      auto null_mask_buffer = rmm::device_buffer{
        null_mask.data(), cudf::bitmask_allocation_size_bytes(val.size()), stream, mr};
      stream.synchronize();
      return std::make_unique<column>(
        dtype, val.size(), data.release(), std::move(null_mask_buffer), null_count);
    }
  };
};

/**
 * @brief Constructs a boolean mask indicating which input columns can participate in statistics
 * (StatsAST) based filtering
 */
class stats_columns_collector : public ast::detail::expression_transformer {
 public:
  stats_columns_collector() = default;

  stats_columns_collector(ast::expression const& expr, cudf::size_type num_columns);

  /**
   * @copydoc ast::detail::expression_transformer::visit(ast::literal const& )
   */
  std::reference_wrapper<ast::expression const> visit(ast::literal const& expr) override;

  /**
   * @copydoc ast::detail::expression_transformer::visit(ast::column_reference const& )
   */
  std::reference_wrapper<ast::expression const> visit(ast::column_reference const& expr) override;

  /**
   * @copydoc ast::detail::expression_transformer::visit(ast::column_name_reference const& )
   */
  std::reference_wrapper<ast::expression const> visit(
    ast::column_name_reference const& expr) override;

  /**
   * @copydoc ast::detail::expression_transformer::visit(ast::operation const& )
   */
  std::reference_wrapper<ast::expression const> visit(ast::operation const& expr) override;

  /**
   * @brief Return a boolean vector indicating input columns that can participate in stats based
   * filtering
   *
   * @return Boolean vector indicating input columns that can participate in stats based filtering
   */
  std::pair<thrust::host_vector<bool>, bool> get_stats_columns_mask() &&;

 protected:
  std::vector<std::reference_wrapper<ast::expression const>> visit_operands(
    cudf::host_span<std::reference_wrapper<ast::expression const> const> operands);

  size_type _num_columns;

 private:
  thrust::host_vector<bool> _columns_mask;
  bool _has_is_null_operator = false;
};

/**
 * @brief Converts AST expression to StatsAST for comparing with column statistics
 *
 * This is used in row group filtering based on predicate.
 * statistics min value of a column is referenced by column_index*3
 * statistics max value of a column is referenced by column_index*3+1
 * statistics is_null value of a column is referenced by column_index*3+2
 */
class stats_expression_converter : public stats_columns_collector {
 public:
  stats_expression_converter(ast::expression const& expr,
                             size_type num_columns,
                             bool has_is_null_operator,
                             rmm::cuda_stream_view stream);

  // Bring all overrides of `visit` from stats_columns_collector into scope
  using stats_columns_collector::visit;

  /**
   * @copydoc ast::detail::expression_transformer::visit(ast::operation const& )
   */
  std::reference_wrapper<ast::expression const> visit(ast::operation const& expr) override;

  /**
   * @brief Returns the AST to apply on Column chunk statistics.
   *
   * @return AST operation expression
   */
  [[nodiscard]] std::reference_wrapper<ast::expression const> get_stats_expr() const;

  /**
   * @brief Delete stats columns mask getter as it's not needed in the derived class
   */
  thrust::host_vector<bool> get_stats_columns_mask() && = delete;

 private:
  ast::tree _stats_expr;
  cudf::size_type _stats_cols_per_column;
  std::unique_ptr<cudf::numeric_scalar<bool>> _always_true_scalar;
  std::unique_ptr<ast::literal> _always_true;
};

}  // namespace cudf::io::parquet::detail
