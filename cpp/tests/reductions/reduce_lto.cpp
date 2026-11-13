/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>

#include <cudf/column/scalar_column_view.hpp>
#include <cudf/detail/utilities/buffer_factories.hpp>
#include <cudf/dictionary/encode.hpp>
#include <cudf/reduction.hpp>
#include <cudf/udf.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_layout.hpp>

#include <rmm/cuda_device.hpp>
#include <rmm/mr/statistics_resource_adaptor.hpp>

#include <cuda/std/tuple>
#include <cuda_runtime_api.h>

#include <cudf_reduce_test_embed.hpp>

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

template <typename T>
using wrapper = cudf::test::fixed_width_column_wrapper<T>;
using cudf::data_type;
using cudf::type_id;

data_type const state_types[]{data_type{type_id::INT64}, data_type{type_id::INT64}};

data_type const output_types[]{state_types[0], state_types[1], data_type{type_id::FLOAT64}};

cudf::lto_fragments lto_fragments()
{
  auto const range = cudf_reduce_test_embed::file_ranges[cudf_reduce_test_embed::fragment];
  auto const bytes = cudf_reduce_test_embed::files.subspan(range[0], range[1]);
  static cudf::lto_fragment const fragments[]{{cudf::lto_binary_type::FATBIN, bytes}};
  return fragments;
}

void expect_sum_count_avg(std::span<std::unique_ptr<cudf::column> const> result,
                          int64_t sum,
                          int64_t count)
{
  ASSERT_EQ(result.size(), std::size(output_types));
  auto const valid   = count != 0;
  auto const average = valid ? static_cast<double>(sum) / count : 0.0;
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result[0], (wrapper<int64_t>{{sum}, {valid}}));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result[1], (wrapper<int64_t>{{count}, {true}}));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result[2], (wrapper<double>{{average}, {valid}}));
}

}  // namespace

struct ReduceUdf : cudf::test::BaseFixture {
  void SetUp() override { CUDF_CUDA_TRY(cudaSetDevice(rmm::get_current_cuda_device().value())); }
};

TEST_F(ReduceUdf, EmptyStateAndOutputs)
{
  cudf::reduce_udf udf{.fragments     = lto_fragments(),
                       .init          = "empty_init",
                       .update        = "empty_update",
                       .merge         = "empty_merge",
                       .finalize      = "empty_finalize",
                       .is_null_aware = cudf::null_aware::NO};
  for (int32_t rows : {0, 100'003}) {
    SCOPED_TRACE(rows);
    std::vector<int32_t> values(rows, 1);
    wrapper<int32_t> input(values.begin(), values.end());
    cudf::reduce_input const inputs[]{input};
    auto result = cudf::reduce(udf, inputs, {}, {});
    EXPECT_TRUE(result.empty());
  }
  udf.finalize = "empty_finalize_error";
  try {
    auto result = cudf::reduce(udf, {}, {}, {});
    FAIL() << "Finalization errors must propagate even without state or outputs";
  } catch (cudf::evaluation_error const& error) {
    EXPECT_EQ(error.error_code(), cudf::errc::DIVISION_BY_ZERO);
  }
}

TEST_F(ReduceUdf, ScalarOnly)
{
  wrapper<int32_t> value{7};
  cudf::reduce_input const inputs[]{cudf::scalar_column_view{value}};
  cudf::reduce_udf const udf{.fragments     = lto_fragments(),
                             .init          = "tpch_q6_sales_init",
                             .update        = "sum_update",
                             .merge         = "tpch_q6_sales_merge",
                             .finalize      = "tpch_q6_sales_finalize",
                             .is_null_aware = cudf::null_aware::YES};

  auto result = cudf::reduce(udf, inputs, state_types, output_types);
  expect_sum_count_avg(result, 0, 0);

  // A filter column determines the row count when update inputs are all broadcast.
  wrapper<bool> keep{true, false, true};
  cudf::reduce_input const filter_inputs[]{keep};
  auto filtered   = udf;
  filtered.filter = "boolean_filter";
  result          = cudf::reduce(filtered, inputs, state_types, output_types, filter_inputs);
  expect_sum_count_avg(result, 14, 2);
}

TEST_F(ReduceUdf, WideOptionalState)
{
  constexpr int32_t fields      = 32;
  constexpr int64_t sum_weights = fields * (fields + 1) / 2;
  std::vector<cudf::data_type> states(fields, cudf::data_type{type_id::INT64});
  cudf::data_type const outputs[]{cudf::data_type{type_id::INT64}};
  cudf::reduce_udf const udf{.fragments     = lto_fragments(),
                             .init          = "wide_init",
                             .update        = "wide_update",
                             .merge         = "wide_merge",
                             .finalize      = "wide_finalize",
                             .is_null_aware = cudf::null_aware::YES};

  for (int32_t rows : {0, 257, 100'003}) {
    SCOPED_TRACE(rows);
    std::vector<int32_t> values(rows, 1);
    wrapper<int32_t> input(values.begin(), values.end());
    cudf::reduce_input const inputs[]{input};
    auto result = cudf::reduce(udf, inputs, states, outputs);
    ASSERT_EQ(result.size(), 1);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result[0], (wrapper<int64_t>{{rows * sum_weights}, {true}}));
  }
}

TEST_F(ReduceUdf, NullHandling)
{
  wrapper<int32_t> values{{10, 99, 5, 15}, {true, false, true, true}};
  wrapper<int32_t> all_null{{10, 20}, {false, false}};

  struct test_case {
    cudf::column_view input;
    int64_t expected_sum;
    int64_t expected_count;
  };

  test_case const cases[]{{values, 30, 3}, {all_null, 0, 0}};

  for (auto const& test : cases) {
    cudf::reduce_input const inputs[]{test.input};
    wrapper<int64_t> expected_raw_sum{{test.expected_sum}, {true}};
    cudf::reduce_udf const udf{.fragments     = lto_fragments(),
                               .init          = "tpch_q6_sales_init",
                               .update        = "sum_update",
                               .merge         = "tpch_q6_sales_merge",
                               .finalize      = "tpch_q6_sales_finalize",
                               .is_null_aware = cudf::null_aware::YES};
    auto result = cudf::reduce(udf, inputs, state_types, output_types);

    expect_sum_count_avg(result, test.expected_sum, test.expected_count);
    auto indexed_filter           = udf;
    indexed_filter.filter_indexed = true;
    result                        = cudf::reduce(indexed_filter, inputs, state_types, output_types);
    expect_sum_count_avg(result, test.expected_sum, test.expected_count);

    // Value-only callbacks skip null rows and preserve their zero identity.
    data_type const types[]{state_types[0]};
    cudf::reduce_udf const raw{.fragments     = lto_fragments(),
                               .init          = "raw_init",
                               .update        = "raw_update",
                               .merge         = "raw_merge",
                               .finalize      = "raw_finalize",
                               .is_null_aware = cudf::null_aware::NO};
    result = cudf::reduce(raw, inputs, types, types);

    ASSERT_EQ(result.size(), std::size(types));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result[0], expected_raw_sum);
  }
}

TEST_F(ReduceUdf, FusedOutput)
{
  // SELECT SUM(amount), COUNT(amount), AVG(amount), SUM(amount) >= 100 FROM input;
  wrapper<int32_t> amount{20, 40, 60};

  constexpr int64_t expected_sum   = 120;
  constexpr int64_t expected_count = 3;

  wrapper<bool> expected_qualifies{{true}, {true}};

  cudf::reduce_input const inputs[]{amount};
  data_type const outputs[]{
    output_types[0], output_types[1], output_types[2], data_type{type_id::BOOL8}};

  cudf::reduce_udf const udf{.fragments     = lto_fragments(),
                             .init          = "tpch_q6_sales_init",
                             .update        = "sum_update",
                             .merge         = "tpch_q6_sales_merge",
                             .finalize      = "fused_finalize",
                             .is_null_aware = cudf::null_aware::YES};

  auto result = cudf::reduce(udf, inputs, state_types, outputs);

  ASSERT_EQ(result.size(), std::size(outputs));
  expect_sum_count_avg(
    std::span{result}.first(std::size(output_types)), expected_sum, expected_count);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result[std::size(output_types)], expected_qualifies);
}

TEST_F(ReduceUdf, FilterPushDownReduction)
{
  wrapper<int32_t> values{10, -99, 20, -99, 30};
  wrapper<bool> keep{{true, false, true, true, true}, {true, true, true, false, true}};

  constexpr int64_t expected_sum   = 60;
  constexpr int64_t expected_count = 3;

  auto packed = cudf::detail::make_device_buffer(
    std::vector<cudf::bitmask_type>{0b10101}, cudf::get_default_stream(), mr());
  cudf::reduce_input const inputs[]{values};

  cudf::reduce_udf const udf{.fragments     = lto_fragments(),
                             .init          = "tpch_q6_sales_init",
                             .update        = "checked_update",
                             .merge         = "tpch_q6_sales_merge",
                             .finalize      = "tpch_q6_sales_finalize",
                             .is_null_aware = cudf::null_aware::YES};

  // Without a selection, negative input values trigger an error in checked_update.
  EXPECT_THROW(cudf::reduce(udf, inputs, state_types, output_types), cudf::evaluation_error);

  // A BOOL8 column supplies the selection; a null predicate excludes its row.
  cudf::reduce_input const boolean_inputs[]{keep};
  cudf::reduce_udf const boolean_udf{.fragments     = lto_fragments(),
                                     .init          = "tpch_q6_sales_init",
                                     .update        = "checked_update",
                                     .merge         = "tpch_q6_sales_merge",
                                     .finalize      = "tpch_q6_sales_finalize",
                                     .filter        = "boolean_filter",
                                     .is_null_aware = cudf::null_aware::YES};

  auto result = cudf::reduce(boolean_udf, inputs, state_types, output_types, boolean_inputs);
  expect_sum_count_avg(result, expected_sum, expected_count);

  // A packed mask has no column type; the filter reads its words through user_data.
  cudf::reduce_input const mask_inputs[]{cudf::user_data{packed.data()}};
  cudf::reduce_udf const mask_udf{.fragments      = lto_fragments(),
                                  .init           = "tpch_q6_sales_init",
                                  .update         = "checked_update",
                                  .merge          = "tpch_q6_sales_merge",
                                  .finalize       = "tpch_q6_sales_finalize",
                                  .filter         = "mask_filter",
                                  .is_null_aware  = cudf::null_aware::YES,
                                  .filter_indexed = true};

  result = cudf::reduce(mask_udf, inputs, state_types, output_types, mask_inputs);
  expect_sum_count_avg(result, expected_sum, expected_count);

  data_type const raw_types[]{state_types[0]};
  cudf::reduce_udf const raw{.fragments      = lto_fragments(),
                             .init           = "raw_init",
                             .update         = "raw_update",
                             .merge          = "raw_merge",
                             .finalize       = "raw_finalize",
                             .filter         = "raw_mask_filter",
                             .is_null_aware  = cudf::null_aware::NO,
                             .filter_indexed = true};
  result = cudf::reduce(raw, inputs, raw_types, raw_types, mask_inputs);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result[0], (wrapper<int64_t>{{expected_sum}, {true}}));
}

TEST_F(ReduceUdf, ErrorHandling)
{
  cudf::reduce_udf const init{.fragments     = lto_fragments(),
                              .init          = "error_init",
                              .update        = "sum_update",
                              .merge         = "tpch_q6_sales_merge",
                              .finalize      = "tpch_q6_sales_finalize",
                              .is_null_aware = cudf::null_aware::YES};
  cudf::reduce_udf const filter{.fragments     = lto_fragments(),
                                .init          = "tpch_q6_sales_init",
                                .update        = "sum_update",
                                .merge         = "tpch_q6_sales_merge",
                                .finalize      = "tpch_q6_sales_finalize",
                                .filter        = "error_filter",
                                .is_null_aware = cudf::null_aware::YES};
  cudf::reduce_udf const update{.fragments     = lto_fragments(),
                                .init          = "tpch_q6_sales_init",
                                .update        = "checked_update",
                                .merge         = "tpch_q6_sales_merge",
                                .finalize      = "tpch_q6_sales_finalize",
                                .is_null_aware = cudf::null_aware::YES};
  cudf::reduce_udf const merge{.fragments     = lto_fragments(),
                               .init          = "tpch_q6_sales_init",
                               .update        = "sum_update",
                               .merge         = "error_merge",
                               .finalize      = "tpch_q6_sales_finalize",
                               .is_null_aware = cudf::null_aware::YES};
  cudf::reduce_udf const finalize{.fragments     = lto_fragments(),
                                  .init          = "tpch_q6_sales_init",
                                  .update        = "sum_update",
                                  .merge         = "tpch_q6_sales_merge",
                                  .finalize      = "error_finalize",
                                  .is_null_aware = cudf::null_aware::YES};

  struct error_case {
    cudf::reduce_udf udf;
    cudf::errc expected_error;
  };

  error_case const cases[]{{init, cudf::errc::ARITHMETIC_OVERFLOW},
                           {filter, cudf::errc::ARITHMETIC_OVERFLOW},
                           {update, cudf::errc::ARITHMETIC_OVERFLOW},
                           {merge, cudf::errc::DIVISION_BY_ZERO},
                           {finalize, cudf::errc::DIVISION_BY_ZERO}};

  std::vector<int32_t> values(100'003, -1);  // Large enough to invoke merge callbacks.
  wrapper<int32_t> input(values.begin(), values.end());

  cudf::reduce_input const inputs[]{input};

  for (auto const& test : cases) {
    try {
      cudf::reduce(test.udf, inputs, state_types, output_types);
      FAIL() << "Expected a callback error";
    } catch (cudf::evaluation_error const& error) {
      EXPECT_EQ(error.error_code(), test.expected_error);
    }
  }
}

TEST_F(ReduceUdf, Assertions)
{
  wrapper<int32_t> values{1, 2, 3};
  wrapper<int32_t> shorter{1, 2};

  cudf::reduce_input const inputs[]{values};
  cudf::reduce_input const mismatched[]{values, shorter};
  wrapper<bool> shorter_mask{true, false};
  cudf::reduce_input const shorter_filter[]{shorter_mask};
  cudf::reduce_udf const udf{.fragments     = lto_fragments(),
                             .init          = "tpch_q6_sales_init",
                             .update        = "sum_update",
                             .merge         = "tpch_q6_sales_merge",
                             .finalize      = "tpch_q6_sales_finalize",
                             .is_null_aware = cudf::null_aware::YES};

  // Input columns must have matching row counts.
  EXPECT_THROW(cudf::reduce(udf, mismatched, state_types, output_types), std::invalid_argument);
  // Supplying filter inputs requires an explicit predicate, even when lengths match.
  EXPECT_THROW(cudf::reduce(udf, inputs, state_types, output_types, inputs), std::invalid_argument);
  auto filtered   = udf;
  filtered.filter = "boolean_filter";
  EXPECT_THROW(cudf::reduce(filtered, inputs, state_types, output_types, shorter_filter),
               std::invalid_argument);
}

TEST_F(ReduceUdf, UnsupportedTypes)
{
  wrapper<int32_t> values{1, 2, 3};
  cudf::reduce_input const inputs[]{values};
  for (auto type : {type_id::EMPTY, type_id::DICTIONARY32, type_id::LIST, type_id::STRUCT}) {
    SCOPED_TRACE(static_cast<int32_t>(type));
    data_type const unsupported[]{data_type{type}};
    cudf::reduce_udf const udf{.fragments     = lto_fragments(),
                               .init          = "tpch_q6_sales_init",
                               .update        = "sum_update",
                               .merge         = "tpch_q6_sales_merge",
                               .finalize      = "tpch_q6_sales_finalize",
                               .is_null_aware = cudf::null_aware::YES};

    // Scalar reductions reject state types other than fixed-width types and strings.
    EXPECT_THROW(cudf::reduce(udf, inputs, unsupported, output_types), std::invalid_argument);

    // Scalar reductions apply the same restriction to output types.
    EXPECT_THROW(cudf::reduce(udf, inputs, state_types, unsupported), std::invalid_argument);
  }
}

TEST_F(ReduceUdf, MultiInput)
{
  // SELECT SUM((amount - discount) * rate), COUNT(...), AVG(...) FROM input;
  wrapper<int32_t> amount{20, 50, 100};
  wrapper<int32_t> discount{{5, 20, 50}, {true, false, true}};
  wrapper<int32_t> rate{3};

  constexpr int64_t expected_sum   = 195;
  constexpr int64_t expected_count = 2;

  cudf::reduce_input const inputs[]{amount, discount, cudf::scalar_column_view{rate}};

  cudf::reduce_udf const udf{.fragments     = lto_fragments(),
                             .init          = "tpch_q6_sales_init",
                             .update        = "discount_update",
                             .merge         = "tpch_q6_sales_merge",
                             .finalize      = "tpch_q6_sales_finalize",
                             .is_null_aware = cudf::null_aware::YES};

  auto result = cudf::reduce(udf, inputs, state_types, output_types);

  expect_sum_count_avg(result, expected_sum, expected_count);
}

TEST_F(ReduceUdf, MultiOutput)
{
  wrapper<int32_t> values{1, 2, 9};

  constexpr int64_t expected_sum   = 12;
  constexpr int64_t expected_count = 3;

  wrapper<int64_t> expected_retained_sum{{expected_sum}, {true}};

  cudf::reduce_input const inputs[]{values};

  cudf::reduce_udf const udf{.fragments     = lto_fragments(),
                             .init          = "tpch_q6_sales_init",
                             .update        = "sum_update",
                             .merge         = "tpch_q6_sales_merge",
                             .finalize      = "tpch_q6_sales_finalize",
                             .is_null_aware = cudf::null_aware::YES};

  auto result = cudf::reduce(udf, inputs, state_types, output_types);

  expect_sum_count_avg(result, expected_sum, expected_count);

  EXPECT_NE(result[0]->view().head(), result[1]->view().head());
  EXPECT_NE(result[0]->view().null_mask(), result[1]->view().null_mask());

  auto sum = std::move(result[0]);

  result.clear();
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*sum, expected_retained_sum);
}

TEST_F(ReduceUdf, DictionaryInput)
{
  wrapper<int32_t> values{{10, 99, 10, 20, -5}, {true, false, true, true, true}};

  constexpr int64_t expected_sum   = 35;
  constexpr int64_t expected_count = 4;

  auto dictionary = cudf::dictionary::encode(values);

  cudf::reduce_input const inputs[]{dictionary->view()};

  cudf::reduce_udf const udf{.fragments     = lto_fragments(),
                             .init          = "tpch_q6_sales_init",
                             .update        = "sum_update",
                             .merge         = "tpch_q6_sales_merge",
                             .finalize      = "tpch_q6_sales_finalize",
                             .is_null_aware = cudf::null_aware::YES};

  auto result = cudf::reduce(udf, inputs, state_types, output_types);

  expect_sum_count_avg(result, expected_sum, expected_count);
}

TEST_F(ReduceUdf, StringReduction)
{
  // SELECT SUM(OCTET_LENGTH(value)), COUNT(value), AVG(OCTET_LENGTH(value)) FROM input;
  cudf::test::strings_column_wrapper strings{{"", "abc", "ignored", "GPU", "\xc3\xa9"},
                                             {true, true, false, true, true}};

  constexpr int64_t expected_sum   = 8;
  constexpr int64_t expected_count = 4;

  auto dictionary = cudf::dictionary::encode(strings);
  cudf::reduce_udf const udf{.fragments     = lto_fragments(),
                             .init          = "tpch_q6_sales_init",
                             .update        = "string_update",
                             .merge         = "tpch_q6_sales_merge",
                             .finalize      = "tpch_q6_sales_finalize",
                             .is_null_aware = cudf::null_aware::YES};

  for (auto input : {cudf::column_view{strings}, dictionary->view()}) {
    cudf::reduce_input const inputs[]{input};

    auto result = cudf::reduce(udf, inputs, state_types, output_types);

    expect_sum_count_avg(result, expected_sum, expected_count);
  }
}

TEST_F(ReduceUdf, StringStateAndOutput)
{
  // SELECT MAX(value), OCTET_LENGTH(MAX(value)) FROM input;
  std::vector<std::string> large_values(100'003, "abc");
  large_values.back() = "zebra";

  struct test_case {
    std::vector<std::string> values;
    std::vector<bool> valid;
    std::string expected_max;
    bool expected_valid;
  };

  test_case const cases[]{
    {{"a", "\xc3\xa9", "", "ignored"}, {true, true, true, false}, "\xc3\xa9", true},
    {{""}, {true}, "", true},
    {{"ignored"}, {false}, "", false},
    {{}, {}, "", false},
    {std::move(large_values), std::vector<bool>(100'003, true), "zebra", true}};

  data_type const states[]{data_type{type_id::STRING}};
  data_type const outputs[]{states[0], data_type{type_id::INT32}};

  for (auto nulls : {cudf::null_aware::YES, cudf::null_aware::NO}) {
    SCOPED_TRACE(nulls == cudf::null_aware::YES ? "optional" : "value-only");
    bool na = nulls == cudf::null_aware::NO;
    cudf::reduce_udf const udf{.fragments = lto_fragments(),
                               .init      = na ? "raw_string_max_init" : "string_max_init",
                               .update    = na ? "raw_string_max_update" : "string_max_update",
                               .merge     = na ? "raw_string_max_merge" : "string_max_merge",
                               .finalize  = na ? "raw_string_max_finalize" : "string_max_finalize",
                               .is_null_aware = nulls};

    for (auto const& test : cases) {
      SCOPED_TRACE(test.values.size());
      auto input = cudf::test::strings_column_wrapper(
                     test.values.begin(), test.values.end(), test.valid.begin())
                     .release();
      auto const valid = test.expected_valid || nulls == cudf::null_aware::NO;
      cudf::test::strings_column_wrapper expected_max({test.expected_max}, {valid});
      wrapper<int32_t> expected_length{{static_cast<int32_t>(test.expected_max.size())}, {valid}};
      cudf::reduce_input const inputs[]{input->view()};

      auto result = cudf::reduce(udf, inputs, states, outputs);
      ASSERT_EQ(result.size(), std::size(outputs));
      if (!test.expected_max.empty()) { EXPECT_NE(result[0]->view().head(), input->view().head()); }
      input.reset();  // The result must own its characters after the input is destroyed.
      CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result[0], expected_max);
      CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result[1], expected_length);
    }
  }
}

TEST_F(ReduceUdf, UserDataInput)
{
  wrapper<int32_t> values{2, 4, 6};
  auto weights = cudf::detail::make_device_buffer(
    std::vector<int64_t>{3, 10, 20, 30}, cudf::get_default_stream(), mr());

  constexpr int64_t expected_sum   = 96;
  constexpr int64_t expected_count = 3;

  cudf::reduce_input const inputs[]{values, cudf::user_data{weights.data()}};

  cudf::reduce_udf const udf{.fragments      = lto_fragments(),
                             .init           = "tpch_q6_sales_init",
                             .update         = "user_data_update",
                             .merge          = "tpch_q6_sales_merge",
                             .finalize       = "tpch_q6_sales_finalize",
                             .is_null_aware  = cudf::null_aware::YES,
                             .update_indexed = true};

  auto result = cudf::reduce(udf, inputs, state_types, output_types);
  expect_sum_count_avg(result, expected_sum, expected_count);

  auto mask = cudf::detail::make_device_buffer(
    std::vector<cudf::bitmask_type>{0b101}, cudf::get_default_stream(), mr());
  cudf::reduce_input const filter_inputs[]{cudf::user_data{mask.data()}};
  auto filtered           = udf;
  filtered.filter         = "mask_filter";
  filtered.filter_indexed = true;
  result = cudf::reduce(filtered, inputs, state_types, output_types, filter_inputs);
  expect_sum_count_avg(result, 64, 2);
}

TEST_F(ReduceUdf, TpchQ6Sales)
{
  // WHERE active AND quantity > 0, then SUM/COUNT/AVG(price * quantity).
  wrapper<int32_t> price{10, 20, 30, 40, 50};
  wrapper<int32_t> quantity{2, 0, 3, -1, 2};
  wrapper<bool> active{true, true, false, true, true};
  constexpr int64_t expected_sum   = 120;
  constexpr int64_t expected_count = 2;

  cudf::reduce_udf const udf{.fragments     = lto_fragments(),
                             .init          = "tpch_q6_sales_init",
                             .update        = "tpch_q6_sales_update",
                             .merge         = "tpch_q6_sales_merge",
                             .finalize      = "tpch_q6_sales_finalize",
                             .is_null_aware = cudf::null_aware::YES};

  cudf::reduce_input const inputs[]{price, quantity, active};

  auto result = cudf::reduce(udf, inputs, state_types, output_types);

  expect_sum_count_avg(result, expected_sum, expected_count);
}

TEST_F(ReduceUdf, CudaSourceTpchQ6SalesNullsAndEmptyInput)
{
  cudf::reduce_udf const udf{.fragments     = lto_fragments(),
                             .init          = "tpch_q6_sales_init",
                             .update        = "tpch_q6_sales_update",
                             .merge         = "tpch_q6_sales_merge",
                             .finalize      = "tpch_q6_sales_finalize",
                             .is_null_aware = cudf::null_aware::YES};

  for (int32_t rows : {0, 3}) {
    // SQL SUM/AVG are NULL and COUNT(expression) is zero for empty or all-null input.

    std::vector<int32_t> prices(rows, 10);
    std::vector<bool> valid(rows, false);
    std::vector<int32_t> quantities(rows, 2);
    std::vector<bool> enabled(rows, true);

    wrapper<int32_t> price(prices.begin(), prices.end(), valid.begin());
    wrapper<int32_t> quantity(quantities.begin(), quantities.end());
    wrapper<bool> active(enabled.begin(), enabled.end());

    constexpr int64_t expected_sum   = 0;
    constexpr int64_t expected_count = 0;

    cudf::reduce_input const inputs[]{price, quantity, active};

    auto result = cudf::reduce(udf, inputs, state_types, output_types);
    expect_sum_count_avg(result, expected_sum, expected_count);
  }
}

TEST_F(ReduceUdf, CudaSourceTpchQ6SalesLargeInput)
{
  // The same fused query over enough rows to require merging partial aggregates.
  constexpr int32_t rows = 100'003;
  std::vector<int32_t> prices(rows, 10);
  std::vector<int32_t> quantities(rows, 2);
  std::vector<bool> enabled(rows, true);

  enabled.back() = false;

  wrapper<int32_t> price(prices.begin(), prices.end());
  wrapper<int32_t> quantity(quantities.begin(), quantities.end());
  wrapper<bool> active(enabled.begin(), enabled.end());

  constexpr int64_t expected_count = rows - 1;
  constexpr int64_t expected_sum   = int64_t{20} * expected_count;

  cudf::reduce_input const inputs[]{price, quantity, active};

  cudf::reduce_udf const udf{.fragments     = lto_fragments(),
                             .init          = "tpch_q6_sales_init",
                             .update        = "tpch_q6_sales_update",
                             .merge         = "tpch_q6_sales_merge",
                             .finalize      = "tpch_q6_sales_finalize",
                             .is_null_aware = cudf::null_aware::YES};

  auto result = cudf::reduce(udf, inputs, state_types, output_types);
  expect_sum_count_avg(result, expected_sum, expected_count);
}
