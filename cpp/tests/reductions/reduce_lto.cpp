/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>

#include <cudf/column/scalar_column_view.hpp>
#include <cudf/detail/utilities/buffer_factories.hpp>
#include <cudf/detail/utilities/layout.hpp>
#include <cudf/dictionary/encode.hpp>
#include <cudf/reduction.hpp>
#include <cudf/udf.hpp>
#include <cudf/utilities/error.hpp>

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
#include <string_view>
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
  static cudf::lto_fragment const fragments[]{
    {cudf::lto_binary_type::FATBIN, cudf_reduce_test_embed::get(cudf_reduce_test_embed::fragment)}};
  return fragments;
}

cudf::reduce_udf sum_count_avg_udf(std::string_view update)
{
  return {.fragments     = lto_fragments(),
          .init          = "sum_count_avg_init",
          .update        = update,
          .merge         = "sum_count_avg_merge",
          .finalize      = "sum_count_avg_finalize",
          .is_null_aware = cudf::null_aware::YES};
}

cudf::reduce_udf segmented_tpch_q6_sales_udf()
{
  return {.fragments        = lto_fragments(),
          .init             = "tpch_q6_sales_segment_init",
          .update           = "tpch_q6_sales_segment_update",
          .merge            = "tpch_q6_sales_segment_merge",
          .finalize         = "tpch_q6_sales_segment_finalize",
          .is_null_aware    = cudf::null_aware::YES,
          .indexed_update   = true,
          .indexed_init     = true,
          .indexed_merge    = true,
          .indexed_finalize = true};
}

void expect_sum_count_avg(std::span<std::unique_ptr<cudf::column> const> result,
                          int64_t sum,
                          int64_t count)
{
  ASSERT_EQ(result.size(), std::size(output_types));
  auto valid   = count != 0;
  auto average = valid ? static_cast<double>(sum) / count : 0.0;
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
    cudf::reduce_input inputs[]{input};
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
  EXPECT_NO_THROW(cudf::reduce(udf, {}, {}, {}, {}, cudf::fallible::NO));
}

TEST_F(ReduceUdf, ScalarOnly)
{
  wrapper<int32_t> value{7};
  cudf::reduce_input inputs[]{cudf::scalar_column_view{value}};
  auto udf = sum_count_avg_udf("sum_update");

  auto result = cudf::reduce(udf, inputs, state_types, output_types);
  expect_sum_count_avg(result, 0, 0);

  // A filter column determines the row count when update inputs are all broadcast.
  wrapper<bool> keep{true, false, true};
  cudf::reduce_input filter_inputs[]{keep};
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
  cudf::data_type outputs[]{cudf::data_type{type_id::INT64}};
  cudf::reduce_udf udf{.fragments     = lto_fragments(),
                       .init          = "wide_init",
                       .update        = "wide_update",
                       .merge         = "wide_merge",
                       .finalize      = "wide_finalize",
                       .is_null_aware = cudf::null_aware::YES};

  for (int32_t rows : {0, 257, 100'003}) {
    SCOPED_TRACE(rows);
    std::vector<int32_t> values(rows, 1);
    wrapper<int32_t> input(values.begin(), values.end());
    cudf::reduce_input inputs[]{input};
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

  for (auto& test : cases) {
    cudf::reduce_input inputs[]{test.input};
    wrapper<int64_t> expected_raw_sum{test.expected_sum};
    auto udf    = sum_count_avg_udf("sum_update");
    auto result = cudf::reduce(udf, inputs, state_types, output_types);

    expect_sum_count_avg(result, test.expected_sum, test.expected_count);
    result = cudf::reduce(udf, inputs, state_types, output_types, {}, cudf::fallible::NO);
    expect_sum_count_avg(result, test.expected_sum, test.expected_count);
    auto indexed_filter           = udf;
    indexed_filter.indexed_filter = true;
    result                        = cudf::reduce(indexed_filter, inputs, state_types, output_types);
    expect_sum_count_avg(result, test.expected_sum, test.expected_count);

    // Value-only callbacks skip null rows and preserve their zero identity.
    data_type types[]{state_types[0]};
    cudf::reduce_udf raw{.fragments     = lto_fragments(),
                         .init          = "plain_sum_init",
                         .update        = "plain_sum_update",
                         .merge         = "plain_sum_merge",
                         .finalize      = "plain_sum_finalize",
                         .is_null_aware = cudf::null_aware::NO};
    result = cudf::reduce(raw, inputs, types, types);

    ASSERT_EQ(result.size(), std::size(types));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result[0], expected_raw_sum);
    result = cudf::reduce(raw, inputs, types, types, {}, cudf::fallible::NO);
    EXPECT_FALSE(result[0]->nullable());
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result[0], expected_raw_sum);
  }
}

TEST_F(ReduceUdf, FusedOutput)
{
  // SELECT SUM(amount), COUNT(amount), AVG(amount), SUM(amount) >= 100 FROM input;
  wrapper<int32_t> amount{20, 40, 60};

  wrapper<bool> expected_qualifies{{true}, {true}};

  cudf::reduce_input inputs[]{amount};
  data_type outputs[]{output_types[0], output_types[1], output_types[2], data_type{type_id::BOOL8}};

  auto udf = sum_count_avg_udf("sum_update");

  udf.finalize = "fused_finalize";

  auto result = cudf::reduce(udf, inputs, state_types, outputs);

  ASSERT_EQ(result.size(), std::size(outputs));
  expect_sum_count_avg(std::span{result}.first(std::size(output_types)), 120, 3);
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
  cudf::reduce_input inputs[]{values};

  auto udf = sum_count_avg_udf("nonnegative_sum_update");

  // Without a selection, negative input values trigger an error in nonnegative_sum_update.
  EXPECT_THROW(cudf::reduce(udf, inputs, state_types, output_types), cudf::evaluation_error);

  // A BOOL8 column supplies the selection; a null predicate excludes its row.
  cudf::reduce_input boolean_inputs[]{keep};
  auto boolean_udf   = sum_count_avg_udf("nonnegative_sum_update");
  boolean_udf.filter = "boolean_filter";

  auto result = cudf::reduce(boolean_udf, inputs, state_types, output_types, boolean_inputs);
  expect_sum_count_avg(result, expected_sum, expected_count);

  // A packed mask has no column type; the filter reads its words through user_data.
  cudf::reduce_input mask_inputs[]{cudf::user_data{packed.data()}};
  auto mask_udf           = sum_count_avg_udf("nonnegative_sum_update");
  mask_udf.filter         = "mask_filter";
  mask_udf.indexed_filter = true;

  result = cudf::reduce(mask_udf, inputs, state_types, output_types, mask_inputs);
  expect_sum_count_avg(result, expected_sum, expected_count);

  data_type raw_types[]{state_types[0]};
  cudf::reduce_udf raw{.fragments      = lto_fragments(),
                       .init           = "plain_sum_init",
                       .update         = "plain_sum_update",
                       .merge          = "plain_sum_merge",
                       .finalize       = "plain_sum_finalize",
                       .filter         = "raw_mask_filter",
                       .is_null_aware  = cudf::null_aware::NO,
                       .indexed_filter = true};
  result = cudf::reduce(raw, inputs, raw_types, raw_types, mask_inputs);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result[0], (wrapper<int64_t>{expected_sum}));
}

TEST_F(ReduceUdf, ErrorHandling)
{
  struct error_case {
    std::string_view cudf::reduce_udf::* callback;
    std::string_view symbol;
    cudf::errc expected_error;
  };

  error_case cases[]{
    {&cudf::reduce_udf::init, "error_init", cudf::errc::ARITHMETIC_OVERFLOW},
    {&cudf::reduce_udf::filter, "error_filter", cudf::errc::ARITHMETIC_OVERFLOW},
    {&cudf::reduce_udf::update, "nonnegative_sum_update", cudf::errc::ARITHMETIC_OVERFLOW},
    {&cudf::reduce_udf::merge, "error_merge", cudf::errc::DIVISION_BY_ZERO},
    {&cudf::reduce_udf::finalize, "error_finalize", cudf::errc::DIVISION_BY_ZERO}};

  std::vector<int32_t> values(100'003, -1);  // Large enough to invoke merge callbacks.
  wrapper<int32_t> input(values.begin(), values.end());

  cudf::reduce_input inputs[]{input};

  for (auto& test : cases) {
    SCOPED_TRACE(test.symbol);
    auto udf           = sum_count_avg_udf("sum_update");
    udf.*test.callback = test.symbol;
    try {
      cudf::reduce(udf, inputs, state_types, output_types);
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

  cudf::reduce_input inputs[]{values};
  cudf::reduce_input mismatched[]{values, shorter};
  wrapper<bool> shorter_mask{true, false};
  cudf::reduce_input shorter_filter[]{shorter_mask};
  auto udf = sum_count_avg_udf("sum_update");

  // Input columns must have matching row counts.
  EXPECT_THROW(cudf::reduce(udf, mismatched, state_types, output_types), std::invalid_argument);
  // Supplying filter inputs requires an explicit predicate, even when lengths match.
  EXPECT_THROW(cudf::reduce(udf, inputs, state_types, output_types, inputs), std::invalid_argument);
  auto filtered   = udf;
  filtered.filter = "boolean_filter";
  EXPECT_THROW(cudf::reduce(filtered, inputs, state_types, output_types, shorter_filter),
               std::invalid_argument);

  auto segmented   = segmented_tpch_q6_sales_udf();
  segmented.update = "sum_segment_update";
  EXPECT_THROW(
    cudf::segmented_reduce(segmented, inputs, {}, state_types, output_types, std::nullopt),
    std::invalid_argument);
  auto offsets =
    cudf::detail::make_device_buffer(std::vector<int32_t>{0, 3}, cudf::get_default_stream(), mr());
  EXPECT_THROW(cudf::segmented_reduce(segmented, inputs, offsets, state_types, output_types, -1),
               std::invalid_argument);
}

TEST_F(ReduceUdf, UnsupportedTypes)
{
  wrapper<int32_t> values{1, 2, 3};
  cudf::reduce_input inputs[]{values};
  for (auto type : {type_id::EMPTY, type_id::DICTIONARY32, type_id::LIST, type_id::STRUCT}) {
    SCOPED_TRACE(static_cast<int32_t>(type));
    data_type unsupported[]{data_type{type}};
    auto udf = sum_count_avg_udf("sum_update");

    // Scalar reductions reject state types other than fixed-width types and strings.
    EXPECT_THROW(cudf::reduce(udf, inputs, unsupported, output_types), std::invalid_argument);

    // Scalar reductions apply the same restriction to output types.
    EXPECT_THROW(cudf::reduce(udf, inputs, state_types, unsupported), std::invalid_argument);

    auto segmented = segmented_tpch_q6_sales_udf();
    auto offsets   = cudf::detail::make_device_buffer(
      std::vector<int32_t>{0, 3}, cudf::get_default_stream(), mr());
    EXPECT_THROW(
      cudf::segmented_reduce(segmented, inputs, offsets, unsupported, output_types, std::nullopt),
      std::invalid_argument);
    EXPECT_THROW(
      cudf::segmented_reduce(segmented, inputs, offsets, state_types, unsupported, std::nullopt),
      std::invalid_argument);
  }
}

TEST_F(ReduceUdf, MultiInput)
{
  // SELECT SUM((amount - discount) * rate), COUNT(...), AVG(...) FROM input;
  wrapper<int32_t> amount{20, 50, 100};
  wrapper<int32_t> discount{{5, 20, 50}, {true, false, true}};
  wrapper<int32_t> rate{3};

  cudf::reduce_input inputs[]{amount, discount, cudf::scalar_column_view{rate}};

  auto udf = sum_count_avg_udf("discount_update");

  auto result = cudf::reduce(udf, inputs, state_types, output_types);

  expect_sum_count_avg(result, 195, 2);
}

TEST_F(ReduceUdf, MultiOutput)
{
  wrapper<int32_t> values{1, 2, 9};

  constexpr int64_t expected_sum = 12;

  wrapper<int64_t> expected_retained_sum{{expected_sum}, {true}};

  cudf::reduce_input inputs[]{values};

  auto udf = sum_count_avg_udf("sum_update");

  auto result = cudf::reduce(udf, inputs, state_types, output_types);

  expect_sum_count_avg(result, expected_sum, 3);

  EXPECT_NE(result[0]->view().head(), result[1]->view().head());
  EXPECT_NE(result[0]->view().null_mask(), result[1]->view().null_mask());

  auto sum = std::move(result[0]);

  result.clear();
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*sum, expected_retained_sum);
}

TEST_F(ReduceUdf, DictionaryInput)
{
  wrapper<int32_t> values{{10, 99, 10, 20, -5}, {true, false, true, true, true}};

  auto dictionary = cudf::dictionary::encode(values);

  cudf::reduce_input inputs[]{dictionary->view()};

  auto udf = sum_count_avg_udf("sum_update");

  auto result = cudf::reduce(udf, inputs, state_types, output_types);

  expect_sum_count_avg(result, 35, 4);
}

TEST_F(ReduceUdf, StringReduction)
{
  // SELECT SUM(OCTET_LENGTH(value)), COUNT(value), AVG(OCTET_LENGTH(value)) FROM input;
  cudf::test::strings_column_wrapper strings{{"", "abc", "ignored", "GPU", "\xc3\xa9"},
                                             {true, true, false, true, true}};

  auto dictionary = cudf::dictionary::encode(strings);
  auto udf        = sum_count_avg_udf("string_length_update");

  for (auto input : {cudf::column_view{strings}, dictionary->view()}) {
    cudf::reduce_input inputs[]{input};

    auto result = cudf::reduce(udf, inputs, state_types, output_types);

    expect_sum_count_avg(result, 8, 4);
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

  test_case cases[]{{{"a", "\xc3\xa9", "", "ignored"}, {true, true, true, false}, "\xc3\xa9", true},
                    {{""}, {true}, "", true},
                    {{"ignored"}, {false}, "", false},
                    {{}, {}, "", false},
                    {std::move(large_values), std::vector<bool>(100'003, true), "zebra", true}};

  data_type states[]{data_type{type_id::STRING}};
  data_type outputs[]{states[0], data_type{type_id::INT32}};

  for (auto nulls : {cudf::null_aware::YES, cudf::null_aware::NO}) {
    SCOPED_TRACE(nulls == cudf::null_aware::YES ? "optional" : "value-only");
    bool na = nulls == cudf::null_aware::NO;
    cudf::reduce_udf udf{.fragments     = lto_fragments(),
                         .init          = na ? "raw_string_max_init" : "string_max_init",
                         .update        = na ? "raw_string_max_update" : "string_max_update",
                         .merge         = na ? "raw_string_max_merge" : "string_max_merge",
                         .finalize      = na ? "raw_string_max_finalize" : "string_max_finalize",
                         .is_null_aware = nulls};

    for (auto& test : cases) {
      SCOPED_TRACE(test.values.size());
      auto input = cudf::test::strings_column_wrapper(
                     test.values.begin(), test.values.end(), test.valid.begin())
                     .release();
      auto valid           = test.expected_valid || nulls == cudf::null_aware::NO;
      auto expected_max    = na ? cudf::test::strings_column_wrapper{test.expected_max}
                                : cudf::test::strings_column_wrapper({test.expected_max}, {valid});
      auto length          = static_cast<int32_t>(test.expected_max.size());
      auto expected_length = na ? wrapper<int32_t>{length} : wrapper<int32_t>{{length}, {valid}};
      cudf::reduce_input inputs[]{input->view()};

      auto result = cudf::reduce(udf, inputs, states, outputs, {}, cudf::fallible::NO);
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

  cudf::reduce_input inputs[]{values, cudf::user_data{weights.data()}};

  auto udf = sum_count_avg_udf("user_data_update");

  udf.indexed_update = true;

  auto result = cudf::reduce(udf, inputs, state_types, output_types);
  expect_sum_count_avg(result, 96, 3);

  auto mask = cudf::detail::make_device_buffer(
    std::vector<cudf::bitmask_type>{0b101}, cudf::get_default_stream(), mr());
  cudf::reduce_input filter_inputs[]{cudf::user_data{mask.data()}};
  auto filtered           = udf;
  filtered.filter         = "mask_filter";
  filtered.indexed_filter = true;
  result = cudf::reduce(filtered, inputs, state_types, output_types, filter_inputs);
  expect_sum_count_avg(result, 64, 2);
}

TEST_F(ReduceUdf, TpchQ6Sales)
{
  // WHERE active AND quantity > 0, then SUM/COUNT/AVG(price * quantity).
  wrapper<int32_t> price{10, 20, 30, 40, 50};
  wrapper<int32_t> quantity{2, 0, 3, -1, 2};
  wrapper<bool> active{true, true, false, true, true};

  auto udf = sum_count_avg_udf("filtered_product_update");

  cudf::reduce_input inputs[]{price, quantity, active};

  auto result = cudf::reduce(udf, inputs, state_types, output_types);

  expect_sum_count_avg(result, 120, 2);
}

TEST_F(ReduceUdf, CudaSourceTpchQ6SalesNullsAndEmptyInput)
{
  auto udf = sum_count_avg_udf("filtered_product_update");

  for (int32_t rows : {0, 3}) {
    // SQL SUM/AVG are NULL and COUNT(expression) is zero for empty or all-null input.

    std::vector<int32_t> prices(rows, 10);
    std::vector<bool> valid(rows, false);
    std::vector<int32_t> quantities(rows, 2);
    std::vector<bool> enabled(rows, true);

    wrapper<int32_t> price(prices.begin(), prices.end(), valid.begin());
    wrapper<int32_t> quantity(quantities.begin(), quantities.end());
    wrapper<bool> active(enabled.begin(), enabled.end());

    cudf::reduce_input inputs[]{price, quantity, active};

    auto result = cudf::reduce(udf, inputs, state_types, output_types);
    expect_sum_count_avg(result, 0, 0);
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

  cudf::reduce_input inputs[]{price, quantity, active};

  auto udf = sum_count_avg_udf("filtered_product_update");

  auto result = cudf::reduce(udf, inputs, state_types, output_types);
  expect_sum_count_avg(result, expected_sum, expected_count);
}

TEST_F(ReduceUdf, SeparateMemoryResources)
{
  wrapper<int32_t> values{1, 2, 3};
  cudf::reduce_input const inputs[]{values};
  rmm::mr::statistics_resource_adaptor output_resource{mr()};
  rmm::mr::statistics_resource_adaptor temporary_resource{mr()};
  cudf::memory_resources resources{rmm::device_async_resource_ref{output_resource},
                                   rmm::device_async_resource_ref{temporary_resource}};
  auto result = cudf::reduce(sum_count_avg_udf("sum_update"),
                             inputs,
                             state_types,
                             output_types,
                             {},
                             cudf::fallible::YES,
                             cudf::get_default_stream(),
                             resources);
  expect_sum_count_avg(result, 6, 3);
  EXPECT_GT(output_resource.get_bytes_counter().value, 0);
  EXPECT_GT(temporary_resource.get_bytes_counter().peak, 0);
  EXPECT_EQ(temporary_resource.get_bytes_counter().value, 0);
}

TEST_F(ReduceUdf, SegmentBoundaries)
{
  wrapper<int32_t> values{1, 2, 3, 4};
  cudf::reduce_input const inputs[]{values};
  auto udf   = segmented_tpch_q6_sales_udf();
  udf.update = "sum_segment_update";
  struct boundary_case {
    std::vector<int32_t> offsets;
    int64_t sum;
    int64_t count;
  };
  boundary_case const cases[]{{{0}, 0, 0}, {{1, 1}, 0, 0}, {{1, 3}, 5, 2}};
  for (auto const& test : cases) {
    SCOPED_TRACE(test.offsets.size());
    auto offsets = cudf::detail::make_device_buffer(test.offsets, cudf::get_default_stream(), mr());
    auto result =
      cudf::segmented_reduce(udf, inputs, offsets, state_types, output_types, std::nullopt);
    if (test.offsets.size() == 1) {
      EXPECT_TRUE(result.empty());
    } else {
      expect_sum_count_avg(result, test.sum, test.count);
    }
  }
}

TEST_F(ReduceUdf, OptionalStateLayout)
{
  auto check = []<typename... T>() {
    struct expected_state {
      cuda::std::tuple<T...> values;
      cudf::errc error;
    };
    auto layout =
      sizeof...(T) == 0 ? cudf::detail::layout_of<cuda::std::tuple<>> : cudf::detail::layout{};
    ((layout = layout.extend(cudf::detail::layout_of<T>)), ...);
    layout = layout.aligned().extend(cudf::detail::layout_of<cudf::errc>).aligned();
    EXPECT_GE(layout.size, sizeof(expected_state));
    EXPECT_GE(layout.alignment, alignof(expected_state));
  };
  check.template operator()<>();
  check.template operator()<cuda::std::optional<int8_t>,
                            cuda::std::optional<int64_t>,
                            cuda::std::optional<int32_t>>();
  check.template operator()<cuda::std::optional<int64_t>, cuda::std::optional<int8_t>>();
}

TEST_F(ReduceUdf, SegmentedFilter)
{
  wrapper<int32_t> values{10, -99, 20, 30, -99, 40};
  constexpr int64_t expected_sums[]{0, 0, 20, 40};
  constexpr int64_t expected_counts[]{0, 0, 1, 1};
  auto packed = cudf::detail::make_device_buffer(
    std::vector<cudf::bitmask_type>{0b100101}, cudf::get_default_stream(), mr());
  auto offsets = cudf::detail::make_device_buffer(
    std::vector<int32_t>{0, 0, 2, 4, 6}, cudf::get_default_stream(), mr());
  cudf::reduce_input const inputs[]{values};
  cudf::reduce_input const filter_inputs[]{cudf::user_data{packed.data()}};
  auto udf           = segmented_tpch_q6_sales_udf();
  udf.filter         = "segment_mask_filter";
  udf.indexed_filter = true;
  udf.update         = "checked_segment_mask_update";
  for (auto hint : {std::optional<cudf::size_type>{}, {2}}) {
    auto result =
      cudf::segmented_reduce(udf, inputs, offsets, state_types, output_types, hint, filter_inputs);
    ASSERT_EQ(result.size(), std::size(expected_sums) * std::size(output_types));
    for (size_t segment = 0; segment < std::size(expected_sums); ++segment) {
      SCOPED_TRACE(segment);
      expect_sum_count_avg(
        std::span{result}.subspan(segment * std::size(output_types), std::size(output_types)),
        expected_sums[segment],
        expected_counts[segment]);
    }
  }
}

TEST_F(ReduceUdf, SegmentedErrorHandling)
{
  auto udf   = segmented_tpch_q6_sales_udf();
  udf.update = "checked_segment_update";
  for (int32_t rows : {3, 100'003}) {
    SCOPED_TRACE(rows);
    std::vector<int32_t> values(rows, 1);
    values.back() = -1;
    wrapper<int32_t> input(values.begin(), values.end());
    cudf::reduce_input const inputs[]{input};
    auto offsets = cudf::detail::make_device_buffer(
      std::vector<int32_t>{0, 0, 1, rows}, cudf::get_default_stream(), mr());
    // Exercise unknown lengths, the small-group path, and a split long segment.
    for (auto hint : {std::optional<cudf::size_type>{}, {rows - 1}}) {
      SCOPED_TRACE(hint.value_or(-1));
      try {
        auto result = cudf::segmented_reduce(udf, inputs, offsets, state_types, output_types, hint);
        FAIL() << "A failure in the third segment must propagate across reduction phases";
      } catch (cudf::evaluation_error const& error) {
        EXPECT_EQ(error.error_code(), cudf::errc::ARITHMETIC_OVERFLOW);
      }
    }
  }
}

TEST_F(ReduceUdf, SegmentedStringStateAndOutput)
{
  // Include empty, null, UTF-8, and empty-string groups, then a group requiring partial merges.
  constexpr char const* expected_max[]{"", "b", "\xc3\xa9", "", "z"};
  constexpr bool expected_valid[]{false, true, true, true, true};
  constexpr int32_t expected_lengths[]{0, 1, 2, 0, 1};
  data_type const states[]{data_type{type_id::STRING}};
  data_type const outputs[]{states[0], data_type{type_id::INT32}};
  cudf::reduce_udf const udf{.fragments        = lto_fragments(),
                             .init             = "string_max_segment_init",
                             .update           = "string_max_segment_update",
                             .merge            = "string_max_segment_merge",
                             .finalize         = "string_max_segment_finalize",
                             .is_null_aware    = cudf::null_aware::YES,
                             .indexed_update   = true,
                             .indexed_init     = true,
                             .indexed_merge    = true,
                             .indexed_finalize = true};

  for (int32_t last_segment_size : {1, 100'000}) {
    SCOPED_TRACE(last_segment_size);
    std::vector<int32_t> const boundaries{0, 0, 2, 4, 5, 5 + last_segment_size};
    std::vector<std::string> values(boundaries.back(), "a");
    std::vector<bool> valid(values.size(), true);
    values[0]     = "b";
    values[1]     = "aa";
    values[2]     = "ignored";
    valid[2]      = false;
    values[3]     = "\xc3\xa9";
    values[4]     = "";
    values.back() = "z";
    cudf::test::strings_column_wrapper input(values.begin(), values.end(), valid.begin());
    cudf::reduce_input const inputs[]{input};
    auto offsets = cudf::detail::make_device_buffer(boundaries, cudf::get_default_stream(), mr());

    for (auto hint : {std::optional<cudf::size_type>{}, {std::max(2, last_segment_size)}}) {
      SCOPED_TRACE(hint.value_or(-1));
      auto result = cudf::segmented_reduce(udf, inputs, offsets, states, outputs, hint);
      ASSERT_EQ(result.size(), std::size(outputs) * std::size(expected_max));
      for (size_t segment = 0; segment < std::size(expected_max); ++segment) {
        SCOPED_TRACE(segment);
        cudf::test::strings_column_wrapper expected({expected_max[segment]},
                                                    {expected_valid[segment]});
        wrapper<int32_t> length{{expected_lengths[segment]}, {expected_valid[segment]}};
        CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result[std::size(outputs) * segment], expected);
        CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result[std::size(outputs) * segment + 1], length);
      }
    }
  }
}

TEST_F(ReduceUdf, SmallSegmented)
{
  std::vector<int32_t> boundaries{0, 0, 1, 3, 6, 11, 44};
  std::vector<int32_t> values(boundaries.back(), 2);

  constexpr int64_t expected_sums[]{0, 2, 4, 6, 10, 66};
  constexpr int64_t expected_counts[]{0, 1, 2, 3, 5, 33};

  wrapper<int32_t> input(values.begin(), values.end());
  cudf::reduce_input const inputs[]{input};

  auto offsets = cudf::detail::make_device_buffer(boundaries, cudf::get_default_stream(), mr());
  auto udf     = segmented_tpch_q6_sales_udf();
  udf.update   = "sum_segment_update";

  for (auto const& callback : {sum_count_avg_udf("sum_update"), udf}) {
    auto result = cudf::segmented_reduce(callback, inputs, offsets, state_types, output_types, 33);

    ASSERT_EQ(result.size(), std::size(output_types) * std::size(expected_sums));

    for (size_t segment = 0; segment < std::size(expected_sums); ++segment) {
      SCOPED_TRACE(segment);
      expect_sum_count_avg(
        std::span{result}.subspan(std::size(output_types) * segment, std::size(output_types)),
        expected_sums[segment],
        expected_counts[segment]);
    }
  }
}

TEST_F(ReduceUdf, LargeSegmented)
{
  std::vector<int32_t> boundaries{0, 3, 100'003, 100'003, 250'003};
  std::vector<int32_t> values(boundaries.back(), 1);

  constexpr int64_t expected_sums[]{3, 100'000, 0, 150'000};
  constexpr int64_t expected_counts[]{3, 100'000, 0, 150'000};

  wrapper<int32_t> input(values.begin(), values.end());

  cudf::reduce_input const inputs[]{input};
  auto offsets = cudf::detail::make_device_buffer(boundaries, cudf::get_default_stream(), mr());

  auto udf   = segmented_tpch_q6_sales_udf();
  udf.update = "sum_segment_update";

  for (auto const& callback : {sum_count_avg_udf("sum_update"), udf}) {
    for (auto hint : {std::optional<cudf::size_type>{}, {150'000}}) {
      auto result =
        cudf::segmented_reduce(callback, inputs, offsets, state_types, output_types, hint);
      ASSERT_EQ(result.size(), std::size(output_types) * std::size(expected_sums));

      for (size_t segment = 0; segment < std::size(expected_sums); ++segment) {
        SCOPED_TRACE(segment);
        expect_sum_count_avg(
          std::span{result}.subspan(std::size(output_types) * segment, std::size(output_types)),
          expected_sums[segment],
          expected_counts[segment]);
      }
    }
  }
}

TEST_F(ReduceUdf, SegmentedTpchQ6Sales)
{
  // Apply the same query to each group, including an empty group and a fully filtered group.
  wrapper<int32_t> price{{10, 20, 30, 40, 50, 60}, {true, false, true, true, true, true}};
  wrapper<int32_t> quantity{{2, 1, 0, 2, -1, 3}, {true, true, true, true, false, true}};
  wrapper<bool> active{{true, true, true, true, true, true}, {true, true, true, false, true, true}};
  std::vector<int32_t> const boundaries{0, 0, 2, 4, 6};
  constexpr int64_t expected_sums[]{0, 20, 0, 180};
  constexpr int64_t expected_counts[]{0, 1, 0, 1};

  cudf::reduce_input const inputs[]{price, quantity, active};

  auto offsets = cudf::detail::make_device_buffer(boundaries, cudf::get_default_stream(), mr());

  for (auto hint : {std::optional<cudf::size_type>{}, {2}}) {
    auto result = cudf::segmented_reduce(
      segmented_tpch_q6_sales_udf(), inputs, offsets, state_types, output_types, hint);

    ASSERT_EQ(result.size(), std::size(output_types) * std::size(expected_sums));

    for (size_t segment = 0; segment < std::size(expected_sums); ++segment) {
      SCOPED_TRACE(segment);
      expect_sum_count_avg(
        std::span{result}.subspan(std::size(output_types) * segment, std::size(output_types)),
        expected_sums[segment],
        expected_counts[segment]);
    }
  }
}
