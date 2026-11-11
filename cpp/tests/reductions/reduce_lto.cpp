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

#include <cudf_reduce_lto_test_fragments.hpp>

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

std::span<uint8_t const> get_fragment(size_t id)
{
  auto const range = cudf_reduce_lto_test_fragments::file_ranges[id];
  return cudf_reduce_lto_test_fragments::files.subspan(range[0], range[1]);
}

cudf::compiled_fragment compile_tpch_q6_sales()
{
  auto const bytes = get_fragment(cudf_reduce_lto_test_fragments::tpch_q6_sales_source);
  std::string const source(bytes.begin(), bytes.end());
  return cudf::compile_cuda_to_lto_ir(source.c_str(), {}, {}, {});
}

cudf::lto_fragments precompiled_fragments()
{
  static cudf::lto_fragment const fragments[]{
    {cudf::lto_binary_type::FATBIN, get_fragment(cudf_reduce_lto_test_fragments::tpch_q6_sales)},
    {cudf::lto_binary_type::FATBIN, get_fragment(cudf_reduce_lto_test_fragments::sum_count)},
    {cudf::lto_binary_type::FATBIN, get_fragment(cudf_reduce_lto_test_fragments::filtering)},
    {cudf::lto_binary_type::FATBIN, get_fragment(cudf_reduce_lto_test_fragments::strings)},
    {cudf::lto_binary_type::FATBIN, get_fragment(cudf_reduce_lto_test_fragments::input_output)},
    {cudf::lto_binary_type::FATBIN, get_fragment(cudf_reduce_lto_test_fragments::errors)},
    {cudf::lto_binary_type::FATBIN, get_fragment(cudf_reduce_lto_test_fragments::wide_state)}};
  return fragments;
}

cudf::reduce_udf tpch_q6_sales_udf()
{
  return {precompiled_fragments(),
          "tpch_q6_sales_init",
          "tpch_q6_sales_update",
          "tpch_q6_sales_merge",
          "tpch_q6_sales_finalize"};
}

cudf::reduce_udf segmented_tpch_q6_sales_udf()
{
  return {precompiled_fragments(),
          "tpch_q6_sales_segment_init",
          "tpch_q6_sales_segment_update",
          "tpch_q6_sales_segment_merge",
          "tpch_q6_sales_segment_finalize"};
}

cudf::reduce_udf sum_udf()
{
  return {precompiled_fragments(),
          "tpch_q6_sales_init",
          "sum_update",
          "tpch_q6_sales_merge",
          "tpch_q6_sales_finalize"};
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

struct ReduceLto : cudf::test::BaseFixture {
  void SetUp() override { CUDF_CUDA_TRY(cudaSetDevice(rmm::get_current_cuda_device().value())); }
};

TEST_F(ReduceLto, EmptyStateAndOutputs)
{
  cudf::reduce_udf udf{
    precompiled_fragments(), "empty_init", "empty_update", "empty_merge", "empty_finalize"};
  for (int32_t rows : {0, 100'003}) {
    SCOPED_TRACE(rows);
    std::vector<int32_t> values(rows, 1);
    wrapper<int32_t> input(values.begin(), values.end());
    cudf::reduce_input const inputs[]{input};
    auto result = cudf::reduce_lto(udf, cudf::null_aware::NO, inputs, {}, {});
    EXPECT_TRUE(result.empty());
  }
  udf.finalize = "empty_finalize_error";
  try {
    auto result = cudf::reduce_lto(udf, cudf::null_aware::NO, {}, {}, {});
    FAIL() << "Finalization errors must propagate even without state or outputs";
  } catch (cudf::evaluation_error const& error) {
    EXPECT_EQ(error.error_code(), cudf::errc::DIVISION_BY_ZERO);
  }
}

TEST_F(ReduceLto, ScalarOnlyFinalizesIdentity)
{
  wrapper<int32_t> value{7};
  cudf::reduce_input const inputs[]{cudf::scalar_column_view{value}};
  auto result =
    cudf::reduce_lto(sum_udf(), cudf::null_aware::YES, inputs, state_types, output_types);
  expect_sum_count_avg(result, 0, 0);
}

TEST_F(ReduceLto, SeparateMemoryResources)
{
  wrapper<int32_t> values{1, 2, 3};
  cudf::reduce_input const inputs[]{values};
  rmm::mr::statistics_resource_adaptor output_resource{mr()};
  rmm::mr::statistics_resource_adaptor temporary_resource{mr()};
  cudf::memory_resources resources{rmm::device_async_resource_ref{output_resource},
                                   rmm::device_async_resource_ref{temporary_resource}};
  auto result = cudf::reduce_lto(sum_udf(),
                                 cudf::null_aware::YES,
                                 inputs,
                                 state_types,
                                 output_types,
                                 cudf::get_default_stream(),
                                 resources);
  expect_sum_count_avg(result, 6, 3);
  EXPECT_GT(output_resource.get_bytes_counter().value, 0);
  EXPECT_GT(temporary_resource.get_bytes_counter().peak, 0);
  EXPECT_EQ(temporary_resource.get_bytes_counter().value, 0);
}

TEST_F(ReduceLto, SegmentBoundaries)
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
    auto result  = cudf::segmented_reduce_lto(
      udf, cudf::null_aware::YES, inputs, offsets, state_types, output_types, std::nullopt);
    if (test.offsets.size() == 1) {
      EXPECT_TRUE(result.empty());
    } else {
      expect_sum_count_avg(result, test.sum, test.count);
    }
  }
}

TEST_F(ReduceLto, OptionalStateLayout)
{
  auto check = []<typename... T>() {
    struct expected_state {
      cuda::std::tuple<T...> values;
      cudf::errc error;
    };
    auto layout = sizeof...(T) == 0 ? cudf::detail::memory_layout_of<cuda::std::tuple<>>
                                    : cudf::detail::memory_layout{};
    ((layout = layout.append(cudf::detail::memory_layout_of<T>)), ...);
    layout = layout.aligned().append(cudf::detail::memory_layout_of<cudf::errc>).aligned();
    EXPECT_GE(layout.size, sizeof(expected_state));
    EXPECT_GE(layout.alignment, alignof(expected_state));
  };
  check.template operator()<>();
  check.template operator()<cuda::std::optional<int8_t>,
                            cuda::std::optional<int64_t>,
                            cuda::std::optional<int32_t>>();
  check.template operator()<cuda::std::optional<int64_t>, cuda::std::optional<int8_t>>();
}

TEST_F(ReduceLto, WideOptionalState)
{
  constexpr int32_t fields      = 32;
  constexpr int64_t sum_weights = fields * (fields + 1) / 2;
  std::vector<cudf::data_type> states(fields, cudf::data_type{type_id::INT64});
  cudf::data_type const outputs[]{cudf::data_type{type_id::INT64}};
  cudf::reduce_udf const udf{
    precompiled_fragments(), "wide_init", "wide_update", "wide_merge", "wide_finalize"};

  for (int32_t rows : {0, 257, 100'003}) {
    SCOPED_TRACE(rows);
    std::vector<int32_t> values(rows, 1);
    wrapper<int32_t> input(values.begin(), values.end());
    cudf::reduce_input const inputs[]{input};
    auto result = cudf::reduce_lto(udf, cudf::null_aware::YES, inputs, states, outputs);
    ASSERT_EQ(result.size(), 1);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result[0], (wrapper<int64_t>{{rows * sum_weights}, {true}}));
  }
}

TEST_F(ReduceLto, NullHandling)
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
    auto result =
      cudf::reduce_lto(sum_udf(), cudf::null_aware::YES, inputs, state_types, output_types);

    expect_sum_count_avg(result, test.expected_sum, test.expected_count);

    // Value-only callbacks skip null rows and preserve their zero identity.
    data_type const types[]{state_types[0]};
    cudf::reduce_udf const raw{
      precompiled_fragments(), "raw_init", "raw_update", "raw_merge", "raw_finalize"};
    result = cudf::reduce_lto(raw, cudf::null_aware::NO, inputs, types, types);

    ASSERT_EQ(result.size(), std::size(types));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result[0], expected_raw_sum);
  }
}

TEST_F(ReduceLto, FusedOutput)
{
  // SELECT SUM(amount), COUNT(amount), AVG(amount), SUM(amount) >= 100 FROM input;
  wrapper<int32_t> amount{20, 40, 60};

  constexpr int64_t expected_sum   = 120;
  constexpr int64_t expected_count = 3;

  wrapper<bool> expected_qualifies{{true}, {true}};

  cudf::reduce_input const inputs[]{amount};
  data_type const outputs[]{
    output_types[0], output_types[1], output_types[2], data_type{type_id::BOOL8}};

  auto udf     = sum_udf();
  udf.finalize = "fused_finalize";

  auto result = cudf::reduce_lto(udf, cudf::null_aware::YES, inputs, state_types, outputs);

  ASSERT_EQ(result.size(), std::size(outputs));
  expect_sum_count_avg(
    std::span{result}.first(std::size(output_types)), expected_sum, expected_count);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result[std::size(output_types)], expected_qualifies);
}

TEST_F(ReduceLto, FilterPushDownReduction)
{
  wrapper<int32_t> values{10, -99, 20, -99, 30};
  wrapper<bool> keep{{true, false, true, true, true}, {true, true, true, false, true}};

  constexpr int64_t expected_sum   = 60;
  constexpr int64_t expected_count = 3;

  auto packed = cudf::detail::make_device_buffer(
    std::vector<cudf::bitmask_type>{0b10101}, cudf::get_default_stream(), mr());
  cudf::reduce_input const inputs[]{values};

  auto udf   = sum_udf();
  udf.update = "checked_update";

  // Without a selection, negative input values trigger an error in checked_update.
  EXPECT_THROW(cudf::reduce_lto(udf, cudf::null_aware::YES, inputs, state_types, output_types),
               cudf::evaluation_error);

  // A BOOL8 column supplies the selection; a null predicate excludes its row.
  cudf::reduce_input const boolean_inputs[]{values, keep};
  udf.filter = "boolean_filter";
  udf.update = "checked_boolean_update";
  auto result =
    cudf::reduce_lto(udf, cudf::null_aware::YES, boolean_inputs, state_types, output_types);
  expect_sum_count_avg(result, expected_sum, expected_count);

  // A packed mask has no column type; the filter reads its words through user_data.
  cudf::reduce_input const mask_inputs[]{values, cudf::user_data{packed.data()}};
  udf.filter = "mask_filter";
  udf.update = "checked_mask_update";
  result     = cudf::reduce_lto(udf, cudf::null_aware::YES, mask_inputs, state_types, output_types);
  expect_sum_count_avg(result, expected_sum, expected_count);

  data_type const raw_types[]{state_types[0]};
  cudf::reduce_udf const raw{precompiled_fragments(),
                             "raw_init",
                             "raw_mask_update",
                             "raw_merge",
                             "raw_finalize",
                             "raw_mask_filter"};
  result = cudf::reduce_lto(raw, cudf::null_aware::NO, mask_inputs, raw_types, raw_types);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result[0], (wrapper<int64_t>{{expected_sum}, {true}}));
}

TEST_F(ReduceLto, SegmentedFilter)
{
  wrapper<int32_t> values{10, -99, 20, 30, -99, 40};
  constexpr int64_t expected_sums[]{0, 0, 20, 40};
  constexpr int64_t expected_counts[]{0, 0, 1, 1};
  auto packed = cudf::detail::make_device_buffer(
    std::vector<cudf::bitmask_type>{0b100101}, cudf::get_default_stream(), mr());
  auto offsets = cudf::detail::make_device_buffer(
    std::vector<int32_t>{0, 0, 2, 4, 6}, cudf::get_default_stream(), mr());
  cudf::reduce_input const inputs[]{values, cudf::user_data{packed.data()}};
  auto udf   = segmented_tpch_q6_sales_udf();
  udf.filter = "segment_mask_filter";
  udf.update = "checked_segment_mask_update";
  for (auto hint : {std::optional<cudf::size_type>{}, {2}}) {
    auto result = cudf::segmented_reduce_lto(
      udf, cudf::null_aware::YES, inputs, offsets, state_types, output_types, hint);
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

TEST_F(ReduceLto, ErrorHandling)
{
  auto init         = sum_udf();
  auto filter       = sum_udf();
  auto update       = sum_udf();
  auto merge        = sum_udf();
  auto finalize     = sum_udf();
  init.init         = "error_init";
  filter.filter     = "error_filter";
  filter.update     = "error_update";
  update.update     = "checked_update";
  merge.merge       = "error_merge";
  finalize.finalize = "error_finalize";

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
      cudf::reduce_lto(test.udf, cudf::null_aware::YES, inputs, state_types, output_types);
      FAIL() << "Expected a callback error";
    } catch (cudf::evaluation_error const& error) {
      EXPECT_EQ(error.error_code(), test.expected_error);
    }
  }
}

TEST_F(ReduceLto, SegmentedErrorHandling)
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
        auto result = cudf::segmented_reduce_lto(
          udf, cudf::null_aware::YES, inputs, offsets, state_types, output_types, hint);
        FAIL() << "A failure in the third segment must propagate across reduction phases";
      } catch (cudf::evaluation_error const& error) {
        EXPECT_EQ(error.error_code(), cudf::errc::ARITHMETIC_OVERFLOW);
      }
    }
  }
}

TEST_F(ReduceLto, Assertions)
{
  wrapper<int32_t> values{1, 2, 3};
  wrapper<int32_t> shorter{1, 2};

  cudf::reduce_input const inputs[]{values};
  cudf::reduce_input const mismatched[]{values, shorter};

  // Input columns must have matching row counts.
  EXPECT_THROW(
    cudf::reduce_lto(sum_udf(), cudf::null_aware::YES, mismatched, state_types, output_types),
    std::invalid_argument);

  auto udf   = segmented_tpch_q6_sales_udf();
  udf.update = "sum_segment_update";

  // Even zero segments require one offset; an empty offsets span is invalid.
  EXPECT_THROW(cudf::segmented_reduce_lto(
                 udf, cudf::null_aware::YES, inputs, {}, state_types, output_types, std::nullopt),
               std::invalid_argument);

  auto offsets =
    cudf::detail::make_device_buffer(std::vector<int32_t>{0, 3}, cudf::get_default_stream(), mr());

  // A maximum segment size hint cannot be negative.
  EXPECT_THROW(cudf::segmented_reduce_lto(
                 udf, cudf::null_aware::YES, inputs, offsets, state_types, output_types, -1),
               std::invalid_argument);
}

TEST_F(ReduceLto, UnsupportedTypes)
{
  wrapper<int32_t> values{1, 2, 3};
  cudf::reduce_input const inputs[]{values};
  auto offsets =
    cudf::detail::make_device_buffer(std::vector<int32_t>{0, 3}, cudf::get_default_stream(), mr());

  for (auto type : {type_id::EMPTY, type_id::DICTIONARY32, type_id::LIST, type_id::STRUCT}) {
    SCOPED_TRACE(static_cast<int32_t>(type));
    data_type const unsupported[]{data_type{type}};

    // Scalar reductions reject state types other than fixed-width types and strings.
    EXPECT_THROW(
      cudf::reduce_lto(sum_udf(), cudf::null_aware::YES, inputs, unsupported, output_types),
      std::invalid_argument);

    // Scalar reductions apply the same restriction to output types.
    EXPECT_THROW(
      cudf::reduce_lto(sum_udf(), cudf::null_aware::YES, inputs, state_types, unsupported),
      std::invalid_argument);

    // Segmented reductions reject each unsupported state type as well.
    EXPECT_THROW(cudf::segmented_reduce_lto(segmented_tpch_q6_sales_udf(),
                                            cudf::null_aware::YES,
                                            inputs,
                                            offsets,
                                            unsupported,
                                            output_types,
                                            std::nullopt),
                 std::invalid_argument);

    // Segmented output types follow the same support rules.
    EXPECT_THROW(cudf::segmented_reduce_lto(segmented_tpch_q6_sales_udf(),
                                            cudf::null_aware::YES,
                                            inputs,
                                            offsets,
                                            state_types,
                                            unsupported,
                                            std::nullopt),
                 std::invalid_argument);
  }
}

TEST_F(ReduceLto, MultiInput)
{
  // SELECT SUM((amount - discount) * rate), COUNT(...), AVG(...) FROM input;
  wrapper<int32_t> amount{20, 50, 100};
  wrapper<int32_t> discount{{5, 20, 50}, {true, false, true}};
  wrapper<int32_t> rate{3};

  constexpr int64_t expected_sum   = 195;
  constexpr int64_t expected_count = 2;

  cudf::reduce_input const inputs[]{amount, discount, cudf::scalar_column_view{rate}};

  auto udf    = sum_udf();
  udf.update  = "discount_update";
  auto result = cudf::reduce_lto(udf, cudf::null_aware::YES, inputs, state_types, output_types);

  expect_sum_count_avg(result, expected_sum, expected_count);
}

TEST_F(ReduceLto, MultiOutput)
{
  wrapper<int32_t> values{1, 2, 9};

  constexpr int64_t expected_sum   = 12;
  constexpr int64_t expected_count = 3;

  wrapper<int64_t> expected_retained_sum{{expected_sum}, {true}};

  cudf::reduce_input const inputs[]{values};

  auto result =
    cudf::reduce_lto(sum_udf(), cudf::null_aware::YES, inputs, state_types, output_types);

  expect_sum_count_avg(result, expected_sum, expected_count);

  EXPECT_NE(result[0]->view().head(), result[1]->view().head());
  EXPECT_NE(result[0]->view().null_mask(), result[1]->view().null_mask());

  auto sum = std::move(result[0]);

  result.clear();
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*sum, expected_retained_sum);
}

TEST_F(ReduceLto, DictionaryInput)
{
  wrapper<int32_t> values{{10, 99, 10, 20, -5}, {true, false, true, true, true}};

  constexpr int64_t expected_sum   = 35;
  constexpr int64_t expected_count = 4;

  auto dictionary = cudf::dictionary::encode(values);

  cudf::reduce_input const inputs[]{dictionary->view()};

  auto result =
    cudf::reduce_lto(sum_udf(), cudf::null_aware::YES, inputs, state_types, output_types);

  expect_sum_count_avg(result, expected_sum, expected_count);
}

TEST_F(ReduceLto, StringReduction)
{
  // SELECT SUM(OCTET_LENGTH(value)), COUNT(value), AVG(OCTET_LENGTH(value)) FROM input;
  cudf::test::strings_column_wrapper strings{{"", "abc", "ignored", "GPU", "\xc3\xa9"},
                                             {true, true, false, true, true}};

  constexpr int64_t expected_sum   = 8;
  constexpr int64_t expected_count = 4;

  auto dictionary = cudf::dictionary::encode(strings);
  auto udf        = sum_udf();
  udf.update      = "string_update";

  for (auto input : {cudf::column_view{strings}, dictionary->view()}) {
    cudf::reduce_input const inputs[]{input};

    auto result = cudf::reduce_lto(udf, cudf::null_aware::YES, inputs, state_types, output_types);

    expect_sum_count_avg(result, expected_sum, expected_count);
  }
}

TEST_F(ReduceLto, StringStateAndOutput)
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
    cudf::reduce_udf udf{precompiled_fragments(),
                         "string_max_init",
                         "string_max_update",
                         "string_max_merge",
                         "string_max_finalize"};
    if (nulls == cudf::null_aware::NO) {
      udf.init     = "raw_string_max_init";
      udf.update   = "raw_string_max_update";
      udf.merge    = "raw_string_max_merge";
      udf.finalize = "raw_string_max_finalize";
    }

    for (auto const& test : cases) {
      SCOPED_TRACE(test.values.size());
      auto input = cudf::test::strings_column_wrapper(
                     test.values.begin(), test.values.end(), test.valid.begin())
                     .release();
      auto const valid = test.expected_valid || nulls == cudf::null_aware::NO;
      cudf::test::strings_column_wrapper expected_max({test.expected_max}, {valid});
      wrapper<int32_t> expected_length{{static_cast<int32_t>(test.expected_max.size())}, {valid}};
      cudf::reduce_input const inputs[]{input->view()};

      auto result = cudf::reduce_lto(udf, nulls, inputs, states, outputs);
      ASSERT_EQ(result.size(), std::size(outputs));
      if (!test.expected_max.empty()) { EXPECT_NE(result[0]->view().head(), input->view().head()); }
      input.reset();  // The result must own its characters after the input is destroyed.
      CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result[0], expected_max);
      CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result[1], expected_length);
    }
  }
}

TEST_F(ReduceLto, SegmentedStringStateAndOutput)
{
  // Include empty, null, UTF-8, and empty-string groups, then a group requiring partial merges.
  constexpr char const* expected_max[]{"", "b", "\xc3\xa9", "", "z"};
  constexpr bool expected_valid[]{false, true, true, true, true};
  constexpr int32_t expected_lengths[]{0, 1, 2, 0, 1};
  data_type const states[]{data_type{type_id::STRING}};
  data_type const outputs[]{states[0], data_type{type_id::INT32}};
  cudf::reduce_udf const udf{precompiled_fragments(),
                             "string_max_segment_init",
                             "string_max_segment_update",
                             "string_max_segment_merge",
                             "string_max_segment_finalize"};

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
      auto result = cudf::segmented_reduce_lto(
        udf, cudf::null_aware::YES, inputs, offsets, states, outputs, hint);
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

TEST_F(ReduceLto, UserDataInput)
{
  wrapper<int32_t> values{2, 4, 6};
  auto weights = cudf::detail::make_device_buffer(
    std::vector<int64_t>{3, 10, 20, 30}, cudf::get_default_stream(), mr());

  constexpr int64_t expected_sum   = 96;
  constexpr int64_t expected_count = 3;

  cudf::reduce_input const inputs[]{values, cudf::user_data{weights.data()}};

  auto udf   = sum_udf();
  udf.update = "user_data_update";

  auto result = cudf::reduce_lto(udf, cudf::null_aware::YES, inputs, state_types, output_types);
  expect_sum_count_avg(result, expected_sum, expected_count);
}

TEST_F(ReduceLto, SmallSegmented)
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

  auto result = cudf::segmented_reduce_lto(
    udf, cudf::null_aware::YES, inputs, offsets, state_types, output_types, 33);

  ASSERT_EQ(result.size(), std::size(output_types) * std::size(expected_sums));

  for (size_t segment = 0; segment < std::size(expected_sums); ++segment) {
    SCOPED_TRACE(segment);
    expect_sum_count_avg(
      std::span{result}.subspan(std::size(output_types) * segment, std::size(output_types)),
      expected_sums[segment],
      expected_counts[segment]);
  }
}

TEST_F(ReduceLto, LargeSegmented)
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

  for (auto hint : {std::optional<cudf::size_type>{}, {150'000}}) {
    auto result = cudf::segmented_reduce_lto(
      udf, cudf::null_aware::YES, inputs, offsets, state_types, output_types, hint);
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

TEST_F(ReduceLto, TpchQ6Sales)
{
  // WHERE active AND quantity > 0, then SUM/COUNT/AVG(price * quantity).
  wrapper<int32_t> price{10, 20, 30, 40, 50};
  wrapper<int32_t> quantity{2, 0, 3, -1, 2};
  wrapper<bool> active{true, true, false, true, true};
  constexpr int64_t expected_sum   = 120;
  constexpr int64_t expected_count = 2;

  cudf::reduce_input const inputs[]{price, quantity, active};

  auto result =
    cudf::reduce_lto(tpch_q6_sales_udf(), cudf::null_aware::YES, inputs, state_types, output_types);

  expect_sum_count_avg(result, expected_sum, expected_count);
}

TEST_F(ReduceLto, SegmentedTpchQ6Sales)
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
    auto result = cudf::segmented_reduce_lto(segmented_tpch_q6_sales_udf(),
                                             cudf::null_aware::YES,
                                             inputs,
                                             offsets,
                                             state_types,
                                             output_types,
                                             hint);

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

TEST_F(ReduceLto, CudaSourceTpchQ6SalesNullsAndEmptyInput)
{
  auto code = compile_tpch_q6_sales();
  cudf::lto_fragment const fragments[]{code.view()};

  auto udf      = tpch_q6_sales_udf();
  udf.fragments = fragments;

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

    auto result = cudf::reduce_lto(udf, cudf::null_aware::YES, inputs, state_types, output_types);
    expect_sum_count_avg(result, expected_sum, expected_count);
  }
}

TEST_F(ReduceLto, CudaSourceTpchQ6SalesLargeInput)
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
  auto code = compile_tpch_q6_sales();

  cudf::lto_fragment const fragments[]{code.view()};
  auto udf      = tpch_q6_sales_udf();
  udf.fragments = fragments;

  auto result = cudf::reduce_lto(udf, cudf::null_aware::YES, inputs, state_types, output_types);
  expect_sum_count_avg(result, expected_sum, expected_count);
}
