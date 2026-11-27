/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <tests/udf.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/cudf_gtest.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/random.hpp>
#include <cudf_test/table_utilities.hpp>
#include <cudf_test/testing_main.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/transform.hpp>
#include <cudf/udf.hpp>

#include <cuda/std/type_traits>

#include <jit/abi_traits.cuh>

#include <array>
#include <format>
#include <memory>
#include <string>
#include <utility>
#include <vector>

struct TransformLTOTest : public cudf::test::BaseFixture {};

template <typename T>
using column_wrapper = cudf::test::fixed_width_column_wrapper<T>;

template <typename T>
using decimal_wrapper = cudf::test::fixed_point_column_wrapper<typename T::rep>;

TEST_F(TransformLTOTest, CompileUdfWithIncludes)
{
  constexpr char source[] = R"CUDA(
#include <cudf/udf_includes.hpp>
#include "scale.cuh"
extern "C" __device__ cudf::errc transform(float* output, float input)
{
  *output = scale * input;
  return cudf::errc::SUCCESS;
}
)CUDA";
  column_wrapper<float> input{{1, 2, 4}};
  cudf::transform_input inputs[]{input};
  cudf::transform_output outputs[]{
    {cudf::data_type{cudf::type_id::FLOAT32}, cudf::output_nullability::ALL_VALID}};
  struct test_case {
    char const* header;
    float factor;
  };
  test_case const cases[]{{"inline constexpr float scale = 3.0f;", 3},
                          {"inline constexpr float scale = 3.0f;", 3},
                          {"inline constexpr float scale = 5.0f;", 5}};
  for (auto const& test : cases) {
    SCOPED_TRACE(test.header);
    char const* names[]{"scale.cuh"};
    char const* headers[]{test.header};
    // Header contents change under the same include name, so disable PCH reuse.
    auto fragment   = cudf::compile_cuda_to_lto_ir(source, names, headers, {}, false);
    auto const view = fragment.view();
    auto result     = cudf::transform(cudf::transform_udf{"transform", {&view, 1}},
                                  inputs,
                                  outputs,
                                      {},
                                  std::nullopt,
                                  cudf::test::get_default_stream());
    column_wrapper<float> expected{{test.factor, 2 * test.factor, 4 * test.factor}};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->get_column(0), expected);
  }
}

TEST_F(TransformLTOTest, NullableDictionary)
{
  constexpr char source[] = R"CUDA(
#include <cudf/udf_includes.hpp>
extern "C" __device__ cudf::errc transform(cuda::std::optional<int>* output,
                                           cuda::std::optional<int> input)
{
  *output = input;
  return cudf::errc::SUCCESS;
}
)CUDA";
  auto fragment           = cudf::compile_cuda_to_lto_ir(source);
  cudf::test::dictionary_column_wrapper<int32_t> input{{1, 2, 3}, {true, false, true}};
  cudf::transform_input inputs[]{input};
  cudf::transform_output outputs[]{
    {cudf::data_type{cudf::type_id::INT32}, cudf::output_nullability::PRESERVE}};
  auto view   = fragment.view();
  auto result = cudf::transform(cudf::transform_udf{"transform", {&view, 1}, cudf::null_aware::YES},
                                inputs,
                                outputs,
                                {},
                                std::nullopt,
                                cudf::test::get_default_stream());
  column_wrapper<int32_t> expected{{1, 2, 3}, {true, false, true}};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->get_column(0), expected);
}

TEST_F(TransformLTOTest, InvSqrt)
{
  column_wrapper<float> input{{1.0f, 4.0f, 9.0f, 16.0f}};

  cudf::transform_input inputs[]   = {input};
  cudf::transform_output outputs[] = {
    {cudf::data_type{cudf::type_id::FLOAT32}, cudf::output_nullability::ALL_VALID}};

  auto result = cudf::transform(cudf::test::get_udf("invsqrt"),
                                inputs,
                                outputs,
                                {},
                                std::nullopt,
                                cudf::test::get_default_stream());

  column_wrapper<float> expected{{1.0f, 0.5f, 0.33333334f, 0.25f}};

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->get_column(0), expected);
}

TEST_F(TransformLTOTest, Distance)
{
  column_wrapper<float> x1{{0, 0, 0, 0}};
  column_wrapper<float> y1{{0.0F, 1.2F, 2.5F, 3.7F}};
  column_wrapper<float> x2{{1.6F, 2.1F, 3.2f, 4.5f}};
  column_wrapper<float> y2{{0, 0, 0, 0}};

  cudf::transform_input inputs[]   = {x1, y1, x2, y2};
  cudf::transform_output outputs[] = {
    {cudf::data_type{cudf::type_id::FLOAT32}, cudf::output_nullability::ALL_VALID}};

  auto result = cudf::transform(cudf::test::get_udf("distance"),
                                inputs,
                                outputs,
                                {},
                                std::nullopt,
                                cudf::test::get_default_stream());

  auto distance = [](float x1, float y1, float x2, float y2) {
    return std::sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1));
  };

  column_wrapper<float> expected{{
    distance(0, 0, 1.6F, 0),
    distance(0, 1.2F, 2.1F, 0),
    distance(0, 2.5F, 3.2F, 0),
    distance(0, 3.7F, 4.5F, 0),
  }};

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->get_column(0), expected);
}

TEST_F(TransformLTOTest, ToUpper)
{
  column_wrapper<uint8_t> input{{65, 66, 97, 98, 48, 49, 32, 33, 127, 255}};

  cudf::transform_input inputs[]   = {input};
  cudf::transform_output outputs[] = {
    {cudf::data_type{cudf::type_id::UINT8}, cudf::output_nullability::ALL_VALID}};

  auto result = cudf::transform(cudf::test::get_udf("to_upper"),
                                inputs,
                                outputs,
                                {},
                                std::nullopt,
                                cudf::test::get_default_stream());

  column_wrapper<uint8_t> expected{{65, 66, 65, 66, 48, 49, 32, 33, 127, 255}};

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->get_column(0), expected);
}

TEST_F(TransformLTOTest, OutputWithPreallocatedOffsets)
{
  auto input                     = cudf::test::strings_column_wrapper{"abc", "", "é", "猫", "xyz"};
  cudf::transform_input inputs[] = {input};
  cudf::transform_output outputs[] = {
    {cudf::data_type{cudf::type_id::STRING}, cudf::output_nullability::ALL_VALID}};
  std::vector<std::unique_ptr<cudf::column>> string_offsets;
  string_offsets.push_back(column_wrapper<int32_t>{0, 3, 3, 5, 8, 11}.release());

  auto result = cudf::transform(cudf::test::get_udf("copy_string"),
                                inputs,
                                outputs,
                                std::move(string_offsets),
                                std::nullopt,
                                cudf::test::get_default_stream());

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(input, result->get_column(0));
}

TEST_F(TransformLTOTest, SumOfSquares)
{
  column_wrapper<float> lhs{{1.0f, 4.0f, 9.0f, 16.0f}};
  column_wrapper<float> rhs{{1.0f, 2.0f, 2.0f, 10.0f}};

  cudf::transform_input inputs[]   = {lhs, rhs};
  cudf::transform_output outputs[] = {
    {cudf::data_type{cudf::type_id::FLOAT32}, cudf::output_nullability::ALL_VALID}};

  auto result = cudf::transform(cudf::test::get_udf("sum_of_squares"),
                                inputs,
                                outputs,
                                {},
                                std::nullopt,
                                cudf::test::get_default_stream());

  column_wrapper<float> expected{{2.0f, 20.0f, 85.0f, 356.0f}};

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->get_column(0), expected);
}

TEST_F(TransformLTOTest, FallibleIntegerLehmerMean)
{
  // computes integer lehmer mean: `(a^2 + b^2) / (a + b)` for each row using checked arithmetic and
  // throws if an overflow occurs

  column_wrapper<int32_t> a{{2, 3, 4, 6, 8}};
  column_wrapper<int32_t> b{{1, 1, 2, 3, 4}};
  column_wrapper<int32_t> a_fail{{2, 3, -2, 6, 8}};

  cudf::transform_output outputs[] = {
    {cudf::data_type{cudf::type_id::INT32}, cudf::output_nullability::ALL_VALID}};

  cudf::transform_input inputs[] = {a, b};

  auto result      = cudf::transform(cudf::test::get_udf("lehmer_mean"),
                                inputs,
                                outputs,
                                     {},
                                std::nullopt,
                                cudf::test::get_default_stream());
  auto lehmer_mean = [](int32_t a, int32_t b) { return (a * a + b * b) / (a + b); };

  column_wrapper<int32_t> expected{{lehmer_mean(2, 1),
                                    lehmer_mean(3, 1),
                                    lehmer_mean(4, 2),
                                    lehmer_mean(6, 3),
                                    lehmer_mean(8, 4)}};

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->get_column(0), expected);

  cudf::transform_input inputs_fail[] = {a_fail, b};

  EXPECT_THROW(cudf::transform(cudf::test::get_udf("lehmer_mean"),
                               inputs_fail,
                               outputs,
                               {},
                               std::nullopt,
                               cudf::test::get_default_stream()),
               cudf::evaluation_error);
}

TEST_F(TransformLTOTest, BankersRounding)
{
  using T = numeric::decimal128;

  decimal_wrapper<T> input{{12450, 12550, 12650, 12750, -12450, -12550, -12650, -12750},
                           numeric::scale_type{-2}};

  cudf::transform_output output{cudf::data_type{cudf::type_to_id<T>(), numeric::scale_type{0}},
                                cudf::output_nullability::ALL_VALID};

  cudf::transform_input inputs[]   = {input};
  cudf::transform_output outputs[] = {output};

  auto result = cudf::transform(cudf::test::get_udf("bankers_round"),
                                inputs,
                                outputs,
                                {},
                                std::nullopt,
                                cudf::test::get_default_stream());

  decimal_wrapper<T> expected{{124, 126, 126, 128, -124, -126, -126, -128}, numeric::scale_type{0}};

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->get_column(0), expected);
}

template <typename Abi, typename Cudf>
inline constexpr void expect_abi_mapping()
{
  static_assert((cuda::std::is_same_v<cudf::abi_to_cudf_type<Abi>, Cudf>),
                "ABI to Cudf type mapping failed");
  static_assert((cuda::std::is_same_v<cudf::cudf_to_abi_type<Cudf>, Abi>),
                "Cudf to ABI type mapping failed");
  static_assert(sizeof(Abi) == sizeof(Cudf), "Size mismatch between ABI and Cudf types");
  static_assert(alignof(Abi) == alignof(Cudf), "Alignment mismatch between ABI and Cudf types");
}

TEST_F(TransformLTOTest, AbiTypes)
{
#define CHECK_ABI_TYPE(Abi, Cudf) expect_abi_mapping<cudf_abi::Abi, Cudf>()
  CHECK_ABI_TYPE(int8_t, int8_t);
  CHECK_ABI_TYPE(int16_t, int16_t);
  CHECK_ABI_TYPE(int32_t, int32_t);
  CHECK_ABI_TYPE(int64_t, int64_t);
  CHECK_ABI_TYPE(uint8_t, uint8_t);
  CHECK_ABI_TYPE(uint16_t, uint16_t);
  CHECK_ABI_TYPE(uint32_t, uint32_t);
  CHECK_ABI_TYPE(uint64_t, uint64_t);
  CHECK_ABI_TYPE(float32_t, float);
  CHECK_ABI_TYPE(float64_t, double);
  CHECK_ABI_TYPE(bool8_t, bool);
  CHECK_ABI_TYPE(size_type, cudf::size_type);
  CHECK_ABI_TYPE(voidptr_t, cudf::voidptr_t);
  CHECK_ABI_TYPE(error_int, cudf::error_int);
  CHECK_ABI_TYPE(bitmask_type, cudf::bitmask_type);
  CHECK_ABI_TYPE(errc, cudf::errc);
  CHECK_ABI_TYPE(string_view, cudf::string_view);
  CHECK_ABI_TYPE(mutable_string_view, cudf::mutable_string_view);
  CHECK_ABI_TYPE(duration_D, cudf::duration_D);
  CHECK_ABI_TYPE(duration_h, cudf::duration_h);
  CHECK_ABI_TYPE(duration_m, cudf::duration_m);
  CHECK_ABI_TYPE(duration_s, cudf::duration_s);
  CHECK_ABI_TYPE(duration_ms, cudf::duration_ms);
  CHECK_ABI_TYPE(duration_us, cudf::duration_us);
  CHECK_ABI_TYPE(duration_ns, cudf::duration_ns);
  CHECK_ABI_TYPE(timestamp_D, cudf::timestamp_D);
  CHECK_ABI_TYPE(timestamp_h, cudf::timestamp_h);
  CHECK_ABI_TYPE(timestamp_m, cudf::timestamp_m);
  CHECK_ABI_TYPE(timestamp_s, cudf::timestamp_s);
  CHECK_ABI_TYPE(timestamp_ms, cudf::timestamp_ms);
  CHECK_ABI_TYPE(timestamp_us, cudf::timestamp_us);
  CHECK_ABI_TYPE(timestamp_ns, cudf::timestamp_ns);
  CHECK_ABI_TYPE(decimal32, numeric::decimal32);
  CHECK_ABI_TYPE(decimal64, numeric::decimal64);
  CHECK_ABI_TYPE(decimal128, numeric::decimal128);
#undef CHECK_ABI_TYPE

  expect_abi_mapping<cudf_abi::optional<cudf_abi::int64_t>, cuda::std::optional<int64_t>>();
  expect_abi_mapping<cudf_abi::optional<cudf_abi::mutable_string_view>,
                     cuda::std::optional<cudf::mutable_string_view>>();
}

CUDF_TEST_PROGRAM_MAIN()
