/*
 * SPDX-FileCopyrightText: Copyright 2018-2019 BlazingDB, Inc.
 * SPDX-FileCopyrightText: Copyright 2018 Christian Noboa Mardini <christian@blazingdb.com>
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/*
 * Copyright 2018-2019 BlazingDB, Inc.
 *     Copyright 2018 Christian Noboa Mardini <christian@blazingdb.com>
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "assert_unary.h"

#include <tests/transform/udf.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/random.hpp>
#include <cudf_test/table_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/detail/iterator.cuh>
#include <cudf/dictionary/encode.hpp>
#include <cudf/transform.hpp>
#include <cudf/udf.hpp>

#include <array>
#include <format>
#include <utility>

namespace transformation {

struct RuntimeSupportTest : public cudf::test::BaseFixture {
 protected:
  cudf::test::fixed_width_column_wrapper<float> a{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

  cudf::test::fixed_width_column_wrapper<float> b{
    0.1F, 0.25F, 0.5F, 0.1F, 0.4F, 0.75F, 0.2F, 0.33F, 0.45F, 0.66F};

  cudf::test::fixed_width_column_wrapper<float> b_nulls{
    {0.1F, 0.25F, 0.5F, 0.1F, 0.4F, 0.75F, 0.2F, 0.33F, 0.45F, 0.66F},
    {true, true, true, true, true, true, true, true, true, false}};

  cudf::test::fixed_width_column_wrapper<float> t{0.5f};

  cudf::test::fixed_width_column_wrapper<float> bad_col{1, 2, 3, 4, 5, 6, 7, 8, 9};

  cudf::test::structs_column_wrapper struct_col{a, b};
};

struct AssertsTest : public RuntimeSupportTest {};

TEST_F(AssertsTest, TypeSupport)
{
  cudf::transform_input inputs[] = {a, b, cudf::scalar_column_view(t)};

  EXPECT_NO_THROW((void)std::move(
    cudf::transform(cudf::test::get_transform_udf("lerp"),
                    inputs,
                    std::array{cudf::transform_output{cudf::data_type{cudf::type_id::FLOAT32},
                                                      cudf::output_nullability::PRESERVE}},
                    {},
                    std::nullopt)
      ->release()
      .front()));

  EXPECT_THROW((void)std::move(cudf::transform(cudf::test::get_transform_udf("lerp"),
                                               inputs,
                                               std::array{cudf::transform_output{
                                                 cudf::data_type{cudf::type_id::STRUCT},
                                                 cudf::output_nullability::PRESERVE}},
                                               {},
                                               std::nullopt)
                                 ->release()
                                 .front()),
               std::invalid_argument);

  cudf::transform_input struct_inputs[] = {struct_col, cudf::scalar_column_view(t)};

  EXPECT_THROW((void)std::move(cudf::transform(cudf::test::get_transform_udf("lerp"),
                                               struct_inputs,
                                               std::array{cudf::transform_output{
                                                 cudf::data_type{cudf::type_id::FLOAT32},
                                                 cudf::output_nullability::PRESERVE}},
                                               {},
                                               std::nullopt)
                                 ->release()
                                 .front()),
               std::invalid_argument);
}

TEST_F(AssertsTest, UnequalRowCount)
{
  cudf::transform_input inputs[] = {a, b, bad_col};

  EXPECT_THROW((void)std::move(cudf::transform(cudf::test::get_transform_udf("lerp"),
                                               inputs,
                                               std::array{cudf::transform_output{
                                                 cudf::data_type{cudf::type_id::FLOAT32},
                                                 cudf::output_nullability::PRESERVE}},
                                               {},
                                               std::nullopt)
                                 ->release()
                                 .front()),
               std::invalid_argument);
}

TEST_F(AssertsTest, NullSupport)
{
  cudf::transform_input inputs[] = {a, b_nulls, cudf::scalar_column_view(t)};
  EXPECT_NO_THROW((void)std::move(
    cudf::transform(cudf::test::get_transform_udf("lerp"),
                    inputs,
                    std::array{cudf::transform_output{cudf::data_type{cudf::type_id::FLOAT32},
                                                      cudf::output_nullability::PRESERVE}},
                    {},
                    std::nullopt)
      ->release()
      .front()));
}

TEST_F(RuntimeSupportTest, TransformProgram)
{
  cudf::transform_input inputs[] = {a, b, cudf::scalar_column_view(t)};
  std::array outputs{cudf::transform_output{cudf::data_type{cudf::type_id::FLOAT32},
                                            cudf::output_nullability::ALL_VALID}};

  cudf::transform_program program{cudf::test::get_transform_udf("lerp"), inputs, outputs, {}};

  std::array input_specs{
    cudf::transform_input_spec{.type = cudf::type_id::FLOAT32},
    cudf::transform_input_spec{.type = cudf::type_id::FLOAT32},
    cudf::transform_input_spec{.type = cudf::type_id::FLOAT32, .is_scalar = true}};
  std::array output_specs{cudf::transform_output_spec{
    .type = cudf::type_id::FLOAT32, .nullability = cudf::output_nullability::ALL_VALID}};
  cudf::transform_program spec_program{
    cudf::test::get_transform_udf("lerp"), input_specs, output_specs};

  auto expected =
    cudf::transform(cudf::test::get_transform_udf("lerp"), inputs, outputs, {}, std::nullopt);
  auto result = program.run(inputs, outputs, {}, std::nullopt);
  CUDF_TEST_EXPECT_TABLES_EQUAL(expected->view(), result->view());

  result = spec_program.run(inputs, outputs, {}, std::nullopt);
  CUDF_TEST_EXPECT_TABLES_EQUAL(expected->view(), result->view());

  std::array compatible_outputs{cudf::transform_output{cudf::data_type{cudf::type_id::FLOAT32},
                                                       cudf::output_nullability::PRESERVE}};
  result = spec_program.run(inputs, compatible_outputs, {}, std::nullopt);
  CUDF_TEST_EXPECT_TABLES_EQUAL(expected->view(), result->view());
}

TEST_F(RuntimeSupportTest, TransformProgramIncompatibleSpecifications)
{
  cudf::transform_input inputs[] = {a, b, cudf::scalar_column_view(t)};
  std::array outputs{cudf::transform_output{cudf::data_type{cudf::type_id::FLOAT32},
                                            cudf::output_nullability::ALL_VALID}};
  cudf::transform_program program{cudf::test::get_transform_udf("lerp"), inputs, outputs, {}};

  auto integers = cudf::test::fixed_width_column_wrapper<int32_t>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  cudf::transform_input different_type[] = {a, integers, cudf::scalar_column_view(t)};
  EXPECT_THROW((void)program.run(different_type, outputs, {}, std::nullopt), std::invalid_argument);

  cudf::transform_input different_input_kind[] = {a, b, a};
  EXPECT_THROW((void)program.run(different_input_kind, outputs, {}, std::nullopt),
               std::invalid_argument);

  std::array<cudf::transform_input, 2> fewer_inputs{a, b};
  EXPECT_THROW((void)program.run(fewer_inputs, outputs, {}, std::nullopt), std::invalid_argument);

  std::array different_output_type{cudf::transform_output{cudf::data_type{cudf::type_id::INT32},
                                                          cudf::output_nullability::ALL_VALID}};
  EXPECT_THROW((void)program.run(inputs, different_output_type, {}, std::nullopt),
               std::invalid_argument);

  std::array<cudf::transform_output, 0> fewer_outputs{};
  EXPECT_THROW((void)program.run(inputs, fewer_outputs, {}, std::nullopt), std::invalid_argument);
}

struct UnaryOperationIntegrationTest : public cudf::test::BaseFixture {};

template <class dtype, class Op, class Data>
void test_udf(cudf::transform_udf udf, Op op, Data data_init, cudf::size_type size)
{
  auto all_valid = cudf::test::iterators::no_nulls();
  auto data_iter = cudf::detail::make_counting_transform_iterator(0, data_init);

  cudf::test::fixed_width_column_wrapper<dtype, typename decltype(data_iter)::value_type> in(
    data_iter, data_iter + size, all_valid);

  cudf::transform_input inputs[]    = {in};
  std::unique_ptr<cudf::column> out = std::move(
    cudf::transform(udf,
                    inputs,
                    std::array{cudf::transform_output{cudf::data_type(cudf::type_to_id<dtype>()),
                                                      cudf::output_nullability::PRESERVE}},
                    {},
                    std::nullopt)
      ->release()
      .front());

  ASSERT_UNARY<dtype, dtype>(out->view(), in, op);
}

TEST_F(UnaryOperationIntegrationTest, Transform_FP32_FP32)
{
  // c = a*a*a*a

  using dtype    = float;
  auto op        = [](dtype a) { return a * a * a * a; };
  auto data_init = [](cudf::size_type row) { return row % 3; };

  test_udf<dtype>(cudf::test::get_transform_udf("fourth_power"), op, data_init, 500);
  test_udf<dtype>(cudf::test::get_ptx_transform_udf("fourth_power"), op, data_init, 500);

  test_udf<dtype>(cudf::test::get_transform_udf("fourth_power"), op, data_init, 0);
  test_udf<dtype>(cudf::test::get_ptx_transform_udf("fourth_power"), op, data_init, 0);
}

TEST_F(UnaryOperationIntegrationTest, Transform_INT32_INT32)
{
  // c = a * a - a

  using dtype    = int;
  auto op        = [](dtype a) { return a * a - a; };
  auto data_init = [](cudf::size_type row) { return row % 78; };

  test_udf<dtype>(cudf::test::get_transform_udf("mulsub"), op, data_init, 500);
  test_udf<dtype>(cudf::test::get_ptx_transform_udf("mulsub"), op, data_init, 500);
}

TEST_F(UnaryOperationIntegrationTest, Transform_INT8_INT8)
{
  // Capitalize all the lower case letters

  using dtype    = int8_t;
  auto op        = [](dtype a) { return std::toupper(a); };
  auto data_init = [](cudf::size_type row) { return 'a' + (row % 26); };

  test_udf<dtype>(cudf::test::get_transform_udf("ascii_upper"), op, data_init, 500);
  test_udf<dtype>(cudf::test::get_ptx_transform_udf("ascii_upper"), op, data_init, 500);
}

TEST_F(UnaryOperationIntegrationTest, Transform_Datetime)
{
  // Add one day to timestamp in microseconds

  using dtype = cudf::timestamp_us;
  auto op     = [](dtype a) {
    using dur = cuda::std::chrono::duration<int32_t, cuda::std::ratio<86400>>;
    return static_cast<cudf::timestamp_us>(a + dur{1});
  };
  auto random_eng = cudf::test::UniformRandomGenerator<cudf::timestamp_us::rep>(0, 100000000);
  auto data_init  = [&random_eng](cudf::size_type row) { return random_eng.generate(); };

  test_udf<dtype>(cudf::test::get_transform_udf("add_day"), op, data_init, 500);
}

TEST_F(UnaryOperationIntegrationTest, TransformDictionaryString)
{
  // non-nullable
  {
    auto a =
      cudf::test::strings_column_wrapper{
        "eee", "aaa", "ddd", "bbb", "ccc", "ccc", "ccc", "eee", "aaa"}
        .release();

    auto a_encoded = cudf::dictionary::encode(a->view());

    cudf::transform_input inputs[] = {*a_encoded};

    auto out = std::move(
      cudf::transform(cudf::test::get_transform_udf("decode_string"),
                      inputs,
                      std::array{cudf::transform_output{cudf::data_type{cudf::type_id::STRING},
                                                        cudf::output_nullability::PRESERVE}},
                      {},
                      std::nullopt)
        ->release()
        .front());

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(out->view(), a->view());
  }

  // nullable
  {
    auto a =
      cudf::test::strings_column_wrapper{
        {"eee", "aaa", "ddd", "bbb", "ccc", "ccc", "ccc", "eee", "aaa"},
        {true, true, true, false, true, true, true, true, true}}
        .release();

    auto a_encoded = cudf::dictionary::encode(a->view());

    cudf::transform_input inputs[] = {*a_encoded};

    auto out = std::move(
      cudf::transform(cudf::test::get_transform_udf("decode_string"),
                      inputs,
                      std::array{cudf::transform_output{cudf::data_type{cudf::type_id::STRING},
                                                        cudf::output_nullability::PRESERVE}},
                      {},
                      std::nullopt)
        ->release()
        .front());

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(out->view(), a->view());
  }
}

TEST_F(UnaryOperationIntegrationTest, TransformDictionaryFloat)
{
  // non-nullable
  {
    auto a = cudf::test::fixed_width_column_wrapper<float>(
               {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 5.0F, 5.0F, 1.0F, 2.0F})
               .release();
    auto a_encoded                 = cudf::dictionary::encode(a->view());
    cudf::transform_input inputs[] = {*a_encoded};

    auto out = std::move(
      cudf::transform(cudf::test::get_transform_udf("decode_float"),
                      inputs,
                      std::array{cudf::transform_output{cudf::data_type{cudf::type_id::FLOAT32},
                                                        cudf::output_nullability::PRESERVE}},
                      {},
                      std::nullopt)
        ->release()
        .front());

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(out->view(), a->view());

    auto out_ptx = std::move(
      cudf::transform(cudf::test::get_ptx_transform_udf("decode_float"),
                      inputs,
                      std::array{cudf::transform_output{cudf::data_type{cudf::type_id::FLOAT32},
                                                        cudf::output_nullability::PRESERVE}},
                      {},
                      std::nullopt)
        ->release()
        .front());

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(out_ptx->view(), a->view());
  }

  // nullable
  {
    auto a = cudf::test::fixed_width_column_wrapper<float>(
               {{1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 5.0F, 5.0F, 1.0F, 2.0F},
                {true, true, true, true, true, false, true, true, true}})
               .release();

    auto a_encoded                 = cudf::dictionary::encode(a->view());
    cudf::transform_input inputs[] = {*a_encoded};

    auto out = std::move(
      cudf::transform(cudf::test::get_transform_udf("decode_float"),
                      inputs,
                      std::array{cudf::transform_output{cudf::data_type{cudf::type_id::FLOAT32},
                                                        cudf::output_nullability::PRESERVE}},
                      {},
                      std::nullopt)
        ->release()
        .front());

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(out->view(), a->view());

    auto out_ptx = std::move(
      cudf::transform(cudf::test::get_ptx_transform_udf("decode_float"),
                      inputs,
                      std::array{cudf::transform_output{cudf::data_type{cudf::type_id::FLOAT32},
                                                        cudf::output_nullability::PRESERVE}},
                      {},
                      std::nullopt)
        ->release()
        .front());

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(out_ptx->view(), a->view());
  }

  // empty column
  {
    auto a_empty                   = cudf::test::fixed_width_column_wrapper<float>({}).release();
    auto a_encoded                 = cudf::dictionary::encode(a_empty->view());
    cudf::transform_input inputs[] = {*a_encoded};

    auto out = std::move(
      cudf::transform(cudf::test::get_transform_udf("decode_float"),
                      inputs,
                      std::array{cudf::transform_output{cudf::data_type{cudf::type_id::FLOAT32},
                                                        cudf::output_nullability::PRESERVE}},
                      {},
                      std::nullopt)
        ->release()
        .front());

    EXPECT_EQ(out->size(), 0);
  }

  // sliced
  {
    auto a = cudf::test::fixed_width_column_wrapper<float>(
               {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 5.0F, 5.0F, 1.0F, 2.0F})
               .release();

    auto a_encoded     = cudf::dictionary::encode(a->view());
    auto sliced_input  = cudf::slice(a_encoded->view(), {2, 7}).front();
    auto sliced_expect = cudf::slice(a->view(), {2, 7}).front();

    cudf::transform_input inputs[] = {sliced_input};

    auto out = std::move(
      cudf::transform(cudf::test::get_transform_udf("decode_float"),
                      inputs,
                      std::array{cudf::transform_output{cudf::data_type{cudf::type_id::FLOAT32},
                                                        cudf::output_nullability::PRESERVE}},
                      {},
                      std::nullopt)
        ->release()
        .front());
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(out->view(), sliced_expect);

    auto out_ptx = std::move(
      cudf::transform(cudf::test::get_ptx_transform_udf("decode_float"),
                      inputs,
                      std::array{cudf::transform_output{cudf::data_type{cudf::type_id::FLOAT32},
                                                        cudf::output_nullability::PRESERVE}},
                      {},
                      std::nullopt)
        ->release()
        .front());

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(out_ptx->view(), sliced_expect);
  }
}

struct TernaryOperationTest : public cudf::test::BaseFixture {};

TEST_F(TernaryOperationTest, TransformWithScalar)
{
  using T = float;

  constexpr T A   = 90;
  constexpr T B   = 100;
  constexpr T C   = 5;
  constexpr T OUT = (A + B) * C;

  std::vector<T> a_host(200, A);
  std::vector<T> b_host(200, B);
  std::vector<T> c_host(1, C);
  std::vector<T> expected_host(200, OUT);

  cudf::test::fixed_width_column_wrapper<T> a(a_host.begin(), a_host.end());
  cudf::test::fixed_width_column_wrapper<T> b(b_host.begin(), b_host.end());
  cudf::test::fixed_width_column_wrapper<T> c(c_host.begin(), c_host.end());
  cudf::test::fixed_width_column_wrapper<T> expected(expected_host.begin(), expected_host.end());

  cudf::transform_input cuda_inputs[]       = {a, b, cudf::scalar_column_view(c)};
  std::unique_ptr<cudf::column> cuda_result = std::move(
    cudf::transform(cudf::test::get_transform_udf("addmul"),
                    cuda_inputs,
                    std::array{cudf::transform_output{cudf::data_type(cudf::type_to_id<T>()),
                                                      cudf::output_nullability::PRESERVE}},
                    {},
                    std::nullopt)
      ->release()
      .front());

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*cuda_result, expected);

  cudf::transform_input ptx_inputs[]       = {a, b, cudf::scalar_column_view(c)};
  std::unique_ptr<cudf::column> ptx_result = std::move(
    cudf::transform(cudf::test::get_ptx_transform_udf("addmul"),
                    ptx_inputs,
                    std::array{cudf::transform_output{cudf::data_type(cudf::type_to_id<T>()),
                                                      cudf::output_nullability::PRESERVE}},
                    {},
                    std::nullopt)
      ->release()
      .front());

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*ptx_result, expected);
}

template <typename T>
struct TernaryDecimalOperationTest : public cudf::test::BaseFixture {};

TYPED_TEST_SUITE(TernaryDecimalOperationTest, cudf::test::FixedPointTypes);

TYPED_TEST(TernaryDecimalOperationTest, TransformDecimalsAndScalar)
{
  using T = TypeParam;

  T const A(10, numeric::scale_type{0});
  T const B(20, numeric::scale_type{-1});
  T const C(5, numeric::scale_type{-2});
  T const RES = ((A + B) * C);

  std::vector<typename T::rep> a_host(200, A.value());
  std::vector<typename T::rep> b_host(200, B.value());
  std::vector<typename T::rep> c_host(1, C.value());
  std::vector<typename T::rep> expected_host(200, RES.value());

  cudf::test::fixed_point_column_wrapper<typename T::rep> a(
    a_host.begin(), a_host.end(), A.scale());
  cudf::test::fixed_point_column_wrapper<typename T::rep> b(
    b_host.begin(), b_host.end(), B.scale());
  cudf::test::fixed_point_column_wrapper<typename T::rep> c(
    c_host.begin(), c_host.end(), C.scale());
  cudf::test::fixed_point_column_wrapper<typename T::rep> expected(
    expected_host.begin(), expected_host.end(), RES.scale());

  cudf::transform_input inputs[]            = {a, b, cudf::scalar_column_view(c)};
  std::unique_ptr<cudf::column> cuda_result = std::move(
    cudf::transform(
      cudf::test::get_transform_udf(
        std::format("decimal_arithmetic_{}", sizeof(typename T::rep) * 8)),
      inputs,
      std::array{cudf::transform_output{cudf::data_type(cudf::type_to_id<T>(), RES.scale()),
                                        cudf::output_nullability::PRESERVE}},
      {},
      std::nullopt)
      ->release()
      .front());

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*cuda_result, expected);
}

struct StringOperationTest : public cudf::test::BaseFixture {};

TEST_F(StringOperationTest, StringComparison)
{
  auto a = cudf::test::strings_column_wrapper{"a", "b", "ccc", "dddd", "eee"};
  auto b = cudf::test::strings_column_wrapper{"a", "b", "d", "dddddd", "e"};
  auto c = cudf::test::strings_column_wrapper{"a", "b", "e", "ddd", "e"};

  auto expected = cudf::test::fixed_width_column_wrapper<bool>{true, true, false, false, true};
  cudf::transform_input inputs[] = {a, b, c};
  auto result                    = std::move(
    cudf::transform(cudf::test::get_transform_udf("compare"),
                    inputs,
                    std::array{cudf::transform_output{cudf::data_type(cudf::type_id::BOOL8),
                                                      cudf::output_nullability::PRESERVE}},
                                       {},
                    std::nullopt)
      ->release()
      .front());

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TEST_F(StringOperationTest, StringContains)
{
  auto a = cudf::test::strings_column_wrapper{"a", "ab", "ccc", "dddd", "eee", "123"};
  auto b = cudf::test::strings_column_wrapper{"aa", "b", "d", "dddddd", "e", "321"};

  auto expected =
    cudf::test::fixed_width_column_wrapper<bool>{false, true, false, false, true, false};
  cudf::transform_input inputs[] = {a, b};
  auto result                    = std::move(
    cudf::transform(cudf::test::get_transform_udf("string_contains"),
                    inputs,
                    std::array{cudf::transform_output{cudf::data_type(cudf::type_id::BOOL8),
                                                      cudf::output_nullability::PRESERVE}},
                                       {},
                    std::nullopt)
      ->release()
      .front());

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TEST_F(StringOperationTest, StringFind)
{
  auto a = cudf::test::strings_column_wrapper{"a", "b", " h", "123"};
  auto b = cudf::test::strings_column_wrapper{"aa", "about", "oh hi", "012345"};

  auto expected = cudf::test::fixed_width_column_wrapper<cudf::size_type>{0, 1, 2, 1};
  cudf::transform_input inputs[] = {a, b};
  auto result                    = std::move(
    cudf::transform(cudf::test::get_transform_udf("string_find"),
                    inputs,
                    std::array{cudf::transform_output{cudf::data_type(cudf::type_id::INT32),
                                                      cudf::output_nullability::PRESERVE}},
                                       {},
                    std::nullopt)
      ->release()
      .front());

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TEST_F(StringOperationTest, MixedTypes)
{
  auto a = cudf::test::strings_column_wrapper{"a", "b", "ccc", "dddd", "eee", "123"};
  auto b = cudf::test::strings_column_wrapper{"a", "b", "d", "dddddd", "e", "321"};
  auto c = cudf::test::fixed_width_column_wrapper<int32_t>{1, 2, 3, 4, 5, 6};
  auto d = cudf::test::fixed_width_column_wrapper<int32_t>{1, 2, 5, 6, 7, 8};

  auto expected =
    cudf::test::fixed_width_column_wrapper<bool>{true, true, false, false, false, false};
  cudf::transform_input inputs[] = {a, b, c, d};
  auto result                    = std::move(
    cudf::transform(cudf::test::get_transform_udf("compare_string_and_integer"),
                    inputs,
                    std::array{cudf::transform_output{cudf::data_type(cudf::type_id::BOOL8),
                                                      cudf::output_nullability::PRESERVE}},
                                       {},
                    std::nullopt)
      ->release()
      .front());

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TEST_F(StringOperationTest, Output)
{
  auto a = cudf::test::strings_column_wrapper{"this", "b", "c", "d", "e", "f"};
  auto b = cudf::test::strings_column_wrapper{"aa", "is", "dd", "ddd", "e", "fff"};
  auto c = cudf::test::strings_column_wrapper{"a", "b", "the", "dddd", "e", "fff"};
  auto d = cudf::test::strings_column_wrapper{"a", "b", "d", "largest", "lexicographical", "test"};
  auto empty = cudf::test::strings_column_wrapper{};

  auto expected =
    cudf::test::strings_column_wrapper{"this", "is", "the", "largest", "lexicographical", "test"};
  cudf::transform_input inputs[] = {a, b, c, d};
  auto result                    = std::move(
    cudf::transform(cudf::test::get_transform_udf("max_string"),
                    inputs,
                    std::array{cudf::transform_output{cudf::data_type(cudf::type_id::STRING),
                                                      cudf::output_nullability::PRESERVE}},
                                       {},
                    std::nullopt)
      ->release()
      .front());

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());

  auto expected_empty                  = cudf::test::strings_column_wrapper{};
  cudf::transform_input empty_inputs[] = {empty, empty, empty, empty};
  auto result_empty                    = std::move(
    cudf::transform(cudf::test::get_transform_udf("max_string"),
                    empty_inputs,
                    std::array{cudf::transform_output{cudf::data_type(cudf::type_id::STRING),
                                                      cudf::output_nullability::PRESERVE}},
                                       {},
                    std::nullopt)
      ->release()
      .front());

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_empty, result_empty->view());
}

TEST_F(StringOperationTest, OutputNullAware)
{
  auto input = cudf::test::strings_column_wrapper{{"alpha", "", "ignored", "omega"},
                                                  {true, true, false, true}};

  cudf::transform_input inputs[]   = {input};
  cudf::transform_output outputs[] = {
    {cudf::data_type{cudf::type_id::STRING}, cudf::output_nullability::PRESERVE}};
  auto result = cudf::transform(
    cudf::test::get_transform_udf("nullable_string_identity", cudf::null_aware::YES),
    inputs,
    outputs,
    {},
    std::nullopt);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(input, result->get_column(0));
}

TEST_F(StringOperationTest, OutputOffsetted)
{
  auto a = cudf::test::strings_column_wrapper{"x", "xx", "xxx", "xxxx", "xxxxx", "xxxxxx"};
  auto b = cudf::test::strings_column_wrapper{"aa", "bb", "cc", "dd", "ee", "ff"};
  auto c = cudf::test::strings_column_wrapper{"1", "2", "3", "4", "5", "6"};

  auto offsets = cudf::test::fixed_width_column_wrapper<int32_t>{0, 4, 9, 15, 22, 30, 39}.release();
  std::vector<std::unique_ptr<cudf::column>> strings_offsets;
  strings_offsets.push_back(std::move(offsets));
  auto expected = cudf::test::strings_column_wrapper{
    "xaa1", "xxbb2", "xxxcc3", "xxxxdd4", "xxxxxee5", "xxxxxxff6"};
  cudf::transform_input inputs[]   = {a, b, c};
  cudf::transform_output outputs[] = {cudf::data_type(cudf::type_id::STRING)};
  auto result                      = cudf::transform(cudf::test::get_transform_udf("concat"),
                                inputs,
                                outputs,
                                std::move(strings_offsets),
                                std::nullopt);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->get_column(0));

  auto program_offsets =
    cudf::test::fixed_width_column_wrapper<int32_t>{0, 4, 9, 15, 22, 30, 39}.release();
  std::vector<std::unique_ptr<cudf::column>> program_string_offsets;
  program_string_offsets.push_back(std::move(program_offsets));
  cudf::transform_program program{
    cudf::test::get_transform_udf("concat"), inputs, outputs, program_string_offsets};

  auto int64_offsets =
    cudf::test::fixed_width_column_wrapper<int64_t>{0, 4, 9, 15, 22, 30, 39}.release();
  std::vector<std::unique_ptr<cudf::column>> int64_string_offsets;
  int64_string_offsets.push_back(std::move(int64_offsets));
  result = program.run(inputs, outputs, std::move(int64_string_offsets), std::nullopt);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->get_column(0));

  EXPECT_THROW((void)program.run(inputs, outputs, {}, std::nullopt), std::invalid_argument);
}

TEST_F(StringOperationTest, OutputOffsettedMixed)
{
  auto a = cudf::test::strings_column_wrapper{"x", "xx", "xxx", "xxxx", "xxxxx", "xxxxxx"};
  auto b = cudf::test::strings_column_wrapper{"aa", "bb", "cc", "dd", "ee", "ff"};
  auto c = cudf::test::strings_column_wrapper{"1", "2", "3", "4", "5", "67"};

  auto offsets = cudf::test::fixed_width_column_wrapper<int32_t>{0, 4, 9, 15, 22, 30, 40}.release();
  std::vector<std::unique_ptr<cudf::column>> strings_offsets;

  strings_offsets.push_back(nullptr);
  strings_offsets.push_back(std::move(offsets));
  strings_offsets.push_back(nullptr);

  auto expected_is_odd =
    cudf::test::fixed_width_column_wrapper<bool>{false, true, false, true, false, false};
  auto expected_strings = cudf::test::strings_column_wrapper{
    "xaa1", "xxbb2", "xxxcc3", "xxxxdd4", "xxxxxee5", "xxxxxxff67"};
  auto expected_sizes = cudf::test::fixed_width_column_wrapper<int32_t>{3, 4, 5, 6, 7, 8};
  auto expected       = cudf::table_view({expected_sizes, expected_strings, expected_is_odd});
  cudf::transform_input inputs[]   = {a, b, c};
  cudf::transform_output outputs[] = {{cudf::data_type(cudf::type_id::INT32)},
                                      {cudf::data_type(cudf::type_id::STRING)},
                                      {cudf::data_type(cudf::type_id::BOOL8)}};
  auto result = cudf::transform(cudf::test::get_transform_udf("concat_with_metadata"),
                                inputs,
                                outputs,
                                std::move(strings_offsets),
                                std::nullopt);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result->view());
}

TEST_F(StringOperationTest, StringConcat)
{
  auto first_name = cudf::test::strings_column_wrapper{
    "John", "Mia", "Abd", "Mendes", "Arya", "John", "François", "José", "Søren", "张"};
  auto last_name = cudf::test::strings_column_wrapper{
    "Doe", "Folk", "Louis", "Xi", "Serenity", "Scott", "Ольга", "Łukasz", "Zoë", "伟"};
  cuda::device_buffer<char> scratch(cudf::get_default_stream(),
                                    cudf::get_current_device_resource_ref(),
                                    100 * static_cast<cudf::column_view>(first_name).size(),
                                    cuda::no_init);
  auto scratch_sizes = cudf::test::fixed_width_column_wrapper<int32_t>{100};

  auto expected                  = cudf::test::strings_column_wrapper{"John Doe",
                                                     "Mia Folk",
                                                     "Abd Louis",
                                                     "Mendes Xi",
                                                     "Arya Serenity",
                                                     "John Scott",
                                                     "François Ольга",
                                                     "José Łukasz",
                                                     "Søren Zoë",
                                                     "张 伟"};
  cudf::transform_input inputs[] = {cudf::user_data{scratch.data()},
                                    first_name,
                                    last_name,
                                    cudf::scalar_column_view(scratch_sizes)};

  auto result = std::move(
    cudf::transform(cudf::test::get_transform_udf("format_name", cudf::null_aware::NO, true),
                    inputs,
                    std::array{cudf::transform_output{cudf::data_type(cudf::type_id::STRING),
                                                      cudf::output_nullability::PRESERVE}},
                    {},
                    std::nullopt)
      ->release()
      .front());

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TEST_F(StringOperationTest, EmptyInput)
{
  auto empty                          = cudf::test::strings_column_wrapper();
  cudf::transform_input bool_inputs[] = {empty, empty};
  auto result                         = std::move(
    cudf::transform(cudf::test::get_transform_udf("string_contains"),
                    bool_inputs,
                    std::array{cudf::transform_output{cudf::data_type(cudf::type_id::BOOL8),
                                                      cudf::output_nullability::PRESERVE}},
                                            {},
                    std::nullopt)
      ->release()
      .front());
  EXPECT_EQ(0, result->size());

  cudf::transform_input str_inputs[] = {empty, empty};
  result                             = std::move(
    cudf::transform(cudf::test::get_transform_udf("matching_string"),
                    str_inputs,
                    std::array{cudf::transform_output{cudf::data_type(cudf::type_id::STRING),
                                                      cudf::output_nullability::PRESERVE}},
                                                {},
                    std::nullopt)
      ->release()
      .front());
  EXPECT_EQ(0, result->size());
}

TEST_F(StringOperationTest, MultiOutput)
{
  auto a = cudf::test::strings_column_wrapper{
    "415 555 1234", "212 867 5309", "214 947 5689", "712 312 1863", "777 214 9063", "964 123 2934"};
  auto expected_area_code =
    cudf::test::strings_column_wrapper{"415", "212", "214", "712", "777", "964"};
  auto expected_prefix =
    cudf::test::strings_column_wrapper{"555", "867", "947", "312", "214", "123"};
  auto expected_line_number =
    cudf::test::strings_column_wrapper{"1234", "5309", "5689", "1863", "9063", "2934"};

  auto expected = cudf::table_view({expected_area_code, expected_prefix, expected_line_number});

  cudf::transform_input inputs[]   = {a};
  cudf::transform_output outputs[] = {{.type = cudf::data_type(cudf::type_id::STRING)},
                                      {.type = cudf::data_type(cudf::type_id::STRING)},
                                      {.type = cudf::data_type(cudf::type_id::STRING)}};
  auto result                      = cudf::transform(
    cudf::test::get_transform_udf("capture_group"), inputs, outputs, {}, std::nullopt);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result->view());
}

struct NullTest : public cudf::test::BaseFixture {
 protected:
  float const LOW         = 100.0F;
  float const HIGH        = 200.0F;
  float const T           = 0.25F;
  float const RES         = LOW - T * LOW + T * HIGH;
  cudf::size_type const N = 200;

  std::vector<float> low_host      = std::vector<float>(N, LOW);
  std::vector<float> high_host     = std::vector<float>(N, HIGH);
  std::vector<float> t_host        = std::vector<float>(N, T);
  std::vector<float> t_scalar_host = std::vector<float>(1, T);
  std::vector<float> expected_host = std::vector<float>(N, RES);

  auto fourth() { return cudf::test::iterators::valids_at_multiples_of(4); }

  auto fifth() { return cudf::test::iterators::valids_at_multiples_of(5); }
};

TEST_F(NullTest, ColumnNulls)
{
  auto low =
    cudf::test::fixed_width_column_wrapper<float>(low_host.begin(), low_host.end(), fourth())
      .release();
  auto high =
    cudf::test::fixed_width_column_wrapper<float>(high_host.begin(), high_host.end(), fifth())
      .release();
  auto t = cudf::test::fixed_width_column_wrapper<float>(t_host.begin(), t_host.end()).release();
  auto expected =
    cudf::test::fixed_width_column_wrapper<float>(expected_host.begin(), expected_host.end())
      .release();

  auto [expected_mask, expect_null_count] = cudf::bitmask_and(cudf::table_view({*low, *high}));

  expected->set_null_mask(std::move(expected_mask), expect_null_count);

  cudf::transform_input cuda_inputs[] = {*low, *high, *t};
  auto cuda_result                    = std::move(
    cudf::transform(cudf::test::get_transform_udf("lerp"),
                    cuda_inputs,
                    std::array{cudf::transform_output{cudf::data_type(cudf::type_id::FLOAT32),
                                                      cudf::output_nullability::PRESERVE}},
                                       {},
                    std::nullopt)
      ->release()
      .front());

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*cuda_result, *expected);

  cudf::transform_input ptx_inputs[] = {*low, *high, *t};
  auto ptx_result                    = std::move(
    cudf::transform(cudf::test::get_ptx_transform_udf("lerp"),
                    ptx_inputs,
                    std::array{cudf::transform_output{cudf::data_type(cudf::type_id::FLOAT32),
                                                      cudf::output_nullability::PRESERVE}},
                                       {},
                    std::nullopt)
      ->release()
      .front());

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*ptx_result, *expected);
}

TEST_F(NullTest, ColumnNulls_And_Scalar)
{
  auto low =
    cudf::test::fixed_width_column_wrapper<float>(low_host.begin(), low_host.end(), fourth())
      .release();
  auto high =
    cudf::test::fixed_width_column_wrapper<float>(high_host.begin(), high_host.end(), fifth())
      .release();
  auto t_scalar =
    cudf::test::fixed_width_column_wrapper<float>(t_scalar_host.begin(), t_scalar_host.end())
      .release();
  auto expected =
    cudf::test::fixed_width_column_wrapper<float>(expected_host.begin(), expected_host.end())
      .release();

  auto [expected_mask, expect_null_count] = cudf::bitmask_and(cudf::table_view({*low, *high}));

  expected->set_null_mask(std::move(expected_mask), expect_null_count);

  cudf::transform_input cuda_inputs[] = {*low, *high, cudf::scalar_column_view(*t_scalar)};
  auto cuda_result                    = std::move(
    cudf::transform(cudf::test::get_transform_udf("lerp"),
                    cuda_inputs,
                    std::array{cudf::transform_output{cudf::data_type(cudf::type_id::FLOAT32),
                                                      cudf::output_nullability::PRESERVE}},
                                       {},
                    std::nullopt)
      ->release()
      .front());

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*cuda_result, *expected);

  cudf::transform_input ptx_inputs[] = {*low, *high, cudf::scalar_column_view(*t_scalar)};
  auto ptx_result                    = std::move(
    cudf::transform(cudf::test::get_ptx_transform_udf("lerp"),
                    ptx_inputs,
                    std::array{cudf::transform_output{cudf::data_type(cudf::type_id::FLOAT32),
                                                      cudf::output_nullability::PRESERVE}},
                                       {},
                    std::nullopt)
      ->release()
      .front());

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*ptx_result, *expected);
}

TEST_F(NullTest, ColumnNulls_And_ScalarNull)
{
  auto low =
    cudf::test::fixed_width_column_wrapper<float>(low_host.begin(), low_host.end(), fourth())
      .release();
  auto high =
    cudf::test::fixed_width_column_wrapper<float>(high_host.begin(), high_host.end(), fifth())
      .release();
  auto t_scalar = cudf::test::fixed_width_column_wrapper<float>(
                    t_scalar_host.begin(), t_scalar_host.end(), {false})
                    .release();
  auto expected =
    cudf::test::fixed_width_column_wrapper<float>(expected_host.begin(), expected_host.end())
      .release();

  expected->set_null_mask(cudf::create_null_mask(low_host.size(), cudf::mask_state::ALL_NULL),
                          low->size());
  cudf::transform_input cuda_inputs[] = {*low, *high, cudf::scalar_column_view(*t_scalar)};

  auto cuda_result = std::move(
    cudf::transform(cudf::test::get_transform_udf("lerp"),
                    cuda_inputs,
                    std::array{cudf::transform_output{cudf::data_type(cudf::type_id::FLOAT32),
                                                      cudf::output_nullability::PRESERVE}},
                    {},
                    std::nullopt)
      ->release()
      .front());

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*cuda_result, *expected);

  cudf::transform_input ptx_inputs[] = {*low, *high, cudf::scalar_column_view(*t_scalar)};

  auto ptx_result = std::move(
    cudf::transform(cudf::test::get_ptx_transform_udf("lerp"),
                    ptx_inputs,
                    std::array{cudf::transform_output{cudf::data_type(cudf::type_id::FLOAT32),
                                                      cudf::output_nullability::PRESERVE}},
                    {},
                    std::nullopt)
      ->release()
      .front());

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*ptx_result, *expected);
}

TEST_F(NullTest, IsNull)
{
  auto value = cudf::test::fixed_width_column_wrapper<float>({1.0f, 2.0f, 3.0f, 4.0f, 5.0f},
                                                             {false, false, true, false, true})
                 .release();

  auto expected = cudf::test::fixed_width_column_wrapper<bool>({true, true, false, true, false});

  cudf::transform_input inputs[] = {*value};
  auto result                    = std::move(
    cudf::transform(cudf::test::get_transform_udf("is_null", cudf::null_aware::YES),
                    inputs,
                    std::array{cudf::transform_output{cudf::data_type(cudf::type_id::BOOL8),
                                                      cudf::output_nullability::ALL_VALID}},
                                       {},
                    std::nullopt)
      ->release()
      .front());

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expected);
}

TEST_F(NullTest, NullProject)
{
  auto low =
    cudf::test::fixed_width_column_wrapper<float>(low_host.begin(), low_host.end(), fourth())
      .release();
  auto high =
    cudf::test::fixed_width_column_wrapper<float>(high_host.begin(), high_host.end(), fifth())
      .release();
  auto t = cudf::test::fixed_width_column_wrapper<float>(t_host.begin(), t_host.end()).release();

  auto expected_iter = cudf::detail::make_counting_transform_iterator(
    0, [&](auto i) { return ((i % 5) == 0) && ((i % 4) == 0) ? expected_host[i] : 0.0F; });

  auto expected =
    cudf::test::fixed_width_column_wrapper<float>(expected_iter, expected_iter + low_host.size())
      .release();

  cudf::transform_input inputs[] = {*low, *high, *t};
  auto cuda_result               = std::move(
    cudf::transform(cudf::test::get_transform_udf("null_lerp", cudf::null_aware::YES),
                    inputs,
                    std::array{cudf::transform_output{cudf::data_type(cudf::type_id::FLOAT32),
                                                      cudf::output_nullability::ALL_VALID}},
                                  {},
                    std::nullopt)
      ->release()
      .front());

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*cuda_result, *expected);
}

TEST_F(UnaryOperationIntegrationTest, Transform_ErrorHandling)
{
  cudf::test::fixed_width_column_wrapper<int32_t> a{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  cudf::test::fixed_width_column_wrapper<int32_t> b_fail{1, 2, 3, 4, 5, 6, 0, 8, 9, 10};
  cudf::test::fixed_width_column_wrapper<int32_t> b{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  cudf::test::fixed_width_column_wrapper<int32_t> expected{1, 1, 1, 1, 1, 1, 1, 1, 1, 1};

  {
    cudf::transform_input inputs[]   = {a, b};
    cudf::transform_output outputs[] = {
      {cudf::data_type(cudf::type_id::INT32), cudf::output_nullability::ALL_VALID}};
    std::unique_ptr<cudf::table> result;
    EXPECT_NO_THROW(
      result = cudf::transform(
        cudf::test::get_transform_udf("divide_checked"), inputs, outputs, {}, std::nullopt));

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->get_column(0), expected);
  }

  {
    cudf::transform_input inputs[]   = {a, b_fail};
    cudf::transform_output outputs[] = {
      {cudf::data_type(cudf::type_id::INT32), cudf::output_nullability::ALL_VALID}};
    std::unique_ptr<cudf::table> result;
    EXPECT_THROW(
      result = cudf::transform(
        cudf::test::get_transform_udf("divide_checked"), inputs, outputs, {}, std::nullopt),
      cudf::evaluation_error);
  }
}

}  // namespace transformation
