/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/random.hpp>
#include <cudf_test/table_utilities.hpp>
#include <cudf_test/testing_main.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/detail/iterator.cuh>
#include <cudf/stream_compaction.hpp>
#include <cudf/transform.hpp>

namespace filters {

struct FilterTestFixture : public cudf::test::BaseFixture {
 protected:
  static constexpr char const* udf =
    R"***(
    template<typename T>
    __device__ void is_equal(bool * out, T a, T b) { *out = (a == b); }
    )***";
};

template <typename T>
struct FilterNumericTest : public FilterTestFixture {};

using NumericTypesNotBool =
  cudf::test::Concat<cudf::test::IntegralTypesNotBool, cudf::test::FloatingPointTypes>;

TYPED_TEST_SUITE(FilterNumericTest, NumericTypesNotBool);

TYPED_TEST(FilterNumericTest, NoAssertions)
{
  using T = TypeParam;

  auto a = cudf::test::fixed_width_column_wrapper<T>{{0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
                                                     {1, 0, 1, 1, 1, 1, 1, 1, 0, 0}};
  auto b = cudf::test::fixed_width_column_wrapper<T>{{0, 1, 2, 3, 8, 5, 6, 7, 4, 9},
                                                     {0, 0, 1, 1, 1, 1, 1, 1, 0, 0}};

  auto expected = cudf::test::fixed_width_column_wrapper<T>{{2, 3, 5, 6, 7}, {1, 1, 1, 1, 1}};

  std::vector<std::unique_ptr<cudf::column>> results;

  cudf::filter_input inputs[] = {a, b};

  EXPECT_NO_THROW(
    results = cudf::filter_extended(
      inputs, this->udf, {a}, cudf::udf_source_type::CUDA, std::nullopt, cudf::null_aware::NO));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, results[0]->view());
}

template <typename T>
struct FilterChronoTest : public FilterTestFixture {};

TYPED_TEST_SUITE(FilterChronoTest, cudf::test::ChronoTypes);

TYPED_TEST(FilterChronoTest, NoAssertions)
{
  using T = TypeParam;

  auto a = cudf::test::fixed_width_column_wrapper<T>{
    {T{}, T{}, T{}, T{}, T{}, T{}, T{}, T{}, T{}, T{}}, {1, 0, 1, 1, 1, 1, 1, 1, 0, 0}};
  auto b = cudf::test::fixed_width_column_wrapper<T>{
    {T{}, T{}, T{}, T{}, T{}, T{}, T{}, T{}, T{}, T{}}, {0, 0, 1, 1, 1, 1, 1, 1, 0, 0}};

  auto expected =
    cudf::test::fixed_width_column_wrapper<T>{{T{}, T{}, T{}, T{}, T{}, T{}}, {1, 1, 1, 1, 1, 1}};

  std::vector<std::unique_ptr<cudf::column>> results;
  cudf::filter_input inputs[] = {a, b};
  EXPECT_NO_THROW(
    results = cudf::filter_extended(
      inputs, this->udf, {a}, cudf::udf_source_type::CUDA, std::nullopt, cudf::null_aware::NO));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, results[0]->view());
}

template <typename T>
struct FilterFixedPointTest : public FilterTestFixture {};

TYPED_TEST_SUITE(FilterFixedPointTest, cudf::test::FixedPointTypes);

TYPED_TEST(FilterFixedPointTest, NoAssertions)
{
  using T = TypeParam;

  auto a = cudf::test::fixed_point_column_wrapper<typename T::rep>{
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, {1, 0, 1, 1, 1, 1, 1, 1, 0, 0}, numeric::scale_type{0}};
  auto b = cudf::test::fixed_point_column_wrapper<typename T::rep>{
    {0, 1, 2, 3, 8, 5, 6, 7, 4, 9}, {0, 0, 1, 1, 1, 1, 1, 1, 0, 0}, numeric::scale_type{0}};

  auto expected = cudf::test::fixed_point_column_wrapper<typename T::rep>{
    {2, 3, 5, 6, 7}, {1, 1, 1, 1, 1}, numeric::scale_type{0}};

  std::vector<std::unique_ptr<cudf::column>> results;
  cudf::filter_input inputs[] = {a, b};

  EXPECT_NO_THROW(
    results = cudf::filter_extended(
      inputs, this->udf, {a}, cudf::udf_source_type::CUDA, std::nullopt, cudf::null_aware::NO));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, results[0]->view());
}

TEST_F(FilterTestFixture, StringNoAssertions)
{
  auto a = cudf::test::strings_column_wrapper{{"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"},
                                              {1, 0, 1, 1, 1, 1, 1, 1, 0, 0}};
  auto b = cudf::test::strings_column_wrapper{{"0", "1", "2", "3", "8", "5", "6", "7", "8", "9"},
                                              {0, 0, 1, 1, 1, 1, 1, 1, 0, 0}};

  auto expected = cudf::test::strings_column_wrapper{{"2", "3", "5", "6", "7"}, {1, 1, 1, 1, 1}};

  std::vector<std::unique_ptr<cudf::column>> results;
  cudf::filter_input inputs[] = {a, b};
  EXPECT_NO_THROW(
    results = cudf::filter_extended(
      inputs, this->udf, {a}, cudf::udf_source_type::CUDA, std::nullopt, cudf::null_aware::NO));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, results[0]->view());
}

struct FilterAssertsTest : public FilterTestFixture {};

struct FilterTest : public FilterTestFixture {};

TEST_F(FilterTest, Basic)
{
  auto a = cudf::test::fixed_width_column_wrapper<int32_t>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
                                                           {1, 1, 1, 1, 1, 1, 1, 0, 0, 0});
  std::string cuda = R"***(
__device__ void is_even(bool* out, int32_t a) { *out = (a % 2 == 0); }
  )***";

  cudf::filter_input inputs[] = {a};

  auto result = cudf::filter_extended(
    inputs, cuda, {a}, cudf::udf_source_type::CUDA, std::nullopt, cudf::null_aware::NO);
  auto expected = cudf::test::fixed_width_column_wrapper<int32_t>{{2, 4, 6}, {1, 1, 1}};

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result[0]->view());

  std::string null_cuda = R"***(
__device__ void is_even(cuda::std::optional<bool>* out, cuda::std::optional<int32_t> a) { *out = a.has_value() && (*a % 2 == 0); }
  )***";

  auto null_result = cudf::filter_extended(
    inputs, null_cuda, {a}, cudf::udf_source_type::CUDA, std::nullopt, cudf::null_aware::YES);
  auto null_expected = cudf::test::fixed_width_column_wrapper<int32_t>{{2, 4, 6}, {1, 1, 1}};

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(null_expected, null_result[0]->view());
}

TEST_F(FilterTest, ScalarFilter)
{
  auto a           = cudf::test::fixed_width_column_wrapper<int32_t>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  auto b           = cudf::test::fixed_width_column_wrapper<int32_t>{2};
  std::string cuda = R"***(
__device__ void is_divisible(bool* out, int32_t a, int32_t b) { *out = ((a % b) == 0); }
  )***";
  auto expected    = cudf::test::fixed_width_column_wrapper<int32_t>{{2, 4, 6, 8, 10}};

  cudf::filter_input inputs[] = {a, cudf::scalar_column_view(b)};

  auto result = cudf::filter_extended(
    inputs, cuda, {a}, cudf::udf_source_type::CUDA, std::nullopt, cudf::null_aware::NO);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result[0]->view());
}

TEST_F(FilterTest, MixedTypes)
{
  auto countries = cudf::test::strings_column_wrapper{
    "USA", "Canada", "Mexico", "Brazil", "Argentina", "France", "Germany", "Italy", "Spain"};
  auto average_tmp      = cudf::test::fixed_width_column_wrapper<float>{0, 0, 1, 1, 1, 1, 1, 1, 1};
  auto average_humidity = cudf::test::fixed_width_column_wrapper<float>{0, 0, 0, 0, 1, 1, 1, 1, 1};
  auto timezones        = cudf::test::strings_column_wrapper{
    "EST", "EST", "CST", "BRT", "ART", "CET", "CET", "CET", "CET"};

  std::string cuda = R"***(
__device__ void filter(bool* out,
                       [[maybe_unused]] cudf::string_view country,
                       cudf::string_view tz,
                       float tmp,
                       float hum,
                       float min_tmp,
                       float max_tmp,
                       float min_hum,
                       float max_hum,
                       cudf::string_view tz1,
                       cudf::string_view tz2)
{
  *out = (tmp >= min_tmp && tmp <= max_tmp) && (hum >= min_hum && hum <= max_hum) &&
         (tz == tz1 || tz == tz2);
}
  )***";

  auto min_tmp   = cudf::test::fixed_width_column_wrapper<float>{0.5};
  auto max_tmp   = cudf::test::fixed_width_column_wrapper<float>{1};
  auto min_hum   = cudf::test::fixed_width_column_wrapper<float>{0.5};
  auto max_hum   = cudf::test::fixed_width_column_wrapper<float>{1};
  auto timezone1 = cudf::test::strings_column_wrapper{"CET"};
  auto timezone2 = cudf::test::strings_column_wrapper{"EST"};

  cudf::filter_input inputs[] = {countries,
                                 timezones,
                                 average_tmp,
                                 average_humidity,
                                 cudf::scalar_column_view(min_tmp),
                                 cudf::scalar_column_view(max_tmp),
                                 cudf::scalar_column_view(min_hum),
                                 cudf::scalar_column_view(max_hum),
                                 cudf::scalar_column_view(timezone1),
                                 cudf::scalar_column_view(timezone2)};

  auto result = cudf::filter_extended(inputs,
                                      cuda,
                                      {countries, timezones},
                                      cudf::udf_source_type::CUDA,
                                      std::nullopt,
                                      cudf::null_aware::NO);

  EXPECT_EQ(result.size(), 2);

  auto expected_countries =
    cudf::test::strings_column_wrapper{"France", "Germany", "Italy", "Spain"};

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_countries, result[0]->view());

  auto expected_timezones = cudf::test::strings_column_wrapper{"CET", "CET", "CET", "CET"};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_timezones, result[1]->view());
}

TEST_F(FilterTest, NullableMixedTypes)
{
  auto countries = cudf::test::strings_column_wrapper(
    {"USA", "Canada", "Mexico", "Brazil", "Argentina", "France", "Germany", "Italy", "Spain"},
    {1, 1, 1, 1, 1, 0, 1, 0, 1});
  auto average_tmp      = cudf::test::fixed_width_column_wrapper<float>{0, 0, 1, 1, 1, 1, 1, 1, 1};
  auto average_humidity = cudf::test::fixed_width_column_wrapper<float>{0, 0, 0, 0, 1, 1, 1, 1, 1};
  auto timezones        = cudf::test::strings_column_wrapper{
    "EST", "EST", "CST", "BRT", "ART", "CET", "CET", "CET", "CET"};

  std::string cuda = R"***(
__device__ void filter(bool* out,
                       [[maybe_unused]] cudf::string_view country,
                       cudf::string_view tz,
                       float tmp,
                       float hum,
                       float min_tmp,
                       float max_tmp,
                       float min_hum,
                       float max_hum,
                       cudf::string_view tz1,
                       cudf::string_view tz2)
{
  *out = (tmp >= min_tmp && tmp <= max_tmp) && (hum >= min_hum && hum <= max_hum) &&
         (tz == tz1 || tz == tz2);
}
)***";

  auto min_tmp   = cudf::test::fixed_width_column_wrapper<float>{0.5};
  auto max_tmp   = cudf::test::fixed_width_column_wrapper<float>{1};
  auto min_hum   = cudf::test::fixed_width_column_wrapper<float>{0.5};
  auto max_hum   = cudf::test::fixed_width_column_wrapper<float>{1};
  auto timezone1 = cudf::test::strings_column_wrapper{"CET"};
  auto timezone2 = cudf::test::strings_column_wrapper{"EST"};

  cudf::filter_input inputs[] = {countries,
                                 timezones,
                                 average_tmp,
                                 average_humidity,
                                 cudf::scalar_column_view(min_tmp),
                                 cudf::scalar_column_view(max_tmp),
                                 cudf::scalar_column_view(min_hum),
                                 cudf::scalar_column_view(max_hum),
                                 cudf::scalar_column_view(timezone1),
                                 cudf::scalar_column_view(timezone2)};

  auto result = cudf::filter_extended(inputs,
                                      cuda,
                                      {countries, timezones},
                                      cudf::udf_source_type::CUDA,
                                      std::nullopt,
                                      cudf::null_aware::NO);

  auto expected_countries = cudf::test::strings_column_wrapper({"Germany", "Spain"}, {true, true});

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_countries, result[0]->view());

  auto expected_timezones = cudf::test::strings_column_wrapper({"CET", "CET"});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_timezones, result[1]->view());
}

struct ast_expression_executor {
  static std::unique_ptr<cudf::table> filter(cudf::ast::expression const& expr,
                                             cudf::table_view const& table)
  {
    auto booleans = cudf::compute_column(table, expr);
    return cudf::apply_retention_mask(table, booleans->view());
  }
};

struct jit_expression_executor {
  static std::unique_ptr<cudf::table> filter(cudf::ast::expression const& expr,
                                             cudf::table_view const& table)
  {
    return cudf::filter(table, expr, table);
  }
};

template <typename T>
struct FilterExpressionTest : public cudf::test::BaseFixture {
  std::unique_ptr<cudf::column> a =
    cudf::test::fixed_width_column_wrapper<int32_t>{{1, 2, 3, 4, 5, 6, 7, 8},
                                                    {1, 1, 1, 1, 1, 1, 1, 0}}
      .release();
  std::unique_ptr<cudf::column> b =
    cudf::test::fixed_width_column_wrapper<int32_t>{{1, 8, 3, 4, 5, 6, 7, 8},
                                                    {1, 1, 1, 1, 1, 1, 1, 0}}
      .release();
  std::unique_ptr<cudf::column> bool_a =
    cudf::test::fixed_width_column_wrapper<bool>{
      {false, false, true, true, false, false, true, true}, {1, 1, 1, 1, 1, 0, 0, 0}}
      .release();
  std::unique_ptr<cudf::column> bool_b =
    cudf::test::fixed_width_column_wrapper<bool>{
      {false, true, false, true, true, true, false, true}, {1, 1, 1, 1, 0, 1, 1, 0}}
      .release();

  cudf::table_view table = cudf::table_view({a->view(), b->view(), bool_a->view(), bool_b->view()});
};

using Executors = cudf::test::Types<ast_expression_executor, jit_expression_executor>;

TYPED_TEST_SUITE(FilterExpressionTest, Executors);

TYPED_TEST(FilterExpressionTest, IsNull)
{
  using Executor = TypeParam;

  auto tree          = cudf::ast::tree();
  auto& ref_0        = tree.push(cudf::ast::column_reference(0));
  auto& is_null_expr = tree.push(cudf::ast::operation(cudf::ast::ast_operator::IS_NULL, ref_0));
  auto& filter_expr  = tree.push(cudf::ast::operation(cudf::ast::ast_operator::NOT, is_null_expr));
  auto result        = Executor::filter(filter_expr, this->table);
  auto expected_filter =
    cudf::test::fixed_width_column_wrapper<bool>{{true, true, true, true, true, true, true, false}};
  auto expected_table = cudf::apply_retention_mask(this->table, expected_filter);
  CUDF_TEST_EXPECT_TABLES_EQUAL(expected_table->view(), result->view());
}

TYPED_TEST(FilterExpressionTest, NullEqual)
{
  using Executor = TypeParam;

  auto tree   = cudf::ast::tree();
  auto& ref_0 = tree.push(cudf::ast::column_reference(0));
  auto& ref_1 = tree.push(cudf::ast::column_reference(1));
  auto& null_equal_expr =
    tree.push(cudf::ast::operation(cudf::ast::ast_operator::NULL_EQUAL, ref_0, ref_1));
  auto result = Executor::filter(null_equal_expr, this->table);
  auto expected_filter =
    cudf::test::fixed_width_column_wrapper<bool>{{true, false, true, true, true, true, true, true}};
  auto expected_table = cudf::apply_retention_mask(this->table, expected_filter);
  CUDF_TEST_EXPECT_TABLES_EQUAL(expected_table->view(), result->view());
}

TYPED_TEST(FilterExpressionTest, NullLogicalAnd)
{
  using Executor = TypeParam;

  auto tree   = cudf::ast::tree();
  auto& ref_2 = tree.push(cudf::ast::column_reference(2));
  auto& ref_3 = tree.push(cudf::ast::column_reference(3));
  auto& and_expr =
    tree.push(cudf::ast::operation(cudf::ast::ast_operator::NULL_LOGICAL_AND, ref_2, ref_3));
  auto result          = Executor::filter(and_expr, this->table);
  auto expected_filter = cudf::test::fixed_width_column_wrapper<bool>{
    {false, false, false, true, false, false, false, false}};
  auto expected_table = cudf::apply_retention_mask(this->table, expected_filter);
  CUDF_TEST_EXPECT_TABLES_EQUAL(expected_table->view(), result->view());
}

TYPED_TEST(FilterExpressionTest, NullLogicalOr)
{
  using Executor = TypeParam;

  auto tree   = cudf::ast::tree();
  auto& ref_2 = tree.push(cudf::ast::column_reference(2));
  auto& ref_3 = tree.push(cudf::ast::column_reference(3));
  auto& or_expr =
    tree.push(cudf::ast::operation(cudf::ast::ast_operator::NULL_LOGICAL_OR, ref_2, ref_3));
  auto result          = Executor::filter(or_expr, this->table);
  auto expected_filter = cudf::test::fixed_width_column_wrapper<bool>{
    {false, true, true, true, false, true, false, false}};
  auto expected_table = cudf::apply_retention_mask(this->table, expected_filter);
  CUDF_TEST_EXPECT_TABLES_EQUAL(expected_table->view(), result->view());
}

}  // namespace filters

CUDF_TEST_PROGRAM_MAIN()
