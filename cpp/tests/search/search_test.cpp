/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/testing_main.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/detail/search.hpp>
#include <cudf/search.hpp>

#include <rmm/device_buffer.hpp>

#include <cuda/iterator>
#include <thrust/host_vector.h>

#include <limits>
#include <numeric>

struct SearchTest : public cudf::test::BaseFixture {};

using cudf::numeric_scalar;
using cudf::size_type;
using cudf::string_scalar;
using cudf::test::fixed_width_column_wrapper;

TEST_F(SearchTest, empty_table)
{
  using element_type = int64_t;

  fixed_width_column_wrapper<element_type> column{};
  fixed_width_column_wrapper<element_type> values{0, 7, 10, 11, 30, 32, 40, 47, 50, 90};
  fixed_width_column_wrapper<size_type> expect{0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  std::unique_ptr<cudf::column> result{};

  EXPECT_NO_THROW(result = cudf::lower_bound({cudf::table_view{{column}}},
                                             {cudf::table_view{{values}}},
                                             {cudf::order::ASCENDING},
                                             {cudf::null_order::BEFORE}));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expect);
}

TEST_F(SearchTest, empty_values)
{
  using element_type = int64_t;

  fixed_width_column_wrapper<element_type> column{10, 20, 30, 40, 50};
  fixed_width_column_wrapper<element_type> values{};
  fixed_width_column_wrapper<size_type> expect{};

  std::unique_ptr<cudf::column> result{};

  EXPECT_NO_THROW(result = cudf::lower_bound({cudf::table_view{{column}}},
                                             {cudf::table_view{{values}}},
                                             {cudf::order::ASCENDING},
                                             {cudf::null_order::BEFORE}));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expect);
}

TEST_F(SearchTest, non_null_column__find_first)
{
  using element_type = int64_t;

  fixed_width_column_wrapper<element_type> column{10, 20, 30, 40, 50};
  fixed_width_column_wrapper<element_type> values{0, 7, 10, 11, 30, 32, 40, 47, 50, 90};
  fixed_width_column_wrapper<size_type> expect{0, 0, 0, 1, 2, 3, 3, 4, 4, 5};

  std::unique_ptr<cudf::column> result{};

  EXPECT_NO_THROW(result = cudf::lower_bound({cudf::table_view{{column}}},
                                             {cudf::table_view{{values}}},
                                             {cudf::order::ASCENDING},
                                             {cudf::null_order::BEFORE}));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expect);
}

TEST_F(SearchTest, non_null_column__find_last)
{
  using element_type = int64_t;

  fixed_width_column_wrapper<element_type> column{10, 20, 30, 40, 50};
  fixed_width_column_wrapper<element_type> values{0, 7, 10, 11, 30, 32, 40, 47, 50, 90};
  fixed_width_column_wrapper<size_type> expect{0, 0, 1, 1, 3, 3, 4, 4, 5, 5};

  std::unique_ptr<cudf::column> result{};

  EXPECT_NO_THROW(result = cudf::upper_bound({cudf::table_view{{column}}},
                                             {cudf::table_view{{values}}},
                                             {cudf::order::ASCENDING},
                                             {cudf::null_order::BEFORE}));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expect);
}

TEST_F(SearchTest, non_null_column_desc__find_first)
{
  using element_type = int64_t;

  fixed_width_column_wrapper<element_type> column{50, 40, 30, 20, 10};
  fixed_width_column_wrapper<element_type> values{0, 7, 10, 11, 30, 32, 40, 47, 50, 90};
  fixed_width_column_wrapper<size_type> expect{5, 5, 4, 4, 2, 2, 1, 1, 0, 0};

  std::unique_ptr<cudf::column> result{};

  EXPECT_NO_THROW(result = cudf::lower_bound({cudf::table_view{{column}}},
                                             {cudf::table_view{{values}}},
                                             {cudf::order::DESCENDING},
                                             {cudf::null_order::BEFORE}));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expect);
}

TEST_F(SearchTest, non_null_column_desc__find_last)
{
  using element_type = int64_t;

  fixed_width_column_wrapper<element_type> column{50, 40, 30, 20, 10};
  fixed_width_column_wrapper<element_type> values{0, 7, 10, 11, 30, 32, 40, 47, 50, 90};
  fixed_width_column_wrapper<size_type> expect{5, 5, 5, 4, 3, 2, 2, 1, 1, 0};

  std::unique_ptr<cudf::column> result{};

  EXPECT_NO_THROW(result = cudf::upper_bound({cudf::table_view{{column}}},
                                             {cudf::table_view{{values}}},
                                             {cudf::order::DESCENDING},
                                             {cudf::null_order::BEFORE}));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expect);
}

TEST_F(SearchTest, nullable_column__find_last__nulls_as_smallest)
{
  using element_type = int64_t;

  fixed_width_column_wrapper<element_type> column{{10, 60, 10, 20, 30, 40, 50},
                                                  {0, 0, 1, 1, 1, 1, 1}};
  fixed_width_column_wrapper<element_type> values{{8, 8, 10, 11, 30, 32, 40, 47, 50, 90},
                                                  {0, 1, 1, 1, 1, 1, 1, 1, 1, 1}};

  fixed_width_column_wrapper<size_type> expect{2, 2, 3, 3, 5, 5, 6, 6, 7, 7};

  std::unique_ptr<cudf::column> result{};

  EXPECT_NO_THROW(result = cudf::upper_bound({cudf::table_view{{column}}},
                                             {cudf::table_view{{values}}},
                                             {cudf::order::ASCENDING},
                                             {cudf::null_order::BEFORE}));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expect);
}

TEST_F(SearchTest, nullable_column__find_first__nulls_as_smallest)
{
  using element_type = int64_t;

  fixed_width_column_wrapper<element_type> column{{10, 60, 10, 20, 30, 40, 50},
                                                  {0, 0, 1, 1, 1, 1, 1}};
  fixed_width_column_wrapper<element_type> values{{8, 8, 10, 11, 30, 32, 40, 47, 50, 90},
                                                  {0, 1, 1, 1, 1, 1, 1, 1, 1, 1}};

  fixed_width_column_wrapper<size_type> expect{0, 2, 2, 3, 4, 5, 5, 6, 6, 7};

  std::unique_ptr<cudf::column> result{};

  EXPECT_NO_THROW(result = cudf::lower_bound({cudf::table_view{{column}}},
                                             {cudf::table_view{{values}}},
                                             {cudf::order::ASCENDING},
                                             {cudf::null_order::BEFORE}));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expect);
}

TEST_F(SearchTest, nullable_column__find_last__nulls_as_largest)
{
  using element_type = int64_t;

  fixed_width_column_wrapper<element_type> column{{10, 20, 30, 40, 50, 10, 60},
                                                  {1, 1, 1, 1, 1, 0, 0}};
  fixed_width_column_wrapper<element_type> values{{8, 10, 11, 30, 32, 40, 47, 50, 90, 8},
                                                  {1, 1, 1, 1, 1, 1, 1, 1, 1, 0}};
  fixed_width_column_wrapper<size_type> expect{0, 1, 1, 3, 3, 4, 4, 5, 5, 7};

  std::unique_ptr<cudf::column> result{};

  EXPECT_NO_THROW(result = cudf::upper_bound({cudf::table_view{{column}}},
                                             {cudf::table_view{{values}}},
                                             {cudf::order::ASCENDING},
                                             {cudf::null_order::AFTER}));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expect);
}

TEST_F(SearchTest, nullable_column__find_first__nulls_as_largest)
{
  using element_type = int64_t;

  fixed_width_column_wrapper<element_type> column{{10, 20, 30, 40, 50, 10, 60},
                                                  {1, 1, 1, 1, 1, 0, 0}};
  fixed_width_column_wrapper<element_type> values{{8, 10, 11, 30, 32, 40, 47, 50, 90, 8},
                                                  {1, 1, 1, 1, 1, 1, 1, 1, 1, 0}};
  fixed_width_column_wrapper<size_type> expect{0, 0, 1, 2, 3, 3, 4, 4, 5, 5};

  std::unique_ptr<cudf::column> result{};

  EXPECT_NO_THROW(result = cudf::lower_bound({cudf::table_view{{column}}},
                                             {cudf::table_view{{values}}},
                                             {cudf::order::ASCENDING},
                                             {cudf::null_order::AFTER}));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expect);
}

TEST_F(SearchTest, table__find_first)
{
  fixed_width_column_wrapper<int32_t> column_0{10, 20, 20, 20, 20, 20, 50};
  fixed_width_column_wrapper<float> column_1{5.0, .5, .5, .7, .7, .7, .7};
  fixed_width_column_wrapper<int8_t> column_2{90, 77, 78, 61, 62, 63, 41};

  fixed_width_column_wrapper<int32_t> values_0{0,  0,  0,  0,  10, 10, 10, 10, 10,
                                               10, 10, 10, 11, 20, 20, 20, 20, 20,
                                               20, 20, 20, 20, 20, 20, 30, 50, 60};
  fixed_width_column_wrapper<float> values_1{0., 0., 6., 5., 0., 5., 5., 5., 5., 6., 6., 6., 9., 0.,
                                             .5, .5, .5, .5, .6, .6, .6, .7, .7, .7, .7, .7, .5};
  fixed_width_column_wrapper<int8_t> values_2{0,  91, 0,  91, 0,  79, 90, 91, 77, 80, 90, 91, 91, 0,
                                              76, 77, 78, 30, 65, 77, 78, 80, 62, 78, 64, 41, 20};

  fixed_width_column_wrapper<size_type> expect{0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1,
                                               1, 1, 2, 1, 3, 3, 3, 6, 4, 6, 6, 6, 7};

  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(column_0.release());
  columns.push_back(column_1.release());
  columns.push_back(column_2.release());

  std::vector<std::unique_ptr<cudf::column>> values;
  values.push_back(values_0.release());
  values.push_back(values_1.release());
  values.push_back(values_2.release());

  cudf::table input_table(std::move(columns));
  cudf::table values_table(std::move(values));

  std::vector<cudf::order> order_flags{
    {cudf::order::ASCENDING, cudf::order::ASCENDING, cudf::order::ASCENDING}};
  std::vector<cudf::null_order> null_order_flags{
    {cudf::null_order::BEFORE, cudf::null_order::BEFORE, cudf::null_order::BEFORE}};

  std::unique_ptr<cudf::column> result{};

  EXPECT_NO_THROW(result =
                    cudf::lower_bound(input_table, values_table, order_flags, null_order_flags));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expect);
}

TEST_F(SearchTest, table__find_last)
{
  fixed_width_column_wrapper<int32_t> column_0{10, 20, 20, 20, 20, 20, 50};
  fixed_width_column_wrapper<float> column_1{5.0, .5, .5, .7, .7, .7, .7};
  fixed_width_column_wrapper<int8_t> column_2{90, 77, 78, 61, 62, 63, 41};

  fixed_width_column_wrapper<int32_t> values_0{0,  0,  0,  0,  10, 10, 10, 10, 10,
                                               10, 10, 10, 11, 20, 20, 20, 20, 20,
                                               20, 20, 20, 20, 20, 20, 30, 50, 60};
  fixed_width_column_wrapper<float> values_1{0., 0., 6., 5., 0., 5., 5., 5., 5., 6., 6., 6., 9., 0.,
                                             .5, .5, .5, .5, .6, .6, .6, .7, .7, .7, .7, .7, .5};
  fixed_width_column_wrapper<int8_t> values_2{0,  91, 0,  91, 0,  79, 90, 91, 77, 80, 90, 91, 91, 0,
                                              76, 77, 78, 30, 65, 77, 78, 80, 62, 78, 64, 41, 20};

  fixed_width_column_wrapper<size_type> expect{0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1,
                                               1, 2, 3, 1, 3, 3, 3, 6, 5, 6, 6, 7, 7};

  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(column_0.release());
  columns.push_back(column_1.release());
  columns.push_back(column_2.release());

  std::vector<std::unique_ptr<cudf::column>> values;
  values.push_back(values_0.release());
  values.push_back(values_1.release());
  values.push_back(values_2.release());

  cudf::table input_table(std::move(columns));
  cudf::table values_table(std::move(values));

  std::vector<cudf::order> order_flags{
    {cudf::order::ASCENDING, cudf::order::ASCENDING, cudf::order::ASCENDING}};
  std::vector<cudf::null_order> null_order_flags{
    {cudf::null_order::BEFORE, cudf::null_order::BEFORE, cudf::null_order::BEFORE}};

  std::unique_ptr<cudf::column> result{};

  EXPECT_NO_THROW(result =
                    cudf::upper_bound(input_table, values_table, order_flags, null_order_flags));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expect);
}

TEST_F(SearchTest, table_partial_desc__find_first)
{
  fixed_width_column_wrapper<int32_t> column_0{50, 20, 20, 20, 20, 20, 10};
  fixed_width_column_wrapper<float> column_1{.7, .5, .5, .7, .7, .7, 5.0};
  fixed_width_column_wrapper<int8_t> column_2{41, 78, 77, 63, 62, 61, 90};

  fixed_width_column_wrapper<int32_t> values_0{0,  0,  0,  0,  10, 10, 10, 10, 10,
                                               10, 10, 10, 11, 20, 20, 20, 20, 20,
                                               20, 20, 20, 20, 20, 20, 30, 50, 60};
  fixed_width_column_wrapper<float> values_1{0., 0., 6., 5., 0., 5., 5., 5., 5., 6., 6., 6., 9., 0.,
                                             .5, .5, .5, .5, .6, .6, .6, .7, .7, .7, .7, .7, .5};
  fixed_width_column_wrapper<int8_t> values_2{0,  91, 0,  91, 0,  79, 90, 91, 77, 80, 90, 91, 91, 0,
                                              76, 77, 78, 30, 65, 77, 78, 80, 62, 78, 64, 41, 20};

  fixed_width_column_wrapper<size_type> expect{7, 7, 7, 7, 6, 7, 6, 6, 7, 7, 7, 7, 6, 1,
                                               3, 2, 1, 3, 3, 3, 3, 3, 4, 3, 1, 0, 0};

  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(column_0.release());
  columns.push_back(column_1.release());
  columns.push_back(column_2.release());

  std::vector<std::unique_ptr<cudf::column>> values;
  values.push_back(values_0.release());
  values.push_back(values_1.release());
  values.push_back(values_2.release());

  cudf::table input_table(std::move(columns));
  cudf::table values_table(std::move(values));

  std::vector<cudf::order> order_flags{
    {cudf::order::DESCENDING, cudf::order::ASCENDING, cudf::order::DESCENDING}};
  std::vector<cudf::null_order> null_order_flags{
    {cudf::null_order::BEFORE, cudf::null_order::BEFORE, cudf::null_order::BEFORE}};

  std::unique_ptr<cudf::column> result{};

  EXPECT_NO_THROW(result =
                    cudf::lower_bound(input_table, values_table, order_flags, null_order_flags));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expect);
}

TEST_F(SearchTest, table_partial_desc__find_last)
{
  fixed_width_column_wrapper<int32_t> column_0{50, 20, 20, 20, 20, 20, 10};
  fixed_width_column_wrapper<float> column_1{.7, .5, .5, .7, .7, .7, 5.0};
  fixed_width_column_wrapper<int8_t> column_2{41, 78, 77, 63, 62, 61, 90};

  fixed_width_column_wrapper<int32_t> values_0{0,  0,  0,  0,  10, 10, 10, 10, 10,
                                               10, 10, 10, 11, 20, 20, 20, 20, 20,
                                               20, 20, 20, 20, 20, 20, 30, 50, 60};
  fixed_width_column_wrapper<float> values_1{0., 0., 6., 5., 0., 5., 5., 5., 5., 6., 6., 6., 9., 0.,
                                             .5, .5, .5, .5, .6, .6, .6, .7, .7, .7, .7, .7, .5};
  fixed_width_column_wrapper<int8_t> values_2{0,  91, 0,  91, 0,  79, 90, 91, 77, 80, 90, 91, 91, 0,
                                              76, 77, 78, 30, 65, 77, 78, 80, 62, 78, 64, 41, 20};

  fixed_width_column_wrapper<size_type> expect{7, 7, 7, 7, 6, 7, 7, 6, 7, 7, 7, 7, 6, 1,
                                               3, 3, 2, 3, 3, 3, 3, 3, 5, 3, 1, 1, 0};

  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(column_0.release());
  columns.push_back(column_1.release());
  columns.push_back(column_2.release());

  std::vector<std::unique_ptr<cudf::column>> values;
  values.push_back(values_0.release());
  values.push_back(values_1.release());
  values.push_back(values_2.release());

  cudf::table input_table(std::move(columns));
  cudf::table values_table(std::move(values));

  std::vector<cudf::order> order_flags{
    {cudf::order::DESCENDING, cudf::order::ASCENDING, cudf::order::DESCENDING}};
  std::vector<cudf::null_order> null_order_flags{
    {cudf::null_order::BEFORE, cudf::null_order::BEFORE, cudf::null_order::BEFORE}};

  std::unique_ptr<cudf::column> result{};

  EXPECT_NO_THROW(result =
                    cudf::upper_bound(input_table, values_table, order_flags, null_order_flags));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expect);
}

TEST_F(SearchTest, table__find_first__nulls_as_smallest)
{
  fixed_width_column_wrapper<int32_t> column_0{{30, 10, 10, 20, 20, 20, 20, 20, 20, 20, 50},
                                               {0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}};
  fixed_width_column_wrapper<float> column_1{{.5, 6.0, 5.0, .5, .5, .5, .5, .7, .7, .7, .7},
                                             {1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1}};
  fixed_width_column_wrapper<int8_t> column_2{{50, 95, 90, 79, 76, 77, 78, 61, 62, 63, 41},
                                              {1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1}};

  fixed_width_column_wrapper<int32_t> values_0{{10, 40, 20}, {1, 0, 1}};
  fixed_width_column_wrapper<float> values_1{{6., .5, .5}, {0, 1, 1}};
  fixed_width_column_wrapper<int8_t> values_2{{95, 50, 77}, {1, 1, 0}};

  fixed_width_column_wrapper<size_type> expect{1, 0, 3};

  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(column_0.release());
  columns.push_back(column_1.release());
  columns.push_back(column_2.release());

  std::vector<std::unique_ptr<cudf::column>> values;
  values.push_back(values_0.release());
  values.push_back(values_1.release());
  values.push_back(values_2.release());

  cudf::table input_table(std::move(columns));
  cudf::table values_table(std::move(values));

  std::vector<cudf::order> order_flags{
    {cudf::order::ASCENDING, cudf::order::ASCENDING, cudf::order::ASCENDING}};
  std::vector<cudf::null_order> null_order_flags{
    {cudf::null_order::BEFORE, cudf::null_order::BEFORE, cudf::null_order::BEFORE}};

  std::unique_ptr<cudf::column> result{};

  EXPECT_NO_THROW(result =
                    cudf::lower_bound(input_table, values_table, order_flags, null_order_flags));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expect);
}

TEST_F(SearchTest, table__find_last__nulls_as_smallest)
{
  fixed_width_column_wrapper<int32_t> column_0{{30, 10, 10, 20, 20, 20, 20, 20, 20, 20, 50},
                                               {0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}};
  fixed_width_column_wrapper<float> column_1{{.5, 6.0, 5.0, .5, .5, .5, .5, .7, .7, .7, .7},
                                             {1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1}};
  fixed_width_column_wrapper<int8_t> column_2{{50, 90, 95, 79, 76, 77, 78, 61, 62, 63, 41},
                                              {1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1}};

  fixed_width_column_wrapper<int32_t> values_0{{10, 40, 20}, {1, 0, 1}};
  fixed_width_column_wrapper<float> values_1{{6., .5, .5}, {0, 1, 1}};
  fixed_width_column_wrapper<int8_t> values_2{{95, 50, 77}, {1, 1, 0}};

  fixed_width_column_wrapper<size_type> expect{2, 1, 5};

  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(column_0.release());
  columns.push_back(column_1.release());
  columns.push_back(column_2.release());

  std::vector<std::unique_ptr<cudf::column>> values;
  values.push_back(values_0.release());
  values.push_back(values_1.release());
  values.push_back(values_2.release());

  cudf::table input_table(std::move(columns));
  cudf::table values_table(std::move(values));

  std::vector<cudf::order> order_flags{
    {cudf::order::ASCENDING, cudf::order::ASCENDING, cudf::order::ASCENDING}};
  std::vector<cudf::null_order> null_order_flags{
    {cudf::null_order::BEFORE, cudf::null_order::BEFORE, cudf::null_order::BEFORE}};

  std::unique_ptr<cudf::column> result{};

  EXPECT_NO_THROW(result =
                    cudf::upper_bound(input_table, values_table, order_flags, null_order_flags));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expect);
}

TEST_F(SearchTest, table__find_first__nulls_as_largest)
{
  fixed_width_column_wrapper<int32_t> column_0{{10, 10, 20, 20, 20, 20, 20, 20, 20, 50, 30},
                                               {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0}};
  fixed_width_column_wrapper<float> column_1{{5.0, 6.0, .5, .5, .5, .5, .7, .7, .7, .7, .5},
                                             {1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1}};
  fixed_width_column_wrapper<int8_t> column_2{{90, 95, 77, 78, 79, 76, 61, 62, 63, 41, 50},
                                              {1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1}};

  fixed_width_column_wrapper<int32_t> values_0{{10, 40, 20}, {1, 0, 1}};
  fixed_width_column_wrapper<float> values_1{{6., .5, .5}, {0, 1, 1}};
  fixed_width_column_wrapper<int8_t> values_2{{95, 50, 77}, {1, 1, 0}};

  fixed_width_column_wrapper<size_type> expect{1, 10, 4};

  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(column_0.release());
  columns.push_back(column_1.release());
  columns.push_back(column_2.release());

  std::vector<std::unique_ptr<cudf::column>> values;
  values.push_back(values_0.release());
  values.push_back(values_1.release());
  values.push_back(values_2.release());

  cudf::table input_table(std::move(columns));
  cudf::table values_table(std::move(values));

  std::vector<cudf::order> order_flags{
    {cudf::order::ASCENDING, cudf::order::ASCENDING, cudf::order::ASCENDING}};
  std::vector<cudf::null_order> null_order_flags{
    {cudf::null_order::AFTER, cudf::null_order::AFTER, cudf::null_order::AFTER}};

  std::unique_ptr<cudf::column> result{};

  EXPECT_NO_THROW(result =
                    cudf::lower_bound(input_table, values_table, order_flags, null_order_flags));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expect);
}

TEST_F(SearchTest, table__find_last__nulls_as_largest)
{
  fixed_width_column_wrapper<int32_t> column_0{{10, 10, 20, 20, 20, 20, 20, 20, 20, 50, 30},
                                               {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0}};
  fixed_width_column_wrapper<float> column_1{{5.0, 6.0, .5, .5, .5, .5, .7, .7, .7, .7, .5},
                                             {1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1}};
  fixed_width_column_wrapper<int8_t> column_2{{90, 95, 77, 78, 79, 76, 61, 62, 63, 41, 50},
                                              {1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1}};

  fixed_width_column_wrapper<int32_t> values_0{{10, 40, 20}, {1, 0, 1}};
  fixed_width_column_wrapper<float> values_1{{6., .5, .5}, {0, 1, 1}};
  fixed_width_column_wrapper<int8_t> values_2{{95, 50, 77}, {1, 1, 0}};

  fixed_width_column_wrapper<size_type> expect{2, 11, 6};

  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(column_0.release());
  columns.push_back(column_1.release());
  columns.push_back(column_2.release());

  std::vector<std::unique_ptr<cudf::column>> values;
  values.push_back(values_0.release());
  values.push_back(values_1.release());
  values.push_back(values_2.release());

  cudf::table input_table(std::move(columns));
  cudf::table values_table(std::move(values));

  std::vector<cudf::order> order_flags{
    {cudf::order::ASCENDING, cudf::order::ASCENDING, cudf::order::ASCENDING}};
  std::vector<cudf::null_order> null_order_flags{
    {cudf::null_order::AFTER, cudf::null_order::AFTER, cudf::null_order::AFTER}};

  std::unique_ptr<cudf::column> result{};

  EXPECT_NO_THROW(result =
                    cudf::upper_bound(input_table, values_table, order_flags, null_order_flags));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expect);
}

TEST_F(SearchTest, contains_true)
{
  using element_type = int64_t;
  bool expect        = true;
  bool result        = false;

  fixed_width_column_wrapper<element_type> column{0, 1, 17, 19, 23, 29, 71};
  numeric_scalar<element_type> scalar{23};

  result = cudf::contains(column, scalar);

  ASSERT_EQ(result, expect);
}

TEST_F(SearchTest, contains_false)
{
  using element_type = int64_t;
  bool expect        = false;
  bool result        = false;

  fixed_width_column_wrapper<element_type> column{0, 1, 17, 19, 23, 29, 71};
  numeric_scalar<element_type> scalar{24};

  result = cudf::contains(column, scalar);

  ASSERT_EQ(result, expect);
}

TEST_F(SearchTest, contains_empty_value)
{
  using element_type = int64_t;
  bool expect        = false;
  bool result        = false;

  fixed_width_column_wrapper<element_type> column{0, 1, 17, 19, 23, 29, 71};
  numeric_scalar<element_type> scalar{23, false};

  result = cudf::contains(column, scalar);

  ASSERT_EQ(result, expect);
}

TEST_F(SearchTest, contains_empty_column)
{
  using element_type = int64_t;
  bool expect        = false;
  bool result        = false;

  fixed_width_column_wrapper<element_type> column{};
  numeric_scalar<element_type> scalar{24};

  result = cudf::contains(column, scalar);

  ASSERT_EQ(result, expect);
}

TEST_F(SearchTest, contains_nullable_column_true)
{
  using element_type = int64_t;
  bool result        = false;
  bool expect        = true;

  fixed_width_column_wrapper<element_type> column{{0, 1, 17, 19, 23, 29, 71},
                                                  {0, 0, 1, 1, 1, 1, 1}};
  numeric_scalar<element_type> scalar{23};

  result = cudf::contains(column, scalar);

  ASSERT_EQ(result, expect);
}

TEST_F(SearchTest, contains_nullable_column_false)
{
  using element_type = int64_t;
  bool result        = false;
  bool expect        = false;

  fixed_width_column_wrapper<element_type> column{{0, 1, 17, 19, 23, 29, 71},
                                                  {0, 0, 1, 1, 0, 1, 1}};
  numeric_scalar<element_type> scalar{23};

  result = cudf::contains(column, scalar);

  ASSERT_EQ(result, expect);
}

TEST_F(SearchTest, empty_table_string)
{
  std::vector<char const*> h_col_strings{};
  std::vector<char const*> h_val_strings{"0", "10", "11", "30", "32", "40", "47", "50", "7", "90"};

  cudf::test::strings_column_wrapper column(
    h_col_strings.begin(),
    h_col_strings.end(),
    cuda::transform_iterator(h_col_strings.begin(), [](auto str) { return str != nullptr; }));

  cudf::test::strings_column_wrapper values(
    h_val_strings.begin(),
    h_val_strings.end(),
    cuda::transform_iterator(h_val_strings.begin(), [](auto str) { return str != nullptr; }));

  fixed_width_column_wrapper<size_type> expect{0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  std::unique_ptr<cudf::column> result{};

  EXPECT_NO_THROW(result = cudf::lower_bound({cudf::table_view{{column}}},
                                             {cudf::table_view{{values}}},
                                             {cudf::order::ASCENDING},
                                             {cudf::null_order::BEFORE}));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expect);
}

TEST_F(SearchTest, empty_values_string)
{
  std::vector<char const*> h_col_strings{"10", "20", "30", "40", "50"};
  std::vector<char const*> h_val_strings{};

  cudf::test::strings_column_wrapper column(
    h_col_strings.begin(),
    h_col_strings.end(),
    cuda::transform_iterator(h_col_strings.begin(), [](auto str) { return str != nullptr; }));

  cudf::test::strings_column_wrapper values(
    h_val_strings.begin(),
    h_val_strings.end(),
    cuda::transform_iterator(h_val_strings.begin(), [](auto str) { return str != nullptr; }));

  fixed_width_column_wrapper<size_type> expect{};

  std::unique_ptr<cudf::column> result{};

  EXPECT_NO_THROW(result = cudf::lower_bound({cudf::table_view{{column}}},
                                             {cudf::table_view{{values}}},
                                             {cudf::order::ASCENDING},
                                             {cudf::null_order::BEFORE}));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expect);
}

TEST_F(SearchTest, non_null_column__find_first_string)
{
  std::vector<char const*> h_col_strings{"10", "20", "30", "40", "50"};
  std::vector<char const*> h_val_strings{
    "00", "07", "10", "11", "30", "32", "40", "47", "50", "90"};

  cudf::test::strings_column_wrapper column(
    h_col_strings.begin(),
    h_col_strings.end(),
    cuda::transform_iterator(h_col_strings.begin(), [](auto str) { return str != nullptr; }));

  cudf::test::strings_column_wrapper values(
    h_val_strings.begin(),
    h_val_strings.end(),
    cuda::transform_iterator(h_val_strings.begin(), [](auto str) { return str != nullptr; }));

  fixed_width_column_wrapper<size_type> expect{0, 0, 0, 1, 2, 3, 3, 4, 4, 5};

  std::unique_ptr<cudf::column> result{};

  EXPECT_NO_THROW(result = cudf::lower_bound({cudf::table_view{{column}}},
                                             {cudf::table_view{{values}}},
                                             {cudf::order::ASCENDING},
                                             {cudf::null_order::BEFORE}));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expect);
}

TEST_F(SearchTest, non_null_column__find_last_string)
{
  std::vector<char const*> h_col_strings{"10", "20", "30", "40", "50"};
  std::vector<char const*> h_val_strings{
    "00", "07", "10", "11", "30", "32", "40", "47", "50", "90"};

  cudf::test::strings_column_wrapper column(
    h_col_strings.begin(),
    h_col_strings.end(),
    cuda::transform_iterator(h_col_strings.begin(), [](auto str) { return str != nullptr; }));

  cudf::test::strings_column_wrapper values(
    h_val_strings.begin(),
    h_val_strings.end(),
    cuda::transform_iterator(h_val_strings.begin(), [](auto str) { return str != nullptr; }));

  fixed_width_column_wrapper<size_type> expect{0, 0, 1, 1, 3, 3, 4, 4, 5, 5};

  std::unique_ptr<cudf::column> result{};

  EXPECT_NO_THROW(result = cudf::upper_bound({cudf::table_view{{column}}},
                                             {cudf::table_view{{values}}},
                                             {cudf::order::ASCENDING},
                                             {cudf::null_order::BEFORE}));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expect);
}

TEST_F(SearchTest, non_null_column_desc__find_first_string)
{
  std::vector<char const*> h_col_strings{"50", "40", "30", "20", "10"};
  std::vector<char const*> h_val_strings{
    "00", "07", "10", "11", "30", "32", "40", "47", "50", "90"};

  cudf::test::strings_column_wrapper column(
    h_col_strings.begin(),
    h_col_strings.end(),
    cuda::transform_iterator(h_col_strings.begin(), [](auto str) { return str != nullptr; }));

  cudf::test::strings_column_wrapper values(
    h_val_strings.begin(),
    h_val_strings.end(),
    cuda::transform_iterator(h_val_strings.begin(), [](auto str) { return str != nullptr; }));

  fixed_width_column_wrapper<size_type> expect{5, 5, 4, 4, 2, 2, 1, 1, 0, 0};

  std::unique_ptr<cudf::column> result{};

  EXPECT_NO_THROW(result = cudf::lower_bound({cudf::table_view{{column}}},
                                             {cudf::table_view{{values}}},
                                             {cudf::order::DESCENDING},
                                             {cudf::null_order::BEFORE}));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expect);
}

TEST_F(SearchTest, non_null_column_desc__find_last_string)
{
  std::vector<char const*> h_col_strings{"50", "40", "30", "20", "10"};
  std::vector<char const*> h_val_strings{
    "00", "07", "10", "11", "30", "32", "40", "47", "50", "90"};

  cudf::test::strings_column_wrapper column(
    h_col_strings.begin(),
    h_col_strings.end(),
    cuda::transform_iterator(h_col_strings.begin(), [](auto str) { return str != nullptr; }));

  cudf::test::strings_column_wrapper values(
    h_val_strings.begin(),
    h_val_strings.end(),
    cuda::transform_iterator(h_val_strings.begin(), [](auto str) { return str != nullptr; }));

  fixed_width_column_wrapper<size_type> expect{5, 5, 5, 4, 3, 2, 2, 1, 1, 0};

  std::unique_ptr<cudf::column> result{};

  EXPECT_NO_THROW(result = cudf::upper_bound({cudf::table_view{{column}}},
                                             {cudf::table_view{{values}}},
                                             {cudf::order::DESCENDING},
                                             {cudf::null_order::BEFORE}));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expect);
}

TEST_F(SearchTest, nullable_column__find_last__nulls_as_smallest_string)
{
  std::vector<char const*> h_col_strings{nullptr, nullptr, "10", "20", "30", "40", "50"};
  std::vector<char const*> h_val_strings{
    nullptr, "08", "10", "11", "30", "32", "40", "47", "50", "90"};

  cudf::test::strings_column_wrapper column(
    h_col_strings.begin(),
    h_col_strings.end(),
    cuda::transform_iterator(h_col_strings.begin(), [](auto str) { return str != nullptr; }));

  cudf::test::strings_column_wrapper values(
    h_val_strings.begin(),
    h_val_strings.end(),
    cuda::transform_iterator(h_val_strings.begin(), [](auto str) { return str != nullptr; }));

  fixed_width_column_wrapper<size_type> expect{2, 2, 3, 3, 5, 5, 6, 6, 7, 7};

  std::unique_ptr<cudf::column> result{};

  EXPECT_NO_THROW(result = cudf::upper_bound({cudf::table_view{{column}}},
                                             {cudf::table_view{{values}}},
                                             {cudf::order::ASCENDING},
                                             {cudf::null_order::BEFORE}));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expect);
}

TEST_F(SearchTest, nullable_column__find_first__nulls_as_smallest_string)
{
  std::vector<char const*> h_col_strings{nullptr, nullptr, "10", "20", "30", "40", "50"};
  std::vector<char const*> h_val_strings{
    nullptr, "08", "10", "11", "30", "32", "40", "47", "50", "90"};

  cudf::test::strings_column_wrapper column(
    h_col_strings.begin(),
    h_col_strings.end(),
    cuda::transform_iterator(h_col_strings.begin(), [](auto str) { return str != nullptr; }));

  cudf::test::strings_column_wrapper values(
    h_val_strings.begin(),
    h_val_strings.end(),
    cuda::transform_iterator(h_val_strings.begin(), [](auto str) { return str != nullptr; }));

  fixed_width_column_wrapper<size_type> expect{0, 2, 2, 3, 4, 5, 5, 6, 6, 7};

  std::unique_ptr<cudf::column> result{};

  EXPECT_NO_THROW(result = cudf::lower_bound({cudf::table_view{{column}}},
                                             {cudf::table_view{{values}}},
                                             {cudf::order::ASCENDING},
                                             {cudf::null_order::BEFORE}));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expect);
}

TEST_F(SearchTest, nullable_column__find_last__nulls_as_largest_string)
{
  std::vector<char const*> h_col_strings{"10", "20", "30", "40", "50", nullptr, nullptr};
  std::vector<char const*> h_val_strings{
    "08", "10", "11", "30", "32", "40", "47", "50", "90", nullptr};

  cudf::test::strings_column_wrapper column(
    h_col_strings.begin(),
    h_col_strings.end(),
    cuda::transform_iterator(h_col_strings.begin(), [](auto str) { return str != nullptr; }));

  cudf::test::strings_column_wrapper values(
    h_val_strings.begin(),
    h_val_strings.end(),
    cuda::transform_iterator(h_val_strings.begin(), [](auto str) { return str != nullptr; }));

  fixed_width_column_wrapper<size_type> expect{0, 1, 1, 3, 3, 4, 4, 5, 5, 7};

  std::unique_ptr<cudf::column> result{};

  EXPECT_NO_THROW(result = cudf::upper_bound({cudf::table_view{{column}}},
                                             {cudf::table_view{{values}}},
                                             {cudf::order::ASCENDING},
                                             {cudf::null_order::AFTER}));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expect);
}

TEST_F(SearchTest, non_null_column__nullable_values__find_last__nulls_as_largest_string)
{
  cudf::test::strings_column_wrapper column({"N", "N", "N", "N", "Y", "Y", "Y", "Y"},
                                            {1, 1, 1, 1, 1, 1, 1, 1});

  cudf::test::strings_column_wrapper values({"Y", "", "N"}, {1, 0, 1});

  fixed_width_column_wrapper<size_type> expect{8, 8, 4};

  std::unique_ptr<cudf::column> result{};

  EXPECT_NO_THROW(result = cudf::upper_bound({cudf::table_view{{column}}},
                                             {cudf::table_view{{values}}},
                                             {cudf::order::ASCENDING},
                                             {cudf::null_order::AFTER}));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expect);
}

TEST_F(SearchTest, nullable_column__find_first__nulls_as_largest_string)
{
  std::vector<char const*> h_col_strings{"10", "20", "30", "40", "50", nullptr, nullptr};
  std::vector<char const*> h_val_strings{
    "08", "10", "11", "30", "32", "40", "47", "50", "90", nullptr};

  cudf::test::strings_column_wrapper column(
    h_col_strings.begin(),
    h_col_strings.end(),
    cuda::transform_iterator(h_col_strings.begin(), [](auto str) { return str != nullptr; }));

  cudf::test::strings_column_wrapper values(
    h_val_strings.begin(),
    h_val_strings.end(),
    cuda::transform_iterator(h_val_strings.begin(), [](auto str) { return str != nullptr; }));

  fixed_width_column_wrapper<size_type> expect{0, 0, 1, 2, 3, 3, 4, 4, 5, 5};

  std::unique_ptr<cudf::column> result{};

  EXPECT_NO_THROW(result = cudf::lower_bound({cudf::table_view{{column}}},
                                             {cudf::table_view{{values}}},
                                             {cudf::order::ASCENDING},
                                             {cudf::null_order::AFTER}));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expect);
}

TEST_F(SearchTest, table__find_first_string)
{
  std::vector<char const*> h_col_0_strings{"10", "20", "20", "20", "20", "20", "50"};
  std::vector<char const*> h_col_2_strings{"90", "77", "78", "61", "62", "63", "41"};

  std::vector<char const*> h_val_0_strings{"0",  "0",  "0",  "0",  "10", "10", "10", "10", "10",
                                           "10", "10", "10", "11", "20", "20", "20", "20", "20",
                                           "20", "20", "20", "20", "20", "20", "30", "50", "60"};
  std::vector<char const*> h_val_2_strings{"0",  "91", "0",  "91", "0",  "79", "90", "91", "77",
                                           "80", "90", "91", "91", "00", "76", "77", "78", "30",
                                           "65", "77", "78", "80", "62", "78", "64", "41", "20"};

  fixed_width_column_wrapper<float> column_1{5.0, .5, .5, .7, .7, .7, .7};
  fixed_width_column_wrapper<float> values_1{0., 0., 6., 5., 0., 5., 5., 5., 5., 6., 6., 6., 9., 0.,
                                             .5, .5, .5, .5, .6, .6, .6, .7, .7, .7, .7, .7, .5};

  fixed_width_column_wrapper<size_type> expect{0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1,
                                               1, 1, 2, 1, 3, 3, 3, 6, 4, 6, 6, 6, 7};

  cudf::test::strings_column_wrapper column_0(
    h_col_0_strings.begin(),
    h_col_0_strings.end(),
    cuda::transform_iterator(h_col_0_strings.begin(), [](auto str) { return str != nullptr; }));

  cudf::test::strings_column_wrapper column_2(
    h_col_2_strings.begin(),
    h_col_2_strings.end(),
    cuda::transform_iterator(h_col_2_strings.begin(), [](auto str) { return str != nullptr; }));

  cudf::test::strings_column_wrapper values_0(
    h_val_0_strings.begin(),
    h_val_0_strings.end(),
    cuda::transform_iterator(h_val_0_strings.begin(), [](auto str) { return str != nullptr; }));

  cudf::test::strings_column_wrapper values_2(
    h_val_2_strings.begin(),
    h_val_2_strings.end(),
    cuda::transform_iterator(h_val_2_strings.begin(), [](auto str) { return str != nullptr; }));

  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(column_0.release());
  columns.push_back(column_1.release());
  columns.push_back(column_2.release());

  std::vector<std::unique_ptr<cudf::column>> values;
  values.push_back(values_0.release());
  values.push_back(values_1.release());
  values.push_back(values_2.release());

  cudf::table input_table(std::move(columns));
  cudf::table values_table(std::move(values));

  std::vector<cudf::order> order_flags{
    {cudf::order::ASCENDING, cudf::order::ASCENDING, cudf::order::ASCENDING}};
  std::vector<cudf::null_order> null_order_flags{
    {cudf::null_order::BEFORE, cudf::null_order::BEFORE, cudf::null_order::BEFORE}};

  std::unique_ptr<cudf::column> result{};

  EXPECT_NO_THROW(result =
                    cudf::lower_bound(input_table, values_table, order_flags, null_order_flags));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expect);
}

TEST_F(SearchTest, table__find_last_string)
{
  std::vector<char const*> h_col_0_strings{"10", "20", "20", "20", "20", "20", "50"};
  std::vector<char const*> h_col_2_strings{"90", "77", "78", "61", "62", "63", "41"};

  std::vector<char const*> h_val_0_strings{"0",  "0",  "0",  "0",  "10", "10", "10", "10", "10",
                                           "10", "10", "10", "11", "20", "20", "20", "20", "20",
                                           "20", "20", "20", "20", "20", "20", "30", "50", "60"};
  std::vector<char const*> h_val_2_strings{"0",  "91", "0",  "91", "0",  "79", "90", "91", "77",
                                           "80", "90", "91", "91", "00", "76", "77", "78", "30",
                                           "65", "77", "78", "80", "62", "78", "64", "41", "20"};

  fixed_width_column_wrapper<float> column_1{5.0, .5, .5, .7, .7, .7, .7};
  fixed_width_column_wrapper<float> values_1{0., 0., 6., 5., 0., 5., 5., 5., 5., 6., 6., 6., 9., 0.,
                                             .5, .5, .5, .5, .6, .6, .6, .7, .7, .7, .7, .7, .5};

  fixed_width_column_wrapper<size_type> expect{0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1,
                                               1, 2, 3, 1, 3, 3, 3, 6, 5, 6, 6, 7, 7};

  cudf::test::strings_column_wrapper column_0(
    h_col_0_strings.begin(),
    h_col_0_strings.end(),
    cuda::transform_iterator(h_col_0_strings.begin(), [](auto str) { return str != nullptr; }));

  cudf::test::strings_column_wrapper column_2(
    h_col_2_strings.begin(),
    h_col_2_strings.end(),
    cuda::transform_iterator(h_col_2_strings.begin(), [](auto str) { return str != nullptr; }));

  cudf::test::strings_column_wrapper values_0(
    h_val_0_strings.begin(),
    h_val_0_strings.end(),
    cuda::transform_iterator(h_val_0_strings.begin(), [](auto str) { return str != nullptr; }));

  cudf::test::strings_column_wrapper values_2(
    h_val_2_strings.begin(),
    h_val_2_strings.end(),
    cuda::transform_iterator(h_val_2_strings.begin(), [](auto str) { return str != nullptr; }));

  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(column_0.release());
  columns.push_back(column_1.release());
  columns.push_back(column_2.release());

  std::vector<std::unique_ptr<cudf::column>> values;
  values.push_back(values_0.release());
  values.push_back(values_1.release());
  values.push_back(values_2.release());

  cudf::table input_table(std::move(columns));
  cudf::table values_table(std::move(values));

  std::vector<cudf::order> order_flags{
    {cudf::order::ASCENDING, cudf::order::ASCENDING, cudf::order::ASCENDING}};
  std::vector<cudf::null_order> null_order_flags{
    {cudf::null_order::BEFORE, cudf::null_order::BEFORE, cudf::null_order::BEFORE}};

  std::unique_ptr<cudf::column> result{};

  EXPECT_NO_THROW(result =
                    cudf::upper_bound(input_table, values_table, order_flags, null_order_flags));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expect);
}

TEST_F(SearchTest, table_partial_desc__find_first_string)
{
  std::vector<char const*> h_col_0_strings{"50", "20", "20", "20", "20", "20", "10"};
  std::vector<char const*> h_col_2_strings{"41", "78", "77", "63", "62", "61", "90"};

  std::vector<char const*> h_val_0_strings{"0",  "0",  "0",  "0",  "10", "10", "10", "10", "10",
                                           "10", "10", "10", "11", "20", "20", "20", "20", "20",
                                           "20", "20", "20", "20", "20", "20", "30", "50", "60"};
  std::vector<char const*> h_val_2_strings{"0",  "91", "0",  "91", "0",  "79", "90", "91", "77",
                                           "80", "90", "91", "91", "00", "76", "77", "78", "30",
                                           "65", "77", "78", "80", "62", "78", "64", "41", "20"};

  fixed_width_column_wrapper<float> column_1{.7, .5, .5, .7, .7, .7, 5.0};
  fixed_width_column_wrapper<float> values_1{0., 0., 6., 5., 0., 5., 5., 5., 5., 6., 6., 6., 9., 0.,
                                             .5, .5, .5, .5, .6, .6, .6, .7, .7, .7, .7, .7, .5};

  fixed_width_column_wrapper<size_type> expect{7, 7, 7, 7, 6, 7, 6, 6, 7, 7, 7, 7, 6, 1,
                                               3, 2, 1, 3, 3, 3, 3, 3, 4, 3, 1, 0, 0};

  cudf::test::strings_column_wrapper column_0(
    h_col_0_strings.begin(),
    h_col_0_strings.end(),
    cuda::transform_iterator(h_col_0_strings.begin(), [](auto str) { return str != nullptr; }));

  cudf::test::strings_column_wrapper column_2(
    h_col_2_strings.begin(),
    h_col_2_strings.end(),
    cuda::transform_iterator(h_col_2_strings.begin(), [](auto str) { return str != nullptr; }));

  cudf::test::strings_column_wrapper values_0(
    h_val_0_strings.begin(),
    h_val_0_strings.end(),
    cuda::transform_iterator(h_val_0_strings.begin(), [](auto str) { return str != nullptr; }));

  cudf::test::strings_column_wrapper values_2(
    h_val_2_strings.begin(),
    h_val_2_strings.end(),
    cuda::transform_iterator(h_val_2_strings.begin(), [](auto str) { return str != nullptr; }));

  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(column_0.release());
  columns.push_back(column_1.release());
  columns.push_back(column_2.release());

  std::vector<std::unique_ptr<cudf::column>> values;
  values.push_back(values_0.release());
  values.push_back(values_1.release());
  values.push_back(values_2.release());

  cudf::table input_table(std::move(columns));
  cudf::table values_table(std::move(values));

  std::vector<cudf::order> order_flags{
    {cudf::order::DESCENDING, cudf::order::ASCENDING, cudf::order::DESCENDING}};
  std::vector<cudf::null_order> null_order_flags{
    {cudf::null_order::BEFORE, cudf::null_order::BEFORE, cudf::null_order::BEFORE}};

  std::unique_ptr<cudf::column> result{};

  EXPECT_NO_THROW(result =
                    cudf::lower_bound(input_table, values_table, order_flags, null_order_flags));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expect);
}

TEST_F(SearchTest, table_partial_desc__find_last_string)
{
  std::vector<char const*> h_col_0_strings{"50", "20", "20", "20", "20", "20", "10"};
  std::vector<char const*> h_col_2_strings{"41", "78", "77", "63", "62", "61", "90"};

  std::vector<char const*> h_val_0_strings{"0",  "0",  "0",  "0",  "10", "10", "10", "10", "10",
                                           "10", "10", "10", "11", "20", "20", "20", "20", "20",
                                           "20", "20", "20", "20", "20", "20", "30", "50", "60"};
  std::vector<char const*> h_val_2_strings{"0",  "91", "0",  "91", "0",  "79", "90", "91", "77",
                                           "80", "90", "91", "91", "00", "76", "77", "78", "30",
                                           "65", "77", "78", "80", "62", "78", "64", "41", "20"};

  fixed_width_column_wrapper<float> column_1{.7, .5, .5, .7, .7, .7, 5.0};

  fixed_width_column_wrapper<float> values_1{0., 0., 6., 5., 0., 5., 5., 5., 5., 6., 6., 6., 9., 0.,
                                             .5, .5, .5, .5, .6, .6, .6, .7, .7, .7, .7, .7, .5};

  fixed_width_column_wrapper<size_type> expect{7, 7, 7, 7, 6, 7, 7, 6, 7, 7, 7, 7, 6, 1,
                                               3, 3, 2, 3, 3, 3, 3, 3, 5, 3, 1, 1, 0};

  cudf::test::strings_column_wrapper column_0(
    h_col_0_strings.begin(),
    h_col_0_strings.end(),
    cuda::transform_iterator(h_col_0_strings.begin(), [](auto str) { return str != nullptr; }));

  cudf::test::strings_column_wrapper column_2(
    h_col_2_strings.begin(),
    h_col_2_strings.end(),
    cuda::transform_iterator(h_col_2_strings.begin(), [](auto str) { return str != nullptr; }));

  cudf::test::strings_column_wrapper values_0(
    h_val_0_strings.begin(),
    h_val_0_strings.end(),
    cuda::transform_iterator(h_val_0_strings.begin(), [](auto str) { return str != nullptr; }));

  cudf::test::strings_column_wrapper values_2(
    h_val_2_strings.begin(),
    h_val_2_strings.end(),
    cuda::transform_iterator(h_val_2_strings.begin(), [](auto str) { return str != nullptr; }));

  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(column_0.release());
  columns.push_back(column_1.release());
  columns.push_back(column_2.release());

  std::vector<std::unique_ptr<cudf::column>> values;
  values.push_back(values_0.release());
  values.push_back(values_1.release());
  values.push_back(values_2.release());

  cudf::table input_table(std::move(columns));
  cudf::table values_table(std::move(values));

  std::vector<cudf::order> order_flags{
    {cudf::order::DESCENDING, cudf::order::ASCENDING, cudf::order::DESCENDING}};
  std::vector<cudf::null_order> null_order_flags{
    {cudf::null_order::BEFORE, cudf::null_order::BEFORE, cudf::null_order::BEFORE}};

  std::unique_ptr<cudf::column> result{};

  EXPECT_NO_THROW(result =
                    cudf::upper_bound(input_table, values_table, order_flags, null_order_flags));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expect);
}

TEST_F(SearchTest, table__find_first__nulls_as_smallest_string)
{
  std::vector<char const*> h_col_0_strings{
    nullptr, "10", "10", "20", "20", "20", "20", "20", "20", "20", "50"};
  std::vector<char const*> h_col_2_strings{
    "50", "95", "90", nullptr, nullptr, "77", "78", "61", "62", "63", "41"};

  std::vector<char const*> h_val_0_strings{"10", nullptr, "20"};
  std::vector<char const*> h_val_2_strings{"95", "50", nullptr};

  fixed_width_column_wrapper<float> column_1{{.5, 6.0, 5.0, .5, .5, .5, .5, .7, .7, .7, .7},
                                             {1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1}};

  fixed_width_column_wrapper<float> values_1{{6., .5, .5}, {0, 1, 1}};

  fixed_width_column_wrapper<size_type> expect{1, 0, 3};

  cudf::test::strings_column_wrapper column_0(
    h_col_0_strings.begin(),
    h_col_0_strings.end(),
    cuda::transform_iterator(h_col_0_strings.begin(), [](auto str) { return str != nullptr; }));

  cudf::test::strings_column_wrapper column_2(
    h_col_2_strings.begin(),
    h_col_2_strings.end(),
    cuda::transform_iterator(h_col_2_strings.begin(), [](auto str) { return str != nullptr; }));

  cudf::test::strings_column_wrapper values_0(
    h_val_0_strings.begin(),
    h_val_0_strings.end(),
    cuda::transform_iterator(h_val_0_strings.begin(), [](auto str) { return str != nullptr; }));

  cudf::test::strings_column_wrapper values_2(
    h_val_2_strings.begin(),
    h_val_2_strings.end(),
    cuda::transform_iterator(h_val_2_strings.begin(), [](auto str) { return str != nullptr; }));

  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(column_0.release());
  columns.push_back(column_1.release());
  columns.push_back(column_2.release());

  std::vector<std::unique_ptr<cudf::column>> values;
  values.push_back(values_0.release());
  values.push_back(values_1.release());
  values.push_back(values_2.release());

  cudf::table input_table(std::move(columns));
  cudf::table values_table(std::move(values));

  std::vector<cudf::order> order_flags{
    {cudf::order::ASCENDING, cudf::order::ASCENDING, cudf::order::ASCENDING}};
  std::vector<cudf::null_order> null_order_flags{
    {cudf::null_order::BEFORE, cudf::null_order::BEFORE, cudf::null_order::BEFORE}};

  std::unique_ptr<cudf::column> result{};

  EXPECT_NO_THROW(result =
                    cudf::lower_bound(input_table, values_table, order_flags, null_order_flags));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expect);
}

TEST_F(SearchTest, table__find_last__nulls_as_smallest_string)
{
  std::vector<char const*> h_col_0_strings{
    nullptr, "10", "10", "20", "20", "20", "20", "20", "20", "20", "50"};
  std::vector<char const*> h_col_2_strings{
    "50", "90", "95", nullptr, nullptr, "77", "78", "61", "62", "63", "41"};

  std::vector<char const*> h_val_0_strings{"10", nullptr, "20"};
  std::vector<char const*> h_val_2_strings{"95", "50", nullptr};

  fixed_width_column_wrapper<float> column_1{{.5, 6.0, 5.0, .5, .5, .5, .5, .7, .7, .7, .7},
                                             {1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1}};

  fixed_width_column_wrapper<float> values_1{{6., .5, .5}, {0, 1, 1}};

  fixed_width_column_wrapper<size_type> expect{2, 1, 5};

  cudf::test::strings_column_wrapper column_0(
    h_col_0_strings.begin(),
    h_col_0_strings.end(),
    cuda::transform_iterator(h_col_0_strings.begin(), [](auto str) { return str != nullptr; }));

  cudf::test::strings_column_wrapper column_2(
    h_col_2_strings.begin(),
    h_col_2_strings.end(),
    cuda::transform_iterator(h_col_2_strings.begin(), [](auto str) { return str != nullptr; }));

  cudf::test::strings_column_wrapper values_0(
    h_val_0_strings.begin(),
    h_val_0_strings.end(),
    cuda::transform_iterator(h_val_0_strings.begin(), [](auto str) { return str != nullptr; }));

  cudf::test::strings_column_wrapper values_2(
    h_val_2_strings.begin(),
    h_val_2_strings.end(),
    cuda::transform_iterator(h_val_2_strings.begin(), [](auto str) { return str != nullptr; }));

  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(column_0.release());
  columns.push_back(column_1.release());
  columns.push_back(column_2.release());

  std::vector<std::unique_ptr<cudf::column>> values;
  values.push_back(values_0.release());
  values.push_back(values_1.release());
  values.push_back(values_2.release());

  cudf::table input_table(std::move(columns));
  cudf::table values_table(std::move(values));

  std::vector<cudf::order> order_flags{
    {cudf::order::ASCENDING, cudf::order::ASCENDING, cudf::order::ASCENDING}};
  std::vector<cudf::null_order> null_order_flags{
    {cudf::null_order::BEFORE, cudf::null_order::BEFORE, cudf::null_order::BEFORE}};

  std::unique_ptr<cudf::column> result{};

  EXPECT_NO_THROW(result =
                    cudf::upper_bound(input_table, values_table, order_flags, null_order_flags));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expect);
}

TEST_F(SearchTest, table__find_first__nulls_as_largest_string)
{
  std::vector<char const*> h_col_0_strings{
    "10", "10", "20", "20", "20", "20", "20", "20", "20", "50", nullptr};
  std::vector<char const*> h_col_2_strings{
    "90", "95", "77", "78", nullptr, nullptr, "61", "62", "63", "41", "50"};

  std::vector<char const*> h_val_0_strings{"10", nullptr, "20"};
  std::vector<char const*> h_val_2_strings{"95", "50", nullptr};

  fixed_width_column_wrapper<float> column_1{{5.0, 6.0, .5, .5, .5, .5, .7, .7, .7, .7, .5},
                                             {1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1}};

  fixed_width_column_wrapper<float> values_1{{6., .5, .5}, {0, 1, 1}};

  fixed_width_column_wrapper<size_type> expect{1, 10, 4};

  cudf::test::strings_column_wrapper column_0(
    h_col_0_strings.begin(),
    h_col_0_strings.end(),
    cuda::transform_iterator(h_col_0_strings.begin(), [](auto str) { return str != nullptr; }));

  cudf::test::strings_column_wrapper column_2(
    h_col_2_strings.begin(),
    h_col_2_strings.end(),
    cuda::transform_iterator(h_col_2_strings.begin(), [](auto str) { return str != nullptr; }));

  cudf::test::strings_column_wrapper values_0(
    h_val_0_strings.begin(),
    h_val_0_strings.end(),
    cuda::transform_iterator(h_val_0_strings.begin(), [](auto str) { return str != nullptr; }));

  cudf::test::strings_column_wrapper values_2(
    h_val_2_strings.begin(),
    h_val_2_strings.end(),
    cuda::transform_iterator(h_val_2_strings.begin(), [](auto str) { return str != nullptr; }));

  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(column_0.release());
  columns.push_back(column_1.release());
  columns.push_back(column_2.release());

  std::vector<std::unique_ptr<cudf::column>> values;
  values.push_back(values_0.release());
  values.push_back(values_1.release());
  values.push_back(values_2.release());

  cudf::table input_table(std::move(columns));
  cudf::table values_table(std::move(values));

  std::vector<cudf::order> order_flags{
    {cudf::order::ASCENDING, cudf::order::ASCENDING, cudf::order::ASCENDING}};
  std::vector<cudf::null_order> null_order_flags{
    {cudf::null_order::AFTER, cudf::null_order::AFTER, cudf::null_order::AFTER}};

  std::unique_ptr<cudf::column> result{};

  EXPECT_NO_THROW(result =
                    cudf::lower_bound(input_table, values_table, order_flags, null_order_flags));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expect);
}

TEST_F(SearchTest, table__find_last__nulls_as_largest_string)
{
  std::vector<char const*> h_col_0_strings{
    "10", "10", "20", "20", "20", "20", "20", "20", "20", "50", nullptr};
  std::vector<char const*> h_col_2_strings{
    "90", "95", "77", "78", nullptr, nullptr, "61", "62", "63", "41", "50"};

  std::vector<char const*> h_val_0_strings{"10", nullptr, "20"};
  std::vector<char const*> h_val_2_strings{"95", "50", nullptr};

  fixed_width_column_wrapper<float> column_1{{5.0, 6.0, .5, .5, .5, .5, .7, .7, .7, .7, .5},
                                             {1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1}};

  fixed_width_column_wrapper<float> values_1{{6., .5, .5}, {0, 1, 1}};

  fixed_width_column_wrapper<size_type> expect{2, 11, 6};

  cudf::test::strings_column_wrapper column_0(
    h_col_0_strings.begin(),
    h_col_0_strings.end(),
    cuda::transform_iterator(h_col_0_strings.begin(), [](auto str) { return str != nullptr; }));

  cudf::test::strings_column_wrapper column_2(
    h_col_2_strings.begin(),
    h_col_2_strings.end(),
    cuda::transform_iterator(h_col_2_strings.begin(), [](auto str) { return str != nullptr; }));

  cudf::test::strings_column_wrapper values_0(
    h_val_0_strings.begin(),
    h_val_0_strings.end(),
    cuda::transform_iterator(h_val_0_strings.begin(), [](auto str) { return str != nullptr; }));

  cudf::test::strings_column_wrapper values_2(
    h_val_2_strings.begin(),
    h_val_2_strings.end(),
    cuda::transform_iterator(h_val_2_strings.begin(), [](auto str) { return str != nullptr; }));

  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(column_0.release());
  columns.push_back(column_1.release());
  columns.push_back(column_2.release());

  std::vector<std::unique_ptr<cudf::column>> values;
  values.push_back(values_0.release());
  values.push_back(values_1.release());
  values.push_back(values_2.release());

  cudf::table input_table(std::move(columns));
  cudf::table values_table(std::move(values));

  std::vector<cudf::order> order_flags{
    {cudf::order::ASCENDING, cudf::order::ASCENDING, cudf::order::ASCENDING}};
  std::vector<cudf::null_order> null_order_flags{
    {cudf::null_order::AFTER, cudf::null_order::AFTER, cudf::null_order::AFTER}};

  std::unique_ptr<cudf::column> result{};

  EXPECT_NO_THROW(result =
                    cudf::upper_bound(input_table, values_table, order_flags, null_order_flags));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expect);
}

TEST_F(SearchTest, contains_true_string)
{
  std::vector<char const*> h_col_strings{"00", "01", "17", "19", "23", "29", "71"};
  string_scalar scalar{"23"};

  cudf::test::strings_column_wrapper column(
    h_col_strings.begin(),
    h_col_strings.end(),
    cuda::transform_iterator(h_col_strings.begin(), [](auto str) { return str != nullptr; }));

  bool expect = true;
  bool result = false;

  result = cudf::contains(column, scalar);

  ASSERT_EQ(result, expect);
}

TEST_F(SearchTest, contains_false_string)
{
  std::vector<char const*> h_col_strings{"0", "1", "17", "19", "23", "29", "71"};
  string_scalar scalar{"24"};

  cudf::test::strings_column_wrapper column(
    h_col_strings.begin(),
    h_col_strings.end(),
    cuda::transform_iterator(h_col_strings.begin(), [](auto str) { return str != nullptr; }));

  bool expect = false;
  bool result = false;

  result = cudf::contains(column, scalar);

  ASSERT_EQ(result, expect);
}

TEST_F(SearchTest, contains_empty_value_string)
{
  std::vector<char const*> h_col_strings{"0", "1", "17", "19", "23", "29", "71"};
  string_scalar scalar{"23", false};

  cudf::test::strings_column_wrapper column(
    h_col_strings.begin(),
    h_col_strings.end(),
    cuda::transform_iterator(h_col_strings.begin(), [](auto str) { return str != nullptr; }));

  bool expect = false;
  bool result = false;

  result = cudf::contains(column, scalar);

  ASSERT_EQ(result, expect);
}

TEST_F(SearchTest, contains_empty_column_string)
{
  std::vector<char const*> h_col_strings{};
  string_scalar scalar{"24"};

  cudf::test::strings_column_wrapper column(
    h_col_strings.begin(),
    h_col_strings.end(),
    cuda::transform_iterator(h_col_strings.begin(), [](auto str) { return str != nullptr; }));

  bool expect = false;
  bool result = false;

  result = cudf::contains(column, scalar);

  ASSERT_EQ(result, expect);
}

TEST_F(SearchTest, contains_nullable_column_true_string)
{
  std::vector<char const*> h_col_strings{nullptr, nullptr, "17", "19", "23", "29", "71"};
  string_scalar scalar{"23"};

  cudf::test::strings_column_wrapper column(
    h_col_strings.begin(),
    h_col_strings.end(),
    cuda::transform_iterator(h_col_strings.begin(), [](auto str) { return str != nullptr; }));

  bool result = false;
  bool expect = true;

  result = cudf::contains(column, scalar);

  ASSERT_EQ(result, expect);
}

TEST_F(SearchTest, contains_nullable_column_false_string)
{
  std::vector<char const*> h_col_strings{nullptr, nullptr, "17", "19", nullptr, "29", "71"};
  string_scalar scalar{"23"};

  cudf::test::strings_column_wrapper column(
    h_col_strings.begin(),
    h_col_strings.end(),
    cuda::transform_iterator(h_col_strings.begin(), [](auto str) { return str != nullptr; }));

  bool result = false;
  bool expect = false;

  result = cudf::contains(column, scalar);

  ASSERT_EQ(result, expect);
}

TEST_F(SearchTest, multi_contains_some)
{
  using element_type = int64_t;

  fixed_width_column_wrapper<element_type> haystack{0, 1, 17, 19, 23, 29, 71};
  fixed_width_column_wrapper<element_type> needles{17, 19, 45, 72};

  fixed_width_column_wrapper<bool> expect{1, 1, 0, 0};

  auto result = cudf::contains(haystack, needles);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expect);
}

TEST_F(SearchTest, multi_contains_none)
{
  using element_type = int64_t;

  fixed_width_column_wrapper<element_type> haystack{0, 1, 17, 19, 23, 29, 71};
  fixed_width_column_wrapper<element_type> needles{2, 3};

  fixed_width_column_wrapper<bool> expect{0, 0};

  auto result = cudf::contains(haystack, needles);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expect);
}

TEST_F(SearchTest, multi_contains_some_string)
{
  std::vector<char const*> h_haystack_strings{"0", "1", "17", "19", "23", "29", "71"};
  std::vector<char const*> h_needles_strings{"17", "19", "45", "72"};

  cudf::test::strings_column_wrapper haystack(h_haystack_strings.begin(), h_haystack_strings.end());

  cudf::test::strings_column_wrapper needles(h_needles_strings.begin(), h_needles_strings.end());

  fixed_width_column_wrapper<bool> expect{1, 1, 0, 0};

  auto result = cudf::contains(haystack, needles);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expect);
}

TEST_F(SearchTest, multi_contains_none_string)
{
  std::vector<char const*> h_haystack_strings{"0", "1", "17", "19", "23", "29", "71"};
  std::vector<char const*> h_needles_strings{"2", "3"};

  cudf::test::strings_column_wrapper haystack(h_haystack_strings.begin(), h_haystack_strings.end());

  cudf::test::strings_column_wrapper needles(h_needles_strings.begin(), h_needles_strings.end());

  fixed_width_column_wrapper<bool> expect{0, 0};

  auto result = cudf::contains(haystack, needles);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expect);
}

TEST_F(SearchTest, multi_contains_some_with_nulls)
{
  using element_type = int64_t;

  fixed_width_column_wrapper<element_type> haystack{{0, 1, 17, 19, 23, 29, 71},
                                                    {1, 1, 0, 1, 1, 1, 1}};
  fixed_width_column_wrapper<element_type> needles{{17, 19, 23, 72}, {1, 0, 1, 1}};

  fixed_width_column_wrapper<bool> expect{{0, 0, 1, 0}, {1, 0, 1, 1}};

  auto result = cudf::contains(haystack, needles);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expect);
}

TEST_F(SearchTest, multi_contains_none_with_nulls)
{
  using element_type = int64_t;

  fixed_width_column_wrapper<element_type> haystack{{0, 1, 17, 19, 23, 29, 71},
                                                    {1, 1, 0, 1, 1, 1, 1}};
  fixed_width_column_wrapper<element_type> needles{{17, 19, 24, 72}, {1, 0, 1, 1}};

  fixed_width_column_wrapper<bool> expect{{0, 0, 0, 0}, {1, 0, 1, 1}};

  auto result = cudf::contains(haystack, needles);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expect);
}

TEST_F(SearchTest, multi_contains_some_string_with_nulls)
{
  std::vector<char const*> h_haystack_strings{"0", "1", nullptr, "19", "23", "29", "71"};
  std::vector<char const*> h_needles_strings{"17", "23", nullptr, "72"};

  fixed_width_column_wrapper<bool> expect{{0, 1, 0, 0}, {1, 1, 0, 1}};

  cudf::test::strings_column_wrapper haystack(
    h_haystack_strings.begin(),
    h_haystack_strings.end(),
    cuda::transform_iterator(h_haystack_strings.begin(), [](auto str) { return str != nullptr; }));

  cudf::test::strings_column_wrapper needles(
    h_needles_strings.begin(),
    h_needles_strings.end(),
    cuda::transform_iterator(h_needles_strings.begin(), [](auto str) { return str != nullptr; }));

  auto result = cudf::contains(haystack, needles);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expect);
}

TEST_F(SearchTest, multi_contains_none_string_with_nulls)
{
  std::vector<char const*> h_haystack_strings{"0", "1", nullptr, "19", "23", "29", "71"};
  std::vector<char const*> h_needles_strings{"2", nullptr};

  fixed_width_column_wrapper<bool> expect{{0, 0}, {1, 0}};

  cudf::test::strings_column_wrapper haystack(
    h_haystack_strings.begin(),
    h_haystack_strings.end(),
    cuda::transform_iterator(h_haystack_strings.begin(), [](auto str) { return str != nullptr; }));

  cudf::test::strings_column_wrapper needles(
    h_needles_strings.begin(),
    h_needles_strings.end(),
    cuda::transform_iterator(h_needles_strings.begin(), [](auto str) { return str != nullptr; }));

  auto result = cudf::contains(haystack, needles);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expect);
}

TEST_F(SearchTest, multi_contains_empty_column)
{
  using element_type = int64_t;

  fixed_width_column_wrapper<element_type> haystack{};
  fixed_width_column_wrapper<element_type> needles{2, 3};

  fixed_width_column_wrapper<bool> expect{0, 0};

  auto result = cudf::contains(haystack, needles);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expect);
}

TEST_F(SearchTest, multi_contains_empty_column_string)
{
  std::vector<char const*> h_haystack_strings{};
  std::vector<char const*> h_needles_strings{"17", "19", "45", "72"};

  cudf::test::strings_column_wrapper haystack(h_haystack_strings.begin(), h_haystack_strings.end());

  cudf::test::strings_column_wrapper needles(h_needles_strings.begin(), h_needles_strings.end());

  fixed_width_column_wrapper<bool> expect{0, 0, 0, 0};

  auto result = cudf::contains(haystack, needles);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expect);
}

TEST_F(SearchTest, multi_contains_empty_input_set)
{
  using element_type = int64_t;

  fixed_width_column_wrapper<element_type> haystack{0, 1, 17, 19, 23, 29, 71};
  fixed_width_column_wrapper<element_type> needles{};

  fixed_width_column_wrapper<bool> expect{};

  auto result = cudf::contains(haystack, needles);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expect);
}

TEST_F(SearchTest, multi_contains_empty_input_set_string)
{
  std::vector<char const*> h_haystack_strings{"0", "1", "17", "19", "23", "29", "71"};
  std::vector<char const*> h_needles_strings{};

  cudf::test::strings_column_wrapper haystack(h_haystack_strings.begin(), h_haystack_strings.end());

  cudf::test::strings_column_wrapper needles(h_needles_strings.begin(), h_needles_strings.end());

  fixed_width_column_wrapper<bool> expect{};

  auto result = cudf::contains(haystack, needles);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expect);
}

TEST_F(SearchTest, multi_contains_primitive_nan_unequal)
{
  auto nan_val = std::numeric_limits<float>::quiet_NaN();

  fixed_width_column_wrapper<float> haystack{1.0f, nan_val, 3.0f};
  fixed_width_column_wrapper<float> needles{nan_val};

  auto result = cudf::detail::contains(cudf::table_view{{haystack}},
                                       cudf::table_view{{needles}},
                                       cudf::null_equality::EQUAL,
                                       cudf::nan_equality::UNEQUAL,
                                       cudf::get_default_stream(),
                                       cudf::get_current_device_resource_ref());

  // With nan_equality::UNEQUAL, NaN should not match NaN
  EXPECT_FALSE(result.front_element(cudf::get_default_stream()));
}

template <typename T>
struct FixedPointTestAllReps : public cudf::test::BaseFixture {};

TYPED_TEST_SUITE(FixedPointTestAllReps, cudf::test::FixedPointTypes);

TYPED_TEST(FixedPointTestAllReps, FixedPointLowerBound)
{
  using namespace numeric;
  using decimalXX = TypeParam;

  auto vec = std::vector<decimalXX>(1000);
  std::iota(std::begin(vec), std::end(vec), decimalXX{});

  auto const values =
    cudf::test::fixed_width_column_wrapper<decimalXX>{decimalXX{200, scale_type{0}},
                                                      decimalXX{400, scale_type{0}},
                                                      decimalXX{600, scale_type{0}},
                                                      decimalXX{800, scale_type{0}}};
  auto const expect = cudf::test::fixed_width_column_wrapper<cudf::size_type>{200, 400, 600, 800};
  auto const column = cudf::test::fixed_width_column_wrapper<decimalXX>(vec.begin(), vec.end());

  auto result = cudf::lower_bound({cudf::table_view{{column}}},
                                  {cudf::table_view{{values}}},
                                  {cudf::order::ASCENDING},
                                  {cudf::null_order::BEFORE});

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expect);
}

TYPED_TEST(FixedPointTestAllReps, FixedPointUpperBound)
{
  using namespace numeric;
  using decimalXX = TypeParam;

  auto vec = std::vector<decimalXX>(1000);
  std::iota(std::begin(vec), std::end(vec), decimalXX{});

  auto const values =
    cudf::test::fixed_width_column_wrapper<decimalXX>{decimalXX{200, scale_type{0}},
                                                      decimalXX{400, scale_type{0}},
                                                      decimalXX{600, scale_type{0}},
                                                      decimalXX{800, scale_type{0}}};
  auto const expect = cudf::test::fixed_width_column_wrapper<cudf::size_type>{201, 401, 601, 801};
  auto const column = cudf::test::fixed_width_column_wrapper<decimalXX>(vec.begin(), vec.end());

  auto result = cudf::upper_bound({cudf::table_view{{column}}},
                                  {cudf::table_view{{values}}},
                                  {cudf::order::ASCENDING},
                                  {cudf::null_order::BEFORE});

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expect);
}

CUDF_TEST_PROGRAM_MAIN()
