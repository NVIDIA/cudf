/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/debug_utilities.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/memory_resource_utilities.hpp>
#include <cudf_test/random.hpp>
#include <cudf_test/table_utilities.hpp>
#include <cudf_test/testing_main.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/copying.hpp>
#include <cudf/detail/iterator.cuh>

#include <cuda/iterator>
#include <thrust/iterator/transform_iterator.h>

#include <sstream>

template <typename T>
struct ColumnUtilitiesTest : public cudf::test::BaseFixture {
  cudf::test::UniformRandomGenerator<cudf::size_type> random;

  ColumnUtilitiesTest() : random{1000, 5000} {}

  auto size() { return random.generate(); }

  auto data_type() { return cudf::data_type{cudf::type_to_id<T>()}; }
};

template <typename T>
struct ColumnUtilitiesTestIntegral : public cudf::test::BaseFixture {};
template <typename T>
struct ColumnUtilitiesTestFloatingPoint : public cudf::test::BaseFixture {};

template <typename T>
struct ColumnUtilitiesTestFixedPoint : public cudf::test::BaseFixture {};

TYPED_TEST_SUITE(ColumnUtilitiesTest, cudf::test::FixedWidthTypes);
TYPED_TEST_SUITE(ColumnUtilitiesTestIntegral, cudf::test::IntegralTypes);
TYPED_TEST_SUITE(ColumnUtilitiesTestFloatingPoint, cudf::test::FloatingPointTypes);
TYPED_TEST_SUITE(ColumnUtilitiesTestFixedPoint, cudf::test::FixedPointTypes);

TYPED_TEST(ColumnUtilitiesTest, NonNullableToHost)
{
  auto stream = cudf::test::get_default_stream();
  auto mr     = this->mr();

  auto sequence = cudf::detail::make_counting_transform_iterator(
    0, [](auto i) { return cudf::test::make_type_param_scalar<TypeParam>(i); });

  auto size = this->size();

  std::vector<TypeParam> data(sequence, sequence + size);
  cudf::test::fixed_width_column_wrapper<TypeParam> col(data.begin(), data.end(), stream, mr);

  auto host_data = cudf::test::to_host<TypeParam>(col, stream, mr);

  EXPECT_TRUE(std::equal(data.begin(), data.end(), host_data.first.begin()));
}

TYPED_TEST(ColumnUtilitiesTest, NonNullableToHostWithOffset)
{
  auto stream = cudf::test::get_default_stream();
  auto mr     = this->mr();

  auto sequence = cudf::detail::make_counting_transform_iterator(
    0, [](auto i) { return cudf::test::make_type_param_scalar<TypeParam>(i); });

  auto const size  = this->size();
  auto const split = 2;

  auto data          = std::vector<TypeParam>(sequence, sequence + size);
  auto expected_data = std::vector<TypeParam>(sequence + split, sequence + size);
  auto col =
    cudf::test::fixed_width_column_wrapper<TypeParam>(data.begin(), data.end(), stream, mr);

  auto const splits = std::vector<cudf::size_type>{split};
  auto result       = cudf::split(col, splits, stream);

  auto host_data = cudf::test::to_host<TypeParam>(result.back(), stream, mr);

  EXPECT_TRUE(std::equal(expected_data.begin(), expected_data.end(), host_data.first.begin()));
}

TYPED_TEST(ColumnUtilitiesTest, NullableToHostWithOffset)
{
  auto stream = cudf::test::get_default_stream();
  auto mr     = this->mr();

  auto sequence = cudf::detail::make_counting_transform_iterator(
    0, [](auto i) { return cudf::test::make_type_param_scalar<TypeParam>(i); });

  auto split = 2;
  auto size  = this->size();
  auto valid = cudf::detail::make_counting_transform_iterator(
    0, [&split](auto i) { return i <= 10 and i > split; });
  std::vector<TypeParam> data(sequence, sequence + size);
  std::vector<TypeParam> expected_data(sequence + split, sequence + size);
  cudf::test::fixed_width_column_wrapper<TypeParam> col(
    data.begin(), data.end(), valid, stream, mr);

  std::vector<cudf::size_type> splits{split};
  std::vector<cudf::column_view> result = cudf::split(col, splits, stream);

  auto host_data = cudf::test::to_host<TypeParam>(result.back(), stream, mr);

  EXPECT_TRUE(std::equal(expected_data.begin(), expected_data.end(), host_data.first.begin()));

  auto masks = std::get<0>(cudf::test::detail::make_null_mask_vector(valid + split, valid + size));

  EXPECT_TRUE(cudf::test::validate_host_masks(masks, host_data.second, expected_data.size()));
}

TYPED_TEST(ColumnUtilitiesTest, NullableToHostAllValid)
{
  auto stream = cudf::test::get_default_stream();
  auto mr     = this->mr();

  auto sequence = cudf::detail::make_counting_transform_iterator(
    0, [](auto i) { return cudf::test::make_type_param_scalar<TypeParam>(i); });

  auto all_valid = cuda::make_constant_iterator<bool>(true);

  auto size = this->size();

  std::vector<TypeParam> data(sequence, sequence + size);
  cudf::test::fixed_width_column_wrapper<TypeParam> col(
    data.begin(), data.end(), all_valid, stream, mr);

  auto host_data = cudf::test::to_host<TypeParam>(col, stream, mr);

  EXPECT_TRUE(std::equal(data.begin(), data.end(), host_data.first.begin()));

  auto masks = std::get<0>(cudf::test::detail::make_null_mask_vector(all_valid, all_valid + size));

  EXPECT_TRUE(cudf::test::validate_host_masks(masks, host_data.second, size));
}

struct ColumnUtilitiesEquivalenceTest : public cudf::test::BaseFixture {};

TEST_F(ColumnUtilitiesEquivalenceTest, DoubleTest)
{
  auto stream = cudf::test::get_default_stream();
  auto mr     = this->mr();

  cudf::test::fixed_width_column_wrapper<double> col1({10. / 3, 22. / 7}, stream, mr);
  cudf::test::fixed_width_column_wrapper<double> col2(
    {31. / 3 - 21. / 3, 19. / 7 + 3. / 7}, stream, mr);

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
    col1, col2, cudf::test::debug_output_level::FIRST_ERROR, cudf::test::default_ulp, stream, mr);
}

TEST_F(ColumnUtilitiesEquivalenceTest, NullabilityTest)
{
  auto stream = cudf::test::get_default_stream();
  auto mr     = this->mr();

  auto all_valid = cudf::test::iterators::no_nulls();
  cudf::test::fixed_width_column_wrapper<double> col1({1, 2, 3}, stream, mr);
  cudf::test::fixed_width_column_wrapper<double> col2({1, 2, 3}, all_valid, stream, mr);

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
    col1, col2, cudf::test::debug_output_level::FIRST_ERROR, cudf::test::default_ulp, stream, mr);
}

TEST_F(ColumnUtilitiesEquivalenceTest, DistinctMemoryResources)
{
  cudf::test::fixed_width_column_wrapper<int32_t> lhs{1, 2, 3, 4};
  cudf::test::fixed_width_column_wrapper<int32_t> rhs{1, 2, 3, 4};
  cudf::test::fixed_width_column_wrapper<int32_t> nullable({1, 2, 3, 4}, {true, false, true, true});
  cudf::test::strings_column_wrapper strings{"one", "two", "three", "four"};
  cudf::test::fixed_point_column_wrapper<int32_t> decimals({123, 456}, numeric::scale_type{-2});
  cudf::column_view const lhs_view = lhs;
  cudf::column_view const rhs_view = rhs;
  auto const sliced                = cudf::slice(nullable, {1, 4}).front();
  auto const table                 = cudf::table_view{{lhs_view, rhs_view}};

  auto harness      = cudf::test::memory_resource_test_harness{};
  auto const mr     = harness.resources();
  auto const stream = cudf::test::get_default_stream();

  CUDF_TEST_EXPECT_COLUMN_PROPERTIES_EQUAL(
    lhs, rhs, cudf::test::debug_output_level::QUIET, stream, mr);
  CUDF_TEST_EXPECT_COLUMN_PROPERTIES_EQUIVALENT(
    lhs, rhs, cudf::test::debug_output_level::QUIET, stream, mr);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(lhs, rhs, cudf::test::debug_output_level::QUIET, stream, mr);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
    lhs, rhs, cudf::test::debug_output_level::QUIET, cudf::test::default_ulp, stream, mr);
  CUDF_TEST_EXPECT_EQUAL_BUFFERS(lhs_view.data<int32_t>(),
                                 rhs_view.data<int32_t>(),
                                 lhs_view.size() * sizeof(int32_t),
                                 stream,
                                 mr);

  static_cast<void>(cudf::test::bitmask_to_host(sliced, stream, mr));
  static_cast<void>(cudf::test::to_host<int32_t>(sliced, stream, mr));
  static_cast<void>(cudf::test::to_host<std::string>(strings, stream, mr));
  static_cast<void>(cudf::test::to_host<numeric::decimal32>(decimals, stream, mr));

  CUDF_TEST_EXPECT_TABLE_PROPERTIES_EQUAL(table, table);
  CUDF_TEST_EXPECT_TABLES_EQUAL(table, table, stream, mr);
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(table, table, stream, mr);

  static_cast<void>(cudf::test::to_strings(sliced, stream, mr));
  static_cast<void>(cudf::test::to_string(sliced, ",", stream, mr));
  std::ostringstream output;
  cudf::test::print(sliced, output, stream, mr);

  // These validation utilities allocate no output and route temporaries to the temporary MR.
  harness.expect_resource_usage(
    /*expected_output_bytes=*/0,
    {cudf::test::output_allocation_expectation::EXACT,
     cudf::test::temporary_allocation_expectation::SOME},
    stream);
  harness.expect_no_live_allocations(stream);
}

struct ColumnUtilitiesStringsTest : public cudf::test::BaseFixture {};

TEST_F(ColumnUtilitiesStringsTest, StringsToHost)
{
  auto stream = cudf::test::get_default_stream();
  auto mr     = this->mr();

  std::vector<char const*> h_strings{"eee", "bb", nullptr, "", "aa", "bbb", "ééé"};
  cudf::test::strings_column_wrapper strings(
    h_strings.begin(),
    h_strings.end(),
    thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; }),
    stream,
    mr);
  auto host_data  = cudf::test::to_host<std::string>(strings, stream, mr);
  auto result_itr = host_data.first.begin();
  for (auto itr = h_strings.begin(); itr != h_strings.end(); ++itr, ++result_itr) {
    if (*itr) { EXPECT_TRUE((*result_itr) == (*itr)); }
  }
}

TEST_F(ColumnUtilitiesStringsTest, StringsToHostAllNulls)
{
  auto stream = cudf::test::get_default_stream();
  auto mr     = this->mr();

  std::vector<char const*> h_strings{nullptr, nullptr, nullptr};
  cudf::test::strings_column_wrapper strings(
    h_strings.begin(),
    h_strings.end(),
    thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; }),
    stream,
    mr);
  auto host_data = cudf::test::to_host<std::string>(strings, stream, mr);
  auto results   = host_data.first;
  EXPECT_EQ(std::size_t{3}, host_data.first.size());
  EXPECT_TRUE(std::all_of(results.begin(), results.end(), [](auto s) { return s.empty(); }));
}

TYPED_TEST(ColumnUtilitiesTestFixedPoint, NonNullableToHost)
{
  auto stream = cudf::test::get_default_stream();
  auto mr     = this->mr();

  using namespace numeric;
  using decimalXX = TypeParam;
  using rep       = cudf::device_storage_type_t<decimalXX>;

  auto const scale = scale_type{-2};
  auto to_fp       = [&](auto i) { return decimalXX{i, scale}; };
  auto to_rep      = [](auto i) { return i * 100; };
  auto fps         = cudf::detail::make_counting_transform_iterator(0, to_fp);
  auto reps        = cudf::detail::make_counting_transform_iterator(0, to_rep);

  auto const size     = 1000;
  auto const expected = std::vector<decimalXX>(fps, fps + size);
  auto const col =
    cudf::test::fixed_point_column_wrapper<rep>(reps, reps + size, scale, stream, mr);
  auto const host_data = cudf::test::to_host<decimalXX>(col, stream, mr);

  EXPECT_TRUE(std::equal(expected.begin(), expected.end(), host_data.first.begin()));
}

TYPED_TEST(ColumnUtilitiesTestFixedPoint, NonNullableToHostWithOffset)
{
  auto stream = cudf::test::get_default_stream();
  auto mr     = this->mr();

  using namespace numeric;
  using decimalXX = TypeParam;
  using rep       = cudf::device_storage_type_t<decimalXX>;

  auto const scale = scale_type{-2};
  auto to_fp       = [&](auto i) { return decimalXX{i, scale}; };
  auto to_rep      = [](auto i) { return i * 100; };
  auto fps         = cudf::detail::make_counting_transform_iterator(0, to_fp);
  auto reps        = cudf::detail::make_counting_transform_iterator(0, to_rep);

  auto const size  = 1000;
  auto const split = cudf::size_type{2};

  auto const expected = std::vector<decimalXX>(fps + split, fps + size);
  auto const col =
    cudf::test::fixed_point_column_wrapper<rep>(reps, reps + size, scale, stream, mr);
  auto const splits = std::vector<cudf::size_type>{split};
  auto result       = cudf::split(col, splits, stream);

  auto host_data = cudf::test::to_host<decimalXX>(result.back(), stream, mr);

  EXPECT_TRUE(std::equal(expected.begin(), expected.end(), host_data.first.begin()));
}

struct ColumnUtilitiesListsTest : public cudf::test::BaseFixture {};

TEST_F(ColumnUtilitiesListsTest, Equivalence)
{
  auto stream = cudf::test::get_default_stream();
  auto mr     = this->mr();

  // list<int>, nullable vs. non-nullable
  {
    auto all_valid = cudf::test::iterators::no_nulls();
    cudf::test::lists_column_wrapper<int> a(
      {{1, 2, 3}, {5, 6}, {8, 9}, {10}, {14, 15}}, stream, mr);
    cudf::test::lists_column_wrapper<int> b(
      {{1, 2, 3}, {5, 6}, {8, 9}, {10}, {14, 15}}, all_valid, stream, mr);

    // properties
    CUDF_TEST_EXPECT_COLUMN_PROPERTIES_EQUIVALENT(
      a, b, cudf::test::debug_output_level::FIRST_ERROR, stream, mr);
    EXPECT_FALSE(cudf::test::detail::expect_column_properties_equal(
      a, b, cudf::test::debug_output_level::QUIET, stream, mr));

    // values
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
      a, b, cudf::test::debug_output_level::FIRST_ERROR, cudf::test::default_ulp, stream, mr);
    EXPECT_FALSE(cudf::test::detail::expect_columns_equal(
      a, b, cudf::test::debug_output_level::QUIET, stream, mr));
  }

  // list<list<int>>, nullable vs. non-nullable
  {
    auto all_valid = cudf::test::iterators::no_nulls();
    cudf::test::lists_column_wrapper<int> a(
      {{{1, 2, 3}, {5, 6}}, {{8, 9}, {10}}, {{14, 15}}}, stream, mr);
    cudf::test::lists_column_wrapper<int> b(
      {{{1, 2, 3}, {5, 6}}, {{8, 9}, {10}}, {{14, 15}}}, all_valid, stream, mr);

    // properties
    CUDF_TEST_EXPECT_COLUMN_PROPERTIES_EQUIVALENT(
      a, b, cudf::test::debug_output_level::FIRST_ERROR, stream, mr);
    EXPECT_FALSE(cudf::test::detail::expect_column_properties_equal(
      a, b, cudf::test::debug_output_level::QUIET, stream, mr));

    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(
      a, b, cudf::test::debug_output_level::FIRST_ERROR, cudf::test::default_ulp, stream, mr);
    EXPECT_FALSE(cudf::test::detail::expect_columns_equal(
      a, b, cudf::test::debug_output_level::QUIET, stream, mr));
  }
}

TEST_F(ColumnUtilitiesListsTest, DifferingRowCounts)
{
  auto stream = cudf::test::get_default_stream();
  auto mr     = this->mr();

  cudf::test::fixed_width_column_wrapper<int> a({1, 1, 1, 1}, stream, mr);
  cudf::test::fixed_width_column_wrapper<int> b({1, 1, 1, 1, 1}, stream, mr);

  EXPECT_FALSE(cudf::test::detail::expect_columns_equal(
    a, b, cudf::test::debug_output_level::QUIET, stream, mr));
  EXPECT_FALSE(cudf::test::detail::expect_column_properties_equal(
    a, b, cudf::test::debug_output_level::QUIET, stream, mr));
  EXPECT_FALSE(cudf::test::detail::expect_columns_equivalent(
    a, b, cudf::test::debug_output_level::QUIET, cudf::test::default_ulp, stream, mr));
  EXPECT_FALSE(cudf::test::detail::expect_column_properties_equivalent(
    a, b, cudf::test::debug_output_level::QUIET, stream, mr));
}

TEST_F(ColumnUtilitiesListsTest, DifferentPhysicalStructureBeforeConstruction)
{
  auto stream = cudf::test::get_default_stream();
  auto mr     = this->mr();

  // list<int>
  {
    std::vector<bool> valids = {0, 0, 1, 0, 1, 0, 0};

    cudf::test::fixed_width_column_wrapper<int> c0_offsets(
      {0, 3, 6, 8, 11, 14, 16, 19}, stream, mr);
    cudf::test::fixed_width_column_wrapper<int> c0_data(
      {1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 7, 7, 7}, stream, mr);

    auto [null_mask, null_count] = cudf::test::detail::make_null_mask(valids.begin(), valids.end());

    auto c0 = [&] {
      auto tmp = make_lists_column(
        7, c0_offsets.release(), c0_data.release(), null_count, std::move(null_mask));
      return cudf::purge_nonempty_nulls(tmp->view());
    }();

    cudf::test::fixed_width_column_wrapper<int> c1_offsets({0, 0, 0, 2, 2, 5, 5, 5}, stream, mr);
    cudf::test::fixed_width_column_wrapper<int> c1_data({3, 3, 5, 5, 5}, stream, mr);
    auto c1 = [&] {
      auto tmp = make_lists_column(
        7,
        c1_offsets.release(),
        c1_data.release(),
        null_count,
        std::get<0>(cudf::test::detail::make_null_mask(valids.begin(), valids.end())));
      return cudf::purge_nonempty_nulls(tmp->view());
    }();

    // properties
    CUDF_TEST_EXPECT_COLUMN_PROPERTIES_EQUAL(
      *c0, *c1, cudf::test::debug_output_level::FIRST_ERROR, stream, mr);

    // values
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(
      *c0, *c1, cudf::test::debug_output_level::FIRST_ERROR, stream, mr);
  }

  // list<list<struct<int, float>>>
  {
    std::vector<bool> level1_valids = {0, 0, 1, 0, 1, 0, 0};

    cudf::test::fixed_width_column_wrapper<int> c0_l1_offsets({0, 1, 2, 4, 4, 7, 7, 7}, stream, mr);
    cudf::test::fixed_width_column_wrapper<int> c0_l2_offsets(
      {0, 1, 2, 5, 6, 7, 10, 14}, stream, mr);
    cudf::test::fixed_width_column_wrapper<int> c0_l3_ints(
      {1, 1, -1, -2, -3, 1, 1, -4, -5, -6, -7, -8, -9, -10}, stream, mr);
    cudf::test::fixed_width_column_wrapper<float> c0_l3_floats(
      {1, 1, 10, 20, 30, 1, 1, 40, 50, 60, 70, 80, 90, 100}, stream, mr);
    cudf::test::structs_column_wrapper c0_l2_data({c0_l3_ints, c0_l3_floats}, {}, stream, mr);
    std::vector<bool> c0_l2_valids = {1, 1, 1, 0, 0, 1, 1};

    auto [null_mask, null_count] =
      cudf::test::detail::make_null_mask(c0_l2_valids.begin(), c0_l2_valids.end());
    auto c0_l2 = [&] {
      auto tmp = make_lists_column(
        7, c0_l2_offsets.release(), c0_l2_data.release(), null_count, std::move(null_mask));
      return cudf::purge_nonempty_nulls(tmp->view());
    }();

    std::tie(null_mask, null_count) =
      cudf::test::detail::make_null_mask(level1_valids.begin(), level1_valids.end());
    auto c0 = [&] {
      auto tmp = make_lists_column(
        7, c0_l1_offsets.release(), std::move(c0_l2), null_count, std::move(null_mask));
      return cudf::purge_nonempty_nulls(tmp->view());
    }();

    cudf::test::fixed_width_column_wrapper<int> c1_l1_offsets({0, 0, 0, 2, 2, 5, 5, 5}, stream, mr);
    cudf::test::fixed_width_column_wrapper<int> c1_l2_offsets({0, 3, 3, 3, 6, 10}, stream, mr);
    cudf::test::fixed_width_column_wrapper<int> c1_l3_ints(
      {-1, -2, -3, -4, -5, -6, -7, -8, -9, -10}, stream, mr);
    cudf::test::fixed_width_column_wrapper<float> c1_l3_floats(
      {10, 20, 30, 40, 50, 60, 70, 80, 90, 100}, stream, mr);
    cudf::test::structs_column_wrapper c1_l2_data({c1_l3_ints, c1_l3_floats}, {}, stream, mr);
    std::vector<bool> c1_l2_valids = {1, 0, 0, 1, 1};

    std::tie(null_mask, null_count) =
      cudf::test::detail::make_null_mask(c1_l2_valids.begin(), c1_l2_valids.end());
    auto c1_l2 = [&] {
      auto tmp = make_lists_column(
        5, c1_l2_offsets.release(), c1_l2_data.release(), null_count, std::move(null_mask));
      return cudf::purge_nonempty_nulls(tmp->view());
    }();

    std::tie(null_mask, null_count) =
      cudf::test::detail::make_null_mask(level1_valids.begin(), level1_valids.end());
    auto c1 = [&] {
      auto tmp = make_lists_column(
        7, c1_l1_offsets.release(), std::move(c1_l2), null_count, std::move(null_mask));
      return cudf::purge_nonempty_nulls(tmp->view());
    }();

    // properties
    CUDF_TEST_EXPECT_COLUMN_PROPERTIES_EQUAL(
      *c0, *c1, cudf::test::debug_output_level::FIRST_ERROR, stream, mr);

    // values
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(
      *c0, *c1, cudf::test::debug_output_level::FIRST_ERROR, stream, mr);
  }
}

struct ColumnUtilitiesStructsTest : public cudf::test::BaseFixture {};

TEST_F(ColumnUtilitiesStructsTest, Properties)
{
  auto stream = cudf::test::get_default_stream();
  auto mr     = this->mr();

  cudf::test::strings_column_wrapper s0_scol0({"mno", "jkl", "ghi", "def", "abc"}, stream, mr);
  cudf::test::fixed_width_column_wrapper<float> s0_scol1({5, 4, 3, 2, 1}, stream, mr);
  cudf::test::strings_column_wrapper s0_sscol0({"5555", "4444", "333", "22", "1"}, stream, mr);
  cudf::test::fixed_width_column_wrapper<float> s0_sscol1({50, 40, 30, 20, 10}, stream, mr);
  cudf::test::lists_column_wrapper<int> s0_sscol2(
    {{1, 2}, {3, 4}, {5}, {6, 7, 8}, {12, 12}}, stream, mr);
  cudf::test::structs_column_wrapper s0_scol2({s0_sscol0, s0_sscol1, s0_sscol2}, {}, stream, mr);
  cudf::test::structs_column_wrapper s_col0({s0_scol0, s0_scol1, s0_scol2}, {}, stream, mr);

  auto all_valid = cuda::make_constant_iterator<bool>(true);

  cudf::test::strings_column_wrapper s1_scol0({"mno", "jkl", "ghi", "def", "abc"}, stream, mr);
  cudf::test::fixed_width_column_wrapper<float> s1_scol1({5, 4, 3, 2, 1}, stream, mr);
  cudf::test::strings_column_wrapper s1_sscol0({"5555", "4444", "333", "22", "1"}, stream, mr);
  cudf::test::fixed_width_column_wrapper<float> s1_sscol1({50, 40, 30, 20, 10}, stream, mr);
  cudf::test::lists_column_wrapper<int> s1_sscol2(
    {{1, 2}, {3, 4}, {5}, {6, 7, 8}, {12, 12}}, all_valid, stream, mr);
  cudf::test::structs_column_wrapper s1_scol2({s1_sscol0, s1_sscol1, s1_sscol2}, {}, stream, mr);
  cudf::test::structs_column_wrapper s_col1({s1_scol0, s1_scol1, s1_scol2}, {}, stream, mr);

  // equivalent, but not equal
  CUDF_TEST_EXPECT_COLUMN_PROPERTIES_EQUIVALENT(
    s_col0, s_col1, cudf::test::debug_output_level::FIRST_ERROR, stream, mr);
  EXPECT_FALSE(cudf::test::detail::expect_column_properties_equal(
    s_col0, s_col1, cudf::test::debug_output_level::QUIET, stream, mr));

  CUDF_TEST_EXPECT_COLUMN_PROPERTIES_EQUAL(
    s_col0, s_col0, cudf::test::debug_output_level::FIRST_ERROR, stream, mr);
  CUDF_TEST_EXPECT_COLUMN_PROPERTIES_EQUAL(
    s_col1, s_col1, cudf::test::debug_output_level::FIRST_ERROR, stream, mr);
}

TEST_F(ColumnUtilitiesStructsTest, Values)
{
  auto stream = cudf::test::get_default_stream();
  auto mr     = this->mr();

  cudf::test::strings_column_wrapper s0_scol0({"mno", "jkl", "ghi", "def", "abc"}, stream, mr);
  cudf::test::fixed_width_column_wrapper<float> s0_scol1({5, 4, 3, 2, 1}, stream, mr);
  cudf::test::strings_column_wrapper s0_sscol0({"5555", "4444", "333", "22", "1"}, stream, mr);
  cudf::test::fixed_width_column_wrapper<float> s0_sscol1({50, 40, 30, 20, 10}, stream, mr);
  cudf::test::lists_column_wrapper<int> s0_sscol2(
    {{1, 2}, {3, 4}, {5}, {6, 7, 8}, {12, 12}}, stream, mr);
  cudf::test::structs_column_wrapper s0_scol2({s0_sscol0, s0_sscol1, s0_sscol2}, {}, stream, mr);
  cudf::test::structs_column_wrapper s_col0({s0_scol0, s0_scol1, s0_scol2}, {}, stream, mr);

  auto all_valid = cuda::make_constant_iterator<bool>(true);

  cudf::test::strings_column_wrapper s1_scol0({"mno", "jkl", "ghi", "def", "abc"}, stream, mr);
  cudf::test::fixed_width_column_wrapper<float> s1_scol1({5, 4, 3, 2, 1}, stream, mr);
  cudf::test::strings_column_wrapper s1_sscol0({"5555", "4444", "333", "22", "1"}, stream, mr);
  cudf::test::fixed_width_column_wrapper<float> s1_sscol1({50, 40, 30, 20, 10}, stream, mr);
  cudf::test::lists_column_wrapper<int> s1_sscol2(
    {{1, 2}, {3, 4}, {5}, {6, 7, 8}, {12, 12}}, all_valid, stream, mr);
  cudf::test::structs_column_wrapper s1_scol2({s1_sscol0, s1_sscol1, s1_sscol2}, {}, stream, mr);
  cudf::test::structs_column_wrapper s_col1({s1_scol0, s1_scol1, s1_scol2}, {}, stream, mr);

  // equivalent, but not equal
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(s_col0,
                                      s_col1,
                                      cudf::test::debug_output_level::FIRST_ERROR,
                                      cudf::test::default_ulp,
                                      stream,
                                      mr);
  EXPECT_FALSE(cudf::test::detail::expect_columns_equal(
    s_col0, s_col1, cudf::test::debug_output_level::QUIET, stream, mr));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    s_col0, s_col0, cudf::test::debug_output_level::FIRST_ERROR, stream, mr);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    s_col1, s_col1, cudf::test::debug_output_level::FIRST_ERROR, stream, mr);
}

CUDF_TEST_PROGRAM_MAIN()
