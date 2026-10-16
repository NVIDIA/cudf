/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <tests/groupby/groupby_test_util.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/cudf_gtest.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/aggregation.hpp>
#include <cudf/copying.hpp>
#include <cudf/sorting.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

template <typename V>
struct groupby_collect_list_test : public cudf::test::BaseFixture {};

using FixedWidthTypesNotBool = cudf::test::Concat<cudf::test::IntegralTypesNotBool,
                                                  cudf::test::FloatingPointTypes,
                                                  cudf::test::TimestampTypes>;
TYPED_TEST_SUITE(groupby_collect_list_test, FixedWidthTypesNotBool);

TYPED_TEST(groupby_collect_list_test, CollectWithoutNulls)
{
  using K = int32_t;
  using V = TypeParam;

  cudf::test::fixed_width_column_wrapper<K, int32_t> keys{1, 1, 1, 2, 2, 2};
  cudf::test::fixed_width_column_wrapper<V, int32_t> values{1, 2, 3, 4, 5, 6};

  cudf::test::fixed_width_column_wrapper<K, int32_t> expect_keys{1, 2};
  cudf::test::lists_column_wrapper<V, int32_t> expect_vals{{1, 2, 3}, {4, 5, 6}};

  auto agg = cudf::make_collect_list_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, values, expect_keys, expect_vals, std::move(agg));
}

TYPED_TEST(groupby_collect_list_test, CollectWithNulls)
{
  using K = int32_t;
  using V = TypeParam;

  cudf::test::fixed_width_column_wrapper<K, int32_t> keys{1, 1, 2, 2, 3, 3};
  cudf::test::fixed_width_column_wrapper<V, int32_t> values{
    {1, 2, 3, 4, 5, 6}, {true, false, true, false, true, false}};

  cudf::test::fixed_width_column_wrapper<K, int32_t> expect_keys{1, 2, 3};

  std::vector<int32_t> validity({true, false});
  cudf::test::lists_column_wrapper<V, int32_t> expect_vals{
    {{1, 2}, validity.begin()}, {{3, 4}, validity.begin()}, {{5, 6}, validity.begin()}};

  auto agg = cudf::make_collect_list_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, values, expect_keys, expect_vals, std::move(agg));
}

TYPED_TEST(groupby_collect_list_test, CollectWithNullExclusion)
{
  using K = int32_t;
  using V = TypeParam;

  cudf::test::fixed_width_column_wrapper<K, int32_t> keys{1, 1, 1, 2, 2, 3, 3, 4, 4};

  cudf::test::fixed_width_column_wrapper<V, int32_t> values{
    {1, 2, 3, 4, 5, 6, 7, 8, 9}, {false, true, false, true, false, false, false, true, true}};

  cudf::test::fixed_width_column_wrapper<K, int32_t> expect_keys{1, 2, 3, 4};

  cudf::test::lists_column_wrapper<V, int32_t> expect_vals{{2}, {4}, {}, {8, 9}};

  auto agg =
    cudf::make_collect_list_aggregation<cudf::groupby_aggregation>(cudf::null_policy::EXCLUDE);
  test_single_agg(keys, values, expect_keys, expect_vals, std::move(agg));
}

using GroupbyCollectListOwnershipTest = groupby_collect_list_test<int32_t>;

TEST_F(GroupbyCollectListOwnershipTest, NullExclusionPreservesValuesForLaterAggregations)
{
  cudf::test::fixed_width_column_wrapper<int32_t> keys{2, 1, 2, 1, 2, 1, 3, 3};
  cudf::test::fixed_width_column_wrapper<int32_t> values{
    {23, 12, 0, 10, 21, 0, 0, 0}, {true, true, false, true, true, false, false, false}};
  cudf::test::fixed_width_column_wrapper<int32_t> expected_keys{1, 2, 3};
  std::array const first_validity{true, true, false};
  std::array const second_validity{true, false, true};
  std::array const all_nulls{false, false};
  cudf::test::lists_column_wrapper<int32_t> expected_include{{{12, 10, 0}, first_validity.begin()},
                                                             {{23, 0, 21}, second_validity.begin()},
                                                             {{0, 0}, all_nulls.begin()}};
  cudf::test::lists_column_wrapper<int32_t> expected_exclude{{12, 10}, {23, 21}, {}};
  cudf::test::fixed_width_column_wrapper<int32_t> expected_nth_include{{10, 0, 0},
                                                                       {true, false, false}};

  for (bool const include_first : {false, true}) {
    SCOPED_TRACE(include_first);
    std::vector<cudf::groupby::aggregation_request> requests(1);
    requests.front().values = values;
    auto& aggregations      = requests.front().aggregations;
    std::vector<cudf::column_view> expected;
    if (include_first) {
      aggregations.push_back(
        cudf::make_collect_list_aggregation<cudf::groupby_aggregation>(cudf::null_policy::INCLUDE));
      expected.push_back(expected_include);
    }
    aggregations.push_back(
      cudf::make_collect_list_aggregation<cudf::groupby_aggregation>(cudf::null_policy::EXCLUDE));
    aggregations.push_back(
      cudf::make_nth_element_aggregation<cudf::groupby_aggregation>(1, cudf::null_policy::INCLUDE));
    expected.push_back(expected_exclude);
    expected.push_back(expected_nth_include);

    cudf::groupby::groupby gb(
      cudf::table_view{{keys}}, cudf::null_policy::EXCLUDE, cudf::sorted::NO);
    auto result      = gb.aggregate(requests);
    auto const order = cudf::sorted_order(result.first->view());
    auto sorted_keys = cudf::gather(result.first->view(), order->view());
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_keys, sorted_keys->get_column(0));
    auto const& outputs = result.second.front().results;
    ASSERT_EQ(outputs.size(), expected.size());
    for (std::size_t i = 0; i < outputs.size(); ++i) {
      auto sorted = cudf::gather(cudf::table_view{{outputs[i]->view()}}, order->view());
      CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected[i], sorted->get_column(0));
    }
  }
}

TYPED_TEST(groupby_collect_list_test, CollectOnEmptyInput)
{
  using K = int32_t;
  using V = TypeParam;

  cudf::test::fixed_width_column_wrapper<K, int32_t> keys{};
  cudf::test::fixed_width_column_wrapper<V, int32_t> values{};

  cudf::test::fixed_width_column_wrapper<K, int32_t> expect_keys{};
  cudf::test::lists_column_wrapper<V, int32_t> expect_vals{};

  auto agg =
    cudf::make_collect_list_aggregation<cudf::groupby_aggregation>(cudf::null_policy::EXCLUDE);
  test_single_agg(keys, values, expect_keys, expect_vals, std::move(agg));
}

TYPED_TEST(groupby_collect_list_test, CollectLists)
{
  using K = int32_t;
  using V = TypeParam;

  cudf::test::fixed_width_column_wrapper<K, int32_t> keys{1, 1, 2, 2, 3, 3};
  cudf::test::lists_column_wrapper<V, int32_t> values{{1, 2}, {3, 4}, {5, 6, 7}, {}, {9, 10}, {11}};

  cudf::test::fixed_width_column_wrapper<K, int32_t> expect_keys{1, 2, 3};

  cudf::test::lists_column_wrapper<V, int32_t> expect_vals{
    {{1, 2}, {3, 4}}, {{5, 6, 7}, {}}, {{9, 10}, {11}}};

  auto agg = cudf::make_collect_list_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, values, expect_keys, expect_vals, std::move(agg));
}

TYPED_TEST(groupby_collect_list_test, CollectListsWithNullExclusion)
{
  using K = int32_t;
  using V = TypeParam;

  using LCW = cudf::test::lists_column_wrapper<V, int32_t>;

  cudf::test::fixed_width_column_wrapper<K, int32_t> keys{1, 1, 2, 2, 3, 3, 4, 4};
  std::array const validity_mask{true, false, false, true, true, true, false, false};
  LCW values{{{1, 2}, {3, 4}, {5, 6, 7}, {}, {9, 10}, {11}, {20, 30, 40}, {}},
             validity_mask.data()};

  cudf::test::fixed_width_column_wrapper<K, int32_t> expect_keys{1, 2, 3, 4};

  LCW expect_vals{{{1, 2}}, {{}}, {{9, 10}, {11}}, {}};

  auto agg =
    cudf::make_collect_list_aggregation<cudf::groupby_aggregation>(cudf::null_policy::EXCLUDE);
  test_single_agg(keys, values, expect_keys, expect_vals, std::move(agg));
}

TYPED_TEST(groupby_collect_list_test, CollectOnEmptyInputLists)
{
  using K = int32_t;
  using V = TypeParam;

  using LCW = cudf::test::lists_column_wrapper<V, int32_t>;

  auto offsets = cudf::data_type{cudf::type_to_id<cudf::size_type>()};

  cudf::test::fixed_width_column_wrapper<K, int32_t> keys{};
  auto values = cudf::make_lists_column(0,
                                        cudf::make_empty_column(offsets),
                                        LCW{}.release(),
                                        0,
                                        cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED));

  cudf::test::fixed_width_column_wrapper<K, int32_t> expect_keys{};

  auto expect_child =
    cudf::make_lists_column(0,
                            cudf::make_empty_column(offsets),
                            LCW{}.release(),
                            0,
                            cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED));
  auto expect_values =
    cudf::make_lists_column(0,
                            cudf::make_empty_column(offsets),
                            std::move(expect_child),
                            0,
                            cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED));

  auto agg = cudf::make_collect_list_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, values->view(), expect_keys, expect_values->view(), std::move(agg));
}

TYPED_TEST(groupby_collect_list_test, CollectOnEmptyInputListsOfStructs)
{
  using K = int32_t;
  using V = TypeParam;

  using LCW = cudf::test::lists_column_wrapper<V, int32_t>;

  cudf::test::fixed_width_column_wrapper<K, int32_t> keys{};
  auto struct_child  = LCW{};
  auto struct_column = cudf::test::structs_column_wrapper{{struct_child}};

  auto values = cudf::make_lists_column(0,
                                        cudf::make_empty_column(cudf::type_id::INT32),
                                        struct_column.release(),
                                        0,
                                        cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED));

  cudf::test::fixed_width_column_wrapper<K, int32_t> expect_keys{};

  auto expect_struct_child  = LCW{};
  auto expect_struct_column = cudf::test::structs_column_wrapper{{expect_struct_child}};

  auto expect_child =
    cudf::make_lists_column(0,
                            cudf::make_empty_column(cudf::type_id::INT32),
                            expect_struct_column.release(),
                            0,
                            cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED));
  auto expect_values =
    cudf::make_lists_column(0,
                            cudf::make_empty_column(cudf::type_id::INT32),
                            std::move(expect_child),
                            0,
                            cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED));

  auto agg = cudf::make_collect_list_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, values->view(), expect_keys, expect_values->view(), std::move(agg));
}

TYPED_TEST(groupby_collect_list_test, dictionary)
{
  using K = int32_t;
  using V = TypeParam;

  cudf::test::fixed_width_column_wrapper<K, int32_t> keys{1, 1, 1, 2, 2, 2};
  cudf::test::dictionary_column_wrapper<V, int32_t> vals{1, 2, 3, 4, 5, 6};

  cudf::test::fixed_width_column_wrapper<K, int32_t> expect_keys{1, 2};
  cudf::test::lists_column_wrapper<V, int32_t> expect_vals_w{{1, 2, 3}, {4, 5, 6}};

  cudf::test::fixed_width_column_wrapper<int32_t> offsets({0, 3, 6});
  auto expect_vals =
    cudf::make_lists_column(cudf::column_view(offsets).size() - 1,
                            std::make_unique<cudf::column>(offsets),
                            std::make_unique<cudf::column>(vals),
                            0,
                            cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED));

  test_single_agg(keys,
                  vals,
                  expect_keys,
                  expect_vals->view(),
                  cudf::make_collect_list_aggregation<cudf::groupby_aggregation>());
}
