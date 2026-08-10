/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <tests/groupby/groupby_test_util.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/type_lists.hpp>

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

  using LCW = cudf::test::lists_column_wrapper<TypeParam, int32_t>;

  cudf::test::fixed_width_column_wrapper<K, int32_t> keys{1, 1, 2, 2, 3, 3};
  cudf::test::lists_column_wrapper<V, int32_t> values{
    {1, 2}, {3, 4}, {5, 6, 7}, LCW{}, {9, 10}, {11}};

  cudf::test::fixed_width_column_wrapper<K, int32_t> expect_keys{1, 2, 3};

  cudf::test::lists_column_wrapper<V, int32_t> expect_vals{
    {{1, 2}, {3, 4}}, {{5, 6, 7}, LCW{}}, {{9, 10}, {11}}};

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
  LCW values{{{1, 2}, {3, 4}, {5, 6, 7}, LCW{}, {9, 10}, {11}, {20, 30, 40}, LCW{}},
             validity_mask.data()};

  cudf::test::fixed_width_column_wrapper<K, int32_t> expect_keys{1, 2, 3, 4};

  LCW expect_vals{{{1, 2}}, {LCW{}}, {{9, 10}, {11}}, {}};

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
  auto values =
    cudf::make_lists_column(0, cudf::make_empty_column(offsets), LCW{}.release(), 0, {});

  cudf::test::fixed_width_column_wrapper<K, int32_t> expect_keys{};

  auto expect_child =
    cudf::make_lists_column(0, cudf::make_empty_column(offsets), LCW{}.release(), 0, {});
  auto expect_values =
    cudf::make_lists_column(0, cudf::make_empty_column(offsets), std::move(expect_child), 0, {});

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

  auto values =
    cudf::make_lists_column(0,
                            cudf::make_empty_column(cudf::type_to_id<cudf::size_type>()),
                            struct_column.release(),
                            0,
                            {});

  cudf::test::fixed_width_column_wrapper<K, int32_t> expect_keys{};

  auto expect_struct_child  = LCW{};
  auto expect_struct_column = cudf::test::structs_column_wrapper{{expect_struct_child}};

  auto expect_child =
    cudf::make_lists_column(0,
                            cudf::make_empty_column(cudf::type_to_id<cudf::size_type>()),
                            expect_struct_column.release(),
                            0,
                            {});
  auto expect_values =
    cudf::make_lists_column(0,
                            cudf::make_empty_column(cudf::type_to_id<cudf::size_type>()),
                            std::move(expect_child),
                            0,
                            {});

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
  auto expect_vals = cudf::make_lists_column(cudf::column_view(offsets).size() - 1,
                                             std::make_unique<cudf::column>(offsets),
                                             std::make_unique<cudf::column>(vals),
                                             0,
                                             rmm::device_buffer{});

  test_single_agg(keys,
                  vals,
                  expect_keys,
                  expect_vals->view(),
                  cudf::make_collect_list_aggregation<cudf::groupby_aggregation>());
}
