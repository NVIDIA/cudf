/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/table_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/copying.hpp>
#include <cudf/groupby.hpp>
#include <cudf/sorting.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

void test_groups(cudf::column_view const& keys,
                 cudf::column_view const& expect_grouped_keys,
                 std::vector<cudf::size_type> const& expect_group_offsets,
                 cudf::column_view const& values                = {},
                 cudf::column_view const& expect_grouped_values = {})
{
  cudf::groupby::groupby gb(cudf::table_view({keys}));
  cudf::groupby::groupby::groups gb_groups;

  if (values.size()) {
    gb_groups = gb.get_groups(cudf::table_view({values}));
  } else {
    gb_groups = gb.get_groups();
  }
  auto const order        = cudf::stable_sorted_order(gb_groups.keys->view());
  auto const ordered_keys = cudf::gather(gb_groups.keys->view(), *order);
  CUDF_TEST_EXPECT_TABLES_EQUAL(cudf::table_view({expect_grouped_keys}), *ordered_keys);

  auto const& got_offsets = gb_groups.offsets;
  ASSERT_EQ(expect_group_offsets.size(), got_offsets.size());
  ASSERT_FALSE(got_offsets.empty());
  EXPECT_EQ(got_offsets.front(), 0);
  EXPECT_EQ(got_offsets.back(), gb_groups.keys->num_rows());
  // Check each returned group boundary, allowing groups to appear in any order.
  cudf::test::fixed_width_column_wrapper<cudf::size_type> starts(got_offsets.begin(),
                                                                 got_offsets.end() - 1);
  auto const representatives = cudf::gather(gb_groups.keys->view(), starts);
  auto const group_order     = cudf::sorted_order(representatives->view());
  auto const [host_group_order, unused_validity] =
    cudf::test::to_host<cudf::size_type>(*group_order);
  for (std::size_t i = 0; i < host_group_order.size(); ++i) {
    auto const group = host_group_order[i];
    EXPECT_EQ(expect_group_offsets[i + 1] - expect_group_offsets[i],
              got_offsets[group + 1] - got_offsets[group]);
    auto const actual_group =
      cudf::slice(gb_groups.keys->view(), {got_offsets[group], got_offsets[group + 1]});
    auto const expected_group = cudf::slice(cudf::table_view{{expect_grouped_keys}},
                                            {expect_group_offsets[i], expect_group_offsets[i + 1]});
    CUDF_TEST_EXPECT_TABLES_EQUAL(expected_group.front(), actual_group.front());
  }

  if (values.size()) {
    auto const ordered_values = cudf::gather(gb_groups.values->view(), *order);
    CUDF_TEST_EXPECT_TABLES_EQUAL(cudf::table_view({expect_grouped_values}), *ordered_values);
  }
}

struct groupby_group_keys_test : public cudf::test::BaseFixture {};

template <typename V>
struct groupby_group_keys_and_values_test : public cudf::test::BaseFixture {};

TYPED_TEST_SUITE(groupby_group_keys_and_values_test, cudf::test::NumericTypes);

TEST_F(groupby_group_keys_test, basic)
{
  using K = int32_t;

  cudf::test::fixed_width_column_wrapper<K> keys{1, 1, 2, 1, 2, 3};
  cudf::test::fixed_width_column_wrapper<K> expect_grouped_keys{1, 1, 1, 2, 2, 3};
  std::vector<cudf::size_type> expect_group_offsets = {0, 3, 5, 6};
  test_groups(keys, expect_grouped_keys, expect_group_offsets);
}

TEST_F(groupby_group_keys_test, empty_keys)
{
  using K = int32_t;

  cudf::test::fixed_width_column_wrapper<K> keys{};
  cudf::test::fixed_width_column_wrapper<K> expect_grouped_keys{};
  std::vector<cudf::size_type> expect_group_offsets = {0};
  test_groups(keys, expect_grouped_keys, expect_group_offsets);
}

TEST_F(groupby_group_keys_test, all_null_keys)
{
  using K = int32_t;

  cudf::test::fixed_width_column_wrapper<K> keys({1, 1, 2, 3, 1, 2},
                                                 cudf::test::iterators::all_nulls());
  cudf::test::fixed_width_column_wrapper<K> expect_grouped_keys{};
  std::vector<cudf::size_type> expect_group_offsets = {0};
  test_groups(keys, expect_grouped_keys, expect_group_offsets);
}

TYPED_TEST(groupby_group_keys_and_values_test, basic_with_values)
{
  using K = int32_t;
  using V = TypeParam;

  cudf::test::fixed_width_column_wrapper<K> keys({5, 4, 3, 2, 1, 0});
  cudf::test::fixed_width_column_wrapper<K> expect_grouped_keys{0, 1, 2, 3, 4, 5};
  cudf::test::fixed_width_column_wrapper<V> values({0, 0, 1, 1, 2, 2});
  cudf::test::fixed_width_column_wrapper<V> expect_grouped_values{2, 2, 1, 1, 0, 0};
  std::vector<cudf::size_type> expect_group_offsets = {0, 1, 2, 3, 4, 5, 6};
  test_groups(keys, expect_grouped_keys, expect_group_offsets, values, expect_grouped_values);
}

TYPED_TEST(groupby_group_keys_and_values_test, some_nulls)
{
  using K = int32_t;
  using V = TypeParam;

  cudf::test::fixed_width_column_wrapper<K> keys({1, 1, 3, 2, 1, 2},
                                                 {true, false, true, false, false, true});
  cudf::test::fixed_width_column_wrapper<K> expect_grouped_keys({1, 2, 3},
                                                                cudf::test::iterators::no_nulls());
  cudf::test::fixed_width_column_wrapper<V> values({1, 2, 3, 4, 5, 6});
  cudf::test::fixed_width_column_wrapper<V> expect_grouped_values({1, 6, 3});
  std::vector<cudf::size_type> expect_group_offsets = {0, 1, 2, 3};
  test_groups(keys, expect_grouped_keys, expect_group_offsets, values, expect_grouped_values);
}
