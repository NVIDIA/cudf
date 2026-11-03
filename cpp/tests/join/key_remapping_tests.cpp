/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>

#include <cudf/column/column.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/join/key_remapping.hpp>
#include <cudf/sorting.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/mr/statistics_resource_adaptor.hpp>

#include <algorithm>
#include <set>
#include <vector>

template <typename T>
using column_wrapper = cudf::test::fixed_width_column_wrapper<T>;
using strcol_wrapper = cudf::test::strings_column_wrapper;

struct KeyRemappingTest : public cudf::test::BaseFixture {
  // Helper to copy column to host vector
  template <typename T>
  std::vector<T> to_host(cudf::column_view const& col)
  {
    return cudf::detail::make_std_vector<T>(
      cudf::device_span<T const>{col.data<T>(), static_cast<std::size_t>(col.size())},
      cudf::get_default_stream());
  }

  // Verify that equal keys in the input map to equal IDs in the output
  // Uses the original keys table and the remapped IDs column
  void verify_equal_keys_have_equal_ids(cudf::table_view const& keys, cudf::column_view const& ids)
  {
    // Sort the keys+ids together by keys, then verify adjacent equal keys have equal ids
    // Right a table with keys + id column for sorting
    std::vector<cudf::column_view> all_cols;
    for (int i = 0; i < keys.num_columns(); ++i) {
      all_cols.push_back(keys.column(i));
    }
    all_cols.push_back(ids);
    cudf::table_view keys_with_ids{all_cols};

    // Sort by keys (all columns except last)
    std::vector<cudf::order> orders(keys.num_columns(), cudf::order::ASCENDING);
    std::vector<cudf::null_order> null_orders(keys.num_columns(), cudf::null_order::BEFORE);
    auto sorted_indices = cudf::sorted_order(keys, orders, null_orders);
    auto sorted_table   = cudf::gather(keys_with_ids, *sorted_indices);

    // Bring sorted ids to host and verify adjacent equal keys have equal ids
    auto sorted_ids = to_host<int32_t>(sorted_table->get_column(keys.num_columns()).view());

    // Extract just the key columns from the sorted table
    std::vector<cudf::column_view> sorted_key_cols_view;
    for (int i = 0; i < keys.num_columns(); ++i) {
      sorted_key_cols_view.push_back(sorted_table->get_column(i).view());
    }
    cudf::table_view sorted_keys{sorted_key_cols_view};

    // Bring all sorted key columns to host for comparison
    // Assuming int32_t keys (as used in all current tests)
    std::vector<std::vector<int32_t>> sorted_key_cols;
    for (int i = 0; i < keys.num_columns(); ++i) {
      sorted_key_cols.push_back(to_host<int32_t>(sorted_keys.column(i)));
    }

    // Verify: if row[i] and row[i+1] have equal keys, they must have equal IDs
    for (size_t i = 1; i < sorted_ids.size(); ++i) {
      // Check if current row has same keys as previous row
      bool keys_equal = true;
      for (int col = 0; col < keys.num_columns(); ++col) {
        // Handle null values - they should be treated as equal to each other or according to
        // null_equality
        if (sorted_key_cols[col][i] != sorted_key_cols[col][i - 1]) {
          keys_equal = false;
          break;
        }
      }

      // If keys are equal, IDs must be equal (both non-negative or both sentinel)
      if (keys_equal) {
        EXPECT_EQ(sorted_ids[i], sorted_ids[i - 1])
          << "Equal keys at rows " << (i - 1) << " and " << i
          << " have different IDs: " << sorted_ids[i - 1] << " vs " << sorted_ids[i];
      }
    }
  }

  // Verify remapping contract: distinct keys get distinct IDs, equal keys get equal IDs
  void verify_remapping_contract(cudf::table_view const& keys,
                                 cudf::column_view const& ids,
                                 cudf::size_type expected_distinct_count)
  {
    auto host_ids = to_host<int32_t>(ids);

    // Collect non-sentinel IDs
    std::set<int32_t> unique_ids;
    for (auto id : host_ids) {
      if (id >= 0) { unique_ids.insert(id); }
    }

    // Number of unique non-negative IDs should equal distinct key count
    EXPECT_EQ(static_cast<cudf::size_type>(unique_ids.size()), expected_distinct_count);

    // All non-negative IDs should be valid (non-negative)
    for (auto id : host_ids) {
      EXPECT_TRUE(id >= 0 || id == cudf::KEY_REMAP_NOT_FOUND || id == cudf::KEY_REMAP_RIGHT_NULL);
    }
  }

  // Verify that left keys matching right keys get the same ID
  void verify_left_matches_right(cudf::table_view const& right_keys,
                                 cudf::column_view const& right_ids,
                                 cudf::table_view const& left_keys,
                                 cudf::column_view const& left_ids)
  {
    auto host_right_ids = to_host<int32_t>(right_ids);
    auto host_left_ids  = to_host<int32_t>(left_ids);

    // For each left row, if its ID is non-negative, there should exist
    // a right row with the same ID (meaning the key was found)
    std::set<int32_t> right_id_set(host_right_ids.begin(), host_right_ids.end());

    for (size_t i = 0; i < host_left_ids.size(); ++i) {
      auto left_id = host_left_ids[i];
      if (left_id >= 0) {
        // This left key matched a right key, so the ID should exist in right
        EXPECT_TRUE(right_id_set.count(left_id) > 0)
          << "Left row " << i << " has ID " << left_id << " not found in right IDs";
      }
    }
  }
};

TEST_F(KeyRemappingTest, BasicIntegerKeys)
{
  // Right table with some duplicate keys: [1, 2, 3, 2, 1]
  column_wrapper<int32_t> right_col{1, 2, 3, 2, 1};
  auto right_table = cudf::table_view{{right_col}};

  cudf::key_remapping remap{right_table};

  // Check distinct count (should be 3: values 1, 2, 3)
  EXPECT_EQ(remap.get_distinct_count(), 3);

  // Check max duplicate count (value 1 and 2 both appear twice)
  EXPECT_EQ(remap.get_max_duplicate_count(), 2);

  // Remap right keys
  auto right_result = remap.remap_right_keys();

  // Verify contract: 3 distinct IDs for 3 distinct keys
  verify_remapping_contract(right_table, *right_result, 3);

  // Verify equal keys get equal IDs by checking specific pairs
  auto host_ids = to_host<int32_t>(*right_result);
  EXPECT_EQ(host_ids[0], host_ids[4]);  // Both have key=1
  EXPECT_EQ(host_ids[1], host_ids[3]);  // Both have key=2
  EXPECT_NE(host_ids[0], host_ids[1]);  // key=1 vs key=2
  EXPECT_NE(host_ids[0], host_ids[2]);  // key=1 vs key=3
  EXPECT_NE(host_ids[1], host_ids[2]);  // key=2 vs key=3
}

TEST_F(KeyRemappingTest, LeftKeys)
{
  column_wrapper<int32_t> right_col{1, 2, 3, 2, 1};
  auto right_table = cudf::table_view{{right_col}};

  cudf::key_remapping remap{right_table};

  auto right_result   = remap.remap_right_keys();
  auto host_right_ids = to_host<int32_t>(*right_result);

  // Left with some matching and non-matching keys
  column_wrapper<int32_t> left_col{3, 1, 5, 2};
  auto left_table = cudf::table_view{{left_col}};

  auto left_result   = remap.remap_left_keys(left_table);
  auto host_left_ids = to_host<int32_t>(*left_result);

  // key=3 in left should match key=3 in right (row 2)
  EXPECT_EQ(host_left_ids[0], host_right_ids[2]);

  // key=1 in left should match key=1 in right (rows 0 and 4 have same ID)
  EXPECT_EQ(host_left_ids[1], host_right_ids[0]);

  // key=5 not in right -> NOT_FOUND
  EXPECT_EQ(host_left_ids[2], cudf::KEY_REMAP_NOT_FOUND);

  // key=2 in left should match key=2 in right (rows 1 and 3 have same ID)
  EXPECT_EQ(host_left_ids[3], host_right_ids[1]);

  // Verify left matches right
  verify_left_matches_right(right_table, *right_result, left_table, *left_result);
}

TEST_F(KeyRemappingTest, StringKeys)
{
  strcol_wrapper right_col{"apple", "banana", "cherry", "banana"};
  auto right_table = cudf::table_view{{right_col}};

  cudf::key_remapping remap{right_table};

  EXPECT_EQ(remap.get_distinct_count(), 3);
  EXPECT_EQ(remap.get_max_duplicate_count(), 2);  // "banana" appears twice

  auto right_result = remap.remap_right_keys();
  verify_remapping_contract(right_table, *right_result, 3);

  // Verify equal keys get equal IDs
  auto host_ids = to_host<int32_t>(*right_result);
  EXPECT_EQ(host_ids[1], host_ids[3]);  // Both "banana"
  EXPECT_NE(host_ids[0], host_ids[1]);  // "apple" vs "banana"
  EXPECT_NE(host_ids[0], host_ids[2]);  // "apple" vs "cherry"

  // Left with matching and non-matching keys
  strcol_wrapper left_col{"cherry", "date", "apple"};
  auto left_table = cudf::table_view{{left_col}};

  auto left_result   = remap.remap_left_keys(left_table);
  auto host_left_ids = to_host<int32_t>(*left_result);

  EXPECT_EQ(host_left_ids[0], host_ids[2]);                // "cherry" matches
  EXPECT_EQ(host_left_ids[1], cudf::KEY_REMAP_NOT_FOUND);  // "date" not found
  EXPECT_EQ(host_left_ids[2], host_ids[0]);                // "apple" matches

  verify_left_matches_right(right_table, *right_result, left_table, *left_result);
}

TEST_F(KeyRemappingTest, MultiColumnKeys)
{
  column_wrapper<int32_t> right_col1{1, 1, 2, 1};
  strcol_wrapper right_col2{"a", "b", "a", "a"};
  auto right_table = cudf::table_view{{right_col1, right_col2}};

  cudf::key_remapping remap{right_table};

  // Distinct keys: (1,"a"), (1,"b"), (2,"a") = 3 distinct
  EXPECT_EQ(remap.get_distinct_count(), 3);
  EXPECT_EQ(remap.get_max_duplicate_count(), 2);  // (1,"a") appears twice

  auto right_result = remap.remap_right_keys();
  verify_remapping_contract(right_table, *right_result, 3);

  // Verify equal keys get equal IDs
  auto host_ids = to_host<int32_t>(*right_result);
  EXPECT_EQ(host_ids[0], host_ids[3]);  // Both (1,"a")
  EXPECT_NE(host_ids[0], host_ids[1]);  // (1,"a") vs (1,"b")
  EXPECT_NE(host_ids[0], host_ids[2]);  // (1,"a") vs (2,"a")

  // Left
  column_wrapper<int32_t> left_col1{2, 1, 3};
  strcol_wrapper left_col2{"a", "b", "c"};
  auto left_table = cudf::table_view{{left_col1, left_col2}};

  auto left_result   = remap.remap_left_keys(left_table);
  auto host_left_ids = to_host<int32_t>(*left_result);

  EXPECT_EQ(host_left_ids[0], host_ids[2]);                // (2,"a") matches
  EXPECT_EQ(host_left_ids[1], host_ids[1]);                // (1,"b") matches
  EXPECT_EQ(host_left_ids[2], cudf::KEY_REMAP_NOT_FOUND);  // (3,"c") not found

  verify_left_matches_right(right_table, *right_result, left_table, *left_result);
}

TEST_F(KeyRemappingTest, NullsEqual)
{
  // Right table with nulls, nulls are equal
  column_wrapper<int32_t> right_col{{1, 2, 0, 2}, {true, true, false, true}};
  auto right_table = cudf::table_view{{right_col}};

  cudf::key_remapping remap{right_table, cudf::null_equality::EQUAL};

  // Distinct: 1, 2, null = 3
  EXPECT_EQ(remap.get_distinct_count(), 3);

  auto right_result = remap.remap_right_keys();
  verify_remapping_contract(right_table, *right_result, 3);

  auto host_ids = to_host<int32_t>(*right_result);
  // Rows with key=2 should have same ID
  EXPECT_EQ(host_ids[1], host_ids[3]);
  // All IDs should be non-negative (nulls are treated as equal, so they get valid IDs)
  for (auto id : host_ids) {
    EXPECT_GE(id, 0);
  }

  // Left with nulls - null should match
  column_wrapper<int32_t> left_col{{0, 1}, {false, true}};
  auto left_table = cudf::table_view{{left_col}};

  auto left_result   = remap.remap_left_keys(left_table);
  auto host_left_ids = to_host<int32_t>(*left_result);

  // null in left should match null in right
  EXPECT_EQ(host_left_ids[0], host_ids[2]);
  // 1 in left should match 1 in right
  EXPECT_EQ(host_left_ids[1], host_ids[0]);
}

TEST_F(KeyRemappingTest, NullsUnequal)
{
  // Right table with nulls, nulls are unequal
  column_wrapper<int32_t> right_col{{1, 2, 0, 2}, {true, true, false, true}};
  auto right_table = cudf::table_view{{right_col}};

  cudf::key_remapping remap{right_table, cudf::null_equality::UNEQUAL};

  // Distinct: 1, 2 = 2 (null is skipped)
  EXPECT_EQ(remap.get_distinct_count(), 2);

  auto right_result = remap.remap_right_keys();
  auto host_ids     = to_host<int32_t>(*right_result);

  // Rows with key=2 should have same ID
  EXPECT_EQ(host_ids[1], host_ids[3]);
  // Null row should get BUILD_NULL sentinel
  EXPECT_EQ(host_ids[2], cudf::KEY_REMAP_RIGHT_NULL);
  // Non-null rows should have non-negative IDs
  EXPECT_GE(host_ids[0], 0);
  EXPECT_GE(host_ids[1], 0);

  // Left with nulls - null should not match
  column_wrapper<int32_t> left_col{{0, 1}, {false, true}};
  auto left_table = cudf::table_view{{left_col}};

  auto left_result   = remap.remap_left_keys(left_table);
  auto host_left_ids = to_host<int32_t>(*left_result);

  // null in left should get NOT_FOUND
  EXPECT_EQ(host_left_ids[0], cudf::KEY_REMAP_NOT_FOUND);
  // 1 in left should match 1 in right
  EXPECT_EQ(host_left_ids[1], host_ids[0]);
}

TEST_F(KeyRemappingTest, EmptyRightTable)
{
  column_wrapper<int32_t> right_col{};
  auto right_table = cudf::table_view{{right_col}};

  cudf::key_remapping remap{right_table};

  EXPECT_EQ(remap.get_distinct_count(), 0);
  EXPECT_EQ(remap.get_max_duplicate_count(), 0);

  // Left should return all NOT_FOUND
  column_wrapper<int32_t> left_col{1, 2, 3};
  auto left_table = cudf::table_view{{left_col}};

  auto left_result   = remap.remap_left_keys(left_table);
  auto host_left_ids = to_host<int32_t>(*left_result);

  for (auto id : host_left_ids) {
    EXPECT_EQ(id, cudf::KEY_REMAP_NOT_FOUND);
  }
}

TEST_F(KeyRemappingTest, EmptyLeftTable)
{
  column_wrapper<int32_t> right_col{1, 2, 3};
  auto right_table = cudf::table_view{{right_col}};

  cudf::key_remapping remap{right_table};

  column_wrapper<int32_t> left_col{};
  auto left_table = cudf::table_view{{left_col}};

  auto left_result = remap.remap_left_keys(left_table);
  EXPECT_EQ(left_result->size(), 0);
}

TEST_F(KeyRemappingTest, AllDuplicates)
{
  // All rows have the same key
  column_wrapper<int32_t> right_col{42, 42, 42, 42, 42};
  auto right_table = cudf::table_view{{right_col}};

  cudf::key_remapping remap{right_table};

  EXPECT_EQ(remap.get_distinct_count(), 1);
  EXPECT_EQ(remap.get_max_duplicate_count(), 5);

  auto right_result = remap.remap_right_keys();
  auto host_ids     = to_host<int32_t>(*right_result);

  // All should have the same ID (whatever it is)
  auto first_id = host_ids[0];
  EXPECT_GE(first_id, 0);
  for (auto id : host_ids) {
    EXPECT_EQ(id, first_id);
  }
}

TEST_F(KeyRemappingTest, AllUnique)
{
  column_wrapper<int32_t> right_col{1, 2, 3, 4, 5};
  auto right_table = cudf::table_view{{right_col}};

  cudf::key_remapping remap{right_table};

  EXPECT_EQ(remap.get_distinct_count(), 5);
  EXPECT_EQ(remap.get_max_duplicate_count(), 1);

  auto right_result = remap.remap_right_keys();
  auto host_ids     = to_host<int32_t>(*right_result);

  // All IDs should be unique and non-negative
  std::set<int32_t> unique_ids(host_ids.begin(), host_ids.end());
  EXPECT_EQ(unique_ids.size(), 5u);
  for (auto id : host_ids) {
    EXPECT_GE(id, 0);
  }
}

TEST_F(KeyRemappingTest, LargeTable)
{
  // Create a larger table to test with 100 distinct values, each appearing 100 times
  std::vector<int32_t> data(10000);
  for (size_t i = 0; i < data.size(); ++i) {
    data[i] = static_cast<int32_t>(i % 100);
  }

  column_wrapper<int32_t> right_col(data.begin(), data.end());
  auto right_table = cudf::table_view{{right_col}};

  cudf::key_remapping remap{right_table};

  EXPECT_EQ(remap.get_distinct_count(), 100);
  EXPECT_EQ(remap.get_max_duplicate_count(), 100);

  auto right_result = remap.remap_right_keys();
  verify_remapping_contract(right_table, *right_result, 100);

  auto host_ids = to_host<int32_t>(*right_result);

  // Verify all rows with the same key have the same ID
  // Group by key value and check IDs
  std::map<int32_t, int32_t> key_to_id;
  for (size_t i = 0; i < data.size(); ++i) {
    auto key = data[i];
    auto id  = host_ids[i];
    if (key_to_id.count(key) == 0) {
      key_to_id[key] = id;
    } else {
      EXPECT_EQ(key_to_id[key], id) << "Key " << key << " has inconsistent IDs";
    }
  }

  // Left with values 0-99 (all exist) and 100-104 (don't exist)
  std::vector<int32_t> left_data;
  for (int i = 0; i < 105; ++i) {
    left_data.push_back(i);
  }
  column_wrapper<int32_t> left_col(left_data.begin(), left_data.end());
  auto left_table = cudf::table_view{{left_col}};

  auto left_result   = remap.remap_left_keys(left_table);
  auto host_left_ids = to_host<int32_t>(*left_result);

  // Keys 0-99 should match right
  for (int i = 0; i < 100; ++i) {
    EXPECT_EQ(host_left_ids[i], key_to_id[i]) << "Left key " << i << " mismatch";
  }
  // Keys 100-104 should be NOT_FOUND
  for (int i = 100; i < 105; ++i) {
    EXPECT_EQ(host_left_ids[i], cudf::KEY_REMAP_NOT_FOUND)
      << "Left key " << i << " should be NOT_FOUND";
  }
}

TEST_F(KeyRemappingTest, StructKeys)
{
  // Test with struct column keys
  column_wrapper<int32_t> child1{1, 1, 2, 1};
  strcol_wrapper child2{"a", "b", "a", "a"};
  auto struct_col  = cudf::test::structs_column_wrapper{{child1, child2}};
  auto right_table = cudf::table_view{{struct_col}};

  cudf::key_remapping remap{right_table};

  // Distinct structs: {1,"a"}, {1,"b"}, {2,"a"} = 3
  EXPECT_EQ(remap.get_distinct_count(), 3);
  EXPECT_EQ(remap.get_max_duplicate_count(), 2);

  auto right_result = remap.remap_right_keys();
  verify_remapping_contract(right_table, *right_result, 3);

  auto host_ids = to_host<int32_t>(*right_result);
  // Rows 0 and 3 have same struct {1,"a"}
  EXPECT_EQ(host_ids[0], host_ids[3]);
  // All different struct values have different IDs
  EXPECT_NE(host_ids[0], host_ids[1]);
  EXPECT_NE(host_ids[0], host_ids[2]);
  EXPECT_NE(host_ids[1], host_ids[2]);
}

TEST_F(KeyRemappingTest, FloatKeys)
{
  // Test with float keys including duplicates
  column_wrapper<float> right_col{1.5f, 2.5f, 3.5f, 2.5f, 1.5f};
  auto right_table = cudf::table_view{{right_col}};

  cudf::key_remapping remap{right_table};

  // Distinct: 1.5, 2.5, 3.5 = 3
  EXPECT_EQ(remap.get_distinct_count(), 3);
  EXPECT_EQ(remap.get_max_duplicate_count(), 2);

  auto right_result = remap.remap_right_keys();
  verify_remapping_contract(right_table, *right_result, 3);

  auto host_ids = to_host<int32_t>(*right_result);
  // Equal keys should have equal IDs
  EXPECT_EQ(host_ids[0], host_ids[4]);  // Both 1.5
  EXPECT_EQ(host_ids[1], host_ids[3]);  // Both 2.5
  // Different keys should have different IDs
  EXPECT_NE(host_ids[0], host_ids[1]);
  EXPECT_NE(host_ids[0], host_ids[2]);
  EXPECT_NE(host_ids[1], host_ids[2]);

  // Left with matching and non-matching keys
  column_wrapper<float> left_col{3.5f, 1.5f, 9.9f, 2.5f};
  auto left_table = cudf::table_view{{left_col}};

  auto left_result   = remap.remap_left_keys(left_table);
  auto host_left_ids = to_host<int32_t>(*left_result);

  EXPECT_EQ(host_left_ids[0], host_ids[2]);                // 3.5 matches
  EXPECT_EQ(host_left_ids[1], host_ids[0]);                // 1.5 matches
  EXPECT_EQ(host_left_ids[2], cudf::KEY_REMAP_NOT_FOUND);  // 9.9 not found
  EXPECT_EQ(host_left_ids[3], host_ids[1]);                // 2.5 matches

  verify_left_matches_right(right_table, *right_result, left_table, *left_result);
}

TEST_F(KeyRemappingTest, DoubleKeys)
{
  // Test with double keys including duplicates
  column_wrapper<double> right_col{1.123456789, 2.987654321, 3.141592653, 2.987654321};
  auto right_table = cudf::table_view{{right_col}};

  cudf::key_remapping remap{right_table};

  // Distinct: 3 values (the second 2.987654321 is a duplicate)
  EXPECT_EQ(remap.get_distinct_count(), 3);
  EXPECT_EQ(remap.get_max_duplicate_count(), 2);

  auto right_result = remap.remap_right_keys();
  verify_remapping_contract(right_table, *right_result, 3);

  auto host_ids = to_host<int32_t>(*right_result);
  // Equal keys should have equal IDs
  EXPECT_EQ(host_ids[1], host_ids[3]);  // Both 2.987654321
  // Different keys should have different IDs
  EXPECT_NE(host_ids[0], host_ids[1]);
  EXPECT_NE(host_ids[0], host_ids[2]);
  EXPECT_NE(host_ids[1], host_ids[2]);

  // Left with matching and non-matching keys
  column_wrapper<double> left_col{3.141592653, 1.123456789, 99.99};
  auto left_table = cudf::table_view{{left_col}};

  auto left_result   = remap.remap_left_keys(left_table);
  auto host_left_ids = to_host<int32_t>(*left_result);

  EXPECT_EQ(host_left_ids[0], host_ids[2]);                // pi matches
  EXPECT_EQ(host_left_ids[1], host_ids[0]);                // 1.123... matches
  EXPECT_EQ(host_left_ids[2], cudf::KEY_REMAP_NOT_FOUND);  // 99.99 not found

  verify_left_matches_right(right_table, *right_result, left_table, *left_result);
}

TEST_F(KeyRemappingTest, FloatWithNulls)
{
  // Test float keys with null values
  column_wrapper<float> right_col{{1.5f, 2.5f, 0.0f, 2.5f}, {true, true, false, true}};
  auto right_table = cudf::table_view{{right_col}};

  cudf::key_remapping remap{right_table, cudf::null_equality::EQUAL};

  // Distinct: 1.5, 2.5, null = 3
  EXPECT_EQ(remap.get_distinct_count(), 3);

  auto right_result = remap.remap_right_keys();
  verify_remapping_contract(right_table, *right_result, 3);

  auto host_ids = to_host<int32_t>(*right_result);
  // Equal keys should have equal IDs
  EXPECT_EQ(host_ids[1], host_ids[3]);  // Both 2.5
  // All IDs should be non-negative (nulls treated as equal)
  for (auto id : host_ids) {
    EXPECT_GE(id, 0);
  }

  // Test with UNEQUAL null semantics
  cudf::key_remapping remap_unequal{right_table, cudf::null_equality::UNEQUAL};

  // Distinct: 1.5, 2.5 = 2 (null skipped)
  EXPECT_EQ(remap_unequal.get_distinct_count(), 2);

  auto right_result_unequal = remap_unequal.remap_right_keys();
  auto host_ids_unequal     = to_host<int32_t>(*right_result_unequal);

  // Null row should get BUILD_NULL sentinel
  EXPECT_EQ(host_ids_unequal[2], cudf::KEY_REMAP_RIGHT_NULL);
}

TEST_F(KeyRemappingTest, DoubleWithNulls)
{
  // Test double keys with null values
  column_wrapper<double> right_col{{1.0, 2.0, 0.0, 2.0}, {true, true, false, true}};
  auto right_table = cudf::table_view{{right_col}};

  cudf::key_remapping remap{right_table, cudf::null_equality::EQUAL};

  // Distinct: 1.0, 2.0, null = 3
  EXPECT_EQ(remap.get_distinct_count(), 3);

  auto right_result = remap.remap_right_keys();
  auto host_ids     = to_host<int32_t>(*right_result);

  // Equal keys should have equal IDs
  EXPECT_EQ(host_ids[1], host_ids[3]);  // Both 2.0
  // All IDs should be non-negative (nulls treated as equal)
  for (auto id : host_ids) {
    EXPECT_GE(id, 0);
  }

  // Test with UNEQUAL null semantics
  cudf::key_remapping remap_unequal{right_table, cudf::null_equality::UNEQUAL};

  // Distinct: 1.0, 2.0 = 2 (null skipped)
  EXPECT_EQ(remap_unequal.get_distinct_count(), 2);

  auto right_result_unequal = remap_unequal.remap_right_keys();
  auto host_ids_unequal     = to_host<int32_t>(*right_result_unequal);

  // Null row should get BUILD_NULL sentinel
  EXPECT_EQ(host_ids_unequal[2], cudf::KEY_REMAP_RIGHT_NULL);
}

// Schema validation tests: left table must match right table schema

TEST_F(KeyRemappingTest, LeftSchemaMismatchColumnCount)
{
  // Right with 2 columns
  column_wrapper<int32_t> right_col1{1, 2, 3};
  column_wrapper<int32_t> right_col2{4, 5, 6};
  auto right_table = cudf::table_view{{right_col1, right_col2}};

  cudf::key_remapping remap{right_table};

  // Left with 1 column - should throw
  column_wrapper<int32_t> left_col{1, 2};
  auto left_table = cudf::table_view{{left_col}};

  EXPECT_THROW((void)remap.remap_left_keys(left_table), std::invalid_argument);
  EXPECT_THROW((void)remap.remap_left_keys(left_table), std::invalid_argument);
}

TEST_F(KeyRemappingTest, LeftSchemaMismatchColumnType)
{
  // Right with INT32
  column_wrapper<int32_t> right_col{1, 2, 3};
  auto right_table = cudf::table_view{{right_col}};

  cudf::key_remapping remap{right_table};

  // Left with INT64 - should throw due to type mismatch
  column_wrapper<int64_t> left_col{1, 2, 3};
  auto left_table = cudf::table_view{{left_col}};

  EXPECT_THROW((void)remap.remap_left_keys(left_table), cudf::data_type_error);
  EXPECT_THROW((void)remap.remap_left_keys(left_table), cudf::data_type_error);
}

TEST_F(KeyRemappingTest, LeftSchemaMismatchNestedVsPrimitive)
{
  // Right with struct column
  column_wrapper<int32_t> child1{1, 2, 3};
  strcol_wrapper child2{"a", "b", "c"};
  auto struct_col  = cudf::test::structs_column_wrapper{{child1, child2}};
  auto right_table = cudf::table_view{{struct_col}};

  cudf::key_remapping remap{right_table};

  // Left with primitive column - should throw due to type mismatch
  column_wrapper<int32_t> left_col{1, 2, 3};
  auto left_table = cudf::table_view{{left_col}};

  EXPECT_THROW((void)remap.remap_left_keys(left_table), cudf::data_type_error);
  EXPECT_THROW((void)remap.remap_left_keys(left_table), cudf::data_type_error);
}

TEST_F(KeyRemappingTest, LeftSchemaMismatchStructFields)
{
  // Right with struct{INT32, STRING}
  column_wrapper<int32_t> right_child1{1, 2, 3};
  strcol_wrapper right_child2{"a", "b", "c"};
  auto right_struct = cudf::test::structs_column_wrapper{{right_child1, right_child2}};
  auto right_table  = cudf::table_view{{right_struct}};

  cudf::key_remapping remap{right_table};

  // Left with struct{INT32, INT32} - different field types, should throw
  column_wrapper<int32_t> left_child1{1, 2, 3};
  column_wrapper<int32_t> left_child2{4, 5, 6};
  auto left_struct = cudf::test::structs_column_wrapper{{left_child1, left_child2}};
  auto left_table  = cudf::table_view{{left_struct}};

  EXPECT_THROW((void)remap.remap_left_keys(left_table), cudf::data_type_error);
  EXPECT_THROW((void)remap.remap_left_keys(left_table), cudf::data_type_error);
}

TEST_F(KeyRemappingTest, EmptyLeftSchemaMismatchColumnCount)
{
  // Right with 2 columns
  column_wrapper<int32_t> right_col1{1, 2, 3};
  column_wrapper<int32_t> right_col2{4, 5, 6};
  auto right_table = cudf::table_view{{right_col1, right_col2}};

  cudf::key_remapping remap{right_table};

  // Empty left with 1 column - should still throw due to column count mismatch
  column_wrapper<int32_t> left_col{};
  auto left_table = cudf::table_view{{left_col}};

  EXPECT_THROW((void)remap.remap_left_keys(left_table), std::invalid_argument);
  EXPECT_THROW((void)remap.remap_left_keys(left_table), std::invalid_argument);
}

// Tests for optional metrics computation

TEST_F(KeyRemappingTest, MemoryResource)
{
  column_wrapper<int32_t> right_col{1, 2, 2, 3};
  auto right_table = cudf::table_view{{right_col}};

  auto mr = rmm::mr::statistics_resource_adaptor(cudf::get_current_device_resource_ref());

  cudf::key_remapping remap{right_table,
                            cudf::null_equality::EQUAL,
                            cudf::compute_metrics::YES,
                            cudf::get_default_stream(),
                            mr};

  EXPECT_GT(mr.get_bytes_counter().peak, 0);

  auto result = remap.remap_right_keys();
  auto ids    = to_host<int32_t>(result->view());

  ASSERT_EQ(ids.size(), 4);
  EXPECT_EQ(ids[1], ids[2]);
  EXPECT_NE(ids[0], ids[1]);
  EXPECT_NE(ids[1], ids[3]);
}

TEST_F(KeyRemappingTest, MetricsEnabled)
{
  column_wrapper<int32_t> right_col{1, 2, 2, 3, 3, 3};
  auto right_table = cudf::table_view{{right_col}};

  // Default: metrics enabled
  cudf::key_remapping remap{right_table};

  EXPECT_TRUE(remap.has_metrics());
  EXPECT_EQ(remap.get_distinct_count(), 3);
  EXPECT_EQ(remap.get_max_duplicate_count(), 3);
}

TEST_F(KeyRemappingTest, MetricsDisabled)
{
  column_wrapper<int32_t> right_col{1, 2, 2, 3, 3, 3};
  auto right_table = cudf::table_view{{right_col}};

  // Explicitly disable metrics
  cudf::key_remapping remap{right_table, cudf::null_equality::EQUAL, cudf::compute_metrics::NO};

  EXPECT_FALSE(remap.has_metrics());
  EXPECT_THROW((void)remap.get_distinct_count(), cudf::logic_error);
  EXPECT_THROW((void)remap.get_max_duplicate_count(), cudf::logic_error);
}

TEST_F(KeyRemappingTest, MetricsDisabledRemapStillWorks)
{
  column_wrapper<int32_t> right_col{10, 20, 20, 30};
  auto right_table = cudf::table_view{{right_col}};

  // Disable metrics but remapping should still work
  cudf::key_remapping remap{right_table, cudf::null_equality::EQUAL, cudf::compute_metrics::NO};

  // Remap right keys
  auto right_result = remap.remap_right_keys();
  auto right_ids    = to_host<int32_t>(right_result->view());

  // Equal keys should have equal IDs
  EXPECT_EQ(right_ids[1], right_ids[2]);  // Both 20s
  EXPECT_NE(right_ids[0], right_ids[1]);  // 10 vs 20
  EXPECT_NE(right_ids[1], right_ids[3]);  // 20 vs 30

  // Remap left keys
  column_wrapper<int32_t> left_col{20, 40, 10};
  auto left_table  = cudf::table_view{{left_col}};
  auto left_result = remap.remap_left_keys(left_table);
  auto left_ids    = to_host<int32_t>(left_result->view());

  EXPECT_EQ(left_ids[0], right_ids[1]);               // 20 matches
  EXPECT_EQ(left_ids[1], cudf::KEY_REMAP_NOT_FOUND);  // 40 not found
  EXPECT_EQ(left_ids[2], right_ids[0]);               // 10 matches
}
