/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "../../src/partitioning/partition_policy.hpp"

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/copying.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/partitioning.hpp>
#include <cudf/sorting.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table.hpp>

#include <cuda/iterator>
#include <cuda/std/tuple>

#include <algorithm>
#include <cstdlib>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>

template <typename T>
class PartitionTest : public cudf::test::BaseFixture {
  using value_type = cudf::test::GetType<T, 0>;
  using map_type   = cudf::test::GetType<T, 1>;
};

using types =
  cudf::test::CrossProduct<cudf::test::FixedWidthTypes, cudf::test::IntegralTypesNotBool>;

// using types = cudf::test::Types<cudf::test::Types<int32_t, int32_t> >;

TYPED_TEST_SUITE(PartitionTest, types);

using cudf::test::fixed_width_column_wrapper;
using cudf::test::strings_column_wrapper;

// Exceptional cases
TYPED_TEST(PartitionTest, EmptyInputs)
{
  using value_type = cudf::test::GetType<TypeParam, 0>;
  using map_type   = cudf::test::GetType<TypeParam, 1>;

  fixed_width_column_wrapper<value_type> empty_column{};
  fixed_width_column_wrapper<map_type> empty_map{};

  auto result = cudf::partition(cudf::table_view{{empty_column}}, empty_map, 10);

  auto result_offsets = result.second;

  EXPECT_EQ(result_offsets.size(), std::size_t{11});

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(empty_column, result.first->get_column(0));
}

TYPED_TEST(PartitionTest, MapInputSizeMismatch)
{
  using value_type = cudf::test::GetType<TypeParam, 0>;
  using map_type   = cudf::test::GetType<TypeParam, 1>;

  fixed_width_column_wrapper<value_type, int32_t> input({1, 2, 3});
  fixed_width_column_wrapper<map_type> map{1, 2};

  EXPECT_THROW(cudf::partition(cudf::table_view{{input}}, map, 3), cudf::logic_error);
}

TYPED_TEST(PartitionTest, MapWithNullsThrows)
{
  using value_type = cudf::test::GetType<TypeParam, 0>;
  using map_type   = cudf::test::GetType<TypeParam, 1>;

  fixed_width_column_wrapper<value_type, int32_t> input({1, 2, 3});
  fixed_width_column_wrapper<map_type> map{{1, 2}, {1, 0}};

  EXPECT_THROW(cudf::partition(cudf::table_view{{input}}, map, 3), cudf::logic_error);
}

/**
 * @brief Verifies that partitions indicated by `offsets` are equal between
 * `expected` and `actual`.
 *
 * The order of rows within each partition may be different, so each partition
 * is first sorted before being compared for equality.
 */
void expect_equal_partitions(cudf::table_view expected,
                             cudf::table_view actual,
                             std::vector<cudf::size_type> const& offsets)
{
  // Need to convert partition offsets into split points by dropping the first
  // and last element
  std::vector<cudf::size_type> split_points;
  std::copy(std::next(offsets.begin()), std::prev(offsets.end()), std::back_inserter(split_points));

  // Split the partitions, sort each partition, then compare for equality
  auto actual_split   = cudf::split(actual, split_points);
  auto expected_split = cudf::split(expected, split_points);

  auto begin =
    cuda::make_zip_iterator(cuda::std::make_tuple(expected_split.begin(), actual_split.begin()));
  auto end =
    cuda::make_zip_iterator(cuda::std::make_tuple(expected_split.end(), actual_split.end()));

  std::for_each(begin, end, [](auto const& zipped) {
    auto [expected, actual] = zipped;
    auto sorted_expected    = cudf::sort(expected);
    auto sorted_actual      = cudf::sort(actual);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*sorted_expected, *sorted_actual);
  });
}

void run_partition_test(cudf::table_view table_to_partition,
                        cudf::column_view partition_map,
                        cudf::size_type num_partitions,
                        cudf::table_view expected_partitioned_table,
                        std::vector<cudf::size_type> const& expected_offsets)
{
  auto result = cudf::partition(table_to_partition, partition_map, num_partitions);
  auto const& actual_partitioned_table = result.first;
  auto const& actual_offsets           = result.second;
  EXPECT_EQ(actual_offsets, expected_offsets);

  expect_equal_partitions(expected_partitioned_table, *actual_partitioned_table, expected_offsets);
}

// Normal cases
TYPED_TEST(PartitionTest, Identity)
{
  using value_type = cudf::test::GetType<TypeParam, 0>;
  using map_type   = cudf::test::GetType<TypeParam, 1>;

  fixed_width_column_wrapper<value_type, int32_t> first({0, 1, 2, 3, 4, 5});
  strings_column_wrapper strings{"this", "is", "a", "column", "of", "strings"};
  auto table_to_partition = cudf::table_view{{first, strings}};

  fixed_width_column_wrapper<map_type> map{0, 1, 2, 3, 4, 5};

  std::vector<cudf::size_type> expected_offsets{0, 1, 2, 3, 4, 5, 6};

  run_partition_test(table_to_partition, map, 6, table_to_partition, expected_offsets);
}

TYPED_TEST(PartitionTest, Struct)
{
  using value_type = cudf::test::GetType<TypeParam, 0>;
  using map_type   = cudf::test::GetType<TypeParam, 1>;

  fixed_width_column_wrapper<value_type, int32_t> A({1, 2}, {0, 1});
  auto struct_col         = cudf::test::structs_column_wrapper({A}, {0, 1}).release();
  auto table_to_partition = cudf::table_view{{*struct_col}};

  fixed_width_column_wrapper<map_type> map{9, 2};

  fixed_width_column_wrapper<value_type, int32_t> A_expected({2, 1}, {1, 0});
  auto struct_expected = cudf::test::structs_column_wrapper({A_expected}, {1, 0}).release();
  auto expected        = cudf::table_view{{*struct_expected}};

  std::vector<cudf::size_type> expected_offsets{0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2};

  // This does not work because we cannot sort a struct right now...
  // run_partition_test(table_to_partition, map, 12, expected, expected_offsets);
  // But there is no ambiguity in the ordering so I'll just copy it all here for now.
  auto num_partitions                  = 12;
  auto result                          = cudf::partition(table_to_partition, map, num_partitions);
  auto const& actual_partitioned_table = result.first;
  auto const& actual_offsets           = result.second;
  EXPECT_EQ(actual_offsets, expected_offsets);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, *actual_partitioned_table);
}

TYPED_TEST(PartitionTest, Reverse)
{
  using value_type = cudf::test::GetType<TypeParam, 0>;
  using map_type   = cudf::test::GetType<TypeParam, 1>;

  fixed_width_column_wrapper<value_type, int32_t> first({0, 1, 3, 7, 5, 13});
  strings_column_wrapper strings{"this", "is", "a", "column", "of", "strings"};
  auto table_to_partition = cudf::table_view{{first, strings}};

  fixed_width_column_wrapper<map_type> map{5, 4, 3, 2, 1, 0};

  std::vector<cudf::size_type> expected_offsets{0, 1, 2, 3, 4, 5, 6};

  fixed_width_column_wrapper<value_type, int32_t> expected_first({13, 5, 7, 3, 1, 0});
  strings_column_wrapper expected_strings{"strings", "of", "column", "a", "is", "this"};
  auto expected_partitioned_table = cudf::table_view{{expected_first, expected_strings}};

  run_partition_test(table_to_partition, map, 6, expected_partitioned_table, expected_offsets);
}

TYPED_TEST(PartitionTest, SinglePartition)
{
  using value_type = cudf::test::GetType<TypeParam, 0>;
  using map_type   = cudf::test::GetType<TypeParam, 1>;

  fixed_width_column_wrapper<value_type, int32_t> first({0, 1, 3, 7, 5, 13});
  strings_column_wrapper strings{"this", "is", "a", "column", "of", "strings"};
  auto table_to_partition = cudf::table_view{{first, strings}};

  fixed_width_column_wrapper<map_type> map{0, 0, 0, 0, 0, 0};

  std::vector<cudf::size_type> expected_offsets{0, 6};

  fixed_width_column_wrapper<value_type, int32_t> expected_first({13, 5, 7, 3, 1, 0});
  strings_column_wrapper expected_strings{"strings", "of", "column", "a", "is", "this"};
  auto expected_partitioned_table = cudf::table_view{{expected_first, expected_strings}};

  run_partition_test(table_to_partition, map, 1, expected_partitioned_table, expected_offsets);
}

TYPED_TEST(PartitionTest, EmptyPartitions)
{
  using value_type = cudf::test::GetType<TypeParam, 0>;
  using map_type   = cudf::test::GetType<TypeParam, 1>;

  fixed_width_column_wrapper<value_type, int32_t> first({0, 1, 3, 7, 5, 13});
  strings_column_wrapper strings{"this", "is", "a", "column", "of", "strings"};
  auto table_to_partition = cudf::table_view{{first, strings}};

  fixed_width_column_wrapper<map_type> map{2, 2, 0, 0, 4, 4};

  std::vector<cudf::size_type> expected_offsets{0, 2, 2, 4, 4, 6};

  fixed_width_column_wrapper<value_type, int32_t> expected_first({3, 7, 0, 1, 5, 13});
  strings_column_wrapper expected_strings{"a", "column", "this", "is", "of", "strings"};
  auto expected_partitioned_table = cudf::table_view{{expected_first, expected_strings}};

  run_partition_test(table_to_partition, map, 5, expected_partitioned_table, expected_offsets);
}

template <typename T>
class PartitionTestFixedPoint : public cudf::test::BaseFixture {};

TYPED_TEST_SUITE(PartitionTestFixedPoint, cudf::test::FixedPointTypes);

TYPED_TEST(PartitionTestFixedPoint, Partition)
{
  using namespace numeric;
  using decimalXX  = TypeParam;
  using RepType    = cudf::device_storage_type_t<decimalXX>;
  using fp_wrapper = cudf::test::fixed_point_column_wrapper<RepType>;

  auto const input    = fp_wrapper({11, 22, 33, 44, 55, 66}, scale_type{-1});
  auto const map      = fixed_width_column_wrapper<int32_t>{0, 1, 2, 3, 4, 5};
  auto const expected = cudf::table_view{{input}};
  auto const offsets  = std::vector<cudf::size_type>{0, 1, 2, 3, 4, 5, 6};

  run_partition_test(expected, map, 6, expected, offsets);
}

TYPED_TEST(PartitionTestFixedPoint, Partition1)
{
  using namespace numeric;
  using decimalXX  = TypeParam;
  using RepType    = cudf::device_storage_type_t<decimalXX>;
  using fp_wrapper = cudf::test::fixed_point_column_wrapper<RepType>;

  auto const input    = fp_wrapper({11, 22, 33, 44, 55, 66}, scale_type{-1});
  auto const map      = fixed_width_column_wrapper<int32_t>{5, 4, 3, 2, 1, 0};
  auto const expected = fp_wrapper({66, 55, 44, 33, 22, 11}, scale_type{-1});
  auto const offsets  = std::vector<cudf::size_type>{0, 1, 2, 3, 4, 5, 6};

  run_partition_test(cudf::table_view{{input}}, map, 6, cudf::table_view{{expected}}, offsets);
}

TYPED_TEST(PartitionTestFixedPoint, Partition2)
{
  using namespace numeric;
  using decimalXX  = TypeParam;
  using RepType    = cudf::device_storage_type_t<decimalXX>;
  using fp_wrapper = cudf::test::fixed_point_column_wrapper<RepType>;

  auto const input    = fp_wrapper({11, 22, 33, 44, 55, 66}, scale_type{-1});
  auto const map      = fixed_width_column_wrapper<int32_t>{2, 1, 0, 2, 1, 0};
  auto const expected = fp_wrapper({33, 66, 22, 55, 11, 44}, scale_type{-1});
  auto const offsets  = std::vector<cudf::size_type>{0, 2, 4, 6};

  run_partition_test(cudf::table_view{{input}}, map, 3, cudf::table_view{{expected}}, offsets);
}

struct PartitionTestNotTyped : public cudf::test::BaseFixture {};

TEST_F(PartitionTestNotTyped, ListOfStringsEmpty)
{
  cudf::test::lists_column_wrapper<cudf::string_view> list{{}, {}};
  auto table_to_partition = cudf::table_view{{list}};
  fixed_width_column_wrapper<int32_t> map{0, 0};

  auto result = cudf::partition(table_to_partition, map, 2);
  CUDF_TEST_EXPECT_TABLES_EQUAL(table_to_partition, result.first->view());
  EXPECT_EQ(3, result.second.size());
}

TEST_F(PartitionTestNotTyped, ListOfListOfIntEmpty)
{
  cudf::test::lists_column_wrapper<int32_t> level_2_list;

  fixed_width_column_wrapper<int32_t> level_1_offsets{0, 0, 0};
  std::unique_ptr<cudf::column> level_1_list =
    cudf::make_lists_column(2,
                            level_1_offsets.release(),
                            level_2_list.release(),
                            0,
                            cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED));

  auto table_to_partition = cudf::table_view{{*level_1_list}};
  fixed_width_column_wrapper<int32_t> map{0, 0};

  auto result = cudf::partition(table_to_partition, map, 2);
  CUDF_TEST_EXPECT_TABLES_EQUAL(table_to_partition, result.first->view());
  EXPECT_EQ(3, result.second.size());
}

TEST_F(PartitionTestNotTyped, ListOfListOfListOfIntEmpty)
{
  cudf::test::lists_column_wrapper<int32_t> level_3_list{};

  fixed_width_column_wrapper<int32_t> level_2_offsets{};
  std::unique_ptr<cudf::column> level_2_list =
    cudf::make_lists_column(0,
                            level_2_offsets.release(),
                            level_3_list.release(),
                            0,
                            cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED));

  fixed_width_column_wrapper<int32_t> level_1_offsets{0, 0};
  std::unique_ptr<cudf::column> level_1_list =
    cudf::make_lists_column(1,
                            level_1_offsets.release(),
                            std::move(level_2_list),
                            0,
                            cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED));

  auto table_to_partition = cudf::table_view{{*level_1_list}};
  fixed_width_column_wrapper<int32_t> map{0};

  auto result = cudf::partition(table_to_partition, map, 2);
  CUDF_TEST_EXPECT_TABLES_EQUAL(table_to_partition, result.first->view());
  EXPECT_EQ(3, result.second.size());
}

TEST_F(PartitionTestNotTyped, HostMetadataPolicy)
{
  namespace policy = cudf::detail::partition_policy;
  int64_t value{};
  cudf::bitmask_type mask{};
  auto const view = [&](cudf::size_type nulls) {
    return cudf::column_view{
      cudf::data_type{cudf::type_id::INT64}, 1'000'000, &value, nulls ? &mask : nullptr, nulls};
  };
  auto const valid  = view(0);
  auto const sparse = view(100'000);
  auto const dense  = view(900'000);
  auto result       = policy::select_partition_plan(cudf::table_view{{valid, valid}}, 2);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->method, policy::algorithm::recompute);
  EXPECT_TRUE(result->direct);
  result = policy::select_partition_plan(cudf::table_view{{sparse, sparse}}, 256);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->method, policy::algorithm::saved);
  result = policy::select_partition_plan(cudf::table_view{{dense, dense, dense}}, 2);
  ASSERT_TRUE(result.has_value());
  EXPECT_TRUE(result->reordered);
  EXPECT_FALSE(policy::select_partition_plan(cudf::table_view{{valid}}, 257).has_value());
  EXPECT_FALSE(policy::select_partition_plan(cudf::table_view{{valid}}, 1).has_value());
}

TEST_F(PartitionTestNotTyped, TrivialPartitionSlices)
{
  for (auto const rows : {1, 1'025}) {
    for (auto const partitions : {1, 17}) {
      if (rows > 1 && partitions > 1) continue;
      std::vector<int32_t> ids(rows + 7);
      std::iota(ids.begin(), ids.end(), 0);
      std::vector<std::string> text(rows + 7, std::string("a\0b", 3));
      fixed_width_column_wrapper<int32_t> numbers(ids.begin(), ids.end());
      strings_column_wrapper strings(text.begin(), text.end());
      auto input = cudf::slice(cudf::table_view{{numbers, strings}}, {7, rows + 7})[0];
      std::vector<int32_t> assignments(rows + 3, partitions / 2);
      fixed_width_column_wrapper<int32_t> owner(assignments.begin(), assignments.end());
      auto map    = cudf::slice(owner, {3, rows + 3})[0];
      auto result = cudf::partition(input, map, partitions);
      std::vector<cudf::size_type> offsets(partitions + 1, 0);
      for (int p = partitions / 2 + 1; p <= partitions; ++p)
        offsets[p] = rows;
      EXPECT_EQ(result.second, offsets);
      CUDF_TEST_EXPECT_TABLES_EQUIVALENT(input, result.first->view());
    }
  }
}

TEST_F(PartitionTestNotTyped, UniformPartitionColumns)
{
  constexpr int rows = 1'025;
  std::vector<int32_t> values(rows, 3), assignments(rows);
  std::vector<bool> invalid(rows, false);
  std::vector<std::string> text(rows, "payload"), empty(rows, "");
  fixed_width_column_wrapper<int32_t> null_numbers(values.begin(), values.end(), invalid.begin());
  strings_column_wrapper null_strings(text.begin(), text.end(), invalid.begin());
  strings_column_wrapper empty_strings(empty.begin(), empty.end());
  std::vector<std::string> slice_values(rows + 2, "");
  slice_values.front() = "prefix";
  slice_values.back()  = "suffix";
  strings_column_wrapper slice_owner(slice_values.begin(), slice_values.end());
  auto const empty_slice = cudf::slice(slice_owner, {1, rows + 1})[0];
  for (auto const partitions : {1, 17}) {
    std::vector<cudf::size_type> expected(partitions + 1, 0);
    for (int i = 0; i < rows; ++i) {
      assignments[i] = (13 * i) % partitions;
      ++expected[assignments[i] + 1];
    }
    std::partial_sum(expected.begin(), expected.end(), expected.begin());
    fixed_width_column_wrapper<int32_t> map(assignments.begin(), assignments.end());
    for (auto const& input : {cudf::table_view{{null_numbers, null_strings}},
                              cudf::table_view{{null_numbers, null_strings, empty_strings}},
                              cudf::table_view{{empty_strings}},
                              cudf::table_view{{empty_strings, empty_strings, empty_strings}},
                              cudf::table_view{{empty_slice}}}) {
      auto result = cudf::partition(input, map, partitions);
      EXPECT_EQ(result.second, expected);
      CUDF_TEST_EXPECT_TABLES_EQUIVALENT(input, result.first->view());
    }
  }
}

TEST_F(PartitionTestNotTyped, NoIntegerOverflow)
{
  auto elements = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i % 2; });
  fixed_width_column_wrapper<int8_t> map(elements, elements + 129);
  auto table_to_partition = cudf::table_view{{map}};

  std::vector<cudf::size_type> expected_offsets{0, 65, 129};

  auto expected_elements =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i / 65; });
  fixed_width_column_wrapper<int8_t> expected(expected_elements, expected_elements + 129);
  auto expected_table = cudf::table_view{{expected}};

  run_partition_test(table_to_partition, map, 2, expected_table, expected_offsets);
}

TEST_F(PartitionTestNotTyped, BatchedStringGatherPermutation)
{
  for (auto const rows : {1'023, 1'024, 1'025}) {
    for (auto const partitions : {2, 256, 257}) {
      for (auto const nullable : {false, true}) {
        SCOPED_TRACE(::testing::Message() << rows << "/" << partitions << "/" << nullable);
        std::vector<std::string> ids(rows + 7), values(rows + 7), expected_ids, expected_values;
        std::vector<int32_t> bins(rows + 7);
        std::vector<bool> valid(rows + 7), expected_valid;
        std::vector<cudf::size_type> offsets{0};
        for (int i = 0; i < rows + 7; ++i) {
          ids[i]    = std::to_string(i);
          values[i] = std::string(i % 259, 'a' + i % 26) + std::string{"\0x", 2};
          bins[i]   = i % 5 ? 0 : (i * 17) % partitions;
          valid[i]  = !nullable || i % 11 != 0;
        }
        for (int p = 0; p < partitions; ++p) {
          for (int i = 7; i < rows + 7; ++i) {
            if (bins[i] != p) continue;
            expected_ids.push_back(ids[i]);
            expected_values.push_back(values[i]);
            expected_valid.push_back(valid[i]);
          }
          offsets.push_back(expected_ids.size());
        }
        strings_column_wrapper id(ids.begin(), ids.end());
        std::vector<std::string> empty(rows);
        strings_column_wrapper empty_column(empty.begin(), empty.end());
        strings_column_wrapper value(values.begin(), values.end(), valid.begin());
        fixed_width_column_wrapper<int32_t> map(bins.begin(), bins.end());
        strings_column_wrapper expected_id(expected_ids.begin(), expected_ids.end());
        strings_column_wrapper expected_value(
          expected_values.begin(), expected_values.end(), expected_valid.begin());
        auto const sliced = cudf::slice(cudf::table_view{{id, value}}, {7, rows + 7})[0];
        auto input        = cudf::table_view{{sliced.column(0), sliced.column(1), empty_column}};
        auto sliced_map   = cudf::slice(cudf::column_view{map}, {7, rows + 7})[0];
        auto result       = cudf::partition(input, sliced_map, partitions);
        EXPECT_EQ(result.second, offsets);
        auto expected = cudf::table_view{{expected_id, expected_value, empty_column}};
        for (int p = 0; p < partitions; ++p) {
          auto actual_partition =
            cudf::slice(result.first->view(), {offsets[p], offsets[p + 1]})[0];
          auto expected_partition = cudf::slice(expected, {offsets[p], offsets[p + 1]})[0];
          auto actual_sorted      = cudf::sort(actual_partition);
          auto expected_sorted    = cudf::sort(expected_partition);
          // An all-valid mask may be omitted by the materialization path.
          CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected_sorted->view(), actual_sorted->view());
        }
      }
    }
  }
}

TEST_F(PartitionTestNotTyped, BatchedStringGatherOffsetThreshold)
{
  struct restore_environment {
    char const* name;
    std::optional<std::string> previous;

    explicit restore_environment(char const* variable) : name(variable)
    {
      if (auto const value = std::getenv(name)) { previous = value; }
    }

    ~restore_environment()
    {
      if (previous) {
        setenv(name, previous->c_str(), 1);
      } else {
        unsetenv(name);
      }
    }
  };
  restore_environment threshold_restore{"LIBCUDF_LARGE_STRINGS_THRESHOLD"};
  restore_environment enabled_restore{"LIBCUDF_LARGE_STRINGS_ENABLED"};
  ASSERT_EQ(setenv("LIBCUDF_LARGE_STRINGS_THRESHOLD", "2048", 1), 0);
  ASSERT_EQ(setenv("LIBCUDF_LARGE_STRINGS_ENABLED", "1", 1), 0);

  constexpr cudf::size_type rows = 1'024;
  std::vector<std::string> values(rows, "a");
  strings_column_wrapper strings(values.begin(), values.end());
  auto const input = cudf::table_view{{strings, strings}};
  auto const ones  = cuda::make_constant_iterator<int32_t>(1);
  fixed_width_column_wrapper<int32_t> map(ones, ones + rows);
  ASSERT_EQ(cudf::strings_column_view(input.column(0)).offsets().type().id(), cudf::type_id::INT32);

  // Change the threshold after construction to keep the input offsets INT32.
  for (auto const threshold : {1'025, 1'024, 1'023}) {
    ASSERT_EQ(setenv("LIBCUDF_LARGE_STRINGS_THRESHOLD", std::to_string(threshold).c_str(), 1), 0);
    for (auto const enabled : {true, false}) {
      SCOPED_TRACE(::testing::Message() << threshold << "/" << enabled);
      ASSERT_EQ(setenv("LIBCUDF_LARGE_STRINGS_ENABLED", enabled ? "1" : "0", 1), 0);
      if (!enabled && rows >= threshold) {
        EXPECT_THROW(cudf::partition(input, map, 3), std::overflow_error);
      } else {
        auto [output, offsets] = cudf::partition(input, map, 3);
        EXPECT_EQ(offsets, (std::vector<cudf::size_type>{0, 0, rows, rows}));
        auto const expected_type = rows >= threshold ? cudf::type_id::INT64 : cudf::type_id::INT32;
        for (auto const& column : output->view()) {
          EXPECT_EQ(cudf::strings_column_view(column).offsets().type().id(), expected_type);
        }
        CUDF_TEST_EXPECT_TABLES_EQUIVALENT(input, output->view());
      }
    }
  }
}

template <typename T>
class StringPartitionMapTypes : public cudf::test::BaseFixture {};
TYPED_TEST_SUITE(StringPartitionMapTypes, cudf::test::IntegralTypesNotBool);
TYPED_TEST(StringPartitionMapTypes, PartialTiles)
{
  constexpr int rows = 2'049;
  std::vector<std::string> text(rows), expected;
  std::vector<TypeParam> bins(rows);
  for (int i = 0; i < rows; ++i)
    text[i] = std::to_string(i);
  strings_column_wrapper input(text.begin(), text.end());
  for (auto const parts : {2, 3, 127}) {
    expected.clear();
    std::vector<cudf::size_type> offsets{0};
    for (int i = 0; i < rows; ++i)
      bins[i] = i % parts;
    for (int p = 0; p < parts; ++p) {
      for (int i = p; i < rows; i += parts)
        expected.push_back(text[i]);
      offsets.push_back(expected.size());
    }
    fixed_width_column_wrapper<TypeParam> map(bins.begin(), bins.end());
    strings_column_wrapper output(expected.begin(), expected.end());
    run_partition_test(
      cudf::table_view{{input, input}}, map, parts, cudf::table_view{{output, output}}, offsets);
    if (parts == 2) {
      run_partition_test(cudf::table_view{{input, input, input, input}},
                         map,
                         parts,
                         cudf::table_view{{output, output, output, output}},
                         offsets);
    }
  }
}

TEST_F(PartitionTestNotTyped, LargeBinaryStringRanks)
{
  constexpr cudf::size_type rows = 4'000'001;
  constexpr cudf::size_type skip = 7;
  std::vector<std::string> const samples{"", "x", "é", std::string(31, 'm'), std::string(129, 'z')};
  auto const value = [&](cudf::size_type i) -> std::string const& {
    return samples[i % 4096 == 0 ? 4 : (i % 257 == 0 ? 3 : i % 3)];
  };
  auto const values = cudf::detail::make_counting_transform_iterator(0, value);
  auto const empty  = cuda::make_constant_iterator("");
  strings_column_wrapper strings(values, values + rows + skip);
  strings_column_wrapper empty_strings(empty, empty + rows + skip);
  std::vector<int32_t> bins(rows + skip);
  for (cudf::size_type i = 0; i < rows + skip; ++i) {
    bins[i] = (i * 13 + i / 7) % 2;
  }
  std::vector<cudf::size_type> order;
  order.reserve(rows);
  std::vector<cudf::size_type> offsets{0};
  for (int partition = 0; partition < 2; ++partition) {
    for (cudf::size_type i = skip; i < rows + skip; ++i) {
      if (bins[i] == partition) { order.push_back(i); }
    }
    offsets.push_back(order.size());
  }
  auto const expected_values = cudf::detail::make_counting_transform_iterator(
    0, [&](auto i) -> std::string const& { return value(order[i]); });
  strings_column_wrapper expected(expected_values, expected_values + rows);
  strings_column_wrapper expected_empty(empty, empty + rows);
  fixed_width_column_wrapper<int32_t> map(bins.begin(), bins.end());
  auto const input =
    cudf::slice(cudf::table_view{{strings, empty_strings}}, {skip, rows + skip})[0];
  auto const sliced_map = cudf::slice(cudf::column_view{map}, {skip, rows + skip})[0];
  run_partition_test(input, sliced_map, 2, cudf::table_view{{expected, expected_empty}}, offsets);
}

TEST_F(PartitionTestNotTyped, BinaryStringSmallSchemas)
{
  constexpr int rows = 1'025;
  std::vector<std::string> text(rows), expected;
  std::vector<int32_t> bins(rows);
  std::vector<cudf::size_type> offsets{0};
  for (int i = 0; i < rows; ++i) {
    text[i] = std::to_string(i) + std::string(i % 259, 'x');
    bins[i] = i % 7 == 0;
  }
  for (int part = 0; part < 2; ++part) {
    for (int i = 0; i < rows; ++i)
      if (bins[i] == part) expected.push_back(text[i]);
    offsets.push_back(expected.size());
  }
  strings_column_wrapper input(text.begin(), text.end());
  strings_column_wrapper output(expected.begin(), expected.end());
  fixed_width_column_wrapper<int32_t> map(bins.begin(), bins.end());
  for (int count : {1, 2, 3}) {
    run_partition_test(cudf::table_view{std::vector<cudf::column_view>(count, input)},
                       map,
                       2,
                       cudf::table_view{std::vector<cudf::column_view>(count, output)},
                       offsets);
  }
}

TEST_F(PartitionTestNotTyped, DirectFixedWidthSlices)
{
  constexpr int rows = 1'025;
  std::vector<int16_t> values(rows + 7), expected;
  std::vector<int32_t> bins(rows);
  std::vector<bool> valid(rows + 7), expected_valid;
  for (int i = 0; i < rows + 7; ++i) {
    values[i] = i;
    valid[i]  = i % 11 != 0;
  }
  std::vector<cudf::size_type> offsets{0};
  for (int i = 0; i < rows; ++i)
    bins[i] = i % 3 == 0;
  for (int part = 0; part < 2; ++part) {
    for (int i = 0; i < rows; ++i)
      if (bins[i] == part) {
        expected.push_back(values[i + 7]);
        expected_valid.push_back(valid[i + 7]);
      }
    offsets.push_back(expected.size());
  }
  fixed_width_column_wrapper<int16_t> owner(values.begin(), values.end(), valid.begin());
  auto input = cudf::slice(cudf::column_view{owner}, {7, rows + 7})[0];
  fixed_width_column_wrapper<int16_t> output(
    expected.begin(), expected.end(), expected_valid.begin());
  fixed_width_column_wrapper<int32_t> map(bins.begin(), bins.end());
  run_partition_test(
    cudf::table_view{{input, input}}, map, 2, cudf::table_view{{output, output}}, offsets);
}

TEST_F(PartitionTestNotTyped, LargeFixedWidthRoutes)
{
  constexpr cudf::size_type rows = 1'000'001;
  constexpr cudf::size_type skip = 7;
  std::vector<int64_t> values(rows + skip);
  std::iota(values.begin(), values.end(), int64_t{1} << 40);
  struct configuration {
    int columns;
    int partitions;
    bool nullable;
    bool dense;
  };
  for (auto const [columns, partitions, nullable, dense] : {configuration{1, 3, false, false},
                                                            configuration{1, 16, false, false},
                                                            configuration{2, 17, true, false},
                                                            configuration{3, 3, false, false},
                                                            configuration{3, 2, true, true}}) {
    SCOPED_TRACE(::testing::Message()
                 << columns << "/" << partitions << "/" << nullable << "/" << dense);
    std::vector<int32_t> bins(rows + skip);
    std::vector<bool> valid(rows + skip);
    for (cudf::size_type i = 0; i < rows + skip; ++i) {
      bins[i]  = i % 5 == 0 ? (i * 13 + i / 7) % partitions : 0;
      valid[i] = !nullable || (dense ? i % 11 == 0 : i % 11 != 0);
    }
    std::vector<int64_t> expected;
    std::vector<bool> expected_valid;
    expected.reserve(rows);
    expected_valid.reserve(rows);
    std::vector<cudf::size_type> offsets{0};
    for (int partition = 0; partition < partitions; ++partition) {
      for (cudf::size_type i = skip; i < rows + skip; ++i) {
        if (bins[i] == partition) {
          expected.push_back(values[i]);
          expected_valid.push_back(valid[i]);
        }
      }
      offsets.push_back(expected.size());
    }
    auto const make_column = [&](auto const& data, auto const& validity) {
      if (nullable) {
        return fixed_width_column_wrapper<int64_t>(data.begin(), data.end(), validity.begin())
          .release();
      }
      return fixed_width_column_wrapper<int64_t>(data.begin(), data.end()).release();
    };
    auto const owner  = make_column(values, valid);
    auto const output = make_column(expected, expected_valid);
    auto const input  = cudf::slice(owner->view(), {skip, rows + skip})[0];
    fixed_width_column_wrapper<int32_t> map(bins.begin(), bins.end());
    auto const sliced_map = cudf::slice(cudf::column_view{map}, {skip, rows + skip})[0];
    run_partition_test(cudf::table_view{std::vector<cudf::column_view>(columns, input)},
                       sliced_map,
                       partitions,
                       cudf::table_view{std::vector<cudf::column_view>(columns, output->view())},
                       offsets);
  }
}
