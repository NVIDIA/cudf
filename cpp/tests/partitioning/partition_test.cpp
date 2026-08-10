/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/copying.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/partitioning.hpp>
#include <cudf/sorting.hpp>
#include <cudf/table/table.hpp>

#include <cuda/iterator>
#include <cuda/std/tuple>

#include <algorithm>

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
    cudf::make_lists_column(2, level_1_offsets.release(), level_2_list.release(), 0, {});

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
    cudf::make_lists_column(0, level_2_offsets.release(), level_3_list.release(), 0, {});

  fixed_width_column_wrapper<int32_t> level_1_offsets{0, 0};
  std::unique_ptr<cudf::column> level_1_list =
    cudf::make_lists_column(1, level_1_offsets.release(), std::move(level_2_list), 0, {});

  auto table_to_partition = cudf::table_view{{*level_1_list}};
  fixed_width_column_wrapper<int32_t> map{0};

  auto result = cudf::partition(table_to_partition, map, 2);
  CUDF_TEST_EXPECT_TABLES_EQUAL(table_to_partition, result.first->view());
  EXPECT_EQ(3, result.second.size());
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
