/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/memory_resource_utilities.hpp>
#include <cudf_test/table_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/copying.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/partitioning.hpp>
#include <cudf/sorting.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/cuda_device.hpp>
#include <rmm/mr/statistics_resource_adaptor.hpp>

#include <cuda/iterator>
#include <cuda/std/tuple>
#include <cuda/stream>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>

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
TYPED_TEST(PartitionTest, SingleFixedWidthTileBoundaries)
{
  using value_type = cudf::test::GetType<TypeParam, 0>;
  using map_type   = cudf::test::GetType<TypeParam, 1>;

  for (auto const rows : {1024, 1025}) {
    for (auto const hot : {false, true}) {
      SCOPED_TRACE(::testing::Message() << rows << "/" << hot);
      auto const values = cudf::detail::make_counting_transform_iterator(
        0, [](auto i) { return (i * 37 + 11) % 97; });
      fixed_width_column_wrapper<value_type, int32_t> data(values, values + rows + 7);
      auto const input = cudf::slice(cudf::table_view{{data}}, {7, rows + 7})[0];
      std::vector<map_type> map(rows + 3, 0);
      std::vector<int32_t> expected_values;
      std::vector<cudf::size_type> offsets{0};
      for (cudf::size_type p = 0; p < 2; ++p) {
        for (cudf::size_type i = 0; i < rows; ++i) {
          map[i + 3] = hot ? 1 : i % 2;
          if (map[i + 3] == static_cast<map_type>(p)) { expected_values.push_back(values[i + 7]); }
        }
        offsets.push_back(expected_values.size());
      }
      fixed_width_column_wrapper<map_type> map_column(map.begin(), map.end());
      auto const map_view = cudf::slice(map_column, {3, rows + 3})[0];
      fixed_width_column_wrapper<value_type, int32_t> expected(expected_values.begin(),
                                                               expected_values.end());
      run_partition_test(input, map_view, 2, cudf::table_view{{expected}}, offsets);
    }
  }
}

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

template <typename MapType>
struct PartitionMapRanksTest : public cudf::test::BaseFixture {};

TYPED_TEST_SUITE(PartitionMapRanksTest, cudf::test::IntegralTypesNotBool);

TYPED_TEST(PartitionMapRanksTest, SingleRowWithSlicedMap)
{
  auto const ids = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i; });
  auto stream    = cuda::stream{cuda::device_ref{rmm::get_current_cuda_device().value()}};
  std::vector<std::string> values(15, "value");
  values[7] = std::string{"x\0y", 3};
  for (auto const null_mode : {0, 1, 2}) {
    auto const numeric_valid = cuda::make_constant_iterator(null_mode == 0);
    auto const string_valid  = cuda::make_constant_iterator(null_mode != 2);
    fixed_width_column_wrapper<int64_t> numeric(ids, ids + 15, numeric_valid);
    strings_column_wrapper strings(values.begin(), values.end(), string_valid);
    auto const input = cudf::slice(cudf::table_view{{numeric, strings}}, {7, 8})[0];
    for (auto const partitions : {2, 100}) {
      SCOPED_TRACE(::testing::Message() << null_mode << "/" << partitions);
      auto const last = cuda::make_constant_iterator<TypeParam>(partitions - 1);
      fixed_width_column_wrapper<TypeParam> map(last, last + 7);
      auto const map_view = cudf::slice(cudf::column_view{map}, {3, 4})[0];
      cudf::get_default_stream().sync();
      auto [output, offsets] = cudf::partition(input, map_view, partitions, stream, this->mr());
      std::vector<cudf::size_type> expected(partitions + 1, 0);
      expected.back() = 1;
      EXPECT_EQ(offsets, expected);
      CUDF_TEST_EXPECT_TABLES_EQUIVALENT(input, output->view(), stream);
    }
  }
}

TEST_F(PartitionTestNotTyped, PruneNullColumnsPreservesSchemaAndRows)
{
  constexpr cudf::size_type rows = 4097;
  auto const ids                 = cuda::counting_iterator<int32_t>{0};
  auto const invalid             = cuda::make_constant_iterator(false);
  auto const valid =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i % 3 != 0; });
  fixed_width_column_wrapper<int32_t> row_ids(ids, ids + rows + 14);
  fixed_width_column_wrapper<int64_t> null_numbers(ids, ids + rows + 14, invalid);
  fixed_width_column_wrapper<int32_t> partial(ids, ids + rows + 14, valid);
  std::vector<std::string> values(rows + 14, "discarded");
  strings_column_wrapper null_strings(values.begin(), values.end(), invalid);
  auto const input =
    cudf::slice(cudf::table_view{{null_numbers, row_ids, null_strings, partial}}, {7, rows + 7})[0];
  auto stream = cuda::stream{cuda::device_ref{rmm::get_current_cuda_device().value()}};
  for (auto const partitions : {2, 100, 257}) {
    SCOPED_TRACE(partitions);
    std::vector<int32_t> assignments(rows);
    std::vector<cudf::size_type> expected(partitions + 1, 0);
    for (auto i = 0; i < rows; ++i) {
      assignments[i] = (i * 17 + 3) % partitions;
      ++expected[assignments[i] + 1];
    }
    std::partial_sum(expected.begin(), expected.end(), expected.begin());
    fixed_width_column_wrapper<int32_t> map(assignments.begin(), assignments.end());
    cudf::get_default_stream().sync();
    auto [output, offsets] = cudf::partition(input, map, partitions, stream, mr());
    EXPECT_EQ(offsets, expected);
    auto const [actual_ids, mask] = cudf::test::to_host<int32_t>(output->get_column(1), stream);
    for (auto p = 0; p < partitions; ++p) {
      for (auto i = offsets[p]; i < offsets[p + 1]; ++i) {
        ASSERT_GE(actual_ids[i], 7);
        ASSERT_LT(actual_ids[i], rows + 7);
        EXPECT_EQ(assignments[actual_ids[i] - 7], p);
      }
    }
    auto const order     = std::vector<cudf::size_type>{1, 0, 2, 3};
    auto expected_sorted = cudf::sort(input.select(order), {}, {}, stream, mr());
    auto actual_sorted   = cudf::sort(output->view().select(order), {}, {}, stream, mr());
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected_sorted->view(), actual_sorted->view(), stream);
  }
}

TEST_F(PartitionTestNotTyped, UniformEmptyStringsAndMixedColumns)
{
  constexpr cudf::size_type rows = 4097;
  auto const ids                 = cuda::counting_iterator<int32_t>{0};
  auto const valid               = cuda::make_constant_iterator(true);
  auto const invalid             = cuda::make_constant_iterator(false);
  auto const partial =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i % 3 != 0; });
  std::vector<std::string> values(rows + 14, "");
  fixed_width_column_wrapper<int32_t> row_ids(ids, ids + rows + 14);
  fixed_width_column_wrapper<int64_t> null_numbers(ids, ids + rows + 14, invalid);
  strings_column_wrapper partial_empty(values.begin(), values.end(), partial);
  auto stream = cuda::stream{cuda::device_ref{rmm::get_current_cuda_device().value()}};
  for (auto wide : {false, true}) {
    auto empty = strings_column_wrapper(values.begin(), values.end(), valid).release();
    if (wide) {
      auto contents = empty->release();
      auto offsets =
        cudf::cast(contents.children[0]->view(), cudf::data_type{cudf::type_id::INT64});
      empty = cudf::make_strings_column(rows + 14,
                                        std::move(offsets),
                                        std::move(*contents.data),
                                        0,
                                        std::move(*contents.null_mask));
    }
    ASSERT_EQ(empty->view().head<char>(), nullptr);
    for (auto const& unsliced :
         {cudf::table_view{{*empty, null_numbers}},
          cudf::table_view{{*empty, partial_empty, row_ids, null_numbers}}}) {
      auto input = cudf::slice(unsliced, {7, rows + 7})[0];
      for (auto partitions : {1, 2, 100, 257}) {
        SCOPED_TRACE(::testing::Message()
                     << wide << "/" << input.num_columns() << "/" << partitions);
        std::vector<int32_t> assignments(rows);
        std::vector<cudf::size_type> expected(partitions + 1, 0);
        for (auto i = 0; i < rows; ++i) {
          assignments[i] = (i * 17 + 3) % partitions;
          ++expected[assignments[i] + 1];
        }
        std::partial_sum(expected.begin(), expected.end(), expected.begin());
        fixed_width_column_wrapper<int32_t> map(assignments.begin(), assignments.end());
        cudf::get_default_stream().sync();
        auto [output, offsets] = cudf::partition(input, map, partitions, stream, mr());
        EXPECT_EQ(offsets, expected);
        EXPECT_EQ(output->get_column(0).null_count(), 0);
        auto expected_sorted = cudf::sort(input, {}, {}, stream, mr());
        auto actual_sorted   = cudf::sort(output->view(), {}, {}, stream, mr());
        CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected_sorted->view(), actual_sorted->view(), stream);
        if (input.num_columns() == 4) {
          auto const [actual_ids, mask] =
            cudf::test::to_host<int32_t>(output->get_column(2), stream);
          for (auto p = 0; p < partitions; ++p)
            for (auto i = offsets[p]; i < offsets[p + 1]; ++i) {
              ASSERT_GE(actual_ids[i], 7);
              ASSERT_LT(actual_ids[i], rows + 7);
              EXPECT_EQ(assignments[actual_ids[i] - 7], p);
            }
        }
      }
    }
  }
}

TEST_F(PartitionTestNotTyped, AllNullColumnsWithoutRowMap)
{
  constexpr cudf::size_type rows = 4097;
  auto const ids     = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i; });
  auto const invalid = cuda::make_constant_iterator(false);
  fixed_width_column_wrapper<int8_t> narrow(ids, ids + rows + 14, invalid);
  fixed_width_column_wrapper<int64_t> wide(ids, ids + rows + 14, invalid);
  std::vector<std::string> values(rows + 14, "unused");
  strings_column_wrapper strings(values.begin(), values.end(), invalid);
  auto const input = cudf::slice(cudf::table_view{{narrow, wide, strings}}, {7, rows + 7})[0];
  auto stream      = cuda::stream{cuda::device_ref{rmm::get_current_cuda_device().value()}};
  for (auto const partitions : {1, 2, 100, 256, 257}) {
    SCOPED_TRACE(partitions);
    std::vector<int32_t> partition_ids(rows + 3, 0);
    std::vector<cudf::size_type> expected(partitions + 1, 0);
    for (cudf::size_type i = 0; i < rows; ++i) {
      auto const p         = (i * 17 + 3) % partitions;
      partition_ids[i + 3] = p;
      ++expected[p + 1];
    }
    std::partial_sum(expected.begin(), expected.end(), expected.begin());
    fixed_width_column_wrapper<int32_t> map(partition_ids.begin(), partition_ids.end());
    auto const map_view = cudf::slice(cudf::column_view{map}, {3, rows + 3})[0];
    cudf::get_default_stream().sync();
    auto [output, offsets] = cudf::partition(input, map_view, partitions, stream, mr());
    EXPECT_EQ(offsets, expected);
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(input, output->view(), stream);
  }
}

TEST_F(PartitionTestNotTyped, SinglePartitionNestedColumn)
{
  cudf::test::lists_column_wrapper<int64_t> lists{{1, 2}, {}, {3}};
  fixed_width_column_wrapper<int32_t> map{0, 0, 0};
  auto const input       = cudf::table_view{{lists}};
  auto [output, offsets] = cudf::partition(input, map, 1);
  EXPECT_EQ(offsets, (std::vector<cudf::size_type>{0, 3}));
  CUDF_TEST_EXPECT_TABLES_EQUAL(input, output->view());
}

TEST_F(PartitionTestNotTyped, ZeroColumnsWithRows)
{
  for (auto const rows : {1, 1025}) {
    auto const input = cudf::table_view{std::vector<cudf::column_view>{}, rows};
    for (auto const partitions : {1, 2, 257}) {
      SCOPED_TRACE(::testing::Message() << rows << "/" << partitions);
      auto const ids = cudf::detail::make_counting_transform_iterator(
        0, [partitions](auto i) { return (i * 17 + 3) % partitions; });
      fixed_width_column_wrapper<int32_t> map(ids, ids + rows);
      std::vector<cudf::size_type> expected(partitions + 1, 0);
      for (auto i = 0; i < rows; ++i) {
        ++expected[ids[i] + 1];
      }
      std::partial_sum(expected.begin(), expected.end(), expected.begin());
      auto [output, offsets] = cudf::partition(input, map, partitions);
      EXPECT_EQ(offsets, expected);
      EXPECT_EQ(output->num_rows(), rows);
      EXPECT_EQ(output->num_columns(), 0);
    }
  }
}

TEST_F(PartitionTestNotTyped, TrivialStringOutputOffsetThreshold)
{
  struct restore_setting {
    char const* name;
    std::optional<std::string> previous;
    explicit restore_setting(char const* key) : name(key)
    {
      if (auto const value = std::getenv(key)) { previous = value; }
    }
    ~restore_setting()
    {
      if (previous) {
        setenv(name, previous->c_str(), 1);
      } else {
        unsetenv(name);
      }
    }
  };
  restore_setting restore_threshold{"LIBCUDF_LARGE_STRINGS_THRESHOLD"};
  restore_setting restore_enabled{"LIBCUDF_LARGE_STRINGS_ENABLED"};
  auto stream = cuda::stream{cuda::device_ref{rmm::get_current_cuda_device().value()}};
  for (auto const rows : {1, 1025}) {
    for (auto const wide : {false, true}) {
      for (auto const nullable : {false, true}) {
        ASSERT_EQ(setenv("LIBCUDF_LARGE_STRINGS_THRESHOLD", wide ? "1" : "65536", 1), 0);
        ASSERT_EQ(setenv("LIBCUDF_LARGE_STRINGS_ENABLED", "1", 1), 0);
        auto const length = rows == 1 ? 4 : 1;
        std::vector<std::string> values(rows + 14, std::string(length, 'a'));
        auto const valid = cudf::detail::make_counting_transform_iterator(
          0, [nullable](auto i) { return !nullable || i % 10 != 0; });
        auto strings = strings_column_wrapper(values.begin(), values.end(), valid).release();
        if (wide) {
          auto const nulls = strings->null_count();
          auto contents    = strings->release();
          auto wide_offsets =
            cudf::cast(contents.children[0]->view(), cudf::data_type{cudf::type_id::INT64});
          strings = cudf::make_strings_column(rows + 14,
                                              std::move(wide_offsets),
                                              std::move(*contents.data),
                                              nulls,
                                              std::move(*contents.null_mask));
        }
        auto const input = cudf::slice(cudf::table_view{{*strings, *strings}}, {7, rows + 7})[0];
        auto const expected_input_type = wide ? cudf::type_id::INT64 : cudf::type_id::INT32;
        ASSERT_EQ(cudf::strings_column_view(input.column(0)).offsets().type().id(),
                  expected_input_type);
        auto const partitions = rows == 1 ? 2 : 1;
        auto const last       = cuda::make_constant_iterator<int32_t>(partitions - 1);
        fixed_width_column_wrapper<int32_t> map(last, last + rows);
        auto const bytes = (rows - input.column(0).null_count()) * length;
        cudf::get_default_stream().sync();
        for (auto const threshold : {bytes - 1, bytes, bytes + 1}) {
          ASSERT_EQ(setenv("LIBCUDF_LARGE_STRINGS_THRESHOLD", std::to_string(threshold).c_str(), 1),
                    0);
          for (auto const enabled : {false, true}) {
            SCOPED_TRACE(::testing::Message() << rows << "/" << wide << "/" << nullable << "/"
                                              << threshold << "/" << enabled);
            ASSERT_EQ(setenv("LIBCUDF_LARGE_STRINGS_ENABLED", enabled ? "1" : "0", 1), 0);
            if (!enabled && bytes >= threshold) {
              EXPECT_THROW(cudf::partition(input, map, partitions, stream, mr()),
                           std::overflow_error);
            } else {
              auto [output, offsets]      = cudf::partition(input, map, partitions, stream, mr());
              auto const expected_offsets = partitions == 1 ? std::vector<cudf::size_type>{0, rows}
                                                            : std::vector<cudf::size_type>{0, 0, 1};
              EXPECT_EQ(offsets, expected_offsets);
              auto const type = bytes >= threshold ? cudf::type_id::INT64 : cudf::type_id::INT32;
              for (auto const& col : output->view()) {
                EXPECT_EQ(cudf::strings_column_view(col).offsets().type().id(), type);
              }
              CUDF_TEST_EXPECT_TABLES_EQUIVALENT(input, output->view(), stream);
            }
          }
        }
      }
    }
  }
}

TEST_F(PartitionTestNotTyped, SingleFixedWidthNullMask)
{
  constexpr cudf::size_type rows       = 1025;
  constexpr cudf::size_type partitions = 100;
  auto const ids = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i; });
  auto const last_partition = cuda::make_constant_iterator<int32_t>(partitions - 1);
  fixed_width_column_wrapper<int32_t> map(last_partition, last_partition + rows);
  std::vector<cudf::size_type> expected_offsets(partitions + 1, 0);
  expected_offsets.back() = rows;
  auto stream             = cuda::stream{cuda::device_ref{rmm::get_current_cuda_device().value()}};

  for (auto const null_mode : {0, 1, 2}) {
    SCOPED_TRACE(null_mode);
    // Cover an all-valid mask, nulls outside the slice, and one null inside the slice.
    auto const valid = cudf::detail::make_counting_transform_iterator(0, [null_mode](auto i) {
      return null_mode == 0 || (i >= 7 && i < rows + 7 && (null_mode != 2 || i != rows / 2 + 7));
    });
    fixed_width_column_wrapper<int64_t> data(ids, ids + rows + 14, valid);
    auto const input = cudf::slice(cudf::column_view{data}, {7, rows + 7})[0];
    ASSERT_TRUE(input.nullable());
    ASSERT_EQ(input.null_count(), null_mode == 2 ? 1 : 0);

    cudf::get_default_stream().sync();

    auto [output, offsets] =
      cudf::partition(cudf::table_view{{input}}, map, partitions, stream, mr());
    EXPECT_EQ(offsets, expected_offsets);
    EXPECT_EQ(output->get_column(0).nullable(), null_mode == 2);
    auto sorted_input  = cudf::sort(cudf::table_view{{input}}, {}, {}, stream, mr());
    auto sorted_output = cudf::sort(output->view(), {}, {}, stream, mr());
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(sorted_input->view(), sorted_output->view(), stream);
  }
}

TEST_F(PartitionTestNotTyped, SingleFixedWidthManyPartitions)
{
  constexpr cudf::size_type rows = 4099;
  auto const ids = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i; });
  fixed_width_column_wrapper<int64_t> data(ids, ids + rows);
  for (auto const partitions : {200, 256, 257}) {
    for (auto const hot : {false, true}) {
      SCOPED_TRACE(::testing::Message() << partitions << "/" << hot);
      std::vector<int32_t> map(rows);
      std::vector<int64_t> expected_values;
      std::vector<cudf::size_type> offsets{0};
      for (cudf::size_type p = 0; p < partitions; ++p) {
        for (cudf::size_type i = 0; i < rows; ++i) {
          map[i] = hot ? partitions - 1 : (i * 17 + 3) % partitions;
          if (map[i] == p) { expected_values.push_back(i); }
        }
        offsets.push_back(expected_values.size());
      }
      fixed_width_column_wrapper<int32_t> map_column(map.begin(), map.end());
      fixed_width_column_wrapper<int64_t> expected(expected_values.begin(), expected_values.end());
      run_partition_test(
        cudf::table_view{{data}}, map_column, partitions, cudf::table_view{{expected}}, offsets);
    }
  }
}

TYPED_TEST(PartitionMapRanksTest, TileBoundariesAndSkew)
{
  using size_type = cudf::size_type;
  auto stream     = cuda::stream{cuda::device_ref{rmm::get_current_cuda_device().value()}};
  for (auto const rows : {1023, 1024, 1025, 2047, 2048, 2049, 4095, 4096, 4097, 4099}) {
    for (auto const partitions : {1, 2, 100, 200, 256, 257}) {
      if (static_cast<std::uint64_t>(partitions - 1) >
          static_cast<std::uint64_t>(std::numeric_limits<TypeParam>::max())) {
        continue;
      }
      for (auto const distribution : {0, 1, 2}) {
        SCOPED_TRACE(::testing::Message() << rows << "/" << partitions << "/" << distribution);
        // Slice both the data and the map, with a different starting offset for each.
        auto const ids =
          cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i; });
        fixed_width_column_wrapper<size_type> row_ids(ids, ids + rows + 7);
        auto const valid =
          cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i % 7 != 0; });
        fixed_width_column_wrapper<size_type> nullable(ids, ids + rows + 7, valid);
        std::vector<std::string> strings(rows + 7);
        std::transform(ids, ids + rows + 7, strings.begin(), [](auto i) {
          return i % 11 == 0 ? std::string{} : std::to_string(i) + "-\xE4\xB8\xAD";
        });
        strings_column_wrapper text(strings.begin(), strings.end(), valid);
        auto input = cudf::slice(cudf::table_view{{row_ids, nullable, text}}, {7, rows + 7})[0];

        std::vector<TypeParam> map(rows + 3, 0);
        std::vector<size_type> expected_offsets(partitions + 1, 0);
        for (size_type i = 0; i < rows; ++i) {
          auto const partition = distribution == 1                  ? partitions - 1
                                 : distribution == 2 && i % 10 != 0 ? 0
                                                                    : (i * 17 + 3) % partitions;
          map[i + 3]           = static_cast<TypeParam>(partition);
          ++expected_offsets[partition + 1];
        }
        std::partial_sum(
          expected_offsets.begin(), expected_offsets.end(), expected_offsets.begin());
        fixed_width_column_wrapper<TypeParam> map_column(map.begin(), map.end());
        auto map_view = cudf::slice(map_column, {3, rows + 3})[0];
        cudf::get_default_stream().sync();

        auto [output, offsets] = cudf::partition(input, map_view, partitions, stream, this->mr());
        EXPECT_EQ(offsets, expected_offsets);
        auto const [actual_ids, mask] =
          cudf::test::to_host<size_type>(output->get_column(0), stream);
        for (size_type p = 0; p < partitions; ++p) {
          for (auto i = offsets[p]; i < offsets[p + 1]; ++i) {
            ASSERT_GE(actual_ids[i], 7);
            ASSERT_LT(actual_ids[i], rows + 7);
            EXPECT_EQ(map[actual_ids[i] - 7 + 3], p);
          }
        }
        // Unique row ids catch dropped/duplicated rows without requiring a stable partition.
        auto sorted = cudf::sort(output->view(), {}, {}, stream, this->mr());
        CUDF_TEST_EXPECT_TABLES_EQUAL(input, sorted->view(), stream);
      }
    }
  }
}

TEST_F(PartitionTestNotTyped, PartitionMemoryResources)
{
  std::vector<std::string> values(1025);
  std::vector<int32_t> partitions(1025, 0);
  for (int32_t i = 0; i < 1025; ++i) {
    values[i] = "value-" + std::to_string(i);
  }
  strings_column_wrapper strings(values.begin(), values.end());
  fixed_width_column_wrapper<int32_t> map(partitions.begin(), partitions.end());
  auto const input  = cudf::table_view{{strings, strings}};
  auto stream       = cuda::stream{cuda::device_ref{rmm::get_current_cuda_device().value()}};
  auto output_mr    = rmm::mr::statistics_resource_adaptor{mr()};
  auto temporary_mr = rmm::mr::statistics_resource_adaptor{mr()};
  cudf::get_default_stream().sync();
  std::unique_ptr<cudf::table> output;
  {
    auto current = cudf::test::scoped_current_device_resource{temporary_mr};
    auto result  = cudf::partition(input, map, 100, stream, output_mr);
    output       = std::move(result.first);
    stream.sync();
  }
  EXPECT_GT(output_mr.get_bytes_counter().value, 0);
  EXPECT_GT(temporary_mr.get_bytes_counter().total, 0);
  EXPECT_EQ(temporary_mr.get_bytes_counter().value, 0);
  auto expected = cudf::sort(input);
  auto actual   = cudf::sort(output->view());
  CUDF_TEST_EXPECT_TABLES_EQUAL(expected->view(), actual->view());
  output.reset();
  stream.sync();
  EXPECT_EQ(output_mr.get_bytes_counter().value, 0);
}

TEST_F(PartitionTestNotTyped, PartitionFixedWidthMemoryResources)
{
  constexpr cudf::size_type rows = 1025;
  auto const ids = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i; });
  fixed_width_column_wrapper<int32_t> data(ids, ids + rows);
  auto const ones = cuda::make_constant_iterator<int32_t>(1);
  fixed_width_column_wrapper<int32_t> map(ones, ones + rows);
  auto stream = cuda::stream{cuda::device_ref{rmm::get_current_cuda_device().value()}};
  cudf::get_default_stream().sync();

  for (auto const& input : {cudf::table_view{{data}}, cudf::table_view{{data, data}}}) {
    SCOPED_TRACE(input.num_columns());
    cudf::test::memory_resource_test_harness resources{mr()};
    std::unique_ptr<cudf::table> output;
    {
      auto current = cudf::test::scoped_current_device_resource{resources.temporary_mr()};
      auto result  = cudf::partition(input, map, 2, stream, resources.output_mr());
      output       = std::move(result.first);
      EXPECT_EQ(result.second, (std::vector<cudf::size_type>{0, 0, rows}));
      stream.sync();
    }
    resources.expect_resource_usage(output->alloc_size(),
                                    {cudf::test::output_allocation_expectation::EXACT,
                                     cudf::test::temporary_allocation_expectation::SOME},
                                    stream);
    auto sorted = cudf::sort(output->view(), {}, {}, stream, mr());
    CUDF_TEST_EXPECT_TABLES_EQUAL(input, sorted->view(), stream);
    output.reset();
    resources.expect_no_live_allocations(stream);
  }
}

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

TEST_F(PartitionTestNotTyped, StringPermutationAndFallbacks)
{
  auto stream = cuda::stream{cuda::device_ref{rmm::get_current_cuda_device().value()}};
  // Two partitions exercise the specialized string paths. Schemas 1/2/3 fall
  // back for actual nulls, mixed types, and INT64 input offsets. Schema 5 checks
  // fused descriptor compaction with empty columns between nonempty columns.
  for (auto const schema : {0, 1, 2, 3, 4, 5}) {
    for (auto const rows : {1025, 4099}) {
      auto const total = rows + 5;
      std::vector<std::string> ids(total), values(total), empty(total);
      std::vector<bool> valid(total);
      std::vector<int32_t> numbers(total), partition_ids(total);
      for (int32_t i = 0; i < total; ++i) {
        ids[i]    = "row-" + std::to_string(i);
        values[i] = i % 13 == 0 ? std::string{"a\0b", 3}
                    : i % 7 == 0
                      ? std::string{}
                      : std::string(i % 67, 'a' + i % 26) + "\xE4\xB8\xAD\xF0\x9F\x8C\x8D";
        if (schema == 4) {
          values[i] = i % 13 == 0  ? std::string{"a\0b", 3}
                      : i % 7 == 0 ? "\xE4\xB8\xAD"
                                   : std::string(i % 17, 'a' + i % 26);
        }
        valid[i]   = i % 11 != 0;
        numbers[i] = i;
      }
      strings_column_wrapper id_column(ids.begin(), ids.end());
      strings_column_wrapper empty_column(empty.begin(), empty.end());
      auto value_column =
        schema == 1 ? strings_column_wrapper(values.begin(), values.end(), valid.begin()).release()
                    : strings_column_wrapper(values.begin(), values.end()).release();
      if (schema == 3) {
        auto contents = value_column->release();
        auto wide_offsets =
          cudf::cast(contents.children[0]->view(), cudf::data_type{cudf::type_id::INT64});
        value_column =
          cudf::make_strings_column(total,
                                    std::move(wide_offsets),
                                    std::move(*contents.data),
                                    0,
                                    cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED));
      }
      fixed_width_column_wrapper<int32_t> numeric(numbers.begin(), numbers.end());
      std::vector<cudf::column_view> columns{id_column, *value_column, empty_column};
      if (schema == 2) { columns.push_back(numeric); }
      if (schema == 5) { columns = {id_column, empty_column, *value_column, empty_column}; }
      auto input = cudf::slice(cudf::table_view{columns}, {5, total})[0];
      for (auto const partitions : {2, 100, 200}) {
        for (auto const skew : {false, true}) {
          SCOPED_TRACE(::testing::Message()
                       << schema << "/" << rows << "/" << partitions << "/" << skew);
          std::vector<int32_t> expected_offsets(partitions + 1, 0);
          for (int32_t i = 0; i < rows; ++i) {
            partition_ids[i + 5] = skew && i % 10 != 0 ? partitions - 1 : (i * 37 + 7) % partitions;
            ++expected_offsets[partition_ids[i + 5] + 1];
          }
          std::partial_sum(
            expected_offsets.begin(), expected_offsets.end(), expected_offsets.begin());
          fixed_width_column_wrapper<int32_t> map(partition_ids.begin(), partition_ids.end());
          auto map_view = cudf::slice(map, {5, total})[0];
          cudf::get_default_stream().sync();
          auto [output, offsets] = cudf::partition(input, map_view, partitions, stream, mr());
          EXPECT_EQ(offsets, expected_offsets);
          auto const [actual_ids, mask] =
            cudf::test::to_host<std::string>(output->get_column(0), stream);
          std::vector<bool> seen(total, false);
          for (int32_t p = 0; p < partitions; ++p) {
            for (auto row = offsets[p]; row < offsets[p + 1]; ++row) {
              auto const id = std::stoi(actual_ids[row].substr(4));
              ASSERT_GE(id, 5);
              ASSERT_LT(id, total);
              EXPECT_FALSE(seen[id]);
              seen[id] = true;
              EXPECT_EQ(partition_ids[id], p);
            }
          }
          auto expected = cudf::sort(input, {}, {}, stream, mr());
          auto actual   = cudf::sort(output->view(), {}, {}, stream, mr());
          CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected->view(), actual->view(), stream);
        }
      }
    }
  }
}

TEST_F(PartitionTestNotTyped, AllEmptyStringPermutation)
{
  std::vector<std::string> empty(1025);
  strings_column_wrapper strings(empty.begin(), empty.end());
  auto input = cudf::table_view{{strings, strings}};
  std::vector<int32_t> partition_ids(1025, 199);
  fixed_width_column_wrapper<int32_t> map(partition_ids.begin(), partition_ids.end());
  auto [output, offsets] = cudf::partition(input, map, 200);
  EXPECT_TRUE(std::all_of(offsets.begin(), offsets.end() - 1, [](auto x) { return x == 0; }));
  EXPECT_EQ(offsets.back(), 1025);
  CUDF_TEST_EXPECT_TABLES_EQUAL(input, output->view());
}

TEST_F(PartitionTestNotTyped, StringPermutationOutputOffsetThreshold)
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

  constexpr cudf::size_type rows = 1024;
  std::vector<std::string> values(rows, "a");
  strings_column_wrapper strings(values.begin(), values.end());
  auto const input = cudf::table_view{{strings, strings}};
  auto const ones  = cuda::make_constant_iterator<int32_t>(1);
  fixed_width_column_wrapper<int32_t> map(ones, ones + rows);
  ASSERT_EQ(cudf::strings_column_view(input.column(0)).offsets().type().id(), cudf::type_id::INT32);

  // Change the threshold after construction to keep the input offsets INT32.
  for (auto const threshold : {1025, 1024, 1023}) {
    ASSERT_EQ(setenv("LIBCUDF_LARGE_STRINGS_THRESHOLD", std::to_string(threshold).c_str(), 1), 0);
    for (auto const enabled : {true, false}) {
      SCOPED_TRACE(::testing::Message() << threshold << "/" << enabled);
      ASSERT_EQ(setenv("LIBCUDF_LARGE_STRINGS_ENABLED", enabled ? "1" : "0", 1), 0);
      if (!enabled && rows >= threshold) {
        EXPECT_THROW(cudf::partition(input, map, 2), std::overflow_error);
      } else {
        auto [output, offsets] = cudf::partition(input, map, 2);
        EXPECT_EQ(offsets, (std::vector<cudf::size_type>{0, 0, rows}));
        auto const expected_type = rows >= threshold ? cudf::type_id::INT64 : cudf::type_id::INT32;
        for (auto const& column : output->view()) {
          EXPECT_EQ(cudf::strings_column_view(column).offsets().type().id(), expected_type);
        }
        CUDF_TEST_EXPECT_TABLES_EQUIVALENT(input, output->view());
      }
    }
  }
}

TEST_F(PartitionTestNotTyped, StringPermutationCopyGridStride)
{
  // One row beyond the capped four-lane character-copy grid. Two columns use
  // rank recomputation; four columns also exercise the fused copy's second pass.
  constexpr cudf::size_type rows = 65'535 * 64 + 1;
  for (auto const width : {4, 129}) {
    auto const value  = std::string(width, 'x');
    auto const values = cuda::make_constant_iterator(value.c_str());
    strings_column_wrapper strings(values, values + rows);
    auto const ones = cuda::make_constant_iterator<int32_t>(1);
    fixed_width_column_wrapper<int32_t> map(ones, ones + rows);
    for (auto const count : {2, 4}) {
      SCOPED_TRACE(::testing::Message() << width << "/" << count);
      std::vector<cudf::column_view> columns(count, strings);
      auto const input       = cudf::table_view{columns};
      auto [output, offsets] = cudf::partition(input, map, 2);
      EXPECT_EQ(offsets, (std::vector<cudf::size_type>{0, 0, rows}));
      CUDF_TEST_EXPECT_TABLES_EQUAL(input, output->view());
    }
  }
}

class PartitionLargeFixedWidthTest
  : public cudf::test::BaseFixture,
    public ::testing::WithParamInterface<std::tuple<int, int, int, bool>> {};

TEST_P(PartitionLargeFixedWidthTest, HostOracle)
{
  // Exceed the large-plan threshold and leave the last tile and mask word incomplete.
  constexpr cudf::size_type rows                    = 1'000'003;
  constexpr cudf::size_type slice_offset            = 7;
  auto const [count, partitions, null_percent, hot] = GetParam();
  auto stream_owner       = cuda::stream{cuda::device_ref{rmm::get_current_cuda_device().value()}};
  cuda::stream_ref stream = stream_owner;
  auto const valid        = [null_percent](int row, int col) {
    return (row * 17 + col * 13) % 100 >= null_percent;
  };
  auto const value = [](int row, int col) { return int64_t{row} * (37 + col * 2) + col * 11; };
  std::vector<std::unique_ptr<cudf::column>> columns;
  for (int c = 0; c < count; ++c) {
    auto const values =
      cudf::detail::make_counting_transform_iterator(0, [=](auto i) { return value(i, c); });
    auto const validity =
      cudf::detail::make_counting_transform_iterator(0, [=](auto i) { return valid(i, c); });
    if (null_percent == 0) {
      columns.push_back(
        fixed_width_column_wrapper<int64_t>(values, values + rows + slice_offset, stream)
          .release());
    } else {
      columns.push_back(
        fixed_width_column_wrapper<int64_t>(values, values + rows + slice_offset, validity, stream)
          .release());
    }
  }
  auto const owner = cudf::table{std::move(columns)};
  auto const input = cudf::slice(owner.view(), {slice_offset, rows + slice_offset}, stream)[0];
  std::vector<int32_t> assignments(rows + 3, 0);
  std::vector<cudf::size_type> expected_offsets(partitions + 1, 0);
  std::vector<cudf::size_type> expected_all_null(partitions, 0);
  for (int i = 0; i < rows; ++i) {
    auto const p       = hot && i % 10 != 0 ? partitions - 1 : (i * 17 + 3) % partitions;
    assignments[i + 3] = p;
    ++expected_offsets[p + 1];
    bool any_valid = false;
    for (int c = 0; c < count; ++c)
      any_valid |= valid(i + slice_offset, c);
    if (!any_valid) ++expected_all_null[p];
  }
  std::partial_sum(expected_offsets.begin(), expected_offsets.end(), expected_offsets.begin());
  fixed_width_column_wrapper<int32_t> map(assignments.begin(), assignments.end(), stream);
  auto const map_view    = cudf::slice(map, {3, rows + 3}, stream)[0];
  auto [output, offsets] = cudf::partition(input, map_view, partitions, stream, mr());
  ASSERT_EQ(offsets, expected_offsets);
  ASSERT_EQ(output->num_rows(), rows);
  ASSERT_EQ(output->num_columns(), count);
  // This oracle uses host values and validity, independent of gather and GPU sorting.
  using host_column = decltype(cudf::test::to_host<int64_t>(output->get_column(0), stream));
  std::vector<host_column> actual;
  for (int c = 0; c < count; ++c) {
    EXPECT_EQ(output->get_column(c).type(), input.column(c).type());
    EXPECT_EQ(output->get_column(c).null_count(), input.column(c).null_count());
    actual.push_back(cudf::test::to_host<int64_t>(output->get_column(c), stream));
  }
  auto const is_valid = [&](int c, int i) {
    auto const& mask = actual[c].second;
    return mask.empty() || (mask[i / 32] & (1u << (i % 32))) != 0;
  };
  std::vector<bool> seen(rows, false);
  for (int p = 0; p < partitions; ++p) {
    cudf::size_type all_null = 0;
    for (int i = offsets[p]; i < offsets[p + 1]; ++i) {
      int first_valid = 0;
      while (first_valid < count && !is_valid(first_valid, i))
        ++first_valid;
      if (first_valid == count) {
        ++all_null;
        continue;
      }
      auto const source =
        (actual[first_valid].first[i] - first_valid * 11) / (37 + first_valid * 2) - slice_offset;
      ASSERT_GE(source, 0);
      ASSERT_LT(source, rows);
      ASSERT_EQ(assignments[source + 3], p);
      ASSERT_FALSE(seen[source]);
      seen[source] = true;
      for (int c = 0; c < count; ++c) {
        ASSERT_EQ(is_valid(c, i), valid(source + slice_offset, c));
        if (is_valid(c, i)) { ASSERT_EQ(actual[c].first[i], value(source + slice_offset, c)); }
      }
    }
    ASSERT_EQ(all_null, expected_all_null[p]);
  }
  for (int i = 0; i < rows; ++i) {
    bool any_valid = false;
    for (int c = 0; c < count; ++c)
      any_valid |= valid(i + slice_offset, c);
    ASSERT_EQ(seen[i], any_valid);
  }
}

INSTANTIATE_TEST_SUITE_P(LargePlans,
                         PartitionLargeFixedWidthTest,
                         ::testing::Combine(::testing::Values(1, 2, 4, 8),
                                            ::testing::Values(2, 100),
                                            ::testing::Values(0, 10, 90),
                                            ::testing::Bool()));

TEST_F(PartitionTestNotTyped, LargeSparseStringCopy)
{
  constexpr cudf::size_type rows = 4'194'307;
  auto const stream              = cudf::get_default_stream();
  auto const first_values        = cudf::detail::make_counting_transform_iterator(0, [](auto i) {
    auto const slot   = i % 1024;
    auto const length = slot == 0 ? 129 : slot == 1 ? 65 : slot == 2 ? 7 : 0;
    return std::string(length, static_cast<char>('a' + (i / 1024) % 23));
  });
  auto const second_values       = cudf::detail::make_counting_transform_iterator(0, [](auto i) {
    auto const slot   = i % 1024;
    auto const length = slot == 0 ? 9 : slot == 1 ? 128 : slot == 3 ? 513 : 0;
    return std::string(length, static_cast<char>('A' + (i / 1024) % 23));
  });
  strings_column_wrapper first(first_values, first_values + rows, stream);
  strings_column_wrapper second(second_values, second_values + rows, stream);
  auto const assignments =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i % 2; });
  fixed_width_column_wrapper<int32_t> map(assignments, assignments + rows, stream);
  auto const all_columns = cudf::table_view{{first, second}};
  constexpr auto half    = (rows + 1) / 2;
  auto const ordered     = cudf::detail::make_counting_transform_iterator(
    0, [](auto i) { return i < half ? i * 2 : (i - half) * 2 + 1; });
  fixed_width_column_wrapper<int32_t> order(ordered, ordered + rows, stream);
  for (auto const count : {1, 2}) {
    SCOPED_TRACE(count);
    auto const input       = count == 1 ? all_columns.select({0}) : all_columns;
    auto [output, offsets] = cudf::partition(input, map, 2, stream, mr());
    ASSERT_EQ(offsets, (std::vector<cudf::size_type>{0, half, rows}));
    auto expected = cudf::gather(input,
                                 order,
                                 cudf::out_of_bounds_policy::DONT_CHECK,
                                 cudf::negative_index_policy::NOT_ALLOWED,
                                 stream,
                                 mr());
    for (auto p = 0; p < 2; ++p) {
      auto expected_part   = cudf::slice(expected->view(), {offsets[p], offsets[p + 1]}, stream)[0];
      auto actual_part     = cudf::slice(output->view(), {offsets[p], offsets[p + 1]}, stream)[0];
      auto expected_sorted = cudf::sort(expected_part, {}, {}, stream, mr());
      auto actual_sorted   = cudf::sort(actual_part, {}, {}, stream, mr());
      CUDF_TEST_EXPECT_TABLES_EQUIVALENT(expected_sorted->view(), actual_sorted->view(), stream);
    }
  }
}

TEST_F(PartitionTestNotTyped, StringVectorCopyBoundaries)
{
  constexpr cudf::size_type rows = 4099;
  std::vector<std::string> values(rows + 14);
  std::vector<int32_t> assignments(rows);
  std::vector<int32_t> gather;
  std::vector<cudf::size_type> offsets{0};
  std::vector<int> lengths{0, 1, 3, 15, 16, 17, 127, 128, 129, 130, 143, 144, 255, 256, 257, 1024};
  for (std::size_t i = 0; i < values.size(); ++i) {
    values[i] = std::string(lengths[i % lengths.size()], static_cast<char>('a' + i % 26));
    if (values[i].size() > 3) { values[i][2] = '\0'; }
  }
  auto const valid = cuda::make_constant_iterator(true);
  strings_column_wrapper strings(values.begin(), values.end(), valid);
  for (auto i = 0; i < rows; ++i) {
    assignments[i] = i % 10 == 0 ? 0 : 1;
  }
  for (auto p = 0; p < 2; ++p) {
    for (auto i = 0; i < rows; ++i) {
      if (assignments[i] == p) { gather.push_back(i); }
    }
    offsets.push_back(gather.size());
  }
  fixed_width_column_wrapper<int32_t> map(assignments.begin(), assignments.end());
  fixed_width_column_wrapper<int32_t> indices(gather.begin(), gather.end());
  auto stream = cuda::stream{cuda::device_ref{rmm::get_current_cuda_device().value()}};
  for (auto const count : {1, 2, 4}) {
    SCOPED_TRACE(count);
    std::vector<cudf::column_view> columns(count, strings);
    auto const input = cudf::slice(cudf::table_view{columns}, {7, rows + 7})[0];
    auto expected    = cudf::gather(input, indices);
    cudf::get_default_stream().sync();
    auto [output, boundaries] = cudf::partition(input, map, 2, stream, mr());
    EXPECT_EQ(boundaries, offsets);
    stream.sync();
    // An all-valid mask may be discarded; compare partition values and null semantics.
    auto const split_points   = std::vector<cudf::size_type>{offsets[1]};
    auto const expected_parts = cudf::split(expected->view(), split_points);
    auto const actual_parts   = cudf::split(output->view(), split_points);
    for (std::size_t i = 0; i < expected_parts.size(); ++i) {
      auto sorted_expected = cudf::sort(expected_parts[i]);
      auto sorted_actual   = cudf::sort(actual_parts[i]);
      CUDF_TEST_EXPECT_TABLES_EQUIVALENT(sorted_expected->view(), sorted_actual->view());
    }
  }
}
