/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/default_stream.hpp>
#include <cudf_test/table_utilities.hpp>
#include <cudf_test/testing_main.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/copying.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/partitioning.hpp>
#include <cudf/sorting.hpp>
#include <cudf/table/table.hpp>

#include <cuda/iterator>

#include <cstdint>
#include <vector>

using cudf::test::fixed_width_column_wrapper;
using cudf::test::strings_column_wrapper;

class PartitionTest : public cudf::test::BaseFixture {};

TEST_F(PartitionTest, MapRanks)
{
  constexpr cudf::size_type rows = 1025;
  auto const stream              = cudf::test::get_default_stream();
  auto const ids = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i; });
  auto const partitions =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i % 2; });
  fixed_width_column_wrapper<int32_t> data(ids, ids + rows, stream);
  fixed_width_column_wrapper<int32_t> map(partitions, partitions + rows, stream);
  for (auto const& input : {cudf::table_view{{data}}, cudf::table_view{{data, data}}}) {
    SCOPED_TRACE(input.num_columns());
    auto [output, offsets] = cudf::partition(input, map, 2, stream, mr());
    ASSERT_EQ(offsets, (std::vector<cudf::size_type>{0, (rows + 1) / 2, rows}));
    auto const [actual, mask] = cudf::test::to_host<int32_t>(output->get_column(0), stream);
    for (cudf::size_type p = 0; p < 2; ++p) {
      for (auto row = offsets[p]; row < offsets[p + 1]; ++row) {
        EXPECT_EQ(actual[row] % 2, p);
      }
    }
    auto sorted = cudf::sort(output->view(), {}, {}, stream, mr());
    CUDF_TEST_EXPECT_TABLES_EQUAL(input, sorted->view(), stream);
  }
}

TEST_F(PartitionTest, HeuristicFixedWidth)
{
  constexpr cudf::size_type rows = 1'000'003;
  auto const stream              = cudf::test::get_default_stream();
  auto const ids                 = cuda::counting_iterator<int64_t>{0};
  for (auto const null_mode : {0, 1, 2}) {
    auto const valid = cudf::detail::make_counting_transform_iterator(0, [null_mode](auto i) {
      return null_mode == 0 || (null_mode == 1 ? i % 100 != 0 : i % 10 == 0);
    });
    fixed_width_column_wrapper<int64_t> data(ids, ids + rows, valid, stream);
    for (auto const partitions : {2, 100, 256}) {
      auto const assignments = cudf::detail::make_counting_transform_iterator(
        0, [partitions](auto i) { return i % partitions; });
      fixed_width_column_wrapper<int32_t> map(assignments, assignments + rows, stream);
      std::vector<cudf::size_type> expected(partitions + 1, 0);
      for (auto p = 0; p < partitions; ++p) {
        expected[p + 1] = expected[p] + rows / partitions + (p < rows % partitions);
      }
      for (auto const count : {1, 2, 8}) {
        SCOPED_TRACE(::testing::Message() << null_mode << "/" << partitions << "/" << count);
        std::vector<cudf::column_view> columns(count, data);
        auto [output, offsets] =
          cudf::partition(cudf::table_view{columns}, map, partitions, stream, mr());
        EXPECT_EQ(offsets, expected);
        EXPECT_EQ(output->num_rows(), rows);
        EXPECT_EQ(output->num_columns(), count);
        for (auto const& col : output->view()) {
          EXPECT_EQ(col.null_count(), cudf::column_view{data}.null_count());
        }
      }
    }
  }
  fixed_width_column_wrapper<int32_t> narrow(ids, ids + rows, stream);
  auto const partitions =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i % 256; });
  fixed_width_column_wrapper<int32_t> map(partitions, partitions + rows, stream);
  std::vector<cudf::column_view> columns(8, narrow);
  auto result = cudf::partition(cudf::table_view{columns}, map, 256, stream, mr());
  EXPECT_EQ(result.second.back(), rows);
}

TEST_F(PartitionTest, TrivialMapValues)
{
  constexpr cudf::size_type rows = 1025;
  auto const stream              = cudf::test::get_default_stream();
  auto const ids                 = cuda::counting_iterator<int64_t>{0};
  auto const invalid             = cuda::make_constant_iterator(false);
  auto const zero                = cuda::make_constant_iterator<int32_t>(0);
  fixed_width_column_wrapper<int64_t> nulls(ids, ids + rows, invalid, stream);
  fixed_width_column_wrapper<int32_t> map(zero, zero + rows, stream);
  auto const text = cuda::make_constant_iterator("value");
  strings_column_wrapper strings(text, text + rows, stream);
  auto const empty_value = cuda::make_constant_iterator("");
  strings_column_wrapper empty(empty_value, empty_value + rows, stream);
  for (auto const& input : {cudf::table_view{{nulls}},
                            cudf::table_view{{nulls, strings}},
                            cudf::table_view{std::vector<cudf::column_view>{}, rows}}) {
    auto [output, offsets] = cudf::partition(input, map, 1, stream, mr());
    EXPECT_EQ(offsets, (std::vector<cudf::size_type>{0, rows}));
    EXPECT_EQ(output->num_rows(), rows);
  }
  auto empty_result = cudf::partition(cudf::table_view{{empty, nulls}}, map, 256, stream, mr());
  EXPECT_EQ(empty_result.second.back(), rows);
  EXPECT_EQ(empty_result.first->get_column(0).null_count(), 0);
  auto mixed_result = cudf::partition(cudf::table_view{{empty, strings}}, map, 256, stream, mr());
  EXPECT_EQ(mixed_result.second.back(), rows);
  auto [output, offsets] = cudf::partition(cudf::table_view{{nulls}}, map, 256, stream, mr());
  EXPECT_EQ(offsets.front(), 0);
  EXPECT_EQ(offsets.back(), rows);
  EXPECT_EQ(output->get_column(0).null_count(), rows);
}

TEST_F(PartitionTest, Struct)
{
  fixed_width_column_wrapper<numeric::decimal32, int32_t> A({1, 2}, {0, 1});
  auto struct_col         = cudf::test::structs_column_wrapper({A}, {0, 1}).release();
  auto table_to_partition = cudf::table_view{{*struct_col}};
  fixed_width_column_wrapper<int32_t> map{9, 2};

  auto num_partitions = 12;
  auto result =
    cudf::partition(table_to_partition, map, num_partitions, cudf::test::get_default_stream());
}

TEST_F(PartitionTest, EmptyInput)
{
  auto const empty_column    = fixed_width_column_wrapper<int32_t>{};
  auto const num_partitions  = 5;
  auto const start_partition = 0;
  auto const [out_table, out_offsets] =
    cudf::round_robin_partition(cudf::table_view{{empty_column}},
                                num_partitions,
                                start_partition,
                                cudf::test::get_default_stream());
}

TEST_F(PartitionTest, ZeroPartitions)
{
  fixed_width_column_wrapper<float> floats({1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f});
  fixed_width_column_wrapper<int16_t> integers({1, 2, 3, 4, 5, 6, 7, 8});
  strings_column_wrapper strings({"a", "bb", "ccc", "d", "ee", "fff", "gg", "h"});
  auto input = cudf::table_view({floats, integers, strings});

  auto columns_to_hash = std::vector<cudf::size_type>({2});

  cudf::size_type const num_partitions = 0;
  auto [output, offsets]               = cudf::hash_partition(input,
                                                columns_to_hash,
                                                num_partitions,
                                                cudf::hash_id::HASH_MURMUR3,
                                                cudf::DEFAULT_HASH_SEED,
                                                cudf::test::get_default_stream());
}

TEST_F(PartitionTest, HeuristicStrings)
{
  auto const stream = cudf::test::get_default_stream();
  for (auto const partitions : {2, 256}) {
    for (auto const rows : {4099, 4'000'003}) {
      auto const value = cuda::make_constant_iterator("a string with enough characters");
      auto const valid = cuda::make_constant_iterator(true);
      strings_column_wrapper strings(value, value + rows, valid, stream);
      auto const values = cudf::detail::make_counting_transform_iterator(
        0, [partitions](auto i) { return i % partitions; });
      fixed_width_column_wrapper<int32_t> map(values, values + rows, stream);
      for (auto const count : {1, 2, 4}) {
        SCOPED_TRACE(::testing::Message() << rows << "/" << count);
        std::vector<cudf::column_view> columns(count, strings);
        auto [output, offsets] =
          cudf::partition(cudf::table_view{columns}, map, partitions, stream, mr());
        std::vector<cudf::size_type> expected(partitions + 1, 0);
        for (auto p = 0; p < partitions; ++p) {
          expected[p + 1] = expected[p] + rows / partitions + (p < rows % partitions);
        }
        EXPECT_EQ(offsets, expected);
        EXPECT_EQ(output->num_rows(), rows);
        EXPECT_EQ(output->num_columns(), count);
      }
    }
  }
}

CUDF_TEST_PROGRAM_MAIN()
