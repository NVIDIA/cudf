/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/default_stream.hpp>
#include <cudf_test/testing_main.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/copying.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/partitioning.hpp>
#include <cudf/sorting.hpp>
#include <cudf/table/table.hpp>

using cudf::test::fixed_width_column_wrapper;
using cudf::test::strings_column_wrapper;

class PartitionTest : public cudf::test::BaseFixture {};

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

CUDF_TEST_PROGRAM_MAIN()
