/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/detail/iterator.cuh>
#include <cudf/partitioning.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <gtest/gtest.h>

#include <memory>
#include <vector>

using cudf::test::fixed_width_column_wrapper;
using cudf::test::strings_column_wrapper;

template <typename T>
class RoundRobinTest : public cudf::test::BaseFixture {};

TYPED_TEST_SUITE(RoundRobinTest, cudf::test::FixedWidthTypes);

struct RoundRobinZeroColumnTest : public cudf::test::BaseFixture {};

TEST_F(RoundRobinZeroColumnTest, PreservesRowCount)
{
  cudf::table_view input{std::vector<cudf::column_view>{}, 7};
  auto const [result, offsets] = cudf::round_robin_partition(input, 3);
  EXPECT_EQ(result->num_columns(), 0);
  EXPECT_EQ(result->num_rows(), 7);
  // 7 rows over 3 partitions deals sizes {3, 2, 2}, so the round-robin boundaries must hold
  // even with no columns.
  EXPECT_EQ(offsets, (std::vector<cudf::size_type>{0, 3, 5, 7}));
}

TYPED_TEST(RoundRobinTest, EmptyInput)
{
  auto const empty_column    = fixed_width_column_wrapper<TypeParam>{};
  auto const num_partitions  = 5;
  auto const start_partition = 0;
  auto const [out_table, out_offsets] =
    cudf::round_robin_partition(cudf::table_view{{empty_column}}, num_partitions, start_partition);

  EXPECT_EQ(out_table->num_rows(), 0);
  EXPECT_EQ(out_offsets.size(), std::size_t{num_partitions + 1});
}

TYPED_TEST(RoundRobinTest, RoundRobinPartitions13_3)
{
  strings_column_wrapper rrColWrap1(
    {"a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m"},
    {true, true, true, true, true, true, true, true, true, true, true, true, false});

  cudf::size_type inputRows = static_cast<cudf::column_view const&>(rrColWrap1).size();

  auto sequence_l = cudf::detail::make_counting_transform_iterator(
    0, [bool8 = (cudf::type_to_id<TypeParam>() == cudf::type_id::BOOL8)](auto row) {
      return bool8 ? static_cast<decltype(row)>(row % 2 == 0) : row;
    });

  cudf::test::fixed_width_column_wrapper<TypeParam, typename decltype(sequence_l)::value_type>
    rrColWrap2(sequence_l, sequence_l + inputRows);

  cudf::table_view rr_view{{rrColWrap1, rrColWrap2}};

  cudf::size_type num_partitions = 3;

  cudf::size_type start_partition = 0;
  {
    std::pair<std::unique_ptr<cudf::table>, std::vector<cudf::size_type>> result;
    EXPECT_NO_THROW(result = cudf::round_robin_partition(rr_view, num_partitions, start_partition));

    auto p_outputTable = std::move(result.first);

    auto output_column_view1{p_outputTable->view().column(0)};
    auto output_column_view2{p_outputTable->view().column(1)};

    strings_column_wrapper expectedDataWrap1(
      {"a", "d", "g", "j", "m", "b", "e", "h", "k", "c", "f", "i", "l"},
      {true, true, true, true, false, true, true, true, true, true, true, true, true});

    auto expected_column_view1{static_cast<cudf::column_view const&>(expectedDataWrap1)};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_column_view1, output_column_view1);

    if (cudf::type_to_id<TypeParam>() == cudf::type_id::BOOL8) {
      fixed_width_column_wrapper<bool> expectedDataWrap2({1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0});
      auto expected_column_view2{static_cast<cudf::column_view const&>(expectedDataWrap2)};

      EXPECT_EQ(inputRows, expected_column_view2.size());
      EXPECT_EQ(inputRows, output_column_view2.size());

      CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_column_view2, output_column_view2);
    } else {
      fixed_width_column_wrapper<TypeParam, int32_t> expectedDataWrap2(
        {0, 3, 6, 9, 12, 1, 4, 7, 10, 2, 5, 8, 11});
      auto expected_column_view2{static_cast<cudf::column_view const&>(expectedDataWrap2)};
      CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_column_view2, output_column_view2);
    }

    std::vector<cudf::size_type> expected_partition_offsets{0, 5, 9, 13};
    EXPECT_EQ(static_cast<std::size_t>(num_partitions + 1), expected_partition_offsets.size());

    EXPECT_EQ(expected_partition_offsets, result.second);
  }

  start_partition = 1;
  {
    std::pair<std::unique_ptr<cudf::table>, std::vector<cudf::size_type>> result;
    EXPECT_NO_THROW(result = cudf::round_robin_partition(rr_view, num_partitions, start_partition));

    auto p_outputTable = std::move(result.first);

    auto output_column_view1{p_outputTable->view().column(0)};
    auto output_column_view2{p_outputTable->view().column(1)};

    strings_column_wrapper expectedDataWrap1(
      {"c", "f", "i", "l", "a", "d", "g", "j", "m", "b", "e", "h", "k"},
      {true, true, true, true, true, true, true, true, false, true, true, true, true});

    auto expected_column_view1{static_cast<cudf::column_view const&>(expectedDataWrap1)};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_column_view1, output_column_view1);

    if (cudf::type_to_id<TypeParam>() == cudf::type_id::BOOL8) {
      fixed_width_column_wrapper<bool> expectedDataWrap2({1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1});
      auto expected_column_view2{static_cast<cudf::column_view const&>(expectedDataWrap2)};

      EXPECT_EQ(inputRows, expected_column_view2.size());
      EXPECT_EQ(inputRows, output_column_view2.size());

      CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_column_view2, output_column_view2);
    } else {
      fixed_width_column_wrapper<TypeParam, int32_t> expectedDataWrap2(
        {2, 5, 8, 11, 0, 3, 6, 9, 12, 1, 4, 7, 10});
      auto expected_column_view2{static_cast<cudf::column_view const&>(expectedDataWrap2)};
      CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_column_view2, output_column_view2);
    }

    std::vector<cudf::size_type> expected_partition_offsets{0, 4, 9, 13};
    EXPECT_EQ(static_cast<std::size_t>(num_partitions + 1), expected_partition_offsets.size());

    EXPECT_EQ(expected_partition_offsets, result.second);
  }

  start_partition = 2;
  {
    std::pair<std::unique_ptr<cudf::table>, std::vector<cudf::size_type>> result;
    EXPECT_NO_THROW(result = cudf::round_robin_partition(rr_view, num_partitions, start_partition));

    auto p_outputTable = std::move(result.first);

    auto output_column_view1{p_outputTable->view().column(0)};
    auto output_column_view2{p_outputTable->view().column(1)};

    strings_column_wrapper expectedDataWrap1(
      {"b", "e", "h", "k", "c", "f", "i", "l", "a", "d", "g", "j", "m"},
      {true, true, true, true, true, true, true, true, true, true, true, true, false});

    auto expected_column_view1{static_cast<cudf::column_view const&>(expectedDataWrap1)};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_column_view1, output_column_view1);

    if (cudf::type_to_id<TypeParam>() == cudf::type_id::BOOL8) {
      fixed_width_column_wrapper<bool> expectedDataWrap2({0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1});
      auto expected_column_view2{static_cast<cudf::column_view const&>(expectedDataWrap2)};

      EXPECT_EQ(inputRows, expected_column_view2.size());
      EXPECT_EQ(inputRows, output_column_view2.size());

      CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_column_view2, output_column_view2);
    } else {
      fixed_width_column_wrapper<TypeParam, int32_t> expectedDataWrap2(
        {1, 4, 7, 10, 2, 5, 8, 11, 0, 3, 6, 9, 12});
      auto expected_column_view2{static_cast<cudf::column_view const&>(expectedDataWrap2)};
      CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_column_view2, output_column_view2);
    }

    std::vector<cudf::size_type> expected_partition_offsets{0, 4, 8, 13};
    EXPECT_EQ(static_cast<std::size_t>(num_partitions + 1), expected_partition_offsets.size());

    EXPECT_EQ(expected_partition_offsets, result.second);
  }
}

TYPED_TEST(RoundRobinTest, RoundRobinPartitions11_3)
{
  strings_column_wrapper rrColWrap1(
    {"a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k"},
    {true, true, true, true, true, true, true, true, true, true, false});

  cudf::size_type inputRows = static_cast<cudf::column_view const&>(rrColWrap1).size();

  auto sequence_l = cudf::detail::make_counting_transform_iterator(
    0, [bool8 = (cudf::type_to_id<TypeParam>() == cudf::type_id::BOOL8)](auto row) {
      return bool8 ? static_cast<decltype(row)>(row % 2 == 0) : row;
    });

  cudf::test::fixed_width_column_wrapper<TypeParam, typename decltype(sequence_l)::value_type>
    rrColWrap2(sequence_l, sequence_l + inputRows);

  cudf::table_view rr_view{{rrColWrap1, rrColWrap2}};

  cudf::size_type num_partitions = 3;

  cudf::size_type start_partition = 0;
  {
    std::pair<std::unique_ptr<cudf::table>, std::vector<cudf::size_type>> result;
    EXPECT_NO_THROW(result = cudf::round_robin_partition(rr_view, num_partitions, start_partition));

    auto p_outputTable = std::move(result.first);

    auto output_column_view1{p_outputTable->view().column(0)};
    auto output_column_view2{p_outputTable->view().column(1)};

    strings_column_wrapper expectedDataWrap1(
      {"a", "d", "g", "j", "b", "e", "h", "k", "c", "f", "i"},
      {true, true, true, true, true, true, true, false, true, true, true});

    auto expected_column_view1{static_cast<cudf::column_view const&>(expectedDataWrap1)};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_column_view1, output_column_view1);

    if (cudf::type_to_id<TypeParam>() == cudf::type_id::BOOL8) {
      fixed_width_column_wrapper<bool> expectedDataWrap2({1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1});
      auto expected_column_view2{static_cast<cudf::column_view const&>(expectedDataWrap2)};

      EXPECT_EQ(inputRows, expected_column_view2.size());
      EXPECT_EQ(inputRows, output_column_view2.size());

      CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_column_view2, output_column_view2);
    } else {
      fixed_width_column_wrapper<TypeParam, int32_t> expectedDataWrap2(
        {0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8});
      auto expected_column_view2{static_cast<cudf::column_view const&>(expectedDataWrap2)};
      CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_column_view2, output_column_view2);
    }

    std::vector<cudf::size_type> expected_partition_offsets{0, 4, 8, 11};
    EXPECT_EQ(static_cast<std::size_t>(num_partitions + 1), expected_partition_offsets.size());

    EXPECT_EQ(expected_partition_offsets, result.second);
  }

  start_partition = 1;
  {
    std::pair<std::unique_ptr<cudf::table>, std::vector<cudf::size_type>> result;
    EXPECT_NO_THROW(result = cudf::round_robin_partition(rr_view, num_partitions, start_partition));

    auto p_outputTable = std::move(result.first);

    auto output_column_view1{p_outputTable->view().column(0)};
    auto output_column_view2{p_outputTable->view().column(1)};

    strings_column_wrapper expectedDataWrap1(
      {"c", "f", "i", "a", "d", "g", "j", "b", "e", "h", "k"},
      {true, true, true, true, true, true, true, true, true, true, false});

    auto expected_column_view1{static_cast<cudf::column_view const&>(expectedDataWrap1)};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_column_view1, output_column_view1);

    if (cudf::type_to_id<TypeParam>() == cudf::type_id::BOOL8) {
      fixed_width_column_wrapper<bool> expectedDataWrap2({1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1});
      auto expected_column_view2{static_cast<cudf::column_view const&>(expectedDataWrap2)};

      EXPECT_EQ(inputRows, expected_column_view2.size());
      EXPECT_EQ(inputRows, output_column_view2.size());

      CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_column_view2, output_column_view2);
    } else {
      fixed_width_column_wrapper<TypeParam, int32_t> expectedDataWrap2(
        {2, 5, 8, 0, 3, 6, 9, 1, 4, 7, 10});
      auto expected_column_view2{static_cast<cudf::column_view const&>(expectedDataWrap2)};
      CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_column_view2, output_column_view2);
    }

    std::vector<cudf::size_type> expected_partition_offsets{0, 3, 7, 11};
    EXPECT_EQ(static_cast<std::size_t>(num_partitions + 1), expected_partition_offsets.size());

    EXPECT_EQ(expected_partition_offsets, result.second);
  }

  start_partition = 2;
  {
    std::pair<std::unique_ptr<cudf::table>, std::vector<cudf::size_type>> result;
    EXPECT_NO_THROW(result = cudf::round_robin_partition(rr_view, num_partitions, start_partition));

    auto p_outputTable = std::move(result.first);

    auto output_column_view1{p_outputTable->view().column(0)};
    auto output_column_view2{p_outputTable->view().column(1)};

    strings_column_wrapper expectedDataWrap1(
      {"b", "e", "h", "k", "c", "f", "i", "a", "d", "g", "j"},
      {true, true, true, false, true, true, true, true, true, true, true});

    auto expected_column_view1{static_cast<cudf::column_view const&>(expectedDataWrap1)};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_column_view1, output_column_view1);

    if (cudf::type_to_id<TypeParam>() == cudf::type_id::BOOL8) {
      fixed_width_column_wrapper<bool> expectedDataWrap2({0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0});
      auto expected_column_view2{static_cast<cudf::column_view const&>(expectedDataWrap2)};

      EXPECT_EQ(inputRows, expected_column_view2.size());
      EXPECT_EQ(inputRows, output_column_view2.size());

      CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_column_view2, output_column_view2);
    } else {
      fixed_width_column_wrapper<TypeParam, int32_t> expectedDataWrap2(
        {1, 4, 7, 10, 2, 5, 8, 0, 3, 6, 9});
      auto expected_column_view2{static_cast<cudf::column_view const&>(expectedDataWrap2)};
      CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_column_view2, output_column_view2);
    }

    std::vector<cudf::size_type> expected_partition_offsets{0, 4, 7, 11};
    EXPECT_EQ(static_cast<std::size_t>(num_partitions + 1), expected_partition_offsets.size());

    EXPECT_EQ(expected_partition_offsets, result.second);
  }
}

TYPED_TEST(RoundRobinTest, RoundRobinDegeneratePartitions11_15)
{
  strings_column_wrapper rrColWrap1(
    {"a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k"},
    {true, true, true, true, true, true, true, true, true, true, false});

  cudf::size_type inputRows = static_cast<cudf::column_view const&>(rrColWrap1).size();

  auto sequence_l = cudf::detail::make_counting_transform_iterator(
    0, [bool8 = (cudf::type_to_id<TypeParam>() == cudf::type_id::BOOL8)](auto row) {
      return bool8 ? static_cast<decltype(row)>(row % 2 == 0) : row;
    });

  cudf::test::fixed_width_column_wrapper<TypeParam, typename decltype(sequence_l)::value_type>
    rrColWrap2(sequence_l, sequence_l + inputRows);

  cudf::table_view rr_view{{rrColWrap1, rrColWrap2}};

  cudf::size_type num_partitions = 15;

  cudf::size_type start_partition = 2;
  {
    std::pair<std::unique_ptr<cudf::table>, std::vector<cudf::size_type>> result;
    EXPECT_NO_THROW(result = cudf::round_robin_partition(rr_view, num_partitions, start_partition));

    auto p_outputTable = std::move(result.first);

    auto output_column_view1{p_outputTable->view().column(0)};
    auto output_column_view2{p_outputTable->view().column(1)};

    strings_column_wrapper expectedDataWrap1(
      {"a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k"},
      {true, true, true, true, true, true, true, true, true, true, false});

    auto expected_column_view1{static_cast<cudf::column_view const&>(expectedDataWrap1)};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_column_view1, output_column_view1);

    if (cudf::type_to_id<TypeParam>() == cudf::type_id::BOOL8) {
      fixed_width_column_wrapper<bool> expectedDataWrap2({1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1});
      auto expected_column_view2{static_cast<cudf::column_view const&>(expectedDataWrap2)};

      EXPECT_EQ(inputRows, expected_column_view2.size());
      EXPECT_EQ(inputRows, output_column_view2.size());

      CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_column_view2, output_column_view2);
    } else {
      fixed_width_column_wrapper<TypeParam, int32_t> expectedDataWrap2(
        {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
      auto expected_column_view2{static_cast<cudf::column_view const&>(expectedDataWrap2)};
      CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_column_view2, output_column_view2);
    }

    std::vector<cudf::size_type> expected_partition_offsets{
      0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 11, 11};
    EXPECT_EQ(static_cast<std::size_t>(num_partitions + 1), expected_partition_offsets.size());

    EXPECT_EQ(expected_partition_offsets, result.second);
  }

  start_partition = 10;
  {
    std::pair<std::unique_ptr<cudf::table>, std::vector<cudf::size_type>> result;
    EXPECT_NO_THROW(result = cudf::round_robin_partition(rr_view, num_partitions, start_partition));

    auto p_outputTable = std::move(result.first);

    auto output_column_view1{p_outputTable->view().column(0)};
    auto output_column_view2{p_outputTable->view().column(1)};

    strings_column_wrapper expectedDataWrap1(
      {"f", "g", "h", "i", "j", "k", "a", "b", "c", "d", "e"},
      {true, true, true, true, true, false, true, true, true, true, true});

    auto expected_column_view1{static_cast<cudf::column_view const&>(expectedDataWrap1)};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_column_view1, output_column_view1);

    if (cudf::type_to_id<TypeParam>() == cudf::type_id::BOOL8) {
      fixed_width_column_wrapper<bool> expectedDataWrap2({0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1});
      auto expected_column_view2{static_cast<cudf::column_view const&>(expectedDataWrap2)};

      EXPECT_EQ(inputRows, expected_column_view2.size());
      EXPECT_EQ(inputRows, output_column_view2.size());

      CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_column_view2, output_column_view2);
    } else {
      fixed_width_column_wrapper<TypeParam, int32_t> expectedDataWrap2(
        {5, 6, 7, 8, 9, 10, 0, 1, 2, 3, 4});
      auto expected_column_view2{static_cast<cudf::column_view const&>(expectedDataWrap2)};
      CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_column_view2, output_column_view2);
    }

    std::vector<cudf::size_type> expected_partition_offsets{
      0, 1, 2, 3, 4, 5, 6, 6, 6, 6, 6, 7, 8, 9, 10, 11};
    EXPECT_EQ(static_cast<std::size_t>(num_partitions + 1), expected_partition_offsets.size());

    EXPECT_EQ(expected_partition_offsets, result.second);
  }

  start_partition = 14;
  {
    std::pair<std::unique_ptr<cudf::table>, std::vector<cudf::size_type>> result;
    EXPECT_NO_THROW(result = cudf::round_robin_partition(rr_view, num_partitions, start_partition));

    auto p_outputTable = std::move(result.first);

    auto output_column_view1{p_outputTable->view().column(0)};
    auto output_column_view2{p_outputTable->view().column(1)};

    strings_column_wrapper expectedDataWrap1(
      {"b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "a"},
      {true, true, true, true, true, true, true, true, true, false, true});

    auto expected_column_view1{static_cast<cudf::column_view const&>(expectedDataWrap1)};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_column_view1, output_column_view1);

    if (cudf::type_to_id<TypeParam>() == cudf::type_id::BOOL8) {
      fixed_width_column_wrapper<bool> expectedDataWrap2({0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1});
      auto expected_column_view2{static_cast<cudf::column_view const&>(expectedDataWrap2)};

      EXPECT_EQ(inputRows, expected_column_view2.size());
      EXPECT_EQ(inputRows, output_column_view2.size());

      CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_column_view2, output_column_view2);
    } else {
      fixed_width_column_wrapper<TypeParam, int32_t> expectedDataWrap2(
        {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0});
      auto expected_column_view2{static_cast<cudf::column_view const&>(expectedDataWrap2)};
      CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_column_view2, output_column_view2);
    }

    std::vector<cudf::size_type> expected_partition_offsets{
      0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 10, 11};
    EXPECT_EQ(static_cast<std::size_t>(num_partitions + 1), expected_partition_offsets.size());

    EXPECT_EQ(expected_partition_offsets, result.second);
  }
}

TYPED_TEST(RoundRobinTest, RoundRobinDegeneratePartitions11_11)
{
  strings_column_wrapper rrColWrap1(
    {"a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k"},
    {true, true, true, true, true, true, true, true, true, true, false});

  cudf::size_type inputRows = static_cast<cudf::column_view const&>(rrColWrap1).size();

  auto sequence_l = cudf::detail::make_counting_transform_iterator(
    0, [bool8 = (cudf::type_to_id<TypeParam>() == cudf::type_id::BOOL8)](auto row) {
      return bool8 ? static_cast<decltype(row)>(row % 2 == 0) : row;
    });

  cudf::test::fixed_width_column_wrapper<TypeParam, typename decltype(sequence_l)::value_type>
    rrColWrap2(sequence_l, sequence_l + inputRows);

  cudf::table_view rr_view{{rrColWrap1, rrColWrap2}};

  cudf::size_type num_partitions = 11;

  cudf::size_type start_partition = 2;
  {
    std::pair<std::unique_ptr<cudf::table>, std::vector<cudf::size_type>> result;
    EXPECT_NO_THROW(result = cudf::round_robin_partition(rr_view, num_partitions, start_partition));

    auto p_outputTable = std::move(result.first);

    auto output_column_view1{p_outputTable->view().column(0)};
    auto output_column_view2{p_outputTable->view().column(1)};

    strings_column_wrapper expectedDataWrap1(
      {"j", "k", "a", "b", "c", "d", "e", "f", "g", "h", "i"},
      {true, false, true, true, true, true, true, true, true, true, true});

    auto expected_column_view1{static_cast<cudf::column_view const&>(expectedDataWrap1)};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_column_view1, output_column_view1);

    if (cudf::type_to_id<TypeParam>() == cudf::type_id::BOOL8) {
      fixed_width_column_wrapper<bool> expectedDataWrap2({0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1});
      auto expected_column_view2{static_cast<cudf::column_view const&>(expectedDataWrap2)};

      EXPECT_EQ(inputRows, expected_column_view2.size());
      EXPECT_EQ(inputRows, output_column_view2.size());

      CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_column_view2, output_column_view2);
    } else {
      fixed_width_column_wrapper<TypeParam, int32_t> expectedDataWrap2(
        {9, 10, 0, 1, 2, 3, 4, 5, 6, 7, 8});
      auto expected_column_view2{static_cast<cudf::column_view const&>(expectedDataWrap2)};
      CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_column_view2, output_column_view2);
    }

    std::vector<cudf::size_type> expected_partition_offsets{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    EXPECT_EQ(static_cast<std::size_t>(num_partitions + 1), expected_partition_offsets.size());

    EXPECT_EQ(expected_partition_offsets, result.second);
  }
}

TYPED_TEST(RoundRobinTest, RoundRobinNPartitionsDivideNRows)
{
  // test the case when nrows `mod` npartitions = 0
  //
  // input:
  // strings: ["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u"],
  // bools:   [1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1],
  // nulls:   [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0])
  //
  strings_column_wrapper rrColWrap1(
    {"a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k",
     "l", "m", "n", "o", "p", "q", "r", "s", "t", "u"},
    {true, true, true, true, true, true, true, true, true, true, true,
     true, true, true, true, true, true, true, true, true, false});

  cudf::size_type inputRows = static_cast<cudf::column_view const&>(rrColWrap1).size();

  auto sequence_l = cudf::detail::make_counting_transform_iterator(
    0, [bool8 = (cudf::type_to_id<TypeParam>() == cudf::type_id::BOOL8)](auto row) {
      return bool8 ? static_cast<decltype(row)>(row % 2 == 0) : row;
    });

  cudf::test::fixed_width_column_wrapper<TypeParam, typename decltype(sequence_l)::value_type>
    rrColWrap2(sequence_l, sequence_l + inputRows);

  cudf::table_view rr_view{{rrColWrap1, rrColWrap2}};

  cudf::size_type num_partitions = 3;

  // expected:
  // indxs:   [0,3,6,9,12,15,18,1,4,7,10,13,16,19,2,5,8,11,14,17,20],
  // strings: ["a","d","g","j","m","p","s","b","e","h","k","n","q","t","c","f","i","l","o","r","u"],
  // bools:   [1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1],
  // nulls:   [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
  // offsets: [0,7,14]
  //
  cudf::size_type start_partition = 0;
  {
    std::pair<std::unique_ptr<cudf::table>, std::vector<cudf::size_type>> result;
    EXPECT_NO_THROW(result = cudf::round_robin_partition(rr_view, num_partitions, start_partition));

    auto p_outputTable = std::move(result.first);

    auto output_column_view1{p_outputTable->view().column(0)};
    auto output_column_view2{p_outputTable->view().column(1)};

    strings_column_wrapper expectedDataWrap1(
      {"a", "d", "g", "j", "m", "p", "s", "b", "e", "h", "k",
       "n", "q", "t", "c", "f", "i", "l", "o", "r", "u"},
      {true, true, true, true, true, true, true, true, true, true, true,
       true, true, true, true, true, true, true, true, true, false});

    auto expected_column_view1{static_cast<cudf::column_view const&>(expectedDataWrap1)};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_column_view1, output_column_view1);

    if (cudf::type_to_id<TypeParam>() == cudf::type_id::BOOL8) {
      fixed_width_column_wrapper<bool> expectedDataWrap2(
        {1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1});
      auto expected_column_view2{static_cast<cudf::column_view const&>(expectedDataWrap2)};

      EXPECT_EQ(inputRows, expected_column_view2.size());
      EXPECT_EQ(inputRows, output_column_view2.size());

      CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_column_view2, output_column_view2);
    } else {
      fixed_width_column_wrapper<TypeParam, int32_t> expectedDataWrap2(
        {0, 3, 6, 9, 12, 15, 18, 1, 4, 7, 10, 13, 16, 19, 2, 5, 8, 11, 14, 17, 20});
      auto expected_column_view2{static_cast<cudf::column_view const&>(expectedDataWrap2)};
      CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_column_view2, output_column_view2);
    }

    std::vector<cudf::size_type> expected_partition_offsets{0, 7, 14, 21};
    EXPECT_EQ(static_cast<std::size_t>(num_partitions + 1), expected_partition_offsets.size());

    EXPECT_EQ(expected_partition_offsets, result.second);
  }

  // expected:
  // indxs:   [2,5,8,11,14,17,20,0,3,6,9,12,15,18,1,4,7,10,13,16,19],
  // strings: ["c","f","i","l","o","r","u","a","d","g","j","m","p","s","b","e","h","k","n","q","t"],
  // bools:   [1,0,1,0,1,0,1,1,0,1,0,1,0,1,0,1,0,1,0,1,0],
  // nulls:   [1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
  // offsets: [0,7,14]
  //
  start_partition = 1;
  {
    std::pair<std::unique_ptr<cudf::table>, std::vector<cudf::size_type>> result;
    EXPECT_NO_THROW(result = cudf::round_robin_partition(rr_view, num_partitions, start_partition));

    auto p_outputTable = std::move(result.first);

    auto output_column_view1{p_outputTable->view().column(0)};
    auto output_column_view2{p_outputTable->view().column(1)};

    strings_column_wrapper expectedDataWrap1(
      {"c", "f", "i", "l", "o", "r", "u", "a", "d", "g", "j",
       "m", "p", "s", "b", "e", "h", "k", "n", "q", "t"},
      {true, true, true, true, true, true, false, true, true, true, true,
       true, true, true, true, true, true, true,  true, true, true});

    auto expected_column_view1{static_cast<cudf::column_view const&>(expectedDataWrap1)};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_column_view1, output_column_view1);

    if (cudf::type_to_id<TypeParam>() == cudf::type_id::BOOL8) {
      fixed_width_column_wrapper<bool> expectedDataWrap2(
        {1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0});
      auto expected_column_view2{static_cast<cudf::column_view const&>(expectedDataWrap2)};

      EXPECT_EQ(inputRows, expected_column_view2.size());
      EXPECT_EQ(inputRows, output_column_view2.size());

      CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_column_view2, output_column_view2);
    } else {
      fixed_width_column_wrapper<TypeParam, int32_t> expectedDataWrap2(
        {2, 5, 8, 11, 14, 17, 20, 0, 3, 6, 9, 12, 15, 18, 1, 4, 7, 10, 13, 16, 19});
      auto expected_column_view2{static_cast<cudf::column_view const&>(expectedDataWrap2)};
      CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_column_view2, output_column_view2);
    }

    std::vector<cudf::size_type> expected_partition_offsets{0, 7, 14, 21};
    EXPECT_EQ(static_cast<std::size_t>(num_partitions + 1), expected_partition_offsets.size());

    EXPECT_EQ(expected_partition_offsets, result.second);
  }
}

TYPED_TEST(RoundRobinTest, RoundRobinSinglePartition)
{
  strings_column_wrapper rrColWrap1(
    {"a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k"},
    {true, true, true, true, true, true, true, true, true, true, false});

  cudf::size_type inputRows = static_cast<cudf::column_view const&>(rrColWrap1).size();

  auto sequence_l = cudf::detail::make_counting_transform_iterator(
    0, [bool8 = (cudf::type_to_id<TypeParam>() == cudf::type_id::BOOL8)](auto row) {
      return bool8 ? static_cast<decltype(row)>(row % 2 == 0) : row;
    });

  cudf::test::fixed_width_column_wrapper<TypeParam, typename decltype(sequence_l)::value_type>
    rrColWrap2(sequence_l, sequence_l + inputRows);

  cudf::table_view rr_view{{rrColWrap1, rrColWrap2}};

  cudf::size_type num_partitions  = 1;
  cudf::size_type start_partition = 0;
  std::pair<std::unique_ptr<cudf::table>, std::vector<cudf::size_type>> result;
  EXPECT_NO_THROW(result = cudf::round_robin_partition(rr_view, num_partitions, start_partition));

  auto p_outputTable = std::move(result.first);

  auto output_column_view1{p_outputTable->view().column(0)};
  auto output_column_view2{p_outputTable->view().column(1)};

  strings_column_wrapper expectedDataWrap1(
    {"a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k"},
    {true, true, true, true, true, true, true, true, true, true, false});

  auto expected_column_view1{static_cast<cudf::column_view const&>(expectedDataWrap1)};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_column_view1, output_column_view1);

  if (cudf::type_to_id<TypeParam>() == cudf::type_id::BOOL8) {
    fixed_width_column_wrapper<bool> expectedDataWrap2({1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1});
    auto expected_column_view2{static_cast<cudf::column_view const&>(expectedDataWrap2)};

    EXPECT_EQ(inputRows, expected_column_view2.size());
    EXPECT_EQ(inputRows, output_column_view2.size());

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_column_view2, output_column_view2);
  } else {
    fixed_width_column_wrapper<TypeParam, int32_t> expectedDataWrap2(
      {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
    auto expected_column_view2{static_cast<cudf::column_view const&>(expectedDataWrap2)};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_column_view2, output_column_view2);
  }

  std::vector<cudf::size_type> expected_partition_offsets{0, 11};
  EXPECT_EQ(static_cast<std::size_t>(num_partitions + 1), expected_partition_offsets.size());
  EXPECT_EQ(expected_partition_offsets, result.second);
}

TYPED_TEST(RoundRobinTest, RoundRobinIncorrectNumPartitions)
{
  strings_column_wrapper rrColWrap1(
    {"a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k"},
    {true, true, true, true, true, true, true, true, true, true, false});

  cudf::size_type inputRows = static_cast<cudf::column_view const&>(rrColWrap1).size();

  auto sequence_l = cudf::detail::make_counting_transform_iterator(0, [](auto row) {
    if (cudf::type_to_id<TypeParam>() == cudf::type_id::BOOL8) {
      return (row % 2 == 0) ? 1 : 0;
    } else
      return row;
  });

  cudf::test::fixed_width_column_wrapper<TypeParam, typename decltype(sequence_l)::value_type>
    rrColWrap2(sequence_l, sequence_l + inputRows);

  cudf::table_view rr_view{{rrColWrap1, rrColWrap2}};

  cudf::size_type num_partitions  = 0;
  cudf::size_type start_partition = 0;

  EXPECT_THROW(cudf::round_robin_partition(rr_view, num_partitions, start_partition),
               cudf::logic_error);
}

TYPED_TEST(RoundRobinTest, RoundRobinIncorrectStartPartition)
{
  strings_column_wrapper rrColWrap1(
    {"a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k"},
    {true, true, true, true, true, true, true, true, true, true, false});

  cudf::size_type inputRows = static_cast<cudf::column_view const&>(rrColWrap1).size();

  auto sequence_l = cudf::detail::make_counting_transform_iterator(0, [](auto row) {
    if (cudf::type_to_id<TypeParam>() == cudf::type_id::BOOL8) {
      return (row % 2 == 0) ? 1 : 0;
    } else
      return row;
  });

  cudf::test::fixed_width_column_wrapper<TypeParam, typename decltype(sequence_l)::value_type>
    rrColWrap2(sequence_l, sequence_l + inputRows);

  cudf::table_view rr_view{{rrColWrap1, rrColWrap2}};

  cudf::size_type num_partitions  = 4;
  cudf::size_type start_partition = 5;

  EXPECT_THROW(cudf::round_robin_partition(rr_view, num_partitions, start_partition),
               cudf::logic_error);
}
