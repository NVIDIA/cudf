/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/table_utilities.hpp>
#include <cudf_test/testing_main.hpp>

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/dictionary/encode.hpp>
#include <cudf/join/hash_join.hpp>
#include <cudf/join/join.hpp>
#include <cudf/join/sort_merge_join.hpp>
#include <cudf/sorting.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/mr/statistics_resource_adaptor.hpp>

#include <algorithm>
#include <future>
#include <iterator>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

namespace {
template <typename T>
using column_wrapper                = cudf::test::fixed_width_column_wrapper<T>;
using strcol_wrapper                = cudf::test::strings_column_wrapper;
using CVector                       = std::vector<std::unique_ptr<cudf::column>>;
using Table                         = cudf::table;
constexpr cudf::size_type NoneValue = cudf::JoinNoMatch;
enum class algorithm { HASH, HASH_PARTITIONED, SORT_MERGE, MERGE };

void expect_match_counts_equal(rmm::device_uvector<cudf::size_type> const& actual_counts,
                               std::vector<cudf::size_type> const& expected_counts,
                               rmm::cuda_stream_view stream)
{
  auto const host_actual_counts = cudf::detail::make_host_vector_async(actual_counts, stream);
  stream.synchronize();

  ASSERT_EQ(host_actual_counts.size(), expected_counts.size());
  EXPECT_TRUE(
    std::equal(host_actual_counts.begin(), host_actual_counts.end(), expected_counts.begin()))
    << "Match counts do not equal expected counts";
}

using JoinResult = std::pair<std::unique_ptr<rmm::device_uvector<cudf::size_type>>,
                             std::unique_ptr<rmm::device_uvector<cudf::size_type>>>;

template <cudf::out_of_bounds_policy oob_policy = cudf::out_of_bounds_policy::DONT_CHECK,
          typename Func>
std::unique_ptr<cudf::table> join_and_gather(
  Func&& join_impl,
  cudf::table_view const& left_input,
  cudf::table_view const& right_input,
  std::vector<cudf::size_type> const& left_on,
  std::vector<cudf::size_type> const& right_on,
  cudf::null_equality compare_nulls,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref())
{
  auto left_selected  = left_input.select(left_on);
  auto right_selected = right_input.select(right_on);
  auto const [left_join_indices, right_join_indices] =
    join_impl(left_selected, right_selected, compare_nulls, stream, mr);

  auto left_indices_span  = cudf::device_span<cudf::size_type const>{*left_join_indices};
  auto right_indices_span = cudf::device_span<cudf::size_type const>{*right_join_indices};

  auto left_indices_col  = cudf::column_view{left_indices_span};
  auto right_indices_col = cudf::column_view{right_indices_span};

  auto left_result  = cudf::gather(left_input, left_indices_col, oob_policy);
  auto right_result = cudf::gather(right_input, right_indices_col, oob_policy);

  auto joined_cols = left_result->release();
  auto right_cols  = right_result->release();
  joined_cols.insert(joined_cols.end(),
                     std::make_move_iterator(right_cols.begin()),
                     std::make_move_iterator(right_cols.end()));
  return std::make_unique<cudf::table>(std::move(joined_cols));
}

std::unique_ptr<cudf::table> inner_join(
  cudf::table_view const& left_input,
  cudf::table_view const& right_input,
  std::vector<cudf::size_type> const& left_on,
  std::vector<cudf::size_type> const& right_on,
  cudf::null_equality compare_nulls = cudf::null_equality::EQUAL,
  algorithm algo                    = algorithm::HASH)
{
  if (algo == algorithm::SORT_MERGE) {
    return join_and_gather(
      [](cudf::table_view const& left,
         cudf::table_view const& right,
         cudf::null_equality compare_nulls,
         rmm::cuda_stream_view stream,
         rmm::device_async_resource_ref mr) {
        cudf::sort_merge_join obj(right, cudf::sorted::NO, compare_nulls, stream);
        return obj.inner_join(left, stream, mr);
      },
      left_input,
      right_input,
      left_on,
      right_on,
      compare_nulls);
  } else if (algo == algorithm::MERGE) {
    return join_and_gather(
      [](cudf::table_view const& left,
         cudf::table_view const& right,
         cudf::null_equality compare_nulls,
         rmm::cuda_stream_view stream,
         rmm::device_async_resource_ref mr) {
        cudf::sort_merge_join obj(right, cudf::sorted::YES, compare_nulls, stream);
        return obj.inner_join(left, stream, mr);
      },
      left_input,
      right_input,
      left_on,
      right_on,
      compare_nulls);
  } else if (algo == algorithm::HASH_PARTITIONED) {
    return join_and_gather(
      [](cudf::table_view const& left,
         cudf::table_view const& right,
         cudf::null_equality compare_nulls,
         rmm::cuda_stream_view stream,
         rmm::device_async_resource_ref mr) {
        cudf::hash_join hash_joiner(right, compare_nulls, stream);
        auto match_ctx = hash_joiner.inner_join_match_context(left, stream, mr);
        auto part_ctx  = cudf::join_partition_context{
          std::make_unique<cudf::join_match_context>(std::move(match_ctx)), 0, left.num_rows()};
        return hash_joiner.partitioned_inner_join(part_ctx, stream, mr);
      },
      left_input,
      right_input,
      left_on,
      right_on,
      compare_nulls);
  }
  return join_and_gather(
    [](cudf::table_view const& left,
       cudf::table_view const& right,
       cudf::null_equality compare_nulls,
       rmm::cuda_stream_view stream,
       rmm::device_async_resource_ref mr) {
      return cudf::inner_join(left, right, compare_nulls, stream, mr);
    },
    left_input,
    right_input,
    left_on,
    right_on,
    compare_nulls);
}

std::vector<cudf::size_type> inner_join_size_per_row(
  cudf::table_view const& left_input,
  cudf::table_view const& right_input,
  std::vector<cudf::size_type> const& left_on,
  std::vector<cudf::size_type> const& right_on,
  cudf::null_equality compare_nulls = cudf::null_equality::EQUAL,
  algorithm algo                    = algorithm::HASH)
{
  // TODO: implement for HASH algo
  auto left_selected  = left_input.select(left_on);
  auto right_selected = right_input.select(right_on);
  auto stream         = cudf::get_default_stream();
  cudf::sort_merge_join obj(right_selected, cudf::sorted::NO, compare_nulls, stream);
  auto per_row_counts =
    obj.inner_join_match_context(left_selected, stream, cudf::get_current_device_resource_ref());

  return cudf::detail::make_std_vector<cudf::size_type>(*per_row_counts->_match_counts, stream);
}

std::unique_ptr<cudf::table> left_join(
  cudf::table_view const& left_input,
  cudf::table_view const& right_input,
  std::vector<cudf::size_type> const& left_on,
  std::vector<cudf::size_type> const& right_on,
  cudf::null_equality compare_nulls = cudf::null_equality::EQUAL,
  algorithm algo                    = algorithm::HASH)
{
  if (algo == algorithm::SORT_MERGE) {
    return join_and_gather<cudf::out_of_bounds_policy::NULLIFY>(
      [](cudf::table_view const& left,
         cudf::table_view const& right,
         cudf::null_equality compare_nulls,
         rmm::cuda_stream_view stream,
         rmm::device_async_resource_ref mr) {
        cudf::sort_merge_join obj(right, cudf::sorted::NO, compare_nulls, stream);
        return obj.left_join(left, stream, mr);
      },
      left_input,
      right_input,
      left_on,
      right_on,
      compare_nulls);
  } else if (algo == algorithm::MERGE) {
    return join_and_gather<cudf::out_of_bounds_policy::NULLIFY>(
      [](cudf::table_view const& left,
         cudf::table_view const& right,
         cudf::null_equality compare_nulls,
         rmm::cuda_stream_view stream,
         rmm::device_async_resource_ref mr) {
        cudf::sort_merge_join obj(right, cudf::sorted::YES, compare_nulls, stream);
        return obj.left_join(left, stream, mr);
      },
      left_input,
      right_input,
      left_on,
      right_on,
      compare_nulls);
  } else if (algo == algorithm::HASH_PARTITIONED) {
    return join_and_gather<cudf::out_of_bounds_policy::NULLIFY>(
      [](cudf::table_view const& left,
         cudf::table_view const& right,
         cudf::null_equality compare_nulls,
         rmm::cuda_stream_view stream,
         rmm::device_async_resource_ref mr) {
        cudf::hash_join hash_joiner(right, compare_nulls, stream);
        auto match_ctx = hash_joiner.left_join_match_context(left, stream, mr);
        auto part_ctx  = cudf::join_partition_context{
          std::make_unique<cudf::join_match_context>(std::move(match_ctx)), 0, left.num_rows()};
        return hash_joiner.partitioned_left_join(part_ctx, stream, mr);
      },
      left_input,
      right_input,
      left_on,
      right_on,
      compare_nulls);
  }
  return join_and_gather<cudf::out_of_bounds_policy::NULLIFY>(
    [](cudf::table_view const& left,
       cudf::table_view const& right,
       cudf::null_equality compare_nulls,
       rmm::cuda_stream_view stream,
       rmm::device_async_resource_ref mr) {
      return cudf::left_join(left, right, compare_nulls, stream, mr);
    },
    left_input,
    right_input,
    left_on,
    right_on,
    compare_nulls);
}

std::unique_ptr<cudf::table> full_join(
  cudf::table_view const& full_input,
  cudf::table_view const& right_input,
  std::vector<cudf::size_type> const& full_on,
  std::vector<cudf::size_type> const& right_on,
  cudf::null_equality compare_nulls = cudf::null_equality::EQUAL,
  algorithm algo                    = algorithm::HASH)
{
  if (algo == algorithm::HASH_PARTITIONED) {
    return join_and_gather<cudf::out_of_bounds_policy::NULLIFY>(
      [](cudf::table_view const& left,
         cudf::table_view const& right,
         cudf::null_equality compare_nulls,
         rmm::cuda_stream_view stream,
         rmm::device_async_resource_ref mr) {
        cudf::hash_join hash_joiner(right, compare_nulls, stream);
        auto match_ctx = hash_joiner.full_join_match_context(left, stream, mr);
        auto part_ctx  = cudf::join_partition_context{
          std::make_unique<cudf::join_match_context>(std::move(match_ctx)), 0, left.num_rows()};
        auto [left_idx, right_idx] = hash_joiner.partitioned_full_join(part_ctx, stream, mr);

        std::vector<cudf::device_span<cudf::size_type const>> left_partials{
          cudf::device_span<cudf::size_type const>{left_idx->data(), left_idx->size()}};
        std::vector<cudf::device_span<cudf::size_type const>> right_partials{
          cudf::device_span<cudf::size_type const>{right_idx->data(), right_idx->size()}};
        return cudf::hash_join::finalize_partitioned_full_join(
          left_partials, right_partials, left.num_rows(), right.num_rows(), stream, mr);
      },
      full_input,
      right_input,
      full_on,
      right_on,
      compare_nulls);
  }
  return join_and_gather<cudf::out_of_bounds_policy::NULLIFY>(
    [](cudf::table_view const& left,
       cudf::table_view const& right,
       cudf::null_equality compare_nulls,
       rmm::cuda_stream_view stream,
       rmm::device_async_resource_ref mr) {
      return cudf::full_join(left, right, compare_nulls, stream, mr);
    },
    full_input,
    right_input,
    full_on,
    right_on,
    compare_nulls);
}

}  // anonymous namespace

struct JoinTest : public cudf::test::BaseFixture {
  std::pair<std::unique_ptr<cudf::table>, std::unique_ptr<cudf::table>> gather_maps_as_tables(
    cudf::column_view const& expected_left_map,
    cudf::column_view const& expected_right_map,
    std::pair<std::unique_ptr<rmm::device_uvector<cudf::size_type>>,
              std::unique_ptr<rmm::device_uvector<cudf::size_type>>> const& result)
  {
    auto result_table =
      cudf::table_view({cudf::column_view{cudf::data_type{cudf::type_id::INT32},
                                          static_cast<cudf::size_type>(result.first->size()),
                                          result.first->data(),
                                          nullptr,
                                          0},
                        cudf::column_view{cudf::data_type{cudf::type_id::INT32},
                                          static_cast<cudf::size_type>(result.second->size()),
                                          result.second->data(),
                                          nullptr,
                                          0}});
    auto result_sort_order = cudf::sorted_order(result_table);
    auto sorted_result     = cudf::gather(result_table, *result_sort_order);

    cudf::table_view gold({expected_left_map, expected_right_map});
    auto gold_sort_order = cudf::sorted_order(gold);
    auto sorted_gold     = cudf::gather(gold, *gold_sort_order);

    return std::pair(std::move(sorted_gold), std::move(sorted_result));
  }
};

TEST_F(JoinTest, InvalidLoadFactor)
{
  column_wrapper<int32_t> col0_0{{3, 1, 2, 0, 3}};
  strcol_wrapper col0_1({"s0", "s1", "s2", "s4", "s1"});

  CVector cols0;
  cols0.push_back(col0_0.release());
  cols0.push_back(col0_1.release());

  Table t0(std::move(cols0));

  // Test load factor of -0.1
  EXPECT_THROW(cudf::hash_join(t0, cudf::nullable_join::NO, cudf::null_equality::EQUAL, -0.1),
               std::invalid_argument);
  // Test load factor of 0
  EXPECT_THROW(cudf::hash_join(t0, cudf::nullable_join::NO, cudf::null_equality::EQUAL, 0.0),
               std::invalid_argument);
  // Test load factor > 1
  EXPECT_THROW(cudf::hash_join(t0, cudf::nullable_join::NO, cudf::null_equality::EQUAL, 1.5),
               std::invalid_argument);
}

struct JoinParameterizedTest : public JoinTest, public testing::WithParamInterface<algorithm> {};
struct JoinParameterizedTestSortedInput : public JoinTest,
                                          public testing::WithParamInterface<algorithm> {};

// Parametrize qualifying join tests for supported algorithms
INSTANTIATE_TEST_CASE_P(InnerJoinParameterizedTest,
                        JoinParameterizedTest,
                        ::testing::Values(algorithm::HASH,
                                          algorithm::HASH_PARTITIONED,
                                          algorithm::SORT_MERGE));

INSTANTIATE_TEST_CASE_P(InnerJoinParameterizedTestSortedInput,
                        JoinParameterizedTestSortedInput,
                        ::testing::Values(algorithm::HASH,
                                          algorithm::HASH_PARTITIONED,
                                          algorithm::MERGE));

TEST_P(JoinParameterizedTestSortedInput, SortedKeys)
{
  auto algo = GetParam();
  column_wrapper<int32_t> col0_0{{0, 1, 2, 2, 3}};
  strcol_wrapper col0_1({"s4", "s1", "s0", "s0", "s1"});
  column_wrapper<int32_t> col0_2{{4, 1, 2, 1, 0}};

  column_wrapper<int32_t> col1_0{{0, 2, 2, 3, 4}};
  strcol_wrapper col1_1({"s1", "s1", "s0", "s1", "s2"});
  column_wrapper<int32_t> col1_2{{1, 1, 0, 1, 2}};

  CVector cols0, cols1;
  cols0.push_back(col0_0.release());
  cols0.push_back(col0_1.release());
  cols0.push_back(col0_2.release());
  cols1.push_back(col1_0.release());
  cols1.push_back(col1_1.release());
  cols1.push_back(col1_2.release());

  Table t0(std::move(cols0));
  Table t1(std::move(cols1));

  auto result            = inner_join(t0, t1, {0}, {0}, cudf::null_equality::EQUAL, algo);
  auto result_sort_order = cudf::sorted_order(result->view());
  auto sorted_result     = cudf::gather(result->view(), *result_sort_order);

  column_wrapper<int32_t> col_gold_0{{3, 2, 2, 0, 2, 2}};
  strcol_wrapper col_gold_1({"s1", "s0", "s0", "s4", "s0", "s0"});
  column_wrapper<int32_t> col_gold_2{{0, 2, 2, 4, 1, 1}};
  column_wrapper<int32_t> col_gold_3{{3, 2, 2, 0, 2, 2}};
  strcol_wrapper col_gold_4({"s1", "s1", "s0", "s1", "s1", "s0"});
  column_wrapper<int32_t> col_gold_5{{1, 1, 0, 1, 1, 0}};
  CVector cols_gold;
  cols_gold.push_back(col_gold_0.release());
  cols_gold.push_back(col_gold_1.release());
  cols_gold.push_back(col_gold_2.release());
  cols_gold.push_back(col_gold_3.release());
  cols_gold.push_back(col_gold_4.release());
  cols_gold.push_back(col_gold_5.release());
  Table gold(std::move(cols_gold));

  auto gold_sort_order = cudf::sorted_order(gold.view());
  auto sorted_gold     = cudf::gather(gold.view(), *gold_sort_order);
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*sorted_gold, *sorted_result);
}

TEST_P(JoinParameterizedTest, InvalidInput)
{
  auto algo                  = GetParam();
  auto const left_first_col  = cudf::test::fixed_width_column_wrapper<int32_t>{1197};
  auto const left_second_col = cudf::test::strings_column_wrapper{"201812"};

  auto const right_first_col  = cudf::test::fixed_width_column_wrapper<int32_t>{1197};
  auto const right_second_col = cudf::test::strings_column_wrapper{"201812"};
  auto const right_third_col  = cudf::test::fixed_width_column_wrapper<int64_t>{2550000371};

  cudf::table_view left({left_first_col, left_second_col});
  cudf::table_view right({right_first_col, right_second_col, right_third_col});

  EXPECT_THROW(inner_join(left, right, {0, 1}, {0, 1, 2}, cudf::null_equality::EQUAL, algo),
               std::invalid_argument);

  EXPECT_THROW(inner_join(left, right, {}, {0, 1, 2}, cudf::null_equality::EQUAL, algo),
               std::invalid_argument);
}

TEST_P(JoinParameterizedTest, EmptySentinelRepro)
{
  auto algo = GetParam();
  // This test reproduced an implementation specific behavior where the combination of these
  // particular values ended up hashing to the empty key sentinel value used by the hash join
  // This test may no longer be relevant if the implementation ever changes.
  auto const left_first_col  = cudf::test::fixed_width_column_wrapper<int32_t>{1197};
  auto const left_second_col = cudf::test::strings_column_wrapper{"201812"};
  auto const left_third_col  = cudf::test::fixed_width_column_wrapper<int64_t>{2550000371};

  auto const right_first_col  = cudf::test::fixed_width_column_wrapper<int32_t>{1197};
  auto const right_second_col = cudf::test::strings_column_wrapper{"201812"};
  auto const right_third_col  = cudf::test::fixed_width_column_wrapper<int64_t>{2550000371};

  cudf::table_view left({left_first_col, left_second_col, left_third_col});
  cudf::table_view right({right_first_col, right_second_col, right_third_col});

  auto result = inner_join(left, right, {0, 1, 2}, {0, 1, 2}, cudf::null_equality::EQUAL, algo);

  EXPECT_EQ(result->num_rows(), 1);
}

TEST_P(JoinParameterizedTest, LeftJoinNoNullsWithNoCommon)
{
  auto algo = GetParam();
  column_wrapper<int32_t> col0_0{{3, 1, 2, 0, 3}};
  strcol_wrapper col0_1({"s0", "s1", "s2", "s4", "s1"});
  column_wrapper<int32_t> col0_2{{0, 1, 2, 4, 1}};

  column_wrapper<int32_t> col1_0{{2, 2, 0, 4, 3}};
  strcol_wrapper col1_1{{"s1", "s0", "s1", "s2", "s1"}};
  column_wrapper<int32_t> col1_2{{1, 0, 1, 2, 1}};

  CVector cols0, cols1;
  cols0.push_back(col0_0.release());
  cols0.push_back(col0_1.release());
  cols0.push_back(col0_2.release());
  cols1.push_back(col1_0.release());
  cols1.push_back(col1_1.release());
  cols1.push_back(col1_2.release());

  Table t0(std::move(cols0));
  Table t1(std::move(cols1));

  auto result            = left_join(t0, t1, {0}, {0}, cudf::null_equality::EQUAL, algo);
  auto result_sort_order = cudf::sorted_order(result->view());
  auto sorted_result     = cudf::gather(result->view(), *result_sort_order);

  column_wrapper<int32_t> col_gold_0{{3, 1, 2, 2, 0, 3}, {true, true, true, true, true, true}};
  strcol_wrapper col_gold_1({"s0", "s1", "s2", "s2", "s4", "s1"},
                            {true, true, true, true, true, true});
  column_wrapper<int32_t> col_gold_2{{0, 1, 2, 2, 4, 1}, {true, true, true, true, true, true}};
  column_wrapper<int32_t> col_gold_3{{3, -1, 2, 2, 0, 3}, {true, false, true, true, true, true}};
  strcol_wrapper col_gold_4({"s1", "", "s1", "s0", "s1", "s1"},
                            {true, false, true, true, true, true});
  column_wrapper<int32_t> col_gold_5{{1, -1, 1, 0, 1, 1}, {true, false, true, true, true, true}};
  CVector cols_gold;
  cols_gold.push_back(col_gold_0.release());
  cols_gold.push_back(col_gold_1.release());
  cols_gold.push_back(col_gold_2.release());
  cols_gold.push_back(col_gold_3.release());
  cols_gold.push_back(col_gold_4.release());
  cols_gold.push_back(col_gold_5.release());
  Table gold(std::move(cols_gold));

  auto gold_sort_order = cudf::sorted_order(gold.view());
  auto sorted_gold     = cudf::gather(gold.view(), *gold_sort_order);

  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*sorted_gold, *sorted_result);
}

TEST_F(JoinTest, FullJoinNoNulls)
{
  column_wrapper<int32_t> col0_0{{3, 1, 2, 0, 3}};
  strcol_wrapper col0_1({"s0", "s1", "s2", "s4", "s1"});
  column_wrapper<int32_t> col0_2{{0, 1, 2, 4, 1}};

  column_wrapper<int32_t> col1_0{{2, 2, 0, 4, 3}};
  strcol_wrapper col1_1{{"s1", "s0", "s1", "s2", "s1"}};
  column_wrapper<int32_t> col1_2{{1, 0, 1, 2, 1}};

  CVector cols0, cols1;
  cols0.push_back(col0_0.release());
  cols0.push_back(col0_1.release());
  cols0.push_back(col0_2.release());
  cols1.push_back(col1_0.release());
  cols1.push_back(col1_1.release());
  cols1.push_back(col1_2.release());

  Table t0(std::move(cols0));
  Table t1(std::move(cols1));

  auto result            = full_join(t0, t1, {0, 1}, {0, 1});
  auto result_sort_order = cudf::sorted_order(result->view());
  auto sorted_result     = cudf::gather(result->view(), *result_sort_order);

  column_wrapper<int32_t> col_gold_0{{3, 1, 2, 0, 3, -1, -1, -1, -1},
                                     {true, true, true, true, true, false, false, false, false}};
  strcol_wrapper col_gold_1({"s0", "s1", "s2", "s4", "s1", "", "", "", ""},
                            {true, true, true, true, true, false, false, false, false});
  column_wrapper<int32_t> col_gold_2{{0, 1, 2, 4, 1, -1, -1, -1, -1},
                                     {true, true, true, true, true, false, false, false, false}};
  column_wrapper<int32_t> col_gold_3{{-1, -1, -1, -1, 3, 2, 2, 0, 4},
                                     {false, false, false, false, true, true, true, true, true}};
  strcol_wrapper col_gold_4({"", "", "", "", "s1", "s1", "s0", "s1", "s2"},
                            {false, false, false, false, true, true, true, true, true});
  column_wrapper<int32_t> col_gold_5{{-1, -1, -1, -1, 1, 1, 0, 1, 2},
                                     {false, false, false, false, true, true, true, true, true}};

  CVector cols_gold;
  cols_gold.push_back(col_gold_0.release());
  cols_gold.push_back(col_gold_1.release());
  cols_gold.push_back(col_gold_2.release());
  cols_gold.push_back(col_gold_3.release());
  cols_gold.push_back(col_gold_4.release());
  cols_gold.push_back(col_gold_5.release());

  Table gold(std::move(cols_gold));

  auto gold_sort_order = cudf::sorted_order(gold.view());
  auto sorted_gold     = cudf::gather(gold.view(), *gold_sort_order);
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*sorted_gold, *sorted_result);
}

TEST_F(JoinTest, FullJoinWithNulls)
{
  column_wrapper<int32_t> col0_0{{3, 1, 2, 0, 3}};
  strcol_wrapper col0_1({"s0", "s1", "s2", "s4", "s1"});
  column_wrapper<int32_t> col0_2{{0, 1, 2, 4, 1}};

  column_wrapper<int32_t> col1_0{{2, 2, 0, 4, 3}, {true, true, true, false, true}};
  strcol_wrapper col1_1{{"s1", "s0", "s1", "s2", "s1"}};
  column_wrapper<int32_t> col1_2{{1, 0, 1, 2, 1}};

  CVector cols0, cols1;
  cols0.push_back(col0_0.release());
  cols0.push_back(col0_1.release());
  cols0.push_back(col0_2.release());
  cols1.push_back(col1_0.release());
  cols1.push_back(col1_1.release());
  cols1.push_back(col1_2.release());

  Table t0(std::move(cols0));
  Table t1(std::move(cols1));

  auto result            = full_join(t0, t1, {0, 1}, {0, 1});
  auto result_sort_order = cudf::sorted_order(result->view());
  auto sorted_result     = cudf::gather(result->view(), *result_sort_order);

  column_wrapper<int32_t> col_gold_0{{3, 1, 2, 0, 3, -1, -1, -1, -1},
                                     {true, true, true, true, true, false, false, false, false}};
  strcol_wrapper col_gold_1({"s0", "s1", "s2", "s4", "s1", "", "", "", ""},
                            {true, true, true, true, true, false, false, false, false});
  column_wrapper<int32_t> col_gold_2{{0, 1, 2, 4, 1, -1, -1, -1, -1},
                                     {true, true, true, true, true, false, false, false, false}};
  column_wrapper<int32_t> col_gold_3{{-1, -1, -1, -1, 3, 2, 2, 0, 4},
                                     {false, false, false, false, true, true, true, true, false}};
  strcol_wrapper col_gold_4({"", "", "", "", "s1", "s1", "s0", "s1", "s2"},
                            {false, false, false, false, true, true, true, true, true});
  column_wrapper<int32_t> col_gold_5{{-1, -1, -1, -1, 1, 1, 0, 1, 2},
                                     {false, false, false, false, true, true, true, true, true}};

  CVector cols_gold;
  cols_gold.push_back(col_gold_0.release());
  cols_gold.push_back(col_gold_1.release());
  cols_gold.push_back(col_gold_2.release());
  cols_gold.push_back(col_gold_3.release());
  cols_gold.push_back(col_gold_4.release());
  cols_gold.push_back(col_gold_5.release());

  Table gold(std::move(cols_gold));

  auto gold_sort_order = cudf::sorted_order(gold.view());
  auto sorted_gold     = cudf::gather(gold.view(), *gold_sort_order);
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*sorted_gold, *sorted_result);
}

TEST_F(JoinTest, FullJoinOnNulls)
{
  // clang-format off
  column_wrapper<int32_t> col0_0{{  3,    1 },
                                 {  true,    false  }};
  strcol_wrapper          col0_1({"s0", "s1" });
  column_wrapper<int32_t> col0_2{{  0,    1 }};

  column_wrapper<int32_t> col1_0{{  2,    5,    3,    7 },
                                 {  true,    true,    true,    false }};
  strcol_wrapper          col1_1({"s1", "s0", "s0", "s1" });
  column_wrapper<int32_t> col1_2{{  1,    4,    2,    8 }};

  CVector cols0, cols1;
  cols0.push_back(col0_0.release());
  cols0.push_back(col0_1.release());
  cols0.push_back(col0_2.release());
  cols1.push_back(col1_0.release());
  cols1.push_back(col1_1.release());
  cols1.push_back(col1_2.release());

  Table t0(std::move(cols0));
  Table t1(std::move(cols1));

  auto result            = full_join(t0, t1, {0, 1}, {0, 1});
  auto result_sort_order = cudf::sorted_order(result->view());
  auto sorted_result     = cudf::gather(result->view(), *result_sort_order);

  column_wrapper<int32_t> col_gold_0{{   3,   -1,   -1,    -1},
                                     {   true,    false,    false,     false}};
  strcol_wrapper          col_gold_1{{ "s0", "s1",  "",    ""},
                                     {   true,    true,    false,     false}};
  column_wrapper<int32_t> col_gold_2{{   0,    1,   -1,    -1},
                                     {   true,    true,    false,     false}};
  column_wrapper<int32_t> col_gold_3{{   3,   -1,    2,     5},
                                     {   true,    false,    true,     true}};
  strcol_wrapper          col_gold_4{{ "s0", "s1", "s1",  "s0"}};
  column_wrapper<int32_t> col_gold_5{{   2,    8,    1,     4}};

  CVector cols_gold;
  cols_gold.push_back(col_gold_0.release());
  cols_gold.push_back(col_gold_1.release());
  cols_gold.push_back(col_gold_2.release());
  cols_gold.push_back(col_gold_3.release());
  cols_gold.push_back(col_gold_4.release());
  cols_gold.push_back(col_gold_5.release());

  Table gold(std::move(cols_gold));

  auto gold_sort_order = cudf::sorted_order(gold.view());
  auto sorted_gold     = cudf::gather(gold.view(), *gold_sort_order);

  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*sorted_gold, *sorted_result);

  // Repeat test with compare_nulls_equal=false,
  // as per SQL standard.

  result            = full_join(t0, t1, {0, 1}, {0, 1}, cudf::null_equality::UNEQUAL);
  result_sort_order = cudf::sorted_order(result->view());
  sorted_result     = cudf::gather(result->view(), *result_sort_order);

  col_gold_0 =               {{   3,   -1,   -1,    -1,   -1},
                              {   true,    false,    false,     false,    false}};
  col_gold_1 = strcol_wrapper{{ "s0", "s1",   "",    "",   ""},
                              {   true,    true,    false,     false,    false}};
  col_gold_2 =               {{   0,    1,   -1,    -1,   -1},
                              {   true,    true,    false,     false,    false}};
  col_gold_3 =               {{   3,   -1,    2,     5,   -1},
                              {   true,    false,    true,     true,    false}};
  col_gold_4 = strcol_wrapper{{ "s0",  "",  "s1",  "s0",  "s1"},
                              {   true,    false,    true,     true,    true}};
  col_gold_5 =               {{   2,   -1,    1,     4,    8},
                              {   true,    false,    true,     true,    true}};

  // clang-format on

  CVector cols_gold_nulls_unequal;
  cols_gold_nulls_unequal.push_back(col_gold_0.release());
  cols_gold_nulls_unequal.push_back(col_gold_1.release());
  cols_gold_nulls_unequal.push_back(col_gold_2.release());
  cols_gold_nulls_unequal.push_back(col_gold_3.release());
  cols_gold_nulls_unequal.push_back(col_gold_4.release());
  cols_gold_nulls_unequal.push_back(col_gold_5.release());

  Table gold_nulls_unequal{std::move(cols_gold_nulls_unequal)};

  gold_sort_order = cudf::sorted_order(gold_nulls_unequal.view());
  sorted_gold     = cudf::gather(gold_nulls_unequal.view(), *gold_sort_order);

  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*sorted_gold, *sorted_result);
}

TEST_P(JoinParameterizedTest, LeftJoinNoNulls)
{
  auto algo = GetParam();
  column_wrapper<int32_t> col0_0({3, 1, 2, 0, 3});
  strcol_wrapper col0_1({"s0", "s1", "s2", "s4", "s1"});
  column_wrapper<int32_t> col0_2({0, 1, 2, 4, 1});

  column_wrapper<int32_t> col1_0({2, 2, 0, 4, 3});
  strcol_wrapper col1_1({"s1", "s0", "s1", "s2", "s1"});
  column_wrapper<int32_t> col1_2({1, 0, 1, 2, 1});

  CVector cols0, cols1;
  cols0.push_back(col0_0.release());
  cols0.push_back(col0_1.release());
  cols0.push_back(col0_2.release());
  cols1.push_back(col1_0.release());
  cols1.push_back(col1_1.release());
  cols1.push_back(col1_2.release());

  Table t0(std::move(cols0));
  Table t1(std::move(cols1));

  auto result            = left_join(t0, t1, {0, 1}, {0, 1}, cudf::null_equality::EQUAL, algo);
  auto result_sort_order = cudf::sorted_order(result->view());
  auto sorted_result     = cudf::gather(result->view(), *result_sort_order);

  column_wrapper<int32_t> col_gold_0({3, 1, 2, 0, 3});
  strcol_wrapper col_gold_1({"s0", "s1", "s2", "s4", "s1"});
  column_wrapper<int32_t> col_gold_2({0, 1, 2, 4, 1});
  column_wrapper<int32_t> col_gold_3{{-1, -1, -1, -1, 3}, {false, false, false, false, true}};
  strcol_wrapper col_gold_4{{"", "", "", "", "s1"}, {false, false, false, false, true}};
  column_wrapper<int32_t> col_gold_5{{-1, -1, -1, -1, 1}, {false, false, false, false, true}};
  CVector cols_gold;
  cols_gold.push_back(col_gold_0.release());
  cols_gold.push_back(col_gold_1.release());
  cols_gold.push_back(col_gold_2.release());
  cols_gold.push_back(col_gold_3.release());
  cols_gold.push_back(col_gold_4.release());
  cols_gold.push_back(col_gold_5.release());
  Table gold(std::move(cols_gold));

  auto gold_sort_order = cudf::sorted_order(gold.view());
  auto sorted_gold     = cudf::gather(gold.view(), *gold_sort_order);
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*sorted_gold, *sorted_result);
}

TEST_P(JoinParameterizedTest, LeftJoinWithNulls)
{
  auto algo = GetParam();
  column_wrapper<int32_t> col0_0{{3, 1, 2, 0, 2}};
  strcol_wrapper col0_1({"s1", "s1", "", "s4", "s0"}, {true, true, false, true, true});
  column_wrapper<int32_t> col0_2{{0, 1, 2, 4, 1}};

  column_wrapper<int32_t> col1_0{{2, 2, 0, 4, 3}};
  strcol_wrapper col1_1({"s1", "s0", "s1", "s2", "s1"});
  column_wrapper<int32_t> col1_2{{1, 0, 1, 2, 1}, {true, false, true, true, true}};

  CVector cols0, cols1;
  cols0.push_back(col0_0.release());
  cols0.push_back(col0_1.release());
  cols0.push_back(col0_2.release());
  cols1.push_back(col1_0.release());
  cols1.push_back(col1_1.release());
  cols1.push_back(col1_2.release());

  Table t0(std::move(cols0));
  Table t1(std::move(cols1));

  auto result            = left_join(t0, t1, {0, 1}, {0, 1}, cudf::null_equality::EQUAL, algo);
  auto result_sort_order = cudf::sorted_order(result->view());
  auto sorted_result     = cudf::gather(result->view(), *result_sort_order);

  column_wrapper<int32_t> col_gold_0{{3, 1, 2, 0, 2}, {true, true, true, true, true}};
  strcol_wrapper col_gold_1({"s1", "s1", "", "s4", "s0"}, {true, true, false, true, true});
  column_wrapper<int32_t> col_gold_2{{0, 1, 2, 4, 1}, {true, true, true, true, true}};
  column_wrapper<int32_t> col_gold_3{{3, -1, -1, -1, 2}, {true, false, false, false, true}};
  strcol_wrapper col_gold_4{{"s1", "", "", "", "s0"}, {true, false, false, false, true}};
  column_wrapper<int32_t> col_gold_5{{1, -1, -1, -1, -1}, {true, false, false, false, false}};

  CVector cols_gold;
  cols_gold.push_back(col_gold_0.release());
  cols_gold.push_back(col_gold_1.release());
  cols_gold.push_back(col_gold_2.release());
  cols_gold.push_back(col_gold_3.release());
  cols_gold.push_back(col_gold_4.release());
  cols_gold.push_back(col_gold_5.release());
  Table gold(std::move(cols_gold));

  auto gold_sort_order = cudf::sorted_order(gold.view());
  auto sorted_gold     = cudf::gather(gold.view(), *gold_sort_order);
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*sorted_gold, *sorted_result);
}

TEST_P(JoinParameterizedTest, LeftJoinWithStructsAndNulls)
{
  auto algo = GetParam();
  column_wrapper<int32_t> col0_0{{3, 1, 2, 0, 2}};
  strcol_wrapper col0_1({"s1", "s1", "", "s4", "s0"}, {true, true, false, true, true});
  column_wrapper<int32_t> col0_2{{0, 1, 2, 4, 1}};
  auto col0_names_col = strcol_wrapper{
    "Samuel Vimes", "Carrot Ironfoundersson", "Detritus", "Samuel Vimes", "Angua von Überwald"};
  auto col0_ages_col = column_wrapper<int32_t>{{48, 27, 351, 31, 25}};

  auto col0_is_human_col =
    column_wrapper<bool>{{true, true, false, false, false}, {true, true, false, true, false}};

  auto col0_3 =
    cudf::test::structs_column_wrapper{{col0_names_col, col0_ages_col, col0_is_human_col}};

  column_wrapper<int32_t> col1_0{{2, 2, 0, 4, 3}};
  strcol_wrapper col1_1({"s1", "s0", "s1", "s2", "s1"});
  column_wrapper<int32_t> col1_2{{1, 0, 1, 2, 1}, {true, false, true, true, true}};
  auto col1_names_col = strcol_wrapper{
    "Samuel Vimes", "Detritus", "Detritus", "Carrot Ironfoundersson", "Angua von Überwald"};
  auto col1_ages_col = column_wrapper<int32_t>{{48, 35, 351, 22, 25}};

  auto col1_is_human_col =
    column_wrapper<bool>{{true, true, false, false, true}, {true, true, false, true, true}};

  auto col1_3 =
    cudf::test::structs_column_wrapper{{col1_names_col, col1_ages_col, col1_is_human_col}};

  CVector cols0, cols1;
  cols0.push_back(col0_0.release());
  cols0.push_back(col0_1.release());
  cols0.push_back(col0_2.release());
  cols0.push_back(col0_3.release());
  cols1.push_back(col1_0.release());
  cols1.push_back(col1_1.release());
  cols1.push_back(col1_2.release());
  cols1.push_back(col1_3.release());

  Table t0(std::move(cols0));
  Table t1(std::move(cols1));

  auto result            = left_join(t0, t1, {3}, {3}, cudf::null_equality::EQUAL, algo);
  auto result_sort_order = cudf::sorted_order(result->view());
  auto sorted_result     = cudf::gather(result->view(), *result_sort_order);

  column_wrapper<int32_t> col_gold_0{{3, 2, 1, 0, 2}, {true, true, true, true, true}};
  strcol_wrapper col_gold_1({"s1", "", "s1", "s4", "s0"}, {true, false, true, true, true});
  column_wrapper<int32_t> col_gold_2{{0, 2, 1, 4, 1}, {true, true, true, true, true}};
  auto col0_gold_names_col = strcol_wrapper{
    "Samuel Vimes", "Detritus", "Carrot Ironfoundersson", "Samuel Vimes", "Angua von Überwald"};
  auto col0_gold_ages_col = column_wrapper<int32_t>{{48, 351, 27, 31, 25}};

  auto col0_gold_is_human_col =
    column_wrapper<bool>{{true, false, true, false, false}, {true, false, true, true, false}};

  auto col_gold_3 = cudf::test::structs_column_wrapper{
    {col0_gold_names_col, col0_gold_ages_col, col0_gold_is_human_col}};

  column_wrapper<int32_t> col_gold_4{{2, 0, -1, -1, -1}, {true, true, false, false, false}};
  strcol_wrapper col_gold_5{{"s1", "s1", "", "", ""}, {true, true, false, false, false}};
  column_wrapper<int32_t> col_gold_6{{1, 1, -1, -1, -1}, {true, true, false, false, false}};
  auto col1_gold_names_col = strcol_wrapper{{
                                              "Samuel Vimes",
                                              "Detritus",
                                              "",
                                              "",
                                              "",
                                            },
                                            {true, true, false, false, false}};
  auto col1_gold_ages_col =
    column_wrapper<int32_t>{{48, 351, -1, -1, -1}, {true, true, false, false, false}};

  auto col1_gold_is_human_col =
    column_wrapper<bool>{{true, false, false, false, false}, {true, false, false, false, false}};

  auto col_gold_7 = cudf::test::structs_column_wrapper{
    {col1_gold_names_col, col1_gold_ages_col, col1_gold_is_human_col},
    {true, true, false, false, false}};

  CVector cols_gold;
  cols_gold.push_back(col_gold_0.release());
  cols_gold.push_back(col_gold_1.release());
  cols_gold.push_back(col_gold_2.release());
  cols_gold.push_back(col_gold_3.release());
  cols_gold.push_back(col_gold_4.release());
  cols_gold.push_back(col_gold_5.release());
  cols_gold.push_back(col_gold_6.release());
  cols_gold.push_back(col_gold_7.release());
  Table gold(std::move(cols_gold));

  auto gold_sort_order = cudf::sorted_order(gold.view());
  auto sorted_gold     = cudf::gather(gold.view(), *gold_sort_order);
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*sorted_gold, *sorted_result);
}

TEST_P(JoinParameterizedTest, LeftJoinOnNulls)
{
  auto algo = GetParam();
  // clang-format off
  column_wrapper<int32_t> col0_0{{  3,    1,    2},
                                 {  true,    false,    true}};
  strcol_wrapper          col0_1({"s0", "s1", "s2" });
  column_wrapper<int32_t> col0_2{{  0,    1,    2 }};

  column_wrapper<int32_t> col1_0{{  2,    5,    3,    7 },
                                 {  true,    true,    true,    false }};
  strcol_wrapper          col1_1({"s1", "s0", "s0", "s1" });
  column_wrapper<int32_t> col1_2{{  1,    4,    2,    8 }};

  CVector cols0, cols1;
  cols0.push_back(col0_0.release());
  cols0.push_back(col0_1.release());
  cols0.push_back(col0_2.release());
  cols1.push_back(col1_0.release());
  cols1.push_back(col1_1.release());
  cols1.push_back(col1_2.release());

  Table t0(std::move(cols0));
  Table t1(std::move(cols1));

  auto result            = left_join(t0, t1, {0, 1}, {0, 1}, cudf::null_equality::EQUAL, algo);
  auto result_sort_order = cudf::sorted_order(result->view());
  auto sorted_result     = cudf::gather(result->view(), *result_sort_order);

  column_wrapper<int32_t> col_gold_0{{   3,    -1,    2},
                                     {   true,     false,    true}};
  strcol_wrapper          col_gold_1({ "s0",  "s1", "s2"},
                                     {   true,     true,    true});
  column_wrapper<int32_t> col_gold_2{{   0,     1,    2},
                                     {   true,     true,    true}};
  column_wrapper<int32_t> col_gold_3{{   3,    -1,   -1},
                                     {   true,     false,    false}};
  strcol_wrapper          col_gold_4({ "s0",  "s1",  ""},
                                     {   true,     true,    false});
  column_wrapper<int32_t> col_gold_5{{   2,     8,   -1},
                                     {   true,     true,    false}};

  CVector cols_gold;
  cols_gold.push_back(col_gold_0.release());
  cols_gold.push_back(col_gold_1.release());
  cols_gold.push_back(col_gold_2.release());
  cols_gold.push_back(col_gold_3.release());
  cols_gold.push_back(col_gold_4.release());
  cols_gold.push_back(col_gold_5.release());
  Table gold(std::move(cols_gold));

  auto gold_sort_order = cudf::sorted_order(gold.view());
  auto sorted_gold     = cudf::gather(gold.view(), *gold_sort_order);

  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*sorted_gold, *sorted_result);

  // Repeat test with compare_nulls_equal=false,
  // as per SQL standard.

  result            = left_join(t0, t1, {0, 1}, {0, 1}, cudf::null_equality::UNEQUAL);
  result_sort_order = cudf::sorted_order(result->view());
  sorted_result     = cudf::gather(result->view(), *result_sort_order);


  col_gold_0 = {{   3,    -1,    2},
                {   true,     false,    true}};
  col_gold_1 = {{ "s0",  "s1", "s2"},
                {   true,     true,    true}};
  col_gold_2 = {{   0,     1,    2},
                {   true,     true,    true}};
  col_gold_3 = {{   3,    -1,   -1},
                {   true,     false,    false}};
  col_gold_4 = {{ "s0",   "",   ""},
                {   true,     false,    false}};
  col_gold_5 = {{   2,    -1,   -1},
                {   true,     false,    false}};

  // clang-format on
  CVector cols_gold_nulls_unequal;
  cols_gold_nulls_unequal.push_back(col_gold_0.release());
  cols_gold_nulls_unequal.push_back(col_gold_1.release());
  cols_gold_nulls_unequal.push_back(col_gold_2.release());
  cols_gold_nulls_unequal.push_back(col_gold_3.release());
  Table gold_nulls_unequal{std::move(cols_gold_nulls_unequal)};

  gold_sort_order = cudf::sorted_order(gold_nulls_unequal.view());
  sorted_gold     = cudf::gather(gold_nulls_unequal.view(), *gold_sort_order);

  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*sorted_gold, *sorted_result);
}

TEST_F(JoinTest, SortMergeInnerJoinSizePerRowNoNulls)
{
  column_wrapper<int32_t> col0_0{{3, 1, 2, 0, 2}};
  strcol_wrapper col0_1({"s1", "s1", "s0", "s4", "s0"});
  column_wrapper<int32_t> col0_2{{0, 1, 2, 4, 1}};

  column_wrapper<int32_t> col1_0{{2, 2, 0, 4, 3}};
  strcol_wrapper col1_1({"s1", "s0", "s1", "s2", "s1"});
  column_wrapper<int32_t> col1_2{{1, 0, 1, 2, 1}};

  CVector cols0, cols1;
  cols0.push_back(col0_0.release());
  cols0.push_back(col0_1.release());
  cols0.push_back(col0_2.release());
  cols1.push_back(col1_0.release());
  cols1.push_back(col1_1.release());
  cols1.push_back(col1_2.release());

  Table t0(std::move(cols0));
  Table t1(std::move(cols1));

  for (auto const eq : {cudf::null_equality::EQUAL, cudf::null_equality::UNEQUAL}) {
    // single column
    {
      auto size_per_row = inner_join_size_per_row(t0, t1, {0}, {0}, eq, algorithm::SORT_MERGE);
      auto expected     = std::vector<cudf::size_type>{1, 0, 2, 1, 2};
      EXPECT_EQ(size_per_row, expected);
    }
    // multi column
    {
      auto result   = inner_join_size_per_row(t0, t1, {0, 1}, {0, 1}, eq, algorithm::SORT_MERGE);
      auto expected = std::vector<cudf::size_type>{1, 0, 1, 0, 1};
      EXPECT_EQ(result, expected);
    }
  }
}

TEST_F(JoinTest, SortMergeInnerJoinMatchContextMemoryResource)
{
  column_wrapper<int32_t> left_key{{3, 1, 2, 0, 2}};
  column_wrapper<int32_t> right_key{{2, 2, 0, 4, 3}};
  auto left   = cudf::table_view{{left_key}};
  auto right  = cudf::table_view{{right_key}};
  auto stream = cudf::get_default_stream();

  auto mr = rmm::mr::statistics_resource_adaptor(cudf::get_current_device_resource_ref());
  cudf::sort_merge_join obj(right, cudf::sorted::NO, cudf::null_equality::EQUAL, stream);
  auto match_context = obj.inner_join_match_context(left, stream, mr);

  EXPECT_GT(mr.get_bytes_counter().peak, 0);
  expect_match_counts_equal(*match_context->_match_counts, {1, 0, 2, 1, 2}, stream);
}

TEST_F(JoinTest, SortMergeInnerJoinSizePerRowWithNulls)
{
  column_wrapper<int32_t> col0_0{{3, 1, 2, 0, 2}};
  strcol_wrapper col0_1({"s1", "s1", "s0", "s4", "s0"}, {true, true, false, true, true});
  column_wrapper<int32_t> col0_2{{0, 1, 2, 4, 1}};

  column_wrapper<int32_t> col1_0{{2, 2, 0, 4, 3}};
  strcol_wrapper col1_1({"s1", "s0", "s1", "s2", "s1"}, {true, false, true, true, true});
  column_wrapper<int32_t> col1_2{{1, 0, 1, 2, 1}, {true, false, true, true, true}};

  CVector cols0, cols1;
  cols0.push_back(col0_0.release());
  cols0.push_back(col0_1.release());
  cols0.push_back(col0_2.release());
  cols1.push_back(col1_0.release());
  cols1.push_back(col1_1.release());
  cols1.push_back(col1_2.release());

  Table t0(std::move(cols0));
  Table t1(std::move(cols1));

  // single column, null_equality::EQUAL
  {
    auto size_per_row =
      inner_join_size_per_row(t0, t1, {0}, {0}, cudf::null_equality::EQUAL, algorithm::SORT_MERGE);
    auto expected = std::vector<cudf::size_type>{1, 0, 2, 1, 2};
    EXPECT_EQ(size_per_row, expected);
  }
  // multi column, null_equality::EQUAL
  {
    auto size_per_row = inner_join_size_per_row(
      t0, t1, {0, 1}, {0, 1}, cudf::null_equality::EQUAL, algorithm::SORT_MERGE);
    auto expected = std::vector<cudf::size_type>{1, 0, 1, 0, 0};
    EXPECT_EQ(size_per_row, expected);
  }
  // single column, null_equality::UNEQUAL
  {
    auto size_per_row = inner_join_size_per_row(
      t0, t1, {0}, {0}, cudf::null_equality::UNEQUAL, algorithm::SORT_MERGE);
    auto expected = std::vector<cudf::size_type>{1, 0, 2, 1, 2};
    EXPECT_EQ(size_per_row, expected);
  }
  // multi column, null_equality::UNEQUAL
  {
    auto size_per_row = inner_join_size_per_row(
      t0, t1, {0, 1}, {0, 1}, cudf::null_equality::UNEQUAL, algorithm::SORT_MERGE);
    auto expected = std::vector<cudf::size_type>{1, 0, 0, 0, 0};
    EXPECT_EQ(size_per_row, expected);
  }
}

TEST_F(JoinTest, PartitionedInnerJoinWithNulls)
{
  column_wrapper<int32_t> col0_0{{3, 1, 2, 0, 2}};
  strcol_wrapper col0_1({"s1", "s1", "s0", "s4", "s0"}, {true, true, false, true, true});
  column_wrapper<int32_t> col0_2{{0, 1, 2, 4, 1}};

  column_wrapper<int32_t> col1_0{{2, 2, 0, 4, 3}};
  strcol_wrapper col1_1({"s1", "s0", "s1", "s2", "s1"}, {true, false, true, true, true});
  column_wrapper<int32_t> col1_2{{1, 0, 1, 2, 1}, {true, false, true, true, true}};

  CVector cols0, cols1;
  cols0.push_back(col0_0.release());
  cols0.push_back(col0_1.release());
  cols0.push_back(col0_2.release());
  cols1.push_back(col1_0.release());
  cols1.push_back(col1_1.release());
  cols1.push_back(col1_2.release());

  Table t0(std::move(cols0));
  Table t1(std::move(cols1));

  auto left_on       = std::vector<cudf::size_type>({0, 1});
  auto right_on      = std::vector<cudf::size_type>({0, 1});
  auto compare_nulls = cudf::null_equality::EQUAL;
  auto expected_result =
    inner_join(t0, t1, left_on, right_on, compare_nulls, algorithm::SORT_MERGE);
  auto expected_result_sort_order = cudf::sorted_order(expected_result->view());
  auto expected_sorted_result = cudf::gather(expected_result->view(), *expected_result_sort_order);

  auto stream = cudf::get_default_stream();

  cudf::sort_merge_join obj(t1.select(right_on), cudf::sorted::NO, compare_nulls, stream);
  auto match_context = obj.inner_join_match_context(
    t0.select(left_on), stream, cudf::get_current_device_resource_ref());
  auto partition_context = cudf::join_partition_context{std::move(match_context), 0, 0};

  auto join_and_gather = [&t0, &t1, &obj, stream](cudf::join_partition_context const& cxt) {
    auto const [left_join_indices, right_join_indices] =
      obj.partitioned_inner_join(cxt, stream, cudf::get_current_device_resource_ref());

    auto left_indices_span  = cudf::device_span<cudf::size_type const>{*left_join_indices};
    auto right_indices_span = cudf::device_span<cudf::size_type const>{*right_join_indices};

    auto left_indices_col  = cudf::column_view{left_indices_span};
    auto right_indices_col = cudf::column_view{right_indices_span};

    auto left_result  = cudf::gather(t0, left_indices_col, cudf::out_of_bounds_policy::DONT_CHECK);
    auto right_result = cudf::gather(t1, right_indices_col, cudf::out_of_bounds_policy::DONT_CHECK);

    auto joined_cols = left_result->release();
    auto right_cols  = right_result->release();
    joined_cols.insert(joined_cols.end(),
                       std::make_move_iterator(right_cols.begin()),
                       std::make_move_iterator(right_cols.end()));
    return std::make_unique<cudf::table>(std::move(joined_cols));
  };

  std::vector<std::unique_ptr<cudf::table>> partial_tables;
  std::vector<cudf::table_view> partial_table_views;
  for (cudf::size_type i = 0; i < t0.num_rows(); i++) {
    partition_context.left_start_idx = i;
    partition_context.left_end_idx   = i + 1;
    partial_tables.push_back(join_and_gather(partition_context));
    partial_table_views.push_back(partial_tables.back()->view());
  }

  auto concatenated_result =
    cudf::concatenate(partial_table_views, stream, cudf::get_current_device_resource_ref());
  auto concatenated_result_sort_order = cudf::sorted_order(concatenated_result->view());
  auto concatenated_sorted_result =
    cudf::gather(concatenated_result->view(), *concatenated_result_sort_order);

  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*expected_sorted_result, *concatenated_sorted_result);
}

TEST_F(JoinTest, PartitionedInnerJoinWithNestedNullsUnequal)
{
  column_wrapper<int32_t> left_child{{10, 20, 30, 40, 50}, {true, false, true, true, true}};
  column_wrapper<int32_t> right_child{{30, 40, 50, 10, 20}, {true, true, true, true, false}};
  auto left_key  = cudf::test::structs_column_wrapper{{left_child}};
  auto right_key = cudf::test::structs_column_wrapper{{right_child}};
  auto left      = cudf::table_view{{left_key}};
  auto right     = cudf::table_view{{right_key}};
  auto stream    = cudf::get_default_stream();
  auto mr        = cudf::get_current_device_resource_ref();

  cudf::sort_merge_join obj(right, cudf::sorted::NO, cudf::null_equality::UNEQUAL, stream);
  auto match_context = obj.inner_join_match_context(left, stream, mr);
  expect_match_counts_equal(*match_context->_match_counts, {1, 0, 1, 1, 1}, stream);
  auto partition_context = cudf::join_partition_context{std::move(match_context), 2, 4};

  auto const [left_indices, right_indices] =
    obj.partitioned_inner_join(partition_context, stream, mr);

  EXPECT_EQ(cudf::detail::make_std_vector<cudf::size_type>(*left_indices, stream),
            std::vector<cudf::size_type>({2, 3}));
  EXPECT_EQ(cudf::detail::make_std_vector<cudf::size_type>(*right_indices, stream),
            std::vector<cudf::size_type>({0, 1}));
}

TEST_F(JoinTest, LeftJoinPreservesNestedNullRowOrderUnequal)
{
  column_wrapper<int32_t> left_child{{10, 20, 30, 40, 50}, {true, false, true, false, true}};
  column_wrapper<int32_t> right_child{{10, 30, 50}};
  auto left_key  = cudf::test::structs_column_wrapper{{left_child}};
  auto right_key = cudf::test::structs_column_wrapper{{right_child}};
  auto left      = cudf::table_view{{left_key}};
  auto right     = cudf::table_view{{right_key}};
  auto stream    = cudf::get_default_stream();
  auto mr        = cudf::get_current_device_resource_ref();

  cudf::sort_merge_join obj(right, cudf::sorted::NO, cudf::null_equality::UNEQUAL, stream);
  auto const [left_indices, right_indices] = obj.left_join(left, stream, mr);

  EXPECT_EQ(cudf::detail::make_std_vector<cudf::size_type>(*left_indices, stream),
            std::vector<cudf::size_type>({0, 1, 2, 3, 4}));
  EXPECT_EQ(cudf::detail::make_std_vector<cudf::size_type>(*right_indices, stream),
            std::vector<cudf::size_type>({0, cudf::JoinNoMatch, 1, cudf::JoinNoMatch, 2}));
}

TEST_P(JoinParameterizedTest, InnerJoinNoNulls)
{
  auto algo = GetParam();
  column_wrapper<int32_t> col0_0{{3, 1, 2, 0, 2}};
  strcol_wrapper col0_1({"s1", "s1", "s0", "s4", "s0"});
  column_wrapper<int32_t> col0_2{{0, 1, 2, 4, 1}};

  column_wrapper<int32_t> col1_0{{2, 2, 0, 4, 3}};
  strcol_wrapper col1_1({"s1", "s0", "s1", "s2", "s1"});
  column_wrapper<int32_t> col1_2{{1, 0, 1, 2, 1}};

  CVector cols0, cols1;
  cols0.push_back(col0_0.release());
  cols0.push_back(col0_1.release());
  cols0.push_back(col0_2.release());
  cols1.push_back(col1_0.release());
  cols1.push_back(col1_1.release());
  cols1.push_back(col1_2.release());

  Table t0(std::move(cols0));
  Table t1(std::move(cols1));

  for (auto const eq : {cudf::null_equality::EQUAL, cudf::null_equality::UNEQUAL}) {
    // single column
    {
      auto result            = inner_join(t0, t1, {0}, {0}, eq, algo);
      auto result_sort_order = cudf::sorted_order(result->view());
      auto sorted_result     = cudf::gather(result->view(), *result_sort_order);

      column_wrapper<int32_t> col_gold_0{{3, 2, 2, 0, 2, 2}};
      strcol_wrapper col_gold_1({"s1", "s0", "s0", "s4", "s0", "s0"});
      column_wrapper<int32_t> col_gold_2{{0, 2, 2, 4, 1, 1}};
      column_wrapper<int32_t> col_gold_3{{3, 2, 2, 0, 2, 2}};
      strcol_wrapper col_gold_4({"s1", "s1", "s0", "s1", "s1", "s0"});
      column_wrapper<int32_t> col_gold_5{{1, 1, 0, 1, 1, 0}};
      CVector cols_gold;
      cols_gold.push_back(col_gold_0.release());
      cols_gold.push_back(col_gold_1.release());
      cols_gold.push_back(col_gold_2.release());
      cols_gold.push_back(col_gold_3.release());
      cols_gold.push_back(col_gold_4.release());
      cols_gold.push_back(col_gold_5.release());
      Table gold(std::move(cols_gold));

      auto gold_sort_order = cudf::sorted_order(gold.view());
      auto sorted_gold     = cudf::gather(gold.view(), *gold_sort_order);
      CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*sorted_gold, *sorted_result);
    }
    // multi column
    {
      auto result            = inner_join(t0, t1, {0, 1}, {0, 1}, eq, algo);
      auto result_sort_order = cudf::sorted_order(result->view());
      auto sorted_result     = cudf::gather(result->view(), *result_sort_order);

      column_wrapper<int32_t> col_gold_0{{3, 2, 2}};
      strcol_wrapper col_gold_1({"s1", "s0", "s0"});
      column_wrapper<int32_t> col_gold_2{{0, 2, 1}};
      column_wrapper<int32_t> col_gold_3{{3, 2, 2}};
      strcol_wrapper col_gold_4({"s1", "s0", "s0"});
      column_wrapper<int32_t> col_gold_5{{1, 0, 0}};
      CVector cols_gold;
      cols_gold.push_back(col_gold_0.release());
      cols_gold.push_back(col_gold_1.release());
      cols_gold.push_back(col_gold_2.release());
      cols_gold.push_back(col_gold_3.release());
      cols_gold.push_back(col_gold_4.release());
      cols_gold.push_back(col_gold_5.release());
      Table gold(std::move(cols_gold));

      auto gold_sort_order = cudf::sorted_order(gold.view());
      auto sorted_gold     = cudf::gather(gold.view(), *gold_sort_order);
      CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*sorted_gold, *sorted_result);
    }
  }
}

TEST_P(JoinParameterizedTest, InnerJoinWithNulls)
{
  auto algo = GetParam();
  column_wrapper<int32_t> col0_0{{3, 1, 2, 0, 2}};
  strcol_wrapper col0_1({"s1", "s1", "s0", "s4", "s0"}, {true, true, false, true, true});
  column_wrapper<int32_t> col0_2{{0, 1, 2, 4, 1}};

  column_wrapper<int32_t> col1_0{{2, 2, 0, 4, 3}};
  strcol_wrapper col1_1({"s1", "s0", "s1", "s2", "s1"});
  column_wrapper<int32_t> col1_2{{1, 0, 1, 2, 1}, {true, false, true, true, true}};

  CVector cols0, cols1;
  cols0.push_back(col0_0.release());
  cols0.push_back(col0_1.release());
  cols0.push_back(col0_2.release());
  cols1.push_back(col1_0.release());
  cols1.push_back(col1_1.release());
  cols1.push_back(col1_2.release());

  Table t0(std::move(cols0));
  Table t1(std::move(cols1));

  auto result            = inner_join(t0, t1, {0, 1}, {0, 1}, cudf::null_equality::EQUAL, algo);
  auto result_sort_order = cudf::sorted_order(result->view());
  auto sorted_result     = cudf::gather(result->view(), *result_sort_order);

  column_wrapper<int32_t> col_gold_0{{3, 2}};
  strcol_wrapper col_gold_1({"s1", "s0"}, {true, true});
  column_wrapper<int32_t> col_gold_2{{0, 1}};
  column_wrapper<int32_t> col_gold_3{{3, 2}};
  strcol_wrapper col_gold_4({"s1", "s0"}, {true, true});
  column_wrapper<int32_t> col_gold_5{{1, -1}, {true, false}};
  CVector cols_gold;
  cols_gold.push_back(col_gold_0.release());
  cols_gold.push_back(col_gold_1.release());
  cols_gold.push_back(col_gold_2.release());
  cols_gold.push_back(col_gold_3.release());
  cols_gold.push_back(col_gold_4.release());
  cols_gold.push_back(col_gold_5.release());
  Table gold(std::move(cols_gold));

  auto gold_sort_order = cudf::sorted_order(gold.view());
  auto sorted_gold     = cudf::gather(gold.view(), *gold_sort_order);
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*sorted_gold, *sorted_result);
}

// TODO: add unequal nulls case here
TEST_P(JoinParameterizedTest, InnerJoinWithStructsAndNulls)
{
  auto algo = GetParam();
  column_wrapper<int32_t> col0_0{{3, 1, 2, 0, 2}};
  strcol_wrapper col0_1({"s1", "s1", "s0", "s4", "s0"}, {true, true, false, true, true});
  column_wrapper<int32_t> col0_2{{0, 1, 2, 4, 1}};
  std::initializer_list<std::string> col0_names = {
    "Samuel Vimes", "Carrot Ironfoundersson", "Detritus", "Samuel Vimes", "Angua von Überwald"};
  auto col0_names_col = strcol_wrapper{col0_names.begin(), col0_names.end()};
  auto col0_ages_col  = column_wrapper<int32_t>{{48, 27, 351, 31, 25}};

  auto col0_is_human_col =
    column_wrapper<bool>{{true, true, false, false, false}, {true, true, false, true, false}};

  auto col0_3 =
    cudf::test::structs_column_wrapper{{col0_names_col, col0_ages_col, col0_is_human_col}};

  column_wrapper<int32_t> col1_0{{2, 2, 0, 4, 3}};
  strcol_wrapper col1_1({"s1", "s0", "s1", "s2", "s1"});
  column_wrapper<int32_t> col1_2{{1, 0, 1, 2, 1}, {true, false, true, true, true}};
  std::initializer_list<std::string> col1_names = {"Carrot Ironfoundersson",
                                                   "Angua von Überwald",
                                                   "Detritus",
                                                   "Carrot Ironfoundersson",
                                                   "Samuel Vimes"};
  auto col1_names_col = strcol_wrapper{col1_names.begin(), col1_names.end()};
  auto col1_ages_col  = column_wrapper<int32_t>{{351, 25, 27, 31, 48}};

  auto col1_is_human_col =
    column_wrapper<bool>{{true, false, false, false, true}, {true, false, false, true, true}};

  auto col1_3 =
    cudf::test::structs_column_wrapper{{col1_names_col, col1_ages_col, col1_is_human_col}};

  CVector cols0, cols1;
  cols0.push_back(col0_0.release());
  cols0.push_back(col0_1.release());
  cols0.push_back(col0_2.release());
  cols0.push_back(col0_3.release());
  cols1.push_back(col1_0.release());
  cols1.push_back(col1_1.release());
  cols1.push_back(col1_2.release());
  cols1.push_back(col1_3.release());

  Table t0(std::move(cols0));
  Table t1(std::move(cols1));

  // Equal nulls
  {
    auto result = inner_join(t0, t1, {0, 1, 3}, {0, 1, 3}, cudf::null_equality::EQUAL, algo);
    auto result_sort_order = cudf::sorted_order(result->view());
    auto sorted_result     = cudf::gather(result->view(), *result_sort_order);

    column_wrapper<int32_t> col_gold_0{{3, 2}};
    strcol_wrapper col_gold_1({"s1", "s0"}, {true, true});
    column_wrapper<int32_t> col_gold_2{{0, 1}};
    auto col_gold_3_names_col = strcol_wrapper{"Samuel Vimes", "Angua von Überwald"};
    auto col_gold_3_ages_col  = column_wrapper<int32_t>{{48, 25}};

    auto col_gold_3_is_human_col = column_wrapper<bool>{{true, false}, {true, false}};

    auto col_gold_3 = cudf::test::structs_column_wrapper{
      {col_gold_3_names_col, col_gold_3_ages_col, col_gold_3_is_human_col}};

    column_wrapper<int32_t> col_gold_4{{3, 2}};
    strcol_wrapper col_gold_5({"s1", "s0"}, {true, true});
    column_wrapper<int32_t> col_gold_6{{1, -1}, {true, false}};
    auto col_gold_7_names_col = strcol_wrapper{"Samuel Vimes", "Angua von Überwald"};
    auto col_gold_7_ages_col  = column_wrapper<int32_t>{{48, 25}};

    auto col_gold_7_is_human_col = column_wrapper<bool>{{true, false}, {true, false}};

    auto col_gold_7 = cudf::test::structs_column_wrapper{
      {col_gold_7_names_col, col_gold_7_ages_col, col_gold_7_is_human_col}};
    CVector cols_gold;
    cols_gold.push_back(col_gold_0.release());
    cols_gold.push_back(col_gold_1.release());
    cols_gold.push_back(col_gold_2.release());
    cols_gold.push_back(col_gold_3.release());
    cols_gold.push_back(col_gold_4.release());
    cols_gold.push_back(col_gold_5.release());
    cols_gold.push_back(col_gold_6.release());
    cols_gold.push_back(col_gold_7.release());
    Table gold(std::move(cols_gold));

    auto gold_sort_order = cudf::sorted_order(gold.view());
    auto sorted_gold     = cudf::gather(gold.view(), *gold_sort_order);
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*sorted_gold, *sorted_result);
  }

  // Unequal nulls
  {
    auto result = inner_join(t0, t1, {0, 1, 3}, {0, 1, 3}, cudf::null_equality::UNEQUAL, algo);
    auto result_sort_order = cudf::sorted_order(result->view());
    auto sorted_result     = cudf::gather(result->view(), *result_sort_order);

    column_wrapper<int32_t> col_gold_0{{3}};
    strcol_wrapper col_gold_1({"s1"});
    column_wrapper<int32_t> col_gold_2{{0}};
    auto col_gold_3_names_col = strcol_wrapper{"Samuel Vimes"};
    auto col_gold_3_ages_col  = column_wrapper<int32_t>{{48}};

    auto col_gold_3_is_human_col = column_wrapper<bool>{{true}, {true}};

    auto col_gold_3 = cudf::test::structs_column_wrapper{
      {col_gold_3_names_col, col_gold_3_ages_col, col_gold_3_is_human_col}};

    column_wrapper<int32_t> col_gold_4{{3}};
    strcol_wrapper col_gold_5({"s1"});
    column_wrapper<int32_t> col_gold_6{{1}};
    auto col_gold_7_names_col = strcol_wrapper{"Samuel Vimes"};
    auto col_gold_7_ages_col  = column_wrapper<int32_t>{{48}};

    auto col_gold_7_is_human_col = column_wrapper<bool>{{true}, {true}};

    auto col_gold_7 = cudf::test::structs_column_wrapper{
      {col_gold_7_names_col, col_gold_7_ages_col, col_gold_7_is_human_col}};
    CVector cols_gold;
    cols_gold.push_back(col_gold_0.release());
    cols_gold.push_back(col_gold_1.release());
    cols_gold.push_back(col_gold_2.release());
    cols_gold.push_back(col_gold_3.release());
    cols_gold.push_back(col_gold_4.release());
    cols_gold.push_back(col_gold_5.release());
    cols_gold.push_back(col_gold_6.release());
    cols_gold.push_back(col_gold_7.release());
    Table gold(std::move(cols_gold));

    auto gold_sort_order = cudf::sorted_order(gold.view());
    auto sorted_gold     = cudf::gather(gold.view(), *gold_sort_order);
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*sorted_gold, *sorted_result);
  }
}

// // Test to check join behavior when join keys are null.
TEST_P(JoinParameterizedTest, InnerJoinOnNulls)
{
  auto algo = GetParam();
  // clang-format off
  column_wrapper<int32_t> col0_0{{  3,    1,    2,    0,    2}};
  strcol_wrapper          col0_1({"s1", "s1", "s8", "s4", "s0"},
                                 {  true,    true,    false,    true,    true});
  column_wrapper<int32_t> col0_2{{  0,    1,    2,    4,    1}};

  column_wrapper<int32_t> col1_0{{  2,    2,    0,    4,    3}};
  strcol_wrapper          col1_1({"s1", "s0", "s1", "s2", "s1"},
                                 {  true,    false,    true,    true,    true});
  column_wrapper<int32_t> col1_2{{  1,    0,    1,    2,    1}};

  CVector cols0, cols1;
  cols0.push_back(col0_0.release());
  cols0.push_back(col0_1.release());
  cols0.push_back(col0_2.release());
  cols1.push_back(col1_0.release());
  cols1.push_back(col1_1.release());
  cols1.push_back(col1_2.release());

  Table t0(std::move(cols0));
  Table t1(std::move(cols1));

  auto result            = inner_join(t0, t1, {0, 1}, {0, 1}, cudf::null_equality::EQUAL, algo);
  auto result_sort_order = cudf::sorted_order(result->view());
  auto sorted_result     = cudf::gather(result->view(), *result_sort_order);

  column_wrapper<int32_t> col_gold_0 {{  3,    2}};
  strcol_wrapper          col_gold_1 ({"s1", "s0"},
                                      {  true,    false});
  column_wrapper<int32_t> col_gold_2{{   0,    2}};
  column_wrapper<int32_t> col_gold_3 {{  3,    2}};
  strcol_wrapper          col_gold_4 ({"s1", "s0"},
                                      {  true,    false});
  column_wrapper<int32_t> col_gold_5{{   1,    0}};
  CVector cols_gold;
  cols_gold.push_back(col_gold_0.release());
  cols_gold.push_back(col_gold_1.release());
  cols_gold.push_back(col_gold_2.release());
  cols_gold.push_back(col_gold_3.release());
  cols_gold.push_back(col_gold_4.release());
  cols_gold.push_back(col_gold_5.release());

  Table gold(std::move(cols_gold));

  auto gold_sort_order = cudf::sorted_order(gold.view());
  auto sorted_gold     = cudf::gather(gold.view(), *gold_sort_order);
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*sorted_gold, *sorted_result);

  // Repeat test with compare_nulls_equal=false,
  // as per SQL standard.

  result            = inner_join(t0, t1, {0, 1}, {0, 1}, cudf::null_equality::UNEQUAL, algo);
  result_sort_order = cudf::sorted_order(result->view());
  sorted_result     = cudf::gather(result->view(), *result_sort_order);

  col_gold_0 =               {{  3}};
  col_gold_1 = strcol_wrapper({"s1"},
                              {  true});
  col_gold_2 =               {{  0}};
  col_gold_3 =               {{  3}};
  col_gold_4 = strcol_wrapper({"s1"},
                              {  true});
  col_gold_5 =               {{  1}};

  // clang-format on

  CVector cols_gold_sql;
  cols_gold_sql.push_back(col_gold_0.release());
  cols_gold_sql.push_back(col_gold_1.release());
  cols_gold_sql.push_back(col_gold_2.release());
  cols_gold_sql.push_back(col_gold_3.release());
  cols_gold_sql.push_back(col_gold_4.release());
  cols_gold_sql.push_back(col_gold_5.release());
  Table gold_sql(std::move(cols_gold_sql));

  gold_sort_order = cudf::sorted_order(gold_sql.view());
  sorted_gold     = cudf::gather(gold_sql.view(), *gold_sort_order);
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*sorted_gold, *sorted_result);
}

TEST_P(JoinParameterizedTest, InnerJoinStructs)
{
  auto algo = GetParam();

  // Left Table: 1 row
  // Key: {"aaaaaaaaaa", false}
  strcol_wrapper left_str_col({"aaaaaaaaaa"});
  column_wrapper<bool> left_bool_col{{false}};
  auto left_struct_col = cudf::test::structs_column_wrapper{{left_str_col, left_bool_col}};
  column_wrapper<int32_t> left_val_col{{100}};  // Non-key payload

  CVector cols0;
  cols0.push_back(left_struct_col.release());
  cols0.push_back(left_val_col.release());
  Table t0(std::move(cols0));

  // Right Table: 17 rows
  // Keys with nulls in the boolean child column
  strcol_wrapper right_str_col({"aaaaaaaaaa",
                                "aaa",
                                "aaa",
                                "aaaa",
                                "aaa",
                                "aaaaaaaaaa",
                                "aaaaaaaaaa",
                                "aaa",
                                "aaa",
                                "aaaaaaaaaa",
                                "aaaaaaaaaa",
                                "aaaa",
                                "aaaaaaaaaa",
                                "aaaa",
                                "aaa",
                                "aaa",
                                "aaaaaaaaaa"});
  column_wrapper<bool> right_bool_col({{false,
                                        true,
                                        true,
                                        false,
                                        true,
                                        false,
                                        false,
                                        true,
                                        true,
                                        false,
                                        false,
                                        false,
                                        false,
                                        false,
                                        true,
                                        true,
                                        true},
                                       {true,
                                        true,
                                        true,
                                        false,
                                        true,
                                        true,
                                        true,
                                        true,
                                        true,
                                        true,
                                        true,
                                        false,
                                        true,
                                        false,
                                        true,
                                        true,
                                        true}});
  auto right_struct_col = cudf::test::structs_column_wrapper{{right_str_col, right_bool_col}};
  column_wrapper<int32_t> right_val_col(
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});  // Non-key payload

  CVector cols1;
  cols1.push_back(right_struct_col.release());
  cols1.push_back(right_val_col.release());
  Table t1(std::move(cols1));

  // The key has one column: the structs column (index 0 for both tables)
  // Join logic:
  // - String match: "aaaaaaaaaa"
  // - Bool match: false
  // - Null equality: EQUAL (only null matches null)
  // Matching Right Rows (Index, String, Bool):
  // 0: "aaaaaaaaaa", false (Match)
  // 5: "aaaaaaaaaa", false (Match)
  // 6: "aaaaaaaaaa", false (Match)
  // 9: "aaaaaaaaaa", false (Match)
  // 10: "aaaaaaaaaa", false (Match)
  // 12: "aaaaaaaaaa", false (Match)
  // 16: "aaaaaaaaaa", true  (No match on bool)

  auto result            = inner_join(t0, t1, {0}, {0}, cudf::null_equality::EQUAL, algo);
  auto result_sort_order = cudf::sorted_order(result->view());
  auto sorted_result     = cudf::gather(result->view(), *result_sort_order);

  // Expected output is the left row (100) repeated 6 times, combined with matching right rows (0,
  // 5, 6, 9, 10, 12)

  // Gold Left Columns (Struct + Val)
  auto gold_left_str_col = strcol_wrapper(
    {"aaaaaaaaaa", "aaaaaaaaaa", "aaaaaaaaaa", "aaaaaaaaaa", "aaaaaaaaaa", "aaaaaaaaaa"});
  auto gold_left_bool_col = column_wrapper<bool>{{false, false, false, false, false, false}};
  auto gold_left_struct_col =
    cudf::test::structs_column_wrapper{{gold_left_str_col, gold_left_bool_col}};
  column_wrapper<int32_t> gold_left_val_col(
    {100, 100, 100, 100, 100, 100});  // Non-key payload from left

  // Gold Right Columns (Struct + Val)
  auto gold_right_str_col = strcol_wrapper(
    {"aaaaaaaaaa", "aaaaaaaaaa", "aaaaaaaaaa", "aaaaaaaaaa", "aaaaaaaaaa", "aaaaaaaaaa"});
  auto gold_right_bool_col = column_wrapper<bool>{{false, false, false, false, false, false}};
  auto gold_right_struct_col =
    cudf::test::structs_column_wrapper{{gold_right_str_col, gold_right_bool_col}};
  column_wrapper<int32_t> gold_right_val_col(
    {0, 5, 6, 9, 10, 12});  // Non-key payload from right (indices of matching rows)

  CVector cols_gold;
  cols_gold.push_back(gold_left_struct_col.release());
  cols_gold.push_back(gold_left_val_col.release());
  cols_gold.push_back(gold_right_struct_col.release());
  cols_gold.push_back(gold_right_val_col.release());
  Table gold(std::move(cols_gold));

  auto gold_sort_order = cudf::sorted_order(gold.view());
  auto sorted_gold     = cudf::gather(gold.view(), *gold_sort_order);

  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*sorted_gold, *sorted_result);
}

// Empty Left Table
TEST_P(JoinParameterizedTest, EmptyLeftTableInnerJoin)
{
  auto algo = GetParam();
  column_wrapper<int32_t> col0_0;
  column_wrapper<int32_t> col0_1;

  column_wrapper<int32_t> col1_0{{2, 2, 0, 4, 3}};
  column_wrapper<int32_t> col1_1{{1, 0, 1, 2, 1}, {true, false, true, true, true}};

  CVector cols0, cols1;
  cols0.push_back(col0_0.release());
  cols0.push_back(col0_1.release());
  cols1.push_back(col1_0.release());
  cols1.push_back(col1_1.release());

  Table empty0(std::move(cols0));
  Table t1(std::move(cols1));

  auto result = inner_join(empty0, t1, {0, 1}, {0, 1}, cudf::null_equality::EQUAL, algo);
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(empty0, *result);
}

TEST_P(JoinParameterizedTest, EmptyLeftTableLeftJoin)
{
  auto algo = GetParam();
  column_wrapper<int32_t> col0_0;
  column_wrapper<int32_t> col0_1;

  column_wrapper<int32_t> col1_0{{2, 2, 0, 4, 3}};
  column_wrapper<int32_t> col1_1{{1, 0, 1, 2, 1}, {true, false, true, true, true}};

  CVector cols0, cols1;
  cols0.push_back(col0_0.release());
  cols0.push_back(col0_1.release());
  cols1.push_back(col1_0.release());
  cols1.push_back(col1_1.release());

  Table empty0(std::move(cols0));
  Table t1(std::move(cols1));

  auto result = left_join(empty0, t1, {0, 1}, {0, 1}, cudf::null_equality::EQUAL, algo);
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(empty0, *result);
}

TEST_F(JoinTest, EmptyLeftTableFullJoin)
{
  column_wrapper<int32_t> col0_0;
  column_wrapper<int32_t> col0_1;

  column_wrapper<int32_t> col1_0{{2, 2, 0, 4, 3}};
  column_wrapper<int32_t> col1_1{{1, 0, 1, 2, 1}, {true, false, true, true, true}};

  CVector cols0, cols1;
  cols0.push_back(col0_0.release());
  cols0.push_back(col0_1.release());
  cols1.push_back(col1_0.release());
  cols1.push_back(col1_1.release());

  Table lhs(std::move(cols0));
  Table rhs(std::move(cols1));

  auto result            = full_join(lhs, rhs, {0, 1}, {0, 1});
  auto result_sort_order = cudf::sorted_order(result->view());
  auto sorted_result     = cudf::gather(result->view(), *result_sort_order);

  column_wrapper<int32_t> col_gold_0{{-1, -1, -1, -1, -1}, {false, false, false, false, false}};
  column_wrapper<int32_t> col_gold_1{{-1, -1, -1, -1, -1}, {false, false, false, false, false}};
  column_wrapper<int32_t> col_gold_2{{2, 2, 0, 4, 3}};
  column_wrapper<int32_t> col_gold_3{{1, 0, 1, 2, 1}, {true, false, true, true, true}};

  CVector cols_gold;
  cols_gold.push_back(col_gold_0.release());
  cols_gold.push_back(col_gold_1.release());
  cols_gold.push_back(col_gold_2.release());
  cols_gold.push_back(col_gold_3.release());
  Table gold(std::move(cols_gold));

  auto gold_sort_order = cudf::sorted_order(gold.view());
  auto sorted_gold     = cudf::gather(gold.view(), *gold_sort_order);

  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*sorted_gold, *sorted_result);
}

// Empty Right Table
TEST_P(JoinParameterizedTest, EmptyRightTableInnerJoin)
{
  auto algo = GetParam();
  column_wrapper<int32_t> col0_0{{2, 2, 0, 4, 3}};
  column_wrapper<int32_t> col0_1{{1, 0, 1, 2, 1}, {true, false, true, true, true}};

  column_wrapper<int32_t> col1_0;
  column_wrapper<int32_t> col1_1;

  CVector cols0, cols1;
  cols0.push_back(col0_0.release());
  cols0.push_back(col0_1.release());
  cols1.push_back(col1_0.release());
  cols1.push_back(col1_1.release());

  Table t0(std::move(cols0));
  Table empty1(std::move(cols1));

  {
    auto result = inner_join(t0, empty1, {0, 1}, {0, 1}, cudf::null_equality::EQUAL, algo);
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(empty1, *result);
  }

  {
    cudf::hash_join hash_join(empty1, cudf::null_equality::EQUAL);

    auto output_size                         = hash_join.inner_join_size(t0);
    std::optional<std::size_t> optional_size = output_size;

    std::size_t const size_gold = 0;
    EXPECT_EQ(output_size, size_gold);

    auto result = hash_join.inner_join(t0, optional_size);
    column_wrapper<int32_t> col_gold_0{};
    column_wrapper<int32_t> col_gold_1{};
    auto const [sorted_gold, sorted_result] = gather_maps_as_tables(col_gold_0, col_gold_1, result);
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*sorted_gold, *sorted_result);
  }
}

TEST_P(JoinParameterizedTest, EmptyRightTableLeftJoin)
{
  auto algo = GetParam();
  column_wrapper<int32_t> col0_0{{2, 2, 0, 4, 3}, {true, true, true, true, true}};
  column_wrapper<int32_t> col0_1{{1, 0, 1, 2, 1}, {true, false, true, true, true}};

  column_wrapper<int32_t> col1_0;
  column_wrapper<int32_t> col1_1;

  CVector cols0, cols1;
  cols0.push_back(col0_0.release());
  cols0.push_back(col0_1.release());
  cols1.push_back(col1_0.release());
  cols1.push_back(col1_1.release());

  Table t0(std::move(cols0));
  Table empty1(std::move(cols1));

  {
    auto result = left_join(t0, empty1, {0, 1}, {0, 1}, cudf::null_equality::EQUAL, algo);
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(t0, *result);
  }

  {
    cudf::hash_join hash_join(empty1, cudf::null_equality::EQUAL);

    auto output_size                         = hash_join.left_join_size(t0);
    std::optional<std::size_t> optional_size = output_size;

    std::size_t const size_gold = 5;
    EXPECT_EQ(output_size, size_gold);

    auto result = hash_join.left_join(t0, optional_size);
    column_wrapper<int32_t> col_gold_0{{0, 1, 2, 3, 4}};
    column_wrapper<int32_t> col_gold_1{{NoneValue, NoneValue, NoneValue, NoneValue, NoneValue}};
    auto const [sorted_gold, sorted_result] = gather_maps_as_tables(col_gold_0, col_gold_1, result);
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*sorted_gold, *sorted_result);
  }
}

TEST_F(JoinTest, EmptyRightTableFullJoin)
{
  column_wrapper<int32_t> col0_0{{2, 2, 0, 4, 3}};
  column_wrapper<int32_t> col0_1{{1, 0, 1, 2, 1}, {true, false, true, true, true}};

  column_wrapper<int32_t> col1_0;
  column_wrapper<int32_t> col1_1;

  CVector cols0, cols1;
  cols0.push_back(col0_0.release());
  cols0.push_back(col0_1.release());
  cols1.push_back(col1_0.release());
  cols1.push_back(col1_1.release());

  Table t0(std::move(cols0));
  Table empty1(std::move(cols1));

  {
    auto result = full_join(t0, empty1, {0, 1}, {0, 1});
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(t0, *result);
  }

  {
    cudf::hash_join hash_join(empty1, cudf::null_equality::EQUAL);

    auto output_size                         = hash_join.full_join_size(t0);
    std::optional<std::size_t> optional_size = output_size;

    std::size_t const size_gold = 5;
    EXPECT_EQ(output_size, size_gold);

    auto result = hash_join.full_join(t0, optional_size);
    column_wrapper<int32_t> col_gold_0{{0, 1, 2, 3, 4}};
    column_wrapper<int32_t> col_gold_1{{NoneValue, NoneValue, NoneValue, NoneValue, NoneValue}};
    auto const [sorted_gold, sorted_result] = gather_maps_as_tables(col_gold_0, col_gold_1, result);
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*sorted_gold, *sorted_result);
  }
}

// Both tables empty
TEST_P(JoinParameterizedTest, BothEmptyInnerJoin)
{
  auto algo = GetParam();
  column_wrapper<int32_t> col0_0;
  column_wrapper<int32_t> col0_1;

  column_wrapper<int32_t> col1_0;
  column_wrapper<int32_t> col1_1;

  CVector cols0, cols1;
  cols0.push_back(col0_0.release());
  cols0.push_back(col0_1.release());
  cols1.push_back(col1_0.release());
  cols1.push_back(col1_1.release());

  Table t0(std::move(cols0));
  Table empty1(std::move(cols1));

  auto result = inner_join(t0, empty1, {0, 1}, {0, 1}, cudf::null_equality::EQUAL, algo);
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(empty1, *result);
}

TEST_P(JoinParameterizedTest, BothEmptyLeftJoin)
{
  auto algo = GetParam();
  column_wrapper<int32_t> col0_0;
  column_wrapper<int32_t> col0_1;

  column_wrapper<int32_t> col1_0;
  column_wrapper<int32_t> col1_1;

  CVector cols0, cols1;
  cols0.push_back(col0_0.release());
  cols0.push_back(col0_1.release());
  cols1.push_back(col1_0.release());
  cols1.push_back(col1_1.release());

  Table t0(std::move(cols0));
  Table empty1(std::move(cols1));

  auto result = left_join(t0, empty1, {0, 1}, {0, 1}, cudf::null_equality::EQUAL, algo);
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(empty1, *result);
}

TEST_F(JoinTest, BothEmptyFullJoin)
{
  column_wrapper<int32_t> col0_0;
  column_wrapper<int32_t> col0_1;

  column_wrapper<int32_t> col1_0;
  column_wrapper<int32_t> col1_1;

  CVector cols0, cols1;
  cols0.push_back(col0_0.release());
  cols0.push_back(col0_1.release());
  cols1.push_back(col1_0.release());
  cols1.push_back(col1_1.release());

  Table t0(std::move(cols0));
  Table empty1(std::move(cols1));

  auto result = full_join(t0, empty1, {0, 1}, {0, 1});
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(empty1, *result);
}

// // EqualValues X Inner,Left,Full

TEST_P(JoinParameterizedTest, EqualValuesInnerJoin)
{
  auto algo = GetParam();
  column_wrapper<int32_t> col0_0{{0, 0}};
  strcol_wrapper col0_1({"s0", "s0"});

  column_wrapper<int32_t> col1_0{{0, 0}};
  strcol_wrapper col1_1({"s0", "s0"});

  CVector cols0, cols1;
  cols0.push_back(col0_0.release());
  cols0.push_back(col0_1.release());
  cols1.push_back(col1_0.release());
  cols1.push_back(col1_1.release());

  Table t0(std::move(cols0));
  Table t1(std::move(cols1));

  auto result = inner_join(t0, t1, {0, 1}, {0, 1}, cudf::null_equality::EQUAL, algo);

  column_wrapper<int32_t> col_gold_0{{0, 0, 0, 0}};
  strcol_wrapper col_gold_1({"s0", "s0", "s0", "s0"});
  column_wrapper<int32_t> col_gold_2{{0, 0, 0, 0}};
  strcol_wrapper col_gold_3({"s0", "s0", "s0", "s0"});

  CVector cols_gold;
  cols_gold.push_back(col_gold_0.release());
  cols_gold.push_back(col_gold_1.release());
  cols_gold.push_back(col_gold_2.release());
  cols_gold.push_back(col_gold_3.release());

  Table gold(std::move(cols_gold));

  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(gold, *result);
}

TEST_P(JoinParameterizedTest, EqualValuesLeftJoin)
{
  auto algo = GetParam();
  column_wrapper<int32_t> col0_0{{0, 0}};
  strcol_wrapper col0_1({"s0", "s0"});

  column_wrapper<int32_t> col1_0{{0, 0}};
  strcol_wrapper col1_1({"s0", "s0"});

  CVector cols0, cols1;
  cols0.push_back(col0_0.release());
  cols0.push_back(col0_1.release());
  cols1.push_back(col1_0.release());
  cols1.push_back(col1_1.release());

  Table t0(std::move(cols0));
  Table t1(std::move(cols1));

  auto result = left_join(t0, t1, {0, 1}, {0, 1}, cudf::null_equality::EQUAL, algo);

  column_wrapper<int32_t> col_gold_0{{0, 0, 0, 0}, {true, true, true, true}};
  strcol_wrapper col_gold_1({"s0", "s0", "s0", "s0"}, {true, true, true, true});
  column_wrapper<int32_t> col_gold_2{{0, 0, 0, 0}, {true, true, true, true}};
  strcol_wrapper col_gold_3({"s0", "s0", "s0", "s0"}, {true, true, true, true});

  CVector cols_gold;
  cols_gold.push_back(col_gold_0.release());
  cols_gold.push_back(col_gold_1.release());
  cols_gold.push_back(col_gold_2.release());
  cols_gold.push_back(col_gold_3.release());
  Table gold(std::move(cols_gold));

  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(gold, *result);
}

TEST_F(JoinTest, EqualValuesFullJoin)
{
  column_wrapper<int32_t> col0_0{{0, 0}};
  strcol_wrapper col0_1({"s0", "s0"});

  column_wrapper<int32_t> col1_0{{0, 0}};
  strcol_wrapper col1_1({"s0", "s0"});

  CVector cols0, cols1;
  cols0.push_back(col0_0.release());
  cols0.push_back(col0_1.release());
  cols1.push_back(col1_0.release());
  cols1.push_back(col1_1.release());

  Table t0(std::move(cols0));
  Table t1(std::move(cols1));

  auto result = full_join(t0, t1, {0, 1}, {0, 1});

  column_wrapper<int32_t> col_gold_0{{0, 0, 0, 0}};
  strcol_wrapper col_gold_1({"s0", "s0", "s0", "s0"});
  column_wrapper<int32_t> col_gold_2{{0, 0, 0, 0}};
  strcol_wrapper col_gold_3({"s0", "s0", "s0", "s0"});

  CVector cols_gold;
  cols_gold.push_back(col_gold_0.release());
  cols_gold.push_back(col_gold_1.release());
  cols_gold.push_back(col_gold_2.release());
  cols_gold.push_back(col_gold_3.release());
  Table gold(std::move(cols_gold));

  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(gold, *result);
}

TEST_P(JoinParameterizedTest, InnerJoinCornerCase)
{
  auto algo = GetParam();
  column_wrapper<int64_t> col0_0{{4, 1, 3, 2, 2, 2, 2}};
  column_wrapper<int64_t> col1_0{{2}};

  CVector cols0, cols1;
  cols0.push_back(col0_0.release());
  cols1.push_back(col1_0.release());

  Table t0(std::move(cols0));
  Table t1(std::move(cols1));

  auto result            = inner_join(t0, t1, {0}, {0}, cudf::null_equality::EQUAL, algo);
  auto result_sort_order = cudf::sorted_order(result->view());
  auto sorted_result     = cudf::gather(result->view(), *result_sort_order);

  column_wrapper<int64_t> col_gold_0{{2, 2, 2, 2}};
  column_wrapper<int64_t> col_gold_1{{2, 2, 2, 2}};
  CVector cols_gold;
  cols_gold.push_back(col_gold_0.release());
  cols_gold.push_back(col_gold_1.release());
  Table gold(std::move(cols_gold));

  auto gold_sort_order = cudf::sorted_order(gold.view());
  auto sorted_gold     = cudf::gather(gold.view(), *gold_sort_order);
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*sorted_gold, *sorted_result);
}

TEST_F(JoinTest, HashJoinSequentialProbes)
{
  CVector cols1;
  cols1.emplace_back(column_wrapper<int32_t>{{2, 2, 0, 4, 3}}.release());
  cols1.emplace_back(strcol_wrapper{{"s1", "s0", "s1", "s2", "s1"}}.release());

  Table t1(std::move(cols1));

  // Test with different load factors
  std::vector<double> load_factors = {0.5, 1.0};

  for (auto desired_load_factor : load_factors) {
    cudf::hash_join hash_join(
      t1, cudf::nullable_join::NO, cudf::null_equality::EQUAL, desired_load_factor);

    // Test full join
    {
      CVector cols0;
      cols0.emplace_back(column_wrapper<int32_t>{{3, 1, 2, 0, 3}}.release());
      cols0.emplace_back(strcol_wrapper({"s0", "s1", "s2", "s4", "s1"}).release());

      Table t0(std::move(cols0));

      auto output_size                         = hash_join.full_join_size(t0);
      std::optional<std::size_t> optional_size = output_size;

      std::size_t const size_gold = 9;
      EXPECT_EQ(output_size, size_gold);

      auto result = hash_join.full_join(t0, optional_size);
      column_wrapper<int32_t> col_gold_0{
        {NoneValue, NoneValue, NoneValue, NoneValue, 4, 0, 1, 2, 3}};
      column_wrapper<int32_t> col_gold_1{
        {0, 1, 2, 3, 4, NoneValue, NoneValue, NoneValue, NoneValue}};
      auto const [sorted_gold, sorted_result] =
        gather_maps_as_tables(col_gold_0, col_gold_1, result);
      CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*sorted_gold, *sorted_result);
    }

    // Test left join
    {
      CVector cols0;
      cols0.emplace_back(column_wrapper<int32_t>{{3, 1, 2, 0, 3}}.release());
      cols0.emplace_back(strcol_wrapper({"s0", "s1", "s2", "s4", "s1"}).release());

      Table t0(std::move(cols0));

      auto output_size                         = hash_join.left_join_size(t0);
      std::optional<std::size_t> optional_size = output_size;

      std::size_t const size_gold = 5;
      EXPECT_EQ(output_size, size_gold);

      auto result = hash_join.left_join(t0, optional_size);
      column_wrapper<int32_t> col_gold_0{{0, 1, 2, 3, 4}};
      column_wrapper<int32_t> col_gold_1{{NoneValue, NoneValue, NoneValue, NoneValue, 4}};
      auto const [sorted_gold, sorted_result] =
        gather_maps_as_tables(col_gold_0, col_gold_1, result);
      CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*sorted_gold, *sorted_result);
    }

    // Test inner join
    {
      CVector cols0;
      cols0.emplace_back(column_wrapper<int32_t>{{3, 1, 2, 0, 2}}.release());
      cols0.emplace_back(strcol_wrapper({"s1", "s1", "s0", "s4", "s0"}).release());

      Table t0(std::move(cols0));

      auto output_size                         = hash_join.inner_join_size(t0);
      std::optional<std::size_t> optional_size = output_size;

      std::size_t const size_gold = 3;
      EXPECT_EQ(output_size, size_gold);

      auto result = hash_join.inner_join(t0, optional_size);
      column_wrapper<int32_t> col_gold_0{{2, 4, 0}};
      column_wrapper<int32_t> col_gold_1{{1, 1, 4}};
      auto const [sorted_gold, sorted_result] =
        gather_maps_as_tables(col_gold_0, col_gold_1, result);
      CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*sorted_gold, *sorted_result);
    }
  }
}

TEST_F(JoinTest, HashJoinWithStructsAndNulls)
{
  auto col0_names_col = strcol_wrapper{
    "Samuel Vimes", "Carrot Ironfoundersson", "Detritus", "Samuel Vimes", "Angua von Überwald"};
  auto col0_ages_col = column_wrapper<int32_t>{{48, 27, 351, 31, 25}};

  auto col0_is_human_col =
    column_wrapper<bool>{{true, true, false, false, false}, {true, true, false, true, false}};

  auto col0 =
    cudf::test::structs_column_wrapper{{col0_names_col, col0_ages_col, col0_is_human_col}};

  auto col1_names_col = strcol_wrapper{
    "Samuel Vimes", "Detritus", "Detritus", "Carrot Ironfoundersson", "Angua von Überwald"};
  auto col1_ages_col = column_wrapper<int32_t>{{48, 35, 351, 22, 25}};

  auto col1_is_human_col =
    column_wrapper<bool>{{true, true, false, false, true}, {true, true, false, true, true}};

  auto col1 =
    cudf::test::structs_column_wrapper{{col1_names_col, col1_ages_col, col1_is_human_col}};

  CVector cols0, cols1;
  cols0.push_back(col0.release());
  cols1.push_back(col1.release());

  Table t0(std::move(cols0));
  Table t1(std::move(cols1));
  auto const has_nulls = cudf::has_nested_nulls(t0) || cudf::has_nested_nulls(t1)
                           ? cudf::nullable_join::YES
                           : cudf::nullable_join::NO;

  double constexpr desired_load_factor = 0.5;
  auto hash_join = cudf::hash_join(t1, has_nulls, cudf::null_equality::EQUAL, desired_load_factor);

  {
    auto output_size = hash_join.left_join_size(t0);
    EXPECT_EQ(5, output_size);
    auto result = hash_join.left_join(t0, output_size);
    column_wrapper<int32_t> col_gold_0{{0, 1, 2, 3, 4}};
    column_wrapper<int32_t> col_gold_1{{0, NoneValue, 2, NoneValue, NoneValue}};
    auto const [sorted_gold, sorted_result] = gather_maps_as_tables(col_gold_0, col_gold_1, result);
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*sorted_gold, *sorted_result);
  }

  {
    auto output_size = hash_join.inner_join_size(t0);
    EXPECT_EQ(2, output_size);
    auto result = hash_join.inner_join(t0, output_size);
    column_wrapper<int32_t> col_gold_0{{0, 2}};
    column_wrapper<int32_t> col_gold_1{{0, 2}};
    auto const [sorted_gold, sorted_result] = gather_maps_as_tables(col_gold_0, col_gold_1, result);
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*sorted_gold, *sorted_result);
  }

  {
    auto output_size = hash_join.full_join_size(t0);
    EXPECT_EQ(8, output_size);
    auto result = hash_join.full_join(t0, output_size);
    column_wrapper<int32_t> col_gold_0{{NoneValue, NoneValue, NoneValue, 0, 1, 2, 3, 4}};
    column_wrapper<int32_t> col_gold_1{{1, 3, 4, 0, NoneValue, 2, NoneValue, NoneValue}};
    auto const [sorted_gold, sorted_result] = gather_maps_as_tables(col_gold_0, col_gold_1, result);
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*sorted_gold, *sorted_result);
  }
}

TEST_F(JoinTest, HashJoinWithNullsOneSide)
{
  auto const t0 = [] {
    column_wrapper<int32_t> col0{2, 2, 0, 4, 3};
    column_wrapper<int32_t> col1{1, 10, 1, 2, 1};
    CVector cols;
    cols.emplace_back(col0.release());
    cols.emplace_back(col1.release());
    return Table{std::move(cols)};
  }();

  auto const t1 = [] {
    column_wrapper<int32_t> col0{1, 2, 3, 4, 5, 2, 2, 0, 4, 3, 1, 2, 3, 4, 5};
    column_wrapper<int32_t> col1{{1, 2, 3, 4, 5, 1, 0, 1, 2, 1, 1, 2, 3, 4, 5},
                                 cudf::test::iterators::null_at(6)};
    CVector cols;
    cols.emplace_back(col0.release());
    cols.emplace_back(col1.release());
    return Table{std::move(cols)};
  }();

  auto const hash_join   = cudf::hash_join(t0, cudf::null_equality::EQUAL);
  auto constexpr invalid = std::numeric_limits<int32_t>::min();  // invalid index sentinel

  auto const sort_result = [](auto const& result) {
    auto const left_cv  = cudf::column_view{cudf::data_type{cudf::type_id::INT32},
                                           static_cast<cudf::size_type>(result.first->size()),
                                           result.first->data(),
                                           nullptr,
                                           0};
    auto const right_cv = cudf::column_view{cudf::data_type{cudf::type_id::INT32},
                                            static_cast<cudf::size_type>(result.second->size()),
                                            result.second->data(),
                                            nullptr,
                                            0};
    auto sorted_left    = cudf::sort(cudf::table_view{{left_cv}});
    auto sorted_right   = cudf::sort(cudf::table_view{{right_cv}});
    return std::pair{std::move(sorted_left), std::move(sorted_right)};
  };

  {
    auto const output_size = hash_join.left_join_size(t1);
    auto const result      = hash_join.left_join(t1, std::optional<std::size_t>{output_size});
    auto const [sorted_left_indices, sorted_right_indices] = sort_result(result);

    auto const expected_left_indices =
      column_wrapper<int32_t>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14};
    auto const expected_right_indices = column_wrapper<int32_t>{invalid,
                                                                invalid,
                                                                invalid,
                                                                invalid,
                                                                invalid,
                                                                invalid,
                                                                invalid,
                                                                invalid,
                                                                invalid,
                                                                invalid,
                                                                invalid,
                                                                0,
                                                                2,
                                                                3,
                                                                4};

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_left_indices, sorted_left_indices->get_column(0));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_right_indices, sorted_right_indices->get_column(0));
  }

  {
    auto const output_size = hash_join.inner_join_size(t1);
    auto const result      = hash_join.inner_join(t1, std::optional<std::size_t>{output_size});
    auto const [sorted_left_indices, sorted_right_indices] = sort_result(result);

    auto const expected_left_indices  = column_wrapper<int32_t>{5, 7, 8, 9};
    auto const expected_right_indices = column_wrapper<int32_t>{0, 2, 3, 4};

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_left_indices, sorted_left_indices->get_column(0));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_right_indices, sorted_right_indices->get_column(0));
  }

  {
    auto const output_size = hash_join.full_join_size(t1);
    auto const result      = hash_join.full_join(t1, std::optional<std::size_t>{output_size});
    auto const [sorted_left_indices, sorted_right_indices] = sort_result(result);

    auto const expected_left_indices =
      column_wrapper<int32_t>{invalid, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14};
    auto const expected_right_indices = column_wrapper<int32_t>{invalid,
                                                                invalid,
                                                                invalid,
                                                                invalid,
                                                                invalid,
                                                                invalid,
                                                                invalid,
                                                                invalid,
                                                                invalid,
                                                                invalid,
                                                                invalid,
                                                                0,
                                                                1,
                                                                2,
                                                                3,
                                                                4};

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_left_indices, sorted_left_indices->get_column(0));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_right_indices, sorted_right_indices->get_column(0));
  }
}

TEST_F(JoinTest, HashJoinLargeOutputSize)
{
  // self-join a table of zeroes to generate an output row count that would overflow int32_t
  std::size_t col_size = 65567;
  rmm::device_buffer zeroes(col_size * sizeof(int32_t), cudf::get_default_stream());
  CUDF_CUDA_TRY(
    cudaMemsetAsync(zeroes.data(), 0, zeroes.size(), cudf::get_default_stream().value()));
  cudf::column_view col_zeros(
    cudf::data_type{cudf::type_id::INT32}, col_size, zeroes.data(), nullptr, 0);
  cudf::table_view tview{{col_zeros}};
  double constexpr desired_load_factor = 0.5;
  cudf::hash_join hash_join(
    tview, cudf::nullable_join::NO, cudf::null_equality::UNEQUAL, desired_load_factor);
  std::size_t output_size = hash_join.inner_join_size(tview);
  EXPECT_EQ(col_size * col_size, output_size);
}

TEST_F(JoinTest, HashJoinMemoryResource)
{
  CVector cols0;
  cols0.emplace_back(column_wrapper<int32_t>{{3, 1, 2, 0, 2}}.release());
  Table t0(std::move(cols0));

  CVector cols1;
  cols1.emplace_back(column_wrapper<int32_t>{{2, 2, 0, 4, 3}}.release());
  Table t1(std::move(cols1));

  auto mr = rmm::mr::statistics_resource_adaptor(cudf::get_current_device_resource_ref());

  cudf::hash_join hash_join(t1, cudf::null_equality::EQUAL, cudf::get_default_stream(), mr);

  EXPECT_GT(mr.get_bytes_counter().peak, 0);

  auto result = hash_join.inner_join(t0);
  column_wrapper<int32_t> col_gold_0{{0, 2, 2, 3, 4, 4}};
  column_wrapper<int32_t> col_gold_1{{4, 0, 1, 2, 0, 1}};
  auto const [sorted_gold, sorted_result] = gather_maps_as_tables(col_gold_0, col_gold_1, result);
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*sorted_gold, *sorted_result);
}

TEST_F(JoinTest, HashJoinInnerMatchContext)
{
  // Test inner join match context functionality with multiple matches and nulls
  // Use the same test data as SortMergeInnerJoinSizePerRowWithNulls for consistency
  column_wrapper<int32_t> col0_0{{3, 1, 2, 0, 2}};
  strcol_wrapper col0_1({"s1", "s1", "s0", "s4", "s0"}, {true, true, false, true, true});
  column_wrapper<int32_t> col0_2{{0, 1, 2, 4, 1}};

  column_wrapper<int32_t> col1_0{{2, 2, 0, 4, 3}};
  strcol_wrapper col1_1({"s1", "s0", "s1", "s2", "s1"}, {true, false, true, true, true});
  column_wrapper<int32_t> col1_2{{1, 0, 1, 2, 1}, {true, false, true, true, true}};

  CVector cols0, cols1;
  cols0.emplace_back(col0_0.release());
  cols0.emplace_back(col0_1.release());
  cols0.emplace_back(col0_2.release());
  cols1.emplace_back(col1_0.release());
  cols1.emplace_back(col1_1.release());
  cols1.emplace_back(col1_2.release());

  Table t0(std::move(cols0));
  Table t1(std::move(cols1));

  auto const stream = cudf::get_default_stream();

  // Test single column join with null_equality::EQUAL
  {
    cudf::hash_join const hash_join(t1.select({0}), cudf::null_equality::EQUAL);
    auto const match_context = hash_join.inner_join_match_context(t0.select({0}), stream);

    expect_match_counts_equal(*match_context._match_counts, {1, 0, 2, 1, 2}, stream);

    auto const host_match_counts =
      cudf::detail::make_host_vector_async(*match_context._match_counts, stream);
    stream.synchronize();
    cudf::size_type const total_matches =
      std::accumulate(host_match_counts.begin(), host_match_counts.end(), cudf::size_type{0});
    auto const inner_join_size = hash_join.inner_join_size(t0.select({0}));
    EXPECT_EQ(total_matches, inner_join_size);
  }

  // Test multi-column join with null_equality::EQUAL
  {
    cudf::hash_join const hash_join(t1.select({0, 1}), cudf::null_equality::EQUAL);
    auto const match_context = hash_join.inner_join_match_context(t0.select({0, 1}), stream);

    expect_match_counts_equal(*match_context._match_counts, {1, 0, 1, 0, 0}, stream);
  }

  // Test single column join with null_equality::UNEQUAL
  {
    cudf::hash_join const hash_join(t1.select({0}), cudf::null_equality::UNEQUAL);
    auto const match_context = hash_join.inner_join_match_context(t0.select({0}), stream);

    expect_match_counts_equal(*match_context._match_counts, {1, 0, 2, 1, 2}, stream);
  }

  // Test multi-column join with null_equality::UNEQUAL
  {
    cudf::hash_join const hash_join(t1.select({0, 1}), cudf::null_equality::UNEQUAL);
    auto const match_context = hash_join.inner_join_match_context(t0.select({0, 1}), stream);

    expect_match_counts_equal(*match_context._match_counts, {1, 0, 0, 0, 0}, stream);
  }
}

TEST_F(JoinTest, HashJoinLeftMatchContext)
{
  column_wrapper<int32_t> col0_0{{3, 1, 2, 0, 2}};
  strcol_wrapper col0_1({"s1", "s1", "s0", "s4", "s0"}, {true, true, false, true, true});

  column_wrapper<int32_t> col1_0{{2, 2, 0, 4, 3}};
  strcol_wrapper col1_1({"s1", "s0", "s1", "s2", "s1"}, {true, false, true, true, true});

  CVector cols0, cols1;
  cols0.emplace_back(col0_0.release());
  cols0.emplace_back(col0_1.release());
  cols1.emplace_back(col1_0.release());
  cols1.emplace_back(col1_1.release());

  Table t0(std::move(cols0));
  Table t1(std::move(cols1));

  auto const stream = cudf::get_default_stream();

  // Test single column join
  {
    cudf::hash_join const hash_join(t1.select({0}), cudf::null_equality::EQUAL);
    auto const match_context = hash_join.left_join_match_context(t0.select({0}), stream);

    expect_match_counts_equal(*match_context._match_counts, {1, 1, 2, 1, 2}, stream);

    auto const host_match_counts =
      cudf::detail::make_host_vector_async(*match_context._match_counts, stream);
    stream.synchronize();
    cudf::size_type const total_matches =
      std::accumulate(host_match_counts.begin(), host_match_counts.end(), cudf::size_type{0});
    auto const left_join_size = hash_join.left_join_size(t0.select({0}));
    EXPECT_EQ(total_matches, left_join_size);
  }

  // Test multi-column join with null_equality::EQUAL
  {
    cudf::hash_join const hash_join(t1.select({0, 1}), cudf::null_equality::EQUAL);
    auto const match_context = hash_join.left_join_match_context(t0.select({0, 1}), stream);

    expect_match_counts_equal(*match_context._match_counts, {1, 1, 1, 1, 1}, stream);
  }

  // Test multi-column join with null_equality::UNEQUAL
  {
    cudf::hash_join const hash_join(t1.select({0, 1}), cudf::null_equality::UNEQUAL);
    auto const match_context = hash_join.left_join_match_context(t0.select({0, 1}), stream);

    expect_match_counts_equal(*match_context._match_counts, {1, 1, 1, 1, 1}, stream);
  }
}

TEST_F(JoinTest, HashJoinFullMatchContext)
{
  column_wrapper<int32_t> col0_0{{3, 1, 2, 0, 2}};
  strcol_wrapper col0_1({"s1", "s1", "s0", "s4", "s0"}, {true, true, false, true, true});

  column_wrapper<int32_t> col1_0{{2, 2, 0, 4, 3}};
  strcol_wrapper col1_1({"s1", "s0", "s1", "s2", "s1"}, {true, false, true, true, true});

  CVector cols0, cols1;
  cols0.emplace_back(col0_0.release());
  cols0.emplace_back(col0_1.release());
  cols1.emplace_back(col1_0.release());
  cols1.emplace_back(col1_1.release());

  Table t0(std::move(cols0));
  Table t1(std::move(cols1));

  auto const stream = cudf::get_default_stream();

  // Test single column join
  {
    cudf::hash_join const hash_join(t1.select({0}), cudf::null_equality::EQUAL);
    auto const match_context = hash_join.full_join_match_context(t0.select({0}), stream);

    expect_match_counts_equal(*match_context._match_counts, {1, 1, 2, 1, 2}, stream);
  }

  // Test multi-column join
  {
    cudf::hash_join const hash_join(t1.select({0, 1}), cudf::null_equality::EQUAL);
    auto const match_context = hash_join.full_join_match_context(t0.select({0, 1}), stream);

    expect_match_counts_equal(*match_context._match_counts, {1, 1, 1, 1, 1}, stream);
  }
}

TEST_F(JoinTest, HashJoinMatchContextEmptyRight)
{
  // Test match context with empty right table
  column_wrapper<int32_t> col0_0{{3, 1, 2}};
  column_wrapper<int32_t> col1_0{};  // Empty

  CVector cols0, cols1;
  cols0.emplace_back(col0_0.release());
  cols1.emplace_back(col1_0.release());

  Table t0(std::move(cols0));
  Table t1(std::move(cols1));

  auto const stream = cudf::get_default_stream();
  cudf::hash_join const hash_join(t1, cudf::null_equality::EQUAL);

  // Test inner join match context
  {
    auto const match_context = hash_join.inner_join_match_context(t0, stream);
    expect_match_counts_equal(*match_context._match_counts, {0, 0, 0}, stream);
  }

  // Test left join match context
  {
    auto const match_context = hash_join.left_join_match_context(t0, stream);
    expect_match_counts_equal(*match_context._match_counts, {1, 1, 1}, stream);
  }

  // Test full join match context
  {
    auto const match_context = hash_join.full_join_match_context(t0, stream);
    expect_match_counts_equal(*match_context._match_counts, {1, 1, 1}, stream);
  }
}

TEST_F(JoinTest, HashJoinMatchContextDuplicatesAndEdgeCases)
{
  // Test with duplicate keys to ensure multiple matches work correctly
  column_wrapper<int32_t> col0_0{{1, 1, 2, 2, 3}};
  strcol_wrapper col0_1({"a", "a", "b", "b", "c"});

  column_wrapper<int32_t> col1_0{{1, 1, 1, 2, 4}};
  strcol_wrapper col1_1({"a", "a", "a", "b", "d"});

  CVector cols0, cols1;
  cols0.emplace_back(col0_0.release());
  cols0.emplace_back(col0_1.release());
  cols1.emplace_back(col1_0.release());
  cols1.emplace_back(col1_1.release());

  Table t0(std::move(cols0));
  Table t1(std::move(cols1));

  auto const stream = cudf::get_default_stream();

  // Test inner join with multiple matches per row
  {
    cudf::hash_join const hash_join(t1.select({0}), cudf::null_equality::EQUAL);
    auto const match_context = hash_join.inner_join_match_context(t0.select({0}), stream);

    expect_match_counts_equal(*match_context._match_counts, {3, 3, 1, 1, 0}, stream);

    auto const host_match_counts =
      cudf::detail::make_host_vector_async(*match_context._match_counts, stream);
    stream.synchronize();
    cudf::size_type const total_matches =
      std::accumulate(host_match_counts.begin(), host_match_counts.end(), cudf::size_type{0});
    auto const inner_join_size = hash_join.inner_join_size(t0.select({0}));
    EXPECT_EQ(total_matches, inner_join_size);
  }

  // Test multi-column join
  {
    cudf::hash_join const hash_join(t1.select({0, 1}), cudf::null_equality::EQUAL);
    auto const match_context = hash_join.inner_join_match_context(t0.select({0, 1}), stream);

    expect_match_counts_equal(*match_context._match_counts, {3, 3, 1, 1, 0}, stream);
  }
}

struct JoinDictionaryTest : public cudf::test::BaseFixture {};

TEST_F(JoinDictionaryTest, LeftJoinNoNulls)
{
  column_wrapper<int32_t> col0_0{{3, 1, 2, 0, 3}};
  strcol_wrapper col0_1_w({"s0", "s1", "s2", "s4", "s1"});
  auto col0_1 = cudf::dictionary::encode(col0_1_w);
  column_wrapper<int32_t> col0_2{{0, 1, 2, 4, 1}};

  column_wrapper<int32_t> col1_0{{2, 2, 0, 4, 3}};
  strcol_wrapper col1_1_w{{"s1", "s0", "s1", "s2", "s1"}};
  auto col1_1 = cudf::dictionary::encode(col1_1_w);
  column_wrapper<int32_t> col1_2{{1, 0, 1, 2, 1}};

  auto t0 = cudf::table_view({col0_0, col0_1->view(), col0_2});
  auto t1 = cudf::table_view({col1_0, col1_1->view(), col1_2});
  auto g0 = cudf::table_view({col0_0, col0_1_w, col0_2});
  auto g1 = cudf::table_view({col1_0, col1_1_w, col1_2});

  auto result      = left_join(t0, t1, {0}, {0});
  auto result_view = result->view();
  auto decoded1    = cudf::dictionary::decode(result_view.column(1));
  auto decoded4    = cudf::dictionary::decode(result_view.column(4));
  std::vector<cudf::column_view> result_decoded({result_view.column(0),
                                                 decoded1->view(),
                                                 result_view.column(2),
                                                 result_view.column(3),
                                                 decoded4->view(),
                                                 result_view.column(5)});
  auto result_sort_order = cudf::sorted_order(cudf::table_view(result_decoded));
  auto sorted_result     = cudf::gather(cudf::table_view(result_decoded), *result_sort_order);

  auto gold            = left_join(g0, g1, {0}, {0});
  auto gold_sort_order = cudf::sorted_order(gold->view());
  auto sorted_gold     = cudf::gather(gold->view(), *gold_sort_order);

  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*sorted_gold, *sorted_result);
}

TEST_F(JoinDictionaryTest, LeftJoinWithNulls)
{
  column_wrapper<int32_t> col0_0{{3, 1, 2, 0, 2}};
  strcol_wrapper col0_1({"s1", "s1", "s0", "s4", "s0"}, {true, true, false, true, true});
  column_wrapper<int32_t> col0_2_w{{0, 1, 2, 4, 1}};
  auto col0_2 = cudf::dictionary::encode(col0_2_w);

  column_wrapper<int32_t> col1_0{{2, 2, 0, 4, 3}};
  strcol_wrapper col1_1({"s1", "s0", "s1", "s2", "s1"});
  column_wrapper<int32_t> col1_2_w{{1, 0, 1, 2, 1}, {true, false, true, true, true}};
  auto col1_2 = cudf::dictionary::encode(col1_2_w);

  auto t0 = cudf::table_view({col0_0, col0_1, col0_2->view()});
  auto t1 = cudf::table_view({col1_0, col1_1, col1_2->view()});

  auto result      = left_join(t0, t1, {0, 1}, {0, 1});
  auto result_view = result->view();
  auto decoded2    = cudf::dictionary::decode(result_view.column(2));
  auto decoded5    = cudf::dictionary::decode(result_view.column(5));
  std::vector<cudf::column_view> result_decoded({result_view.column(0),
                                                 result_view.column(1),
                                                 decoded2->view(),
                                                 result_view.column(3),
                                                 result_view.column(4),
                                                 decoded5->view()});
  auto result_sort_order = cudf::sorted_order(cudf::table_view(result_decoded));
  auto sorted_result     = cudf::gather(cudf::table_view(result_decoded), *result_sort_order);

  auto g0              = cudf::table_view({col0_0, col0_1, col0_2_w});
  auto g1              = cudf::table_view({col1_0, col1_1, col1_2_w});
  auto gold            = left_join(g0, g1, {0, 1}, {0, 1});
  auto gold_sort_order = cudf::sorted_order(gold->view());
  auto sorted_gold     = cudf::gather(gold->view(), *gold_sort_order);

  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*sorted_gold, *sorted_result);
}

TEST_P(JoinParameterizedTest, DictionaryInnerJoinNoNulls)
{
  auto algo = GetParam();
  column_wrapper<int32_t> col0_0{{3, 1, 2, 0, 2}};
  strcol_wrapper col0_1_w({"s1", "s1", "s0", "s4", "s0"});
  auto col0_1 = cudf::dictionary::encode(col0_1_w);
  column_wrapper<int32_t> col0_2{{0, 1, 2, 4, 1}};

  column_wrapper<int32_t> col1_0{{2, 2, 0, 4, 3}};
  strcol_wrapper col1_1_w({"s1", "s0", "s1", "s2", "s1"});
  auto col1_1 = cudf::dictionary::encode(col1_1_w);
  column_wrapper<int32_t> col1_2{{1, 0, 1, 2, 1}};

  auto t0 = cudf::table_view({col0_0, col0_1->view(), col0_2});
  auto t1 = cudf::table_view({col1_0, col1_1->view(), col1_2});

  auto result      = inner_join(t0, t1, {0, 1}, {0, 1}, cudf::null_equality::EQUAL, algo);
  auto result_view = result->view();
  auto decoded1    = cudf::dictionary::decode(result_view.column(1));
  auto decoded4    = cudf::dictionary::decode(result_view.column(4));
  std::vector<cudf::column_view> result_decoded({result_view.column(0),
                                                 decoded1->view(),
                                                 result_view.column(2),
                                                 result_view.column(3),
                                                 decoded4->view(),
                                                 result_view.column(5)});
  auto result_sort_order = cudf::sorted_order(cudf::table_view(result_decoded));
  auto sorted_result     = cudf::gather(cudf::table_view(result_decoded), *result_sort_order);

  auto g0              = cudf::table_view({col0_0, col0_1_w, col0_2});
  auto g1              = cudf::table_view({col1_0, col1_1_w, col1_2});
  auto gold            = inner_join(g0, g1, {0, 1}, {0, 1});
  auto gold_sort_order = cudf::sorted_order(gold->view());
  auto sorted_gold     = cudf::gather(gold->view(), *gold_sort_order);

  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*sorted_gold, *sorted_result);
}

TEST_P(JoinParameterizedTest, DictionaryInnerJoinWithNulls)
{
  auto algo = GetParam();
  column_wrapper<int32_t> col0_0{{3, 1, 2, 0, 2}};
  strcol_wrapper col0_1({"s1", "s1", "s0", "s4", "s0"}, {true, true, false, true, true});
  column_wrapper<int32_t> col0_2_w{{0, 1, 2, 4, 1}};
  auto col0_2 = cudf::dictionary::encode(col0_2_w);

  column_wrapper<int32_t> col1_0{{2, 2, 0, 4, 3}};
  strcol_wrapper col1_1({"s1", "s0", "s1", "s2", "s1"});
  column_wrapper<int32_t> col1_2_w{{1, 0, 1, 2, 1}, {true, false, true, true, true}};
  auto col1_2 = cudf::dictionary::encode(col1_2_w);

  auto t0 = cudf::table_view({col0_0, col0_1, col0_2->view()});
  auto t1 = cudf::table_view({col1_0, col1_1, col1_2->view()});

  auto result      = inner_join(t0, t1, {0, 1}, {0, 1}, cudf::null_equality::EQUAL, algo);
  auto result_view = result->view();
  auto decoded2    = cudf::dictionary::decode(result_view.column(2));
  auto decoded5    = cudf::dictionary::decode(result_view.column(5));
  std::vector<cudf::column_view> result_decoded({result_view.column(0),
                                                 result_view.column(1),
                                                 decoded2->view(),
                                                 result_view.column(3),
                                                 result_view.column(4),
                                                 decoded5->view()});
  auto result_sort_order = cudf::sorted_order(cudf::table_view(result_decoded));
  auto sorted_result     = cudf::gather(cudf::table_view(result_decoded), *result_sort_order);

  auto g0              = cudf::table_view({col0_0, col0_1, col0_2_w});
  auto g1              = cudf::table_view({col1_0, col1_1, col1_2_w});
  auto gold            = inner_join(g0, g1, {0, 1}, {0, 1});
  auto gold_sort_order = cudf::sorted_order(gold->view());
  auto sorted_gold     = cudf::gather(gold->view(), *gold_sort_order);

  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*sorted_gold, *sorted_result);
}

TEST_F(JoinDictionaryTest, FullJoinNoNulls)
{
  column_wrapper<int32_t> col0_0{{3, 1, 2, 0, 3}};
  strcol_wrapper col0_1_w({"s0", "s1", "s2", "s4", "s1"});
  auto col0_1 = cudf::dictionary::encode(col0_1_w);
  column_wrapper<int32_t> col0_2{{0, 1, 2, 4, 1}};

  column_wrapper<int32_t> col1_0{{2, 2, 0, 4, 3}};
  strcol_wrapper col1_1_w{{"s1", "s0", "s1", "s2", "s1"}};
  auto col1_1 = cudf::dictionary::encode(col1_1_w);
  column_wrapper<int32_t> col1_2{{1, 0, 1, 2, 1}};

  auto t0 = cudf::table_view({col0_0, col0_1->view(), col0_2});
  auto t1 = cudf::table_view({col1_0, col1_1->view(), col1_2});

  auto result      = full_join(t0, t1, {0, 1}, {0, 1});
  auto result_view = result->view();
  auto decoded1    = cudf::dictionary::decode(result_view.column(1));
  auto decoded4    = cudf::dictionary::decode(result_view.column(4));
  std::vector<cudf::column_view> result_decoded({result_view.column(0),
                                                 decoded1->view(),
                                                 result_view.column(2),
                                                 result_view.column(3),
                                                 decoded4->view(),
                                                 result_view.column(5)});
  auto result_sort_order = cudf::sorted_order(cudf::table_view(result_decoded));
  auto sorted_result     = cudf::gather(cudf::table_view(result_decoded), *result_sort_order);

  auto g0              = cudf::table_view({col0_0, col0_1_w, col0_2});
  auto g1              = cudf::table_view({col1_0, col1_1_w, col1_2});
  auto gold            = full_join(g0, g1, {0, 1}, {0, 1});
  auto gold_sort_order = cudf::sorted_order(gold->view());
  auto sorted_gold     = cudf::gather(gold->view(), *gold_sort_order);

  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*sorted_gold, *sorted_result);
}

TEST_F(JoinDictionaryTest, FullJoinWithNulls)
{
  column_wrapper<int32_t> col0_0_w{{3, 1, 2, 0, 3}};
  auto col0_0 = cudf::dictionary::encode(col0_0_w);
  strcol_wrapper col0_1({"s0", "s1", "s2", "s4", "s1"});
  column_wrapper<int32_t> col0_2{{0, 1, 2, 4, 1}};

  column_wrapper<int32_t> col1_0_w{{2, 2, 0, 4, 3}, {true, true, true, false, true}};
  auto col1_0 = cudf::dictionary::encode(col1_0_w);
  strcol_wrapper col1_1{{"s1", "s0", "s1", "s2", "s1"}};
  column_wrapper<int32_t> col1_2{{1, 0, 1, 2, 1}};

  auto t0 = cudf::table_view({col0_0->view(), col0_1, col0_2});
  auto t1 = cudf::table_view({col1_0->view(), col1_1, col1_2});

  auto result      = full_join(t0, t1, {0, 1}, {0, 1});
  auto result_view = result->view();
  auto decoded0    = cudf::dictionary::decode(result_view.column(0));
  auto decoded3    = cudf::dictionary::decode(result_view.column(3));
  std::vector<cudf::column_view> result_decoded({decoded0->view(),
                                                 result_view.column(1),
                                                 result_view.column(2),
                                                 decoded3->view(),
                                                 result_view.column(4),
                                                 result_view.column(5)});
  auto result_sort_order = cudf::sorted_order(cudf::table_view(result_decoded));
  auto sorted_result     = cudf::gather(cudf::table_view(result_decoded), *result_sort_order);

  auto g0              = cudf::table_view({col0_0_w, col0_1, col0_2});
  auto g1              = cudf::table_view({col1_0_w, col1_1, col1_2});
  auto gold            = full_join(g0, g1, {0, 1}, {0, 1});
  auto gold_sort_order = cudf::sorted_order(gold->view());
  auto sorted_gold     = cudf::gather(gold->view(), *gold_sort_order);

  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*sorted_gold, *sorted_result);
}

TEST_F(JoinTest, FullJoinWithStructsAndNulls)
{
  column_wrapper<int32_t> col0_0{{3, 1, 2, 0, 3}};
  strcol_wrapper col0_1({"s0", "s1", "s2", "s4", "s1"});
  column_wrapper<int32_t> col0_2{{0, 1, 2, 4, 1}};

  std::initializer_list<std::string> col0_names = {"Samuel Vimes",
                                                   "Carrot Ironfoundersson",
                                                   "Angua von Überwald",
                                                   "Detritus",
                                                   "Carrot Ironfoundersson"};
  auto col0_names_col = strcol_wrapper{col0_names.begin(), col0_names.end()};
  auto col0_ages_col  = column_wrapper<int32_t>{{48, 27, 25, 31, 351}};

  auto col0_is_human_col =
    column_wrapper<bool>{{true, true, false, false, false}, {true, true, false, true, true}};

  auto col0_3 = cudf::test::structs_column_wrapper{
    {col0_names_col, col0_ages_col, col0_is_human_col}, {true, true, true, true, true}};

  column_wrapper<int32_t> col1_0{{2, 2, 0, 4, 3}, {true, true, true, false, true}};
  strcol_wrapper col1_1{{"s1", "s0", "s1", "s2", "s1"}};
  column_wrapper<int32_t> col1_2{{1, 0, 1, 2, 1}};

  std::initializer_list<std::string> col1_names = {"Carrot Ironfoundersson",
                                                   "Samuel Vimes",
                                                   "Carrot Ironfoundersson",
                                                   "Angua von Überwald",
                                                   "Carrot Ironfoundersson"};
  auto col1_names_col = strcol_wrapper{col1_names.begin(), col1_names.end()};
  auto col1_ages_col  = column_wrapper<int32_t>{{27, 48, 27, 25, 27}};

  auto col1_is_human_col =
    column_wrapper<bool>{{true, true, true, false, true}, {true, true, true, false, true}};

  auto col1_3 =
    cudf::test::structs_column_wrapper{{col1_names_col, col1_ages_col, col1_is_human_col}};

  CVector cols0, cols1;
  cols0.push_back(col0_0.release());
  cols0.push_back(col0_1.release());
  cols0.push_back(col0_2.release());
  cols0.push_back(col0_3.release());
  cols1.push_back(col1_0.release());
  cols1.push_back(col1_1.release());
  cols1.push_back(col1_2.release());
  cols1.push_back(col1_3.release());

  Table t0(std::move(cols0));
  Table t1(std::move(cols1));

  auto result            = full_join(t0, t1, {0, 1, 3}, {0, 1, 3});
  auto result_sort_order = cudf::sorted_order(result->view());
  auto sorted_result     = cudf::gather(result->view(), *result_sort_order);

  column_wrapper<int32_t> col_gold_0{
    {3, 1, 2, 0, 3, -1, -1, -1, -1, -1},
    {true, true, true, true, true, false, false, false, false, false}};
  strcol_wrapper col_gold_1({"s0", "s1", "s2", "s4", "s1", "", "", "", "", ""},
                            {true, true, true, true, true, false, false, false, false, false});
  column_wrapper<int32_t> col_gold_2{
    {0, 1, 2, 4, 1, -1, -1, -1, -1, -1},
    {true, true, true, true, true, false, false, false, false, false}};
  auto gold_names0_col =
    strcol_wrapper{{"Samuel Vimes",
                    "Carrot Ironfoundersson",
                    "Angua von Überwald",
                    "Detritus",
                    "Carrot Ironfoundersson",
                    "",
                    "",
                    "",
                    "",
                    ""},
                   {true, true, true, true, true, false, false, false, false, false}};
  auto gold_ages0_col =
    column_wrapper<int32_t>{{48, 27, 25, 31, 351, -1, -1, -1, -1, -1},
                            {true, true, true, true, true, false, false, false, false, false}};

  auto gold_is_human0_col =
    column_wrapper<bool>{{true, true, false, false, false, false, false, false, false, false},
                         {true, true, false, true, true, false, false, false, false, false}};

  auto col_gold_3 = cudf::test::structs_column_wrapper{
    {gold_names0_col, gold_ages0_col, gold_is_human0_col},
    {true, true, true, true, true, false, false, false, false, false}};

  column_wrapper<int32_t> col_gold_4{
    {-1, -1, -1, -1, -1, 3, 2, 2, 0, 4},
    {false, false, false, false, false, true, true, true, true, false}};
  strcol_wrapper col_gold_5({"", "", "", "", "", "s1", "s1", "s0", "s1", "s2"},
                            {false, false, false, false, false, true, true, true, true, true});
  column_wrapper<int32_t> col_gold_6{
    {-1, -1, -1, -1, -1, 1, 1, 0, 1, 2},
    {false, false, false, false, false, true, true, true, true, true}};
  auto gold_names1_col =
    strcol_wrapper{{"",
                    "",
                    "",
                    "",
                    "",
                    "Carrot Ironfoundersson",
                    "Carrot Ironfoundersson",
                    "Samuel Vimes",
                    "Carrot Ironfoundersson",
                    "Angua von Überwald"},
                   {false, false, false, false, false, true, true, true, true, true}};
  auto gold_ages1_col =
    column_wrapper<int32_t>{{-1, -1, -1, -1, -1, 27, 27, 48, 27, 25},
                            {false, false, false, false, false, true, true, true, true, true}};

  auto gold_is_human1_col =
    column_wrapper<bool>{{false, false, false, false, false, true, true, true, true, false},
                         {false, false, false, false, false, true, true, true, true, false}};

  auto col_gold_7 = cudf::test::structs_column_wrapper{
    {gold_names1_col, gold_ages1_col, gold_is_human1_col},
    {false, false, false, false, false, true, true, true, true, true}};

  CVector cols_gold;
  cols_gold.push_back(col_gold_0.release());
  cols_gold.push_back(col_gold_1.release());
  cols_gold.push_back(col_gold_2.release());
  cols_gold.push_back(col_gold_3.release());
  cols_gold.push_back(col_gold_4.release());
  cols_gold.push_back(col_gold_5.release());
  cols_gold.push_back(col_gold_6.release());
  cols_gold.push_back(col_gold_7.release());

  Table gold(std::move(cols_gold));

  auto gold_sort_order = cudf::sorted_order(gold.view());
  auto sorted_gold     = cudf::gather(gold.view(), *gold_sort_order);
  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*sorted_gold, *sorted_result);
}

using lcw = cudf::test::lists_column_wrapper<int32_t>;
using cudf::test::iterators::null_at;

struct JoinTestLists : public cudf::test::BaseFixture {
  /*
    [
      NULL,      0
      [1],       1
      [2, NULL], 2
      [],        3
      [5, 6]     4
  */
  lcw right{{{0}, {1}, {{2, 0}, null_at(1)}, {}, {5, 6}}, null_at(0)};

  /*
    [
      [1],       0
      [3],       1
      NULL,      2
      [],        3
      [2, NULL], 4
      [5],       5
      [6]        6
    ]
  */
  lcw left{{{1}, {3}, {0}, {}, {{2, 0}, null_at(1)}, {5}, {6}}, null_at(2)};

  auto column_view_from_device_uvector(rmm::device_uvector<cudf::size_type> const& vector)
  {
    auto const indices_span = cudf::device_span<cudf::size_type const>{vector};
    return cudf::column_view{indices_span};
  }

  auto sort_and_gather(
    cudf::table_view table,
    cudf::column_view gather_map,
    cudf::out_of_bounds_policy oob_policy = cudf::out_of_bounds_policy::DONT_CHECK)
  {
    auto const gather_table = cudf::gather(table, gather_map, oob_policy);
    auto const sort_order   = cudf::sorted_order(*gather_table);
    return cudf::gather(*gather_table, *sort_order);
  }

  template <typename JoinFunc>
  void join(cudf::column_view left_gold_map,
            cudf::column_view right_gold_map,
            cudf::null_equality nulls_equal,
            JoinFunc join_func,
            cudf::out_of_bounds_policy oob_policy)
  {
    auto const right_tv = cudf::table_view{{right}};
    auto const left_tv  = cudf::table_view{{left}};

    auto const [left_result_map, right_result_map] =
      join_func(right_tv,
                left_tv,
                nulls_equal,
                cudf::get_default_stream(),
                cudf::get_current_device_resource_ref());

    auto const left_result_table =
      sort_and_gather(right_tv, column_view_from_device_uvector(*left_result_map), oob_policy);
    auto const right_result_table =
      sort_and_gather(left_tv, column_view_from_device_uvector(*right_result_map), oob_policy);

    auto const left_gold_table  = sort_and_gather(right_tv, left_gold_map, oob_policy);
    auto const right_gold_table = sort_and_gather(left_tv, right_gold_map, oob_policy);

    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*left_result_table, *left_gold_table);
    CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*right_result_table, *right_gold_table);
  }

  void inner_join(cudf::column_view left_gold_map,
                  cudf::column_view right_gold_map,
                  cudf::null_equality nulls_equal)
  {
    join(left_gold_map,
         right_gold_map,
         nulls_equal,
         cudf::inner_join,
         cudf::out_of_bounds_policy::DONT_CHECK);
  }

  void full_join(cudf::column_view left_gold_map,
                 cudf::column_view right_gold_map,
                 cudf::null_equality nulls_equal)
  {
    join(left_gold_map,
         right_gold_map,
         nulls_equal,
         cudf::full_join,
         cudf::out_of_bounds_policy::NULLIFY);
  }

  void left_join(cudf::column_view left_gold_map,
                 cudf::column_view right_gold_map,
                 cudf::null_equality nulls_equal)
  {
    join(left_gold_map,
         right_gold_map,
         nulls_equal,
         cudf::left_join,
         cudf::out_of_bounds_policy::NULLIFY);
  }
};

struct JoinParameterizedTestLists : public JoinTestLists,
                                    public testing::WithParamInterface<algorithm> {
  void inner_join(cudf::column_view left_gold_map,
                  cudf::column_view right_gold_map,
                  cudf::null_equality nulls_equal,
                  algorithm algo)
  {
    auto join_lambda = [](cudf::table_view const& left,
                          cudf::table_view const& right,
                          cudf::null_equality compare_nulls,
                          rmm::cuda_stream_view stream,
                          rmm::device_async_resource_ref mr) -> JoinResult {
      cudf::sort_merge_join obj(right, cudf::sorted::NO, compare_nulls, stream);
      return obj.inner_join(left, stream, mr);
    };
    join(left_gold_map,
         right_gold_map,
         nulls_equal,
         algo == algorithm::HASH ? cudf::inner_join : join_lambda,
         cudf::out_of_bounds_policy::DONT_CHECK);
  }

  void left_join(cudf::column_view left_gold_map,
                 cudf::column_view right_gold_map,
                 cudf::null_equality nulls_equal,
                 algorithm algo)
  {
    auto join_lambda = [](cudf::table_view const& left,
                          cudf::table_view const& right,
                          cudf::null_equality compare_nulls,
                          rmm::cuda_stream_view stream,
                          rmm::device_async_resource_ref mr) -> JoinResult {
      cudf::sort_merge_join obj(right, cudf::sorted::NO, compare_nulls, stream);
      return obj.left_join(left, stream, mr);
    };
    join(left_gold_map,
         right_gold_map,
         nulls_equal,
         algo == algorithm::HASH ? cudf::left_join : join_lambda,
         cudf::out_of_bounds_policy::NULLIFY);
  }
};
// Parametrize qualifying join tests for supported algorithms
INSTANTIATE_TEST_CASE_P(InnerJoinParameterizedTestLists,
                        JoinParameterizedTestLists,
                        ::testing::Values(algorithm::HASH, algorithm::SORT_MERGE));

TEST_P(JoinParameterizedTestLists, ListWithNullsEqualInnerJoin)
{
  auto algo                 = GetParam();
  auto const left_gold_map  = column_wrapper<int32_t>({0, 1, 2, 3});
  auto const right_gold_map = column_wrapper<int32_t>({0, 2, 3, 4});
  this->inner_join(left_gold_map, right_gold_map, cudf::null_equality::EQUAL, algo);
}

TEST_P(JoinParameterizedTestLists, ListWithNullsUnequalInnerJoin)
{
  auto algo                 = GetParam();
  auto const left_gold_map  = column_wrapper<int32_t>({1, 3});
  auto const right_gold_map = column_wrapper<int32_t>({0, 3});
  this->inner_join(left_gold_map, right_gold_map, cudf::null_equality::UNEQUAL, algo);
}

TEST_F(JoinTestLists, ListWithNullsEqualFullJoin)
{
  auto const left_gold_map =
    column_wrapper<int32_t>({0, 1, 2, 3, 4, NoneValue, NoneValue, NoneValue});
  auto const right_gold_map = column_wrapper<int32_t>({2, 0, 4, 3, NoneValue, 1, 5, 6});
  this->full_join(left_gold_map, right_gold_map, cudf::null_equality::EQUAL);
}

TEST_F(JoinTestLists, ListWithNullsUnequalFullJoin)
{
  auto const left_gold_map =
    column_wrapper<int32_t>({0, 1, 2, 3, 4, NoneValue, NoneValue, NoneValue, NoneValue, NoneValue});
  auto const right_gold_map =
    column_wrapper<int32_t>({NoneValue, 0, NoneValue, 3, NoneValue, 1, 5, 6, 2, 4});
  this->full_join(left_gold_map, right_gold_map, cudf::null_equality::UNEQUAL);
}

TEST_P(JoinParameterizedTestLists, ListWithNullsEqualLeftJoin)
{
  auto algo                 = GetParam();
  auto const left_gold_map  = column_wrapper<int32_t>({0, 1, 2, 3, 4});
  auto const right_gold_map = column_wrapper<int32_t>({2, 0, 4, 3, NoneValue});
  this->left_join(left_gold_map, right_gold_map, cudf::null_equality::EQUAL, algo);
}

TEST_P(JoinParameterizedTestLists, ListWithNullsUnequalLeftJoin)
{
  auto algo                 = GetParam();
  auto const left_gold_map  = column_wrapper<int32_t>({0, 1, 2, 3, 4});
  auto const right_gold_map = column_wrapper<int32_t>({NoneValue, 0, NoneValue, 3, NoneValue});
  this->left_join(left_gold_map, right_gold_map, cudf::null_equality::UNEQUAL, algo);
}

// Thread safety tests for sort_merge_join
struct SortMergeJoinThreadSafetyTest : public cudf::test::BaseFixture {
  void SetUp() override
  {
    // Create test data
    col0 = column_wrapper<int32_t>{{3, 1, 2, 0, 2}};
    col1 = column_wrapper<int32_t>{{2, 2, 0, 4, 3}};
  }

  column_wrapper<int32_t> col0;
  column_wrapper<int32_t> col1;
};

TEST_F(SortMergeJoinThreadSafetyTest, ConcurrentMatchContext)
{
  // Test that multiple threads can call inner_join_match_context() concurrently
  auto const t0 = cudf::table_view{{col0}};
  auto const t1 = cudf::table_view{{col1}};

  cudf::sort_merge_join join_obj(t1, cudf::sorted::NO, cudf::null_equality::EQUAL);

  // Get expected result from single-threaded execution
  auto expected_ctx    = join_obj.inner_join_match_context(t0);
  auto expected_counts = cudf::detail::make_std_vector<cudf::size_type>(
    *expected_ctx->_match_counts, cudf::get_default_stream());

  // Run concurrent calls
  constexpr int num_threads = 4;
  std::vector<std::future<std::vector<cudf::size_type>>> futures;

  for (int i = 0; i < num_threads; ++i) {
    futures.push_back(std::async(std::launch::async, [&]() {
      auto ctx = join_obj.inner_join_match_context(
        t0, cudf::get_default_stream(), cudf::get_current_device_resource_ref());
      auto counts = cudf::detail::make_std_vector<cudf::size_type>(*ctx->_match_counts,
                                                                   cudf::get_default_stream());
      return counts;
    }));
  }

  // Verify all threads got correct results
  for (auto& future : futures) {
    auto counts = future.get();
    EXPECT_EQ(counts, expected_counts);
  }
}

TEST_F(SortMergeJoinThreadSafetyTest, ConcurrentPartitionedJoins)
{
  // Test that multiple threads can do full partitioned join workflow concurrently
  auto const t0     = cudf::table_view{{col0}};
  auto const t1     = cudf::table_view{{col1}};
  auto const stream = cudf::get_default_stream();

  cudf::sort_merge_join join_obj(t1, cudf::sorted::NO, cudf::null_equality::EQUAL, stream);

  // Get expected result from single-threaded inner_join
  auto [expected_left, expected_right] =
    join_obj.inner_join(t0, stream, cudf::get_current_device_resource_ref());
  auto expected_size = expected_left->size();

  // Run concurrent partitioned joins
  constexpr int num_threads = 4;
  std::vector<std::future<size_t>> futures;

  for (int i = 0; i < num_threads; ++i) {
    futures.push_back(std::async(std::launch::async, [&]() {
      // Each thread does full partitioned workflow
      auto match_ctx =
        join_obj.inner_join_match_context(t0, stream, cudf::get_current_device_resource_ref());

      cudf::join_partition_context part_ctx{std::move(match_ctx), 0, 0};

      size_t total_results = 0;
      for (cudf::size_type row = 0; row < t0.num_rows(); ++row) {
        part_ctx.left_start_idx            = row;
        part_ctx.left_end_idx              = row + 1;
        auto [left_indices, right_indices] = join_obj.partitioned_inner_join(
          part_ctx, stream, cudf::get_current_device_resource_ref());
        total_results += left_indices->size();
      }
      return total_results;
    }));
  }

  // Verify all threads got correct total result count
  for (auto& future : futures) {
    auto result_size = future.get();
    EXPECT_EQ(result_size, expected_size);
  }
}

TEST_F(JoinTest, HashJoinPartitionedInnerJoin)
{
  column_wrapper<int32_t> col0_0{{3, 1, 2, 0, 2}};
  strcol_wrapper col0_1({"s1", "s1", "s0", "s4", "s0"}, {true, true, false, true, true});
  column_wrapper<int32_t> col0_2{{0, 1, 2, 4, 1}};

  column_wrapper<int32_t> col1_0{{2, 2, 0, 4, 3}};
  strcol_wrapper col1_1({"s1", "s0", "s1", "s2", "s1"}, {true, false, true, true, true});
  column_wrapper<int32_t> col1_2{{1, 0, 1, 2, 1}, {true, false, true, true, true}};

  CVector cols0, cols1;
  cols0.push_back(col0_0.release());
  cols0.push_back(col0_1.release());
  cols0.push_back(col0_2.release());
  cols1.push_back(col1_0.release());
  cols1.push_back(col1_1.release());
  cols1.push_back(col1_2.release());

  Table t0(std::move(cols0));
  Table t1(std::move(cols1));

  auto const left_on       = std::vector<cudf::size_type>({0, 1});
  auto const right_on      = std::vector<cudf::size_type>({0, 1});
  auto const compare_nulls = cudf::null_equality::EQUAL;
  auto const stream        = cudf::get_default_stream();
  auto const mr            = cudf::get_current_device_resource_ref();

  // Reference result from full inner join
  auto expected_result     = inner_join(t0, t1, left_on, right_on, compare_nulls);
  auto expected_sort_order = cudf::sorted_order(expected_result->view());
  auto expected_sorted     = cudf::gather(expected_result->view(), *expected_sort_order);

  // Partitioned inner join
  cudf::hash_join hash_joiner(t1.select(right_on), compare_nulls, stream);
  auto match_ctx = hash_joiner.inner_join_match_context(t0.select(left_on), stream, mr);
  auto part_ctx  = cudf::join_partition_context{
    std::make_unique<cudf::join_match_context>(std::move(match_ctx)), 0, 0};

  auto join_and_gather = [&](cudf::join_partition_context const& ctx) {
    auto const [left_idx, right_idx] = hash_joiner.partitioned_inner_join(ctx, stream, mr);
    auto left_col  = cudf::column_view{cudf::device_span<cudf::size_type const>{*left_idx}};
    auto right_col = cudf::column_view{cudf::device_span<cudf::size_type const>{*right_idx}};
    auto left_res  = cudf::gather(t0, left_col, cudf::out_of_bounds_policy::DONT_CHECK);
    auto right_res = cudf::gather(t1, right_col, cudf::out_of_bounds_policy::DONT_CHECK);
    auto joined    = left_res->release();
    auto right_c   = right_res->release();
    joined.insert(joined.end(),
                  std::make_move_iterator(right_c.begin()),
                  std::make_move_iterator(right_c.end()));
    return std::make_unique<cudf::table>(std::move(joined));
  };

  // Process row by row
  std::vector<std::unique_ptr<cudf::table>> partials;
  std::vector<cudf::table_view> partial_views;
  for (cudf::size_type i = 0; i < t0.num_rows(); i++) {
    part_ctx.left_start_idx = i;
    part_ctx.left_end_idx   = i + 1;
    partials.push_back(join_and_gather(part_ctx));
    partial_views.push_back(partials.back()->view());
  }

  auto concat_result     = cudf::concatenate(partial_views, stream, mr);
  auto concat_sort_order = cudf::sorted_order(concat_result->view());
  auto concat_sorted     = cudf::gather(concat_result->view(), *concat_sort_order);

  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*expected_sorted, *concat_sorted);
}

TEST_F(JoinTest, HashJoinPartitionedLeftJoin)
{
  column_wrapper<int32_t> col0_0{{3, 1, 2, 0, 2}};
  strcol_wrapper col0_1({"s1", "s1", "s0", "s4", "s0"}, {true, true, false, true, true});
  column_wrapper<int32_t> col0_2{{0, 1, 2, 4, 1}};

  column_wrapper<int32_t> col1_0{{2, 2, 0, 4, 3}};
  strcol_wrapper col1_1({"s1", "s0", "s1", "s2", "s1"}, {true, false, true, true, true});
  column_wrapper<int32_t> col1_2{{1, 0, 1, 2, 1}, {true, false, true, true, true}};

  CVector cols0, cols1;
  cols0.push_back(col0_0.release());
  cols0.push_back(col0_1.release());
  cols0.push_back(col0_2.release());
  cols1.push_back(col1_0.release());
  cols1.push_back(col1_1.release());
  cols1.push_back(col1_2.release());

  Table t0(std::move(cols0));
  Table t1(std::move(cols1));

  auto const left_on       = std::vector<cudf::size_type>({0, 1});
  auto const right_on      = std::vector<cudf::size_type>({0, 1});
  auto const compare_nulls = cudf::null_equality::EQUAL;
  auto const stream        = cudf::get_default_stream();
  auto const mr            = cudf::get_current_device_resource_ref();

  // Reference result from full left join
  auto expected_result     = left_join(t0, t1, left_on, right_on, compare_nulls);
  auto expected_sort_order = cudf::sorted_order(expected_result->view());
  auto expected_sorted     = cudf::gather(expected_result->view(), *expected_sort_order);

  // Partitioned left join
  cudf::hash_join hash_joiner(t1.select(right_on), compare_nulls, stream);
  auto match_ctx = hash_joiner.left_join_match_context(t0.select(left_on), stream, mr);
  auto part_ctx  = cudf::join_partition_context{
    std::make_unique<cudf::join_match_context>(std::move(match_ctx)), 0, 0};

  auto join_and_gather = [&](cudf::join_partition_context const& ctx) {
    auto const [left_idx, right_idx] = hash_joiner.partitioned_left_join(ctx, stream, mr);
    auto left_col  = cudf::column_view{cudf::device_span<cudf::size_type const>{*left_idx}};
    auto right_col = cudf::column_view{cudf::device_span<cudf::size_type const>{*right_idx}};
    auto left_res  = cudf::gather(t0, left_col, cudf::out_of_bounds_policy::NULLIFY);
    auto right_res = cudf::gather(t1, right_col, cudf::out_of_bounds_policy::NULLIFY);
    auto joined    = left_res->release();
    auto right_c   = right_res->release();
    joined.insert(joined.end(),
                  std::make_move_iterator(right_c.begin()),
                  std::make_move_iterator(right_c.end()));
    return std::make_unique<cudf::table>(std::move(joined));
  };

  std::vector<std::unique_ptr<cudf::table>> partials;
  std::vector<cudf::table_view> partial_views;
  for (cudf::size_type i = 0; i < t0.num_rows(); i++) {
    part_ctx.left_start_idx = i;
    part_ctx.left_end_idx   = i + 1;
    partials.push_back(join_and_gather(part_ctx));
    partial_views.push_back(partials.back()->view());
  }

  auto concat_result     = cudf::concatenate(partial_views, stream, mr);
  auto concat_sort_order = cudf::sorted_order(concat_result->view());
  auto concat_sorted     = cudf::gather(concat_result->view(), *concat_sort_order);

  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*expected_sorted, *concat_sorted);
}

TEST_F(JoinTest, HashJoinPartitionedFullJoin)
{
  column_wrapper<int32_t> col0_0{{3, 1, 2, 0, 2}};
  strcol_wrapper col0_1({"s1", "s1", "s0", "s4", "s0"}, {true, true, false, true, true});
  column_wrapper<int32_t> col0_2{{0, 1, 2, 4, 1}};

  column_wrapper<int32_t> col1_0{{2, 2, 0, 4, 3}};
  strcol_wrapper col1_1({"s1", "s0", "s1", "s2", "s1"}, {true, false, true, true, true});
  column_wrapper<int32_t> col1_2{{1, 0, 1, 2, 1}, {true, false, true, true, true}};

  CVector cols0, cols1;
  cols0.push_back(col0_0.release());
  cols0.push_back(col0_1.release());
  cols0.push_back(col0_2.release());
  cols1.push_back(col1_0.release());
  cols1.push_back(col1_1.release());
  cols1.push_back(col1_2.release());

  Table t0(std::move(cols0));
  Table t1(std::move(cols1));

  auto const left_on       = std::vector<cudf::size_type>({0, 1});
  auto const right_on      = std::vector<cudf::size_type>({0, 1});
  auto const compare_nulls = cudf::null_equality::EQUAL;
  auto const stream        = cudf::get_default_stream();
  auto const mr            = cudf::get_current_device_resource_ref();

  // Reference result from full join
  auto expected_result     = full_join(t0, t1, left_on, right_on, compare_nulls);
  auto expected_sort_order = cudf::sorted_order(expected_result->view());
  auto expected_sorted     = cudf::gather(expected_result->view(), *expected_sort_order);

  // Partitioned full join (probe side)
  cudf::hash_join hash_joiner(t1.select(right_on), compare_nulls, stream);
  auto match_ctx = hash_joiner.full_join_match_context(t0.select(left_on), stream, mr);
  auto part_ctx  = cudf::join_partition_context{
    std::make_unique<cudf::join_match_context>(std::move(match_ctx)), 0, 0};

  // Collect per-partition (left, right) indices for finalization.
  std::vector<std::unique_ptr<rmm::device_uvector<cudf::size_type>>> left_idx_parts;
  std::vector<std::unique_ptr<rmm::device_uvector<cudf::size_type>>> right_idx_parts;
  for (cudf::size_type i = 0; i < t0.num_rows(); i++) {
    part_ctx.left_start_idx    = i;
    part_ctx.left_end_idx      = i + 1;
    auto [left_idx, right_idx] = hash_joiner.partitioned_full_join(part_ctx, stream, mr);
    left_idx_parts.push_back(std::move(left_idx));
    right_idx_parts.push_back(std::move(right_idx));
  }

  std::vector<cudf::device_span<cudf::size_type const>> left_partials;
  std::vector<cudf::device_span<cudf::size_type const>> right_partials;
  left_partials.reserve(left_idx_parts.size());
  right_partials.reserve(right_idx_parts.size());
  for (std::size_t i = 0; i < left_idx_parts.size(); ++i) {
    left_partials.emplace_back(left_idx_parts[i]->data(), left_idx_parts[i]->size());
    right_partials.emplace_back(right_idx_parts[i]->data(), right_idx_parts[i]->size());
  }

  auto [final_left, final_right] =
    cudf::hash_join::finalize_partitioned_full_join(left_partials,
                                                    right_partials,
                                                    t0.select(left_on).num_rows(),
                                                    t1.select(right_on).num_rows(),
                                                    stream,
                                                    mr);

  auto left_col  = cudf::column_view{cudf::device_span<cudf::size_type const>{*final_left}};
  auto right_col = cudf::column_view{cudf::device_span<cudf::size_type const>{*final_right}};
  auto left_res  = cudf::gather(t0, left_col, cudf::out_of_bounds_policy::NULLIFY);
  auto right_res = cudf::gather(t1, right_col, cudf::out_of_bounds_policy::NULLIFY);
  auto joined    = left_res->release();
  auto right_c   = right_res->release();
  joined.insert(
    joined.end(), std::make_move_iterator(right_c.begin()), std::make_move_iterator(right_c.end()));
  auto concat_result     = std::make_unique<cudf::table>(std::move(joined));
  auto concat_sort_order = cudf::sorted_order(concat_result->view());
  auto concat_sorted     = cudf::gather(concat_result->view(), *concat_sort_order);

  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*expected_sorted, *concat_sorted);
}

TEST_F(JoinTest, HashJoinPartitionedEmptyPartition)
{
  column_wrapper<int32_t> col0{{3, 1, 2, 0, 2}};
  column_wrapper<int32_t> col1{{2, 2, 0, 4, 3}};

  CVector cols0, cols1;
  cols0.push_back(col0.release());
  cols1.push_back(col1.release());
  Table t0(std::move(cols0));
  Table t1(std::move(cols1));

  auto const stream = cudf::get_default_stream();
  auto const mr     = cudf::get_current_device_resource_ref();

  cudf::hash_join hash_joiner(t1.select({0}), cudf::null_equality::EQUAL, stream);
  auto match_ctx = hash_joiner.inner_join_match_context(t0.select({0}), stream, mr);
  auto part_ctx  = cudf::join_partition_context{
    std::make_unique<cudf::join_match_context>(std::move(match_ctx)), 0, 0};

  // Empty partition (start == end)
  part_ctx.left_start_idx    = 2;
  part_ctx.left_end_idx      = 2;
  auto [left_idx, right_idx] = hash_joiner.partitioned_inner_join(part_ctx, stream, mr);
  EXPECT_EQ(left_idx->size(), 0);
  EXPECT_EQ(right_idx->size(), 0);
}

TEST_F(JoinTest, HashJoinPartitionedWholeTable)
{
  column_wrapper<int32_t> col0{{3, 1, 2, 0, 2}};
  column_wrapper<int32_t> col1{{2, 2, 0, 4, 3}};

  CVector cols0, cols1;
  cols0.push_back(col0.release());
  cols1.push_back(col1.release());
  Table t0(std::move(cols0));
  Table t1(std::move(cols1));

  auto const stream = cudf::get_default_stream();
  auto const mr     = cudf::get_current_device_resource_ref();

  // Reference: full inner join
  auto expected       = inner_join(t0, t1, {0}, {0});
  auto expected_order = cudf::sorted_order(expected->view());
  auto expected_sort  = cudf::gather(expected->view(), *expected_order);

  // Partitioned: entire table as one partition
  cudf::hash_join hash_joiner(t1.select({0}), cudf::null_equality::EQUAL, stream);
  auto match_ctx = hash_joiner.inner_join_match_context(t0.select({0}), stream, mr);
  auto part_ctx  = cudf::join_partition_context{
    std::make_unique<cudf::join_match_context>(std::move(match_ctx)), 0, t0.num_rows()};

  auto [left_idx, right_idx] = hash_joiner.partitioned_inner_join(part_ctx, stream, mr);
  auto left_col  = cudf::column_view{cudf::device_span<cudf::size_type const>{*left_idx}};
  auto right_col = cudf::column_view{cudf::device_span<cudf::size_type const>{*right_idx}};
  auto left_res  = cudf::gather(t0, left_col, cudf::out_of_bounds_policy::DONT_CHECK);
  auto right_res = cudf::gather(t1, right_col, cudf::out_of_bounds_policy::DONT_CHECK);
  auto joined    = left_res->release();
  auto right_c   = right_res->release();
  joined.insert(
    joined.end(), std::make_move_iterator(right_c.begin()), std::make_move_iterator(right_c.end()));
  auto result       = std::make_unique<cudf::table>(std::move(joined));
  auto result_order = cudf::sorted_order(result->view());
  auto result_sort  = cudf::gather(result->view(), *result_order);

  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*expected_sort, *result_sort);
}

// Exercises both a sliced (non-zero offset) left view and a partition size large enough
// to span multiple kernel blocks.
TEST_F(JoinTest, HashJoinPartitionedSlicedMultiBlock)
{
  auto constexpr left_full_rows = 4000;
  auto constexpr left_offset    = 1234;
  auto constexpr left_rows      = 2500;
  auto constexpr right_rows     = 300;

  std::vector<int32_t> left_vals(left_full_rows);
  for (cudf::size_type i = 0; i < left_full_rows; ++i) {
    left_vals[i] = i % 200;
  }
  std::vector<int32_t> right_vals(right_rows);
  for (cudf::size_type i = 0; i < right_rows; ++i) {
    right_vals[i] = i;
  }

  column_wrapper<int32_t> left_col(left_vals.begin(), left_vals.end());
  column_wrapper<int32_t> right_col(right_vals.begin(), right_vals.end());

  CVector cols_left, cols_right;
  cols_left.push_back(left_col.release());
  cols_right.push_back(right_col.release());
  Table left_full(std::move(cols_left));
  Table right(std::move(cols_right));

  auto const stream = cudf::get_default_stream();
  auto const mr     = cudf::get_current_device_resource_ref();

  // Sliced left view with non-zero offset
  auto const left_view = cudf::slice(left_full.view(), {left_offset, left_offset + left_rows})[0];

  // Reference: full inner join on the sliced left
  auto expected       = inner_join(cudf::table_view{left_view}, right, {0}, {0});
  auto expected_order = cudf::sorted_order(expected->view());
  auto expected_sort  = cudf::gather(expected->view(), *expected_order);

  cudf::hash_join hash_joiner(right.select({0}), cudf::null_equality::EQUAL, stream);
  auto match_ctx = hash_joiner.inner_join_match_context(left_view, stream, mr);
  auto part_ctx  = cudf::join_partition_context{
    std::make_unique<cudf::join_match_context>(std::move(match_ctx)), 0, 0};

  // Two partitions covering the sliced left; each is large enough to span multiple GPU blocks.
  auto const mid                                                            = left_rows / 2;
  std::vector<std::pair<cudf::size_type, cudf::size_type>> const partitions = {{0, mid},
                                                                               {mid, left_rows}};

  std::vector<std::unique_ptr<cudf::table>> partials;
  std::vector<cudf::table_view> partial_views;
  for (auto [s, e] : partitions) {
    part_ctx.left_start_idx          = s;
    part_ctx.left_end_idx            = e;
    auto const [left_idx, right_idx] = hash_joiner.partitioned_inner_join(part_ctx, stream, mr);
    auto left_col_view  = cudf::column_view{cudf::device_span<cudf::size_type const>{*left_idx}};
    auto right_col_view = cudf::column_view{cudf::device_span<cudf::size_type const>{*right_idx}};
    auto left_res       = cudf::gather(
      cudf::table_view{left_view}, left_col_view, cudf::out_of_bounds_policy::DONT_CHECK);
    auto right_res = cudf::gather(right, right_col_view, cudf::out_of_bounds_policy::DONT_CHECK);
    auto joined    = left_res->release();
    auto right_c   = right_res->release();
    joined.insert(joined.end(),
                  std::make_move_iterator(right_c.begin()),
                  std::make_move_iterator(right_c.end()));
    partials.push_back(std::make_unique<cudf::table>(std::move(joined)));
    partial_views.push_back(partials.back()->view());
  }

  auto concat       = cudf::concatenate(partial_views, stream, mr);
  auto concat_order = cudf::sorted_order(concat->view());
  auto concat_sort  = cudf::gather(concat->view(), *concat_order);

  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(*expected_sort, *concat_sort);
}

CUDF_TEST_PROGRAM_MAIN()
