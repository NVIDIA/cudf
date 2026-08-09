/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/transform.hpp>
#include <cudf/types.hpp>

#include <rmm/exec_policy.hpp>

#include <cuda/functional>
#include <cuda/iterator>
#include <thrust/fill.h>
#include <thrust/reduce.h>
#include <thrust/tabulate.h>
#include <thrust/transform.h>

#include <numeric>

// Reuse function defined in `row_bit_count_test.cu`.
namespace row_bit_count_test {
template <typename T>
std::pair<std::unique_ptr<cudf::column>, std::unique_ptr<cudf::column>> build_list_column();
std::pair<std::unique_ptr<cudf::column>, std::unique_ptr<cudf::column>> build_struct_column();
std::unique_ptr<cudf::column> build_nested_column1(std::vector<bool> const& struct_validity);
std::unique_ptr<cudf::column> build_nested_column2(std::vector<bool> const& struct_validity);
}  // namespace row_bit_count_test

namespace {

// Compute row bit count, then sum up sizes for each segment of rows.
std::pair<std::unique_ptr<cudf::column>, std::unique_ptr<cudf::column>>
compute_segmented_row_bit_count(cudf::table_view const& input, cudf::size_type segment_length)
{
  // The expected values are computed with the assumption that
  // the outputs of `cudf::row_bit_count` are correct.
  // This should be fine as they are verified by their own unit tests in `row_bit_count_test.cu`.
  auto const row_sizes    = cudf::row_bit_count(input);
  auto const num_segments = cudf::util::div_rounding_up_safe(row_sizes->size(), segment_length);
  auto expected = cudf::make_fixed_width_column(
    cudf::data_type{cudf::type_to_id<cudf::size_type>()}, num_segments);

  thrust::transform(
    rmm::exec_policy_nosync(cudf::get_default_stream()),
    cuda::counting_iterator<cudf::size_type>{0},
    cuda::counting_iterator{num_segments},
    expected->mutable_view().begin<cudf::size_type>(),
    cuda::proclaim_return_type<cudf::size_type>(
      [segment_length,
       num_segments,
       num_rows = row_sizes->size(),
       d_sizes  = row_sizes->view().begin<cudf::size_type>()] __device__(auto const segment_idx) {
        // Since the number of rows may not divisible by segment_length,
        // the last segment may be shorter than the others.
        auto const size_begin = d_sizes + segment_idx * segment_length;
        auto const size_end   = cuda::std::min(size_begin + segment_length, d_sizes + num_rows);
        return thrust::reduce(thrust::seq, size_begin, size_end);
      }));

  auto actual = cudf::segmented_row_bit_count(input, segment_length);
  return {std::move(expected), std::move(actual)};
}

}  // namespace

struct SegmentedRowBitCount : public cudf::test::BaseFixture {};

TEST_F(SegmentedRowBitCount, Lists)
{
  auto const col   = std::get<0>(row_bit_count_test::build_list_column<int32_t>());
  auto const input = cudf::table_view({*col});

  auto constexpr segment_length = 3;
  auto const [expected, actual] = compute_segmented_row_bit_count(input, segment_length);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*expected, *actual);
}

TEST_F(SegmentedRowBitCount, StringsWithNulls)
{
  // clang-format off
  std::vector<std::string> const strings { "daïs", "def", "", "z", "bananas", "warp", "", "zing" };
  std::vector<bool>        const valids  {  1,      0,    0,  1,   0,          1,      1,  1 };
  // clang-format on
  cudf::test::strings_column_wrapper const col(strings.begin(), strings.end(), valids.begin());
  auto const input = cudf::table_view({col});

  auto constexpr segment_length = 2;
  auto const [expected, actual] = compute_segmented_row_bit_count(input, segment_length);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*expected, *actual);
}

TEST_F(SegmentedRowBitCount, StructsWithNulls)
{
  auto const col   = std::get<0>(row_bit_count_test::build_struct_column());
  auto const input = cudf::table_view({*col});

  auto constexpr segment_length = 2;
  auto const [expected, actual] = compute_segmented_row_bit_count(input, segment_length);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*expected, *actual);
}

TEST_F(SegmentedRowBitCount, NestedTypes)
{
  auto constexpr segment_length = 2;

  {
    // List<Struct<List<int>, float, List<int>, int16>
    auto const col   = row_bit_count_test::build_nested_column1({1, 1, 1, 1, 1, 1, 1, 1});
    auto const input = cudf::table_view({*col});
    auto const [expected, actual] = compute_segmented_row_bit_count(input, segment_length);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*expected, *actual);
  }
  {
    // List<Struct<List<int>, float, List<int>, int16>
    auto const col   = row_bit_count_test::build_nested_column1({0, 0, 1, 1, 1, 1, 1, 1});
    auto const input = cudf::table_view({*col});
    auto const [expected, actual] = compute_segmented_row_bit_count(input, segment_length);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*expected, *actual);
  }

  {
    // List<Struct<List<List<int>>, Struct<int16>>>
    auto const col                = row_bit_count_test::build_nested_column2({1, 1, 1});
    auto const input              = cudf::table_view({*col});
    auto const [expected, actual] = compute_segmented_row_bit_count(input, segment_length);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*expected, *actual);
  }
  {
    // List<Struct<List<List<int>>, Struct<int16>>>
    auto const col                = row_bit_count_test::build_nested_column2({1, 0, 1});
    auto const input              = cudf::table_view({*col});
    auto const [expected, actual] = compute_segmented_row_bit_count(input, segment_length);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*expected, *actual);
  }
}

TEST_F(SegmentedRowBitCount, NestedTypesTable)
{
  auto const col0  = row_bit_count_test::build_nested_column1({1, 1, 1, 1, 1, 1, 1, 1});
  auto const col1  = std::get<0>(row_bit_count_test::build_struct_column());
  auto const col2  = std::get<0>(row_bit_count_test::build_list_column<int16_t>());
  auto const input = cudf::table_view({*col0, *col1, *col2});

  {
    auto const segment_length     = 2;
    auto const [expected, actual] = compute_segmented_row_bit_count(input, segment_length);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*expected, *actual);
  }

  {
    auto const segment_length     = 4;
    auto const [expected, actual] = compute_segmented_row_bit_count(input, segment_length);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*expected, *actual);
  }

  {
    auto const segment_length     = 5;
    auto const [expected, actual] = compute_segmented_row_bit_count(input, segment_length);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*expected, *actual);
  }
}

TEST_F(SegmentedRowBitCount, EmptyInput)
{
  {
    auto const input = cudf::table_view{};
    {
      auto const result = cudf::segmented_row_bit_count(input, 0);
      EXPECT_TRUE(result != nullptr && result->size() == 0);
    }
    {
      auto const result = cudf::segmented_row_bit_count(input, 1000);
      EXPECT_TRUE(result != nullptr && result->size() == 0);
    }
  }

  {
    auto const strings = cudf::make_empty_column(cudf::type_id::STRING);
    auto const ints    = cudf::make_empty_column(cudf::type_id::INT32);
    auto const input   = cudf::table_view{{*strings, *ints}};
    {
      auto const result = cudf::segmented_row_bit_count(input, 0);
      EXPECT_TRUE(result != nullptr && result->size() == 0);
    }
    {
      auto const result = cudf::segmented_row_bit_count(input, 1000);
      EXPECT_TRUE(result != nullptr && result->size() == 0);
    }
  }
}

TEST_F(SegmentedRowBitCount, InvalidSegment)
{
  auto const col = cudf::make_fixed_width_column(cudf::data_type{cudf::type_to_id<int32_t>()}, 16);
  auto const input = cudf::table_view({*col});

  EXPECT_NO_THROW(cudf::segmented_row_bit_count(input, 1));
  EXPECT_NO_THROW(cudf::segmented_row_bit_count(input, input.num_rows()));
  EXPECT_THROW(cudf::segmented_row_bit_count(input, -1), std::invalid_argument);
  EXPECT_THROW(cudf::segmented_row_bit_count(input, 0), std::invalid_argument);
  EXPECT_THROW(cudf::segmented_row_bit_count(input, input.num_rows() + 1), std::invalid_argument);
  EXPECT_THROW(cudf::segmented_row_bit_count(input, 1000), std::invalid_argument);
}

TEST_F(SegmentedRowBitCount, EdgeCases)
{
  auto const col0  = row_bit_count_test::build_nested_column1({1, 1, 1, 1, 1, 1, 1, 1});
  auto const col1  = std::get<0>(row_bit_count_test::build_struct_column());
  auto const col2  = std::get<0>(row_bit_count_test::build_list_column<int16_t>());
  auto const input = cudf::table_view({*col0, *col1, *col2});

  {
    auto const segment_length     = 1;
    auto const [expected, actual] = compute_segmented_row_bit_count(input, segment_length);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*expected, *actual);
  }

  {
    EXPECT_EQ(input.num_rows(), 6);
    auto const segment_length     = 4;  // input.num_rows()==6, not divisible by segment_length .
    auto const [expected, actual] = compute_segmented_row_bit_count(input, segment_length);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*expected, *actual);
  }

  {
    auto const segment_length     = input.num_rows();
    auto const [expected, actual] = compute_segmented_row_bit_count(input, segment_length);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*expected, *actual);
  }
}
