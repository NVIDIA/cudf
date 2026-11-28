/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/row_operator/equality.cuh>
#include <cudf/dictionary/encode.hpp>
#include <cudf/table/equality.hpp>
#include <cudf/utilities/type_checks.hpp>

#include <rmm/exec_policy.hpp>

#include <cuda/buffer>
#include <cuda/iterator>
#include <thrust/transform.h>

#include <vector>

namespace {

template <typename Comparator>
struct corresponding_row_equality {
  Comparator equal;

  __device__ bool operator()(cudf::size_type row) const
  {
    return equal(cudf::detail::row::lhs_index_type{row}, cudf::detail::row::rhs_index_type{row});
  }
};

template <typename T>
struct EqualScaleTypeValidationTest : cudf::test::BaseFixtureWithHarness {};

TYPED_TEST_SUITE(EqualScaleTypeValidationTest, cudf::test::FixedPointTypes);

TYPED_TEST(EqualScaleTypeValidationTest, ExactTypeGateIncludesRecursiveDecimalScales)
{
  using Rep         = cudf::device_storage_type_t<TypeParam>;
  auto const stream = this->stream();
  auto const mr     = this->resources();
  auto lhs =
    cudf::test::fixed_point_column_wrapper<Rep>{{1, 2, 3}, numeric::scale_type{-1}, stream, mr};
  auto rhs =
    cudf::test::fixed_point_column_wrapper<Rep>{{10, 20, 30}, numeric::scale_type{-2}, stream, mr};

  // The general comparator must retain logical comparisons outside the exact-type gate.
  auto const comparison = cudf::detail::row::equality::two_table_comparator{
    cudf::table_view{{lhs}}, cudf::table_view{{rhs}}, stream, mr.get_temporary_mr()};
  auto const equal = comparison.equal_to<false>(cudf::nullate::NO{});
  auto output = cudf::test::fixed_width_column_wrapper<bool>{{false, false, false}, stream, mr};
  thrust::transform(rmm::exec_policy_nosync(stream, mr.get_temporary_mr()),
                    cuda::counting_iterator<cudf::size_type>{0},
                    cuda::counting_iterator<cudf::size_type>{3},
                    cudf::mutable_column_view{output}.data<bool>(),
                    corresponding_row_equality{equal});
  auto const expected =
    cudf::test::fixed_width_column_wrapper<bool>{{true, true, true}, stream, mr};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    expected, output, cudf::test::debug_output_level::FIRST_ERROR, stream, mr);

  auto left_dictionary =
    cudf::dictionary::encode(lhs, cudf::data_type{cudf::type_id::INT32}, stream, mr);
  auto right_dictionary =
    cudf::dictionary::encode(rhs, cudf::data_type{cudf::type_id::INT32}, stream, mr);
  auto left_struct     = cudf::test::structs_column_wrapper{{lhs}, {}, stream, mr};
  auto right_struct    = cudf::test::structs_column_wrapper{{rhs}, {}, stream, mr};
  auto const make_list = [&](cudf::column_view child) {
    auto offsets =
      cudf::test::fixed_width_column_wrapper<cudf::size_type>{{0, 1, 2, 3}, stream, mr};
    return cudf::make_lists_column(
      3,
      offsets.release(),
      std::make_unique<cudf::column>(child, stream, mr.get_output_mr()),
      0,
      cuda::device_buffer<cudf::bitmask_type>{stream, mr.get_output_mr(), 0, cuda::no_init});
  };
  auto left_list  = make_list(lhs);
  auto right_list = make_list(rhs);
  std::vector<cudf::column_view> const left{
    lhs, left_dictionary->view(), left_struct, left_list->view()};
  std::vector<cudf::column_view> const right{
    rhs, right_dictionary->view(), right_struct, right_list->view()};
  for (std::size_t i = 0; i < left.size(); ++i) {
    // These logical values match, so rejection must come from the recursive types.
    EXPECT_FALSE(cudf::have_same_types(left[i], right[i]));
    EXPECT_FALSE(cudf::have_same_types(cudf::table_view{{left[i]}}, cudf::table_view{{right[i]}}));
    EXPECT_TRUE(cudf::have_same_types(left[i], left[i]));
    EXPECT_FALSE(cudf::tables_equal(cudf::table_view{{left[i]}},
                                    cudf::table_view{{right[i]}},
                                    cudf::null_equality::EQUAL,
                                    stream));
  }
}

}  // namespace
