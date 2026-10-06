/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/aggregation.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/reduction.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <limits>
#include <utility>
#include <vector>

#define XXX 0  // null placeholder

template <typename T>
struct SegmentedReductionTest : public cudf::test::BaseFixture {};

struct SegmentedReductionTestUntyped : public cudf::test::BaseFixture {};

TYPED_TEST_CASE(SegmentedReductionTest, cudf::test::NumericTypes);

TYPED_TEST(SegmentedReductionTest, SumExcludeNulls)
{
  // [1, 2, 3], [1, null, 3], [1], [null], [null, null], []
  // values:   {1, 2, 3, 1, XXX, 3, 1, XXX, XXX, XXX}
  // offsets:  {0, 3, 6, 7, 8, 10, 10}
  // nullmask: {1, 1, 1, 1, 0, 1, 1, 0, 0, 0}
  // outputs:  {6, 4, 1, XXX, XXX, XXX}
  // output nullmask: {1, 1, 1, 0, 0, 0}
  auto const input = cudf::test::fixed_width_column_wrapper<TypeParam>{
    {1, 2, 3, 1, XXX, 3, 1, XXX, XXX, XXX}, {1, 1, 1, 1, 0, 1, 1, 0, 0, 0}};
  auto const offsets   = std::vector<cudf::size_type>{0, 3, 6, 7, 8, 10, 10};
  auto const d_offsets = cudf::detail::make_device_uvector_async(
    offsets, cudf::get_default_stream(), cudf::get_current_device_resource_ref());
  auto const expect =
    cudf::test::fixed_width_column_wrapper<TypeParam>{{6, 4, 1, XXX, XXX, XXX}, {1, 1, 1, 0, 0, 0}};

  auto res =
    cudf::segmented_reduce(input,
                           d_offsets,
                           *cudf::make_sum_aggregation<cudf::segmented_reduce_aggregation>(),
                           cudf::data_type{cudf::type_to_id<TypeParam>()},
                           cudf::null_policy::EXCLUDE);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*res, expect);

  // Test with initial value
  auto const init_scalar = cudf::make_fixed_width_scalar<TypeParam>(3);
  auto const init_expect =
    cudf::test::fixed_width_column_wrapper<TypeParam>{{9, 7, 4, 3, 3, 3}, {1, 1, 1, 1, 1, 1}};

  res = cudf::segmented_reduce(input,
                               d_offsets,
                               *cudf::make_sum_aggregation<cudf::segmented_reduce_aggregation>(),
                               cudf::data_type{cudf::type_to_id<TypeParam>()},
                               cudf::null_policy::EXCLUDE,
                               *init_scalar);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*res, init_expect);

  // Test with null initial value
  init_scalar->set_valid_async(false);
  res = cudf::segmented_reduce(input,
                               d_offsets,
                               *cudf::make_sum_aggregation<cudf::segmented_reduce_aggregation>(),
                               cudf::data_type{cudf::type_to_id<TypeParam>()},
                               cudf::null_policy::EXCLUDE,
                               *init_scalar);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*res, expect);
}

TYPED_TEST(SegmentedReductionTest, ProductExcludeNulls)
{
  // [1, 3, 5], [null, 3, 5], [1], [null], [null, null], []
  // values:    {1, 3, 5, XXX, 3, 5, 1, XXX, XXX, XXX}
  // offsets:   {0, 3, 6, 7, 8, 10, 10}
  // nullmask:  {1, 1, 1, 0, 1, 1, 1, 0, 0, 0}
  // outputs:   {15, 15, 1, XXX, XXX, XXX}
  // output nullmask: {1, 1, 1, 0, 0, 0}
  auto const input = cudf::test::fixed_width_column_wrapper<TypeParam>{
    {1, 3, 5, XXX, 3, 5, 1, XXX, XXX, XXX}, {1, 1, 1, 0, 1, 1, 1, 0, 0, 0}};
  auto const offsets   = std::vector<cudf::size_type>{0, 3, 6, 7, 8, 10, 10};
  auto const d_offsets = cudf::detail::make_device_uvector_async(
    offsets, cudf::get_default_stream(), cudf::get_current_device_resource_ref());
  auto const expect = cudf::test::fixed_width_column_wrapper<TypeParam>{{15, 15, 1, XXX, XXX, XXX},
                                                                        {1, 1, 1, 0, 0, 0}};

  auto res =
    cudf::segmented_reduce(input,
                           d_offsets,
                           *cudf::make_product_aggregation<cudf::segmented_reduce_aggregation>(),
                           cudf::data_type{cudf::type_to_id<TypeParam>()},
                           cudf::null_policy::EXCLUDE);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*res, expect);

  // Test with initial value
  auto const init_scalar = cudf::make_fixed_width_scalar<TypeParam>(3);
  auto const init_expect =
    cudf::test::fixed_width_column_wrapper<TypeParam>{{45, 45, 3, 3, 3, 3}, {1, 1, 1, 1, 1, 1}};

  res =
    cudf::segmented_reduce(input,
                           d_offsets,
                           *cudf::make_product_aggregation<cudf::segmented_reduce_aggregation>(),
                           cudf::data_type{cudf::type_to_id<TypeParam>()},
                           cudf::null_policy::EXCLUDE,
                           *init_scalar);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*res, init_expect);

  // Test with null initial value
  init_scalar->set_valid_async(false);
  res =
    cudf::segmented_reduce(input,
                           d_offsets,
                           *cudf::make_product_aggregation<cudf::segmented_reduce_aggregation>(),
                           cudf::data_type{cudf::type_to_id<TypeParam>()},
                           cudf::null_policy::EXCLUDE,
                           *init_scalar);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*res, expect);
}

TYPED_TEST(SegmentedReductionTest, MaxExcludeNulls)
{
  // [1, 2, 3], [1, null, 3], [1], [null], [null, null], []
  // values:    {1, 2, 3, 1, XXX, 3, 1, XXX, XXX, XXX}
  // offsets:   {0, 3, 6, 7, 8, 10, 10}
  // nullmask:  {1, 1, 1, 1, 0, 1, 1, 0, 0, 0}
  // outputs:   {3, 3, 1, XXX, XXX, XXX}
  // output nullmask: {1, 1, 1, 0, 0, 0}
  auto const input = cudf::test::fixed_width_column_wrapper<TypeParam>{
    {1, 2, 3, 1, XXX, 3, 1, XXX, XXX, XXX}, {1, 1, 1, 1, 0, 1, 1, 0, 0, 0}};
  auto const offsets   = std::vector<cudf::size_type>{0, 3, 6, 7, 8, 10, 10};
  auto const d_offsets = cudf::detail::make_device_uvector_async(
    offsets, cudf::get_default_stream(), cudf::get_current_device_resource_ref());
  auto const expect =
    cudf::test::fixed_width_column_wrapper<TypeParam>{{3, 3, 1, XXX, XXX, XXX}, {1, 1, 1, 0, 0, 0}};

  auto res =
    cudf::segmented_reduce(input,
                           d_offsets,
                           *cudf::make_max_aggregation<cudf::segmented_reduce_aggregation>(),
                           cudf::data_type{cudf::type_to_id<TypeParam>()},
                           cudf::null_policy::EXCLUDE);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*res, expect);

  // Test with initial value
  auto const init_scalar = cudf::make_fixed_width_scalar<TypeParam>(2);
  auto const init_expect =
    cudf::test::fixed_width_column_wrapper<TypeParam>{{3, 3, 2, 2, 2, 2}, {1, 1, 1, 1, 1, 1}};

  res = cudf::segmented_reduce(input,
                               d_offsets,
                               *cudf::make_max_aggregation<cudf::segmented_reduce_aggregation>(),
                               cudf::data_type{cudf::type_to_id<TypeParam>()},
                               cudf::null_policy::EXCLUDE,
                               *init_scalar);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*res, init_expect);

  // Test with null initial value
  init_scalar->set_valid_async(false);
  res = cudf::segmented_reduce(input,
                               d_offsets,
                               *cudf::make_max_aggregation<cudf::segmented_reduce_aggregation>(),
                               cudf::data_type{cudf::type_to_id<TypeParam>()},
                               cudf::null_policy::EXCLUDE,
                               *init_scalar);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*res, expect);
}

TYPED_TEST(SegmentedReductionTest, MinExcludeNulls)
{
  // [1, 2, 3], [1, null, 3], [1], [null], [null, null], []
  // values:   {1, 2, 3, 1, XXX, 3, 1, XXX, XXX, XXX}
  // offsets:  {0, 3, 6, 7, 8, 10, 10}
  // nullmask: {1, 1, 1, 1, 0, 1, 1, 0, 0, 0}
  // outputs:  {1, 1, 1, XXX, XXX, XXX}
  // output nullmask: {1, 1, 1, 0, 0, 0}
  auto const input = cudf::test::fixed_width_column_wrapper<TypeParam>{
    {1, 2, 3, 1, XXX, 3, 1, XXX, XXX, XXX}, {1, 1, 1, 1, 0, 1, 1, 0, 0, 0}};
  auto const offsets   = std::vector<cudf::size_type>{0, 3, 6, 7, 8, 10, 10};
  auto const d_offsets = cudf::detail::make_device_uvector_async(
    offsets, cudf::get_default_stream(), cudf::get_current_device_resource_ref());
  auto const expect =
    cudf::test::fixed_width_column_wrapper<TypeParam>{{1, 1, 1, XXX, XXX, XXX}, {1, 1, 1, 0, 0, 0}};

  auto res =
    cudf::segmented_reduce(input,
                           d_offsets,
                           *cudf::make_min_aggregation<cudf::segmented_reduce_aggregation>(),
                           cudf::data_type{cudf::type_to_id<TypeParam>()},
                           cudf::null_policy::EXCLUDE);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*res, expect);

  // Test with initial value
  auto const init_scalar = cudf::make_fixed_width_scalar<TypeParam>(2);
  auto const init_expect =
    cudf::test::fixed_width_column_wrapper<TypeParam>{{1, 1, 1, 2, 2, 2}, {1, 1, 1, 1, 1, 1}};

  res = cudf::segmented_reduce(input,
                               d_offsets,
                               *cudf::make_min_aggregation<cudf::segmented_reduce_aggregation>(),
                               cudf::data_type{cudf::type_to_id<TypeParam>()},
                               cudf::null_policy::EXCLUDE,
                               *init_scalar);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*res, init_expect);

  // Test with null initial value
  init_scalar->set_valid_async(false);
  res = cudf::segmented_reduce(input,
                               d_offsets,
                               *cudf::make_min_aggregation<cudf::segmented_reduce_aggregation>(),
                               cudf::data_type{cudf::type_to_id<TypeParam>()},
                               cudf::null_policy::EXCLUDE,
                               *init_scalar);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*res, expect);
}

TYPED_TEST(SegmentedReductionTest, AnyExcludeNulls)
{
  // [0, 0, 0], [0, null, 0], [0, 1, 0], [1, null, 0], [], [0], [1], [null], [null, null]
  // values:  {0, 0, 0, 0, XXX, 0, 0, 1, 0, 1, XXX, 0, 0, 1, XXX, XXX, XXX}
  // offsets: {0, 3, 6, 9, 12, 12, 13, 14, 15, 17}
  // nullmask:{1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0}
  // outputs: {0, 0, 1, 1, XXX, 0, 1, XXX, XXX}
  // output nullmask: {1, 1, 1, 1, 0, 1, 1, 0, 0}
  auto const input = cudf::test::fixed_width_column_wrapper<TypeParam>{
    {0, 0, 0, 0, XXX, 0, 0, 1, 0, 1, XXX, 0, 0, 1, XXX, XXX, XXX},
    {1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0}};
  auto const offsets   = std::vector<cudf::size_type>{0, 3, 6, 9, 12, 12, 13, 14, 15, 17};
  auto const d_offsets = cudf::detail::make_device_uvector_async(
    offsets, cudf::get_default_stream(), cudf::get_current_device_resource_ref());
  auto const expect = cudf::test::fixed_width_column_wrapper<bool>{
    {false, false, true, true, bool{XXX}, false, true, bool{XXX}, bool{XXX}},
    {true, true, true, true, false, true, true, false, false}};

  auto const agg         = cudf::make_any_aggregation<cudf::segmented_reduce_aggregation>();
  auto const output_type = cudf::data_type{cudf::type_id::BOOL8};
  auto const policy      = cudf::null_policy::EXCLUDE;

  auto res = cudf::segmented_reduce(input, d_offsets, *agg, output_type, policy);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*res, expect);

  // Test with initial value
  auto const init_scalar = cudf::make_fixed_width_scalar<TypeParam>(0);
  auto const init_expect = cudf::test::fixed_width_column_wrapper<bool>{
    {false, false, true, true, false, false, true, false, false},
    {true, true, true, true, true, true, true, true, true}};

  res = cudf::segmented_reduce(input, d_offsets, *agg, output_type, policy, *init_scalar);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*res, init_expect);

  // Test with null initial value
  init_scalar->set_valid_async(false);
  res = cudf::segmented_reduce(input, d_offsets, *agg, output_type, policy, *init_scalar);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*res, expect);
}

TYPED_TEST(SegmentedReductionTest, AllExcludeNulls)
{
  // [1, 2, 3], [1, null, 3], [], [1], [null], [null, null], [1, 0, 3], [1, null, 0], [0]
  // values: {1, 2, 3, 1, XXX, 3, 1, XXX, XXX, XXX, 1, 0, 3, 1, XXX, 0, 0}
  // offsets: {0, 3, 6, 6, 7, 8, 10, 13, 16, 17}
  // nullmask: {1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1}
  // outputs: {true, true, XXX, true, XXX, XXX, false, false, false}
  // output nullmask: {1, 1, 0, 1, 0, 0, 1, 1, 1}
  auto const input = cudf::test::fixed_width_column_wrapper<TypeParam>{
    {1, 2, 3, 1, XXX, 3, 1, XXX, XXX, XXX, 1, 0, 3, 1, XXX, 0, 0},
    {1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1}};
  auto const offsets   = std::vector<cudf::size_type>{0, 3, 6, 6, 7, 8, 10, 13, 16, 17};
  auto const d_offsets = cudf::detail::make_device_uvector_async(
    offsets, cudf::get_default_stream(), cudf::get_current_device_resource_ref());
  auto const expect = cudf::test::fixed_width_column_wrapper<bool>{
    {true, true, bool{XXX}, true, bool{XXX}, bool{XXX}, false, false, false},
    {true, true, false, true, false, false, true, true, true}};

  auto res =
    cudf::segmented_reduce(input,
                           d_offsets,
                           *cudf::make_all_aggregation<cudf::segmented_reduce_aggregation>(),
                           cudf::data_type{cudf::type_id::BOOL8},
                           cudf::null_policy::EXCLUDE);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*res, expect);

  // Test with initial value
  auto const init_scalar = cudf::make_fixed_width_scalar<TypeParam>(1);
  auto const init_expect = cudf::test::fixed_width_column_wrapper<bool>{
    {true, true, true, true, true, true, false, false, false},
    {true, true, true, true, true, true, true, true, true}};

  res = cudf::segmented_reduce(input,
                               d_offsets,
                               *cudf::make_all_aggregation<cudf::segmented_reduce_aggregation>(),
                               cudf::data_type{cudf::type_id::BOOL8},
                               cudf::null_policy::EXCLUDE,
                               *init_scalar);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*res, init_expect);

  // Test with null initial value
  init_scalar->set_valid_async(false);
  res = cudf::segmented_reduce(input,
                               d_offsets,
                               *cudf::make_all_aggregation<cudf::segmented_reduce_aggregation>(),
                               cudf::data_type{cudf::type_id::BOOL8},
                               cudf::null_policy::EXCLUDE,
                               *init_scalar);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*res, expect);
}

TYPED_TEST(SegmentedReductionTest, SumIncludeNulls)
{
  // [1, 2, 3], [1, null, 3], [1], [null], [null, null], []
  // values:   {1, 2, 3, 1, XXX, 3, 1, XXX, XXX, XXX}
  // offsets:  {0, 3, 6, 7, 8, 10, 10}
  // nullmask: {1, 1, 1, 1, 0, 1, 1, 0, 0, 0}
  // outputs:  {6, XXX, 1, XXX, XXX, XXX}
  // output nullmask: {1, 0, 1, 0, 0, 0}
  auto const input = cudf::test::fixed_width_column_wrapper<TypeParam>{
    {1, 2, 3, 1, XXX, 3, 1, XXX, XXX, XXX}, {1, 1, 1, 1, 0, 1, 1, 0, 0, 0}};
  auto const offsets   = std::vector<cudf::size_type>{0, 3, 6, 7, 8, 10, 10};
  auto const d_offsets = cudf::detail::make_device_uvector_async(
    offsets, cudf::get_default_stream(), cudf::get_current_device_resource_ref());
  auto const expect = cudf::test::fixed_width_column_wrapper<TypeParam>{{6, XXX, 1, XXX, XXX, XXX},
                                                                        {1, 0, 1, 0, 0, 0}};

  auto res =
    cudf::segmented_reduce(input,
                           d_offsets,
                           *cudf::make_sum_aggregation<cudf::segmented_reduce_aggregation>(),
                           cudf::data_type{cudf::type_to_id<TypeParam>()},
                           cudf::null_policy::INCLUDE);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*res, expect);

  // Test with initial value
  auto const init_scalar = cudf::make_fixed_width_scalar<TypeParam>(3);
  auto const init_expect =
    cudf::test::fixed_width_column_wrapper<TypeParam>{{9, XXX, 4, XXX, XXX, 3}, {1, 0, 1, 0, 0, 1}};

  res = cudf::segmented_reduce(input,
                               d_offsets,
                               *cudf::make_sum_aggregation<cudf::segmented_reduce_aggregation>(),
                               cudf::data_type{cudf::type_to_id<TypeParam>()},
                               cudf::null_policy::INCLUDE,
                               *init_scalar);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*res, init_expect);

  // Test with null initial value
  init_scalar->set_valid_async(false);
  auto null_init_expect = cudf::test::fixed_width_column_wrapper<TypeParam>{
    {XXX, XXX, XXX, XXX, XXX, XXX}, {0, 0, 0, 0, 0, 0}};

  res = cudf::segmented_reduce(input,
                               d_offsets,
                               *cudf::make_sum_aggregation<cudf::segmented_reduce_aggregation>(),
                               cudf::data_type{cudf::type_to_id<TypeParam>()},
                               cudf::null_policy::INCLUDE,
                               *init_scalar);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*res, null_init_expect);
}

TYPED_TEST(SegmentedReductionTest, ProductIncludeNulls)
{
  // [1, 3, 5], [null, 3, 5], [1], [null], [null, null], []
  // values:    {1, 3, 5, XXX, 3, 5, 1, XXX, XXX, XXX}
  // offsets:   {0, 3, 6, 7, 8, 10, 10}
  // nullmask:  {1, 1, 1, 0, 1, 1, 1, 0, 0, 0}
  // outputs:   {15, XXX, 1, XXX, XXX, XXX}
  // output nullmask: {1, 0, 1, 0, 0, 0}
  auto const input = cudf::test::fixed_width_column_wrapper<TypeParam>{
    {1, 3, 5, XXX, 3, 5, 1, XXX, XXX, XXX}, {1, 1, 1, 0, 1, 1, 1, 0, 0, 0}};
  auto const offsets   = std::vector<cudf::size_type>{0, 3, 6, 7, 8, 10, 10};
  auto const d_offsets = cudf::detail::make_device_uvector_async(
    offsets, cudf::get_default_stream(), cudf::get_current_device_resource_ref());
  auto const expect = cudf::test::fixed_width_column_wrapper<TypeParam>{{15, XXX, 1, XXX, XXX, XXX},
                                                                        {1, 0, 1, 0, 0, 0}};

  auto res =
    cudf::segmented_reduce(input,
                           d_offsets,
                           *cudf::make_product_aggregation<cudf::segmented_reduce_aggregation>(),
                           cudf::data_type{cudf::type_to_id<TypeParam>()},
                           cudf::null_policy::INCLUDE);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*res, expect);

  // Test with initial value
  auto const init_scalar = cudf::make_fixed_width_scalar<TypeParam>(3);
  auto const init_expect = cudf::test::fixed_width_column_wrapper<TypeParam>{
    {45, XXX, 3, XXX, XXX, 3}, {1, 0, 1, 0, 0, 1}};

  res =
    cudf::segmented_reduce(input,
                           d_offsets,
                           *cudf::make_product_aggregation<cudf::segmented_reduce_aggregation>(),
                           cudf::data_type{cudf::type_to_id<TypeParam>()},
                           cudf::null_policy::INCLUDE,
                           *init_scalar);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*res, init_expect);

  // Test with null initial value
  init_scalar->set_valid_async(false);
  auto null_init_expect = cudf::test::fixed_width_column_wrapper<TypeParam>{
    {XXX, XXX, XXX, XXX, XXX, XXX}, {0, 0, 0, 0, 0, 0}};

  res =
    cudf::segmented_reduce(input,
                           d_offsets,
                           *cudf::make_product_aggregation<cudf::segmented_reduce_aggregation>(),
                           cudf::data_type{cudf::type_to_id<TypeParam>()},
                           cudf::null_policy::INCLUDE,
                           *init_scalar);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*res, null_init_expect);
}

TYPED_TEST(SegmentedReductionTest, MaxIncludeNulls)
{
  // [1, 2, 3], [1, null, 3], [1], [null], [null, null], []
  // values:    {1, 2, 3, 1, XXX, 3, 1, XXX, XXX, XXX}
  // offsets:   {0, 3, 6, 7, 8, 10, 10}
  // nullmask:  {1, 1, 1, 1, 0, 1, 1, 0, 0, 0}
  // outputs:   {3, XXX, 1, XXX, XXX, XXX}
  // output nullmask: {1, 0, 1, 0, 0, 0}
  auto const input = cudf::test::fixed_width_column_wrapper<TypeParam>{
    {1, 2, 3, 1, XXX, 3, 1, XXX, XXX, XXX}, {1, 1, 1, 1, 0, 1, 1, 0, 0, 0}};
  auto const offsets   = std::vector<cudf::size_type>{0, 3, 6, 7, 8, 10, 10};
  auto const d_offsets = cudf::detail::make_device_uvector_async(
    offsets, cudf::get_default_stream(), cudf::get_current_device_resource_ref());
  auto const expect = cudf::test::fixed_width_column_wrapper<TypeParam>{{3, XXX, 1, XXX, XXX, XXX},
                                                                        {1, 0, 1, 0, 0, 0}};

  auto res =
    cudf::segmented_reduce(input,
                           d_offsets,
                           *cudf::make_max_aggregation<cudf::segmented_reduce_aggregation>(),
                           cudf::data_type{cudf::type_to_id<TypeParam>()},
                           cudf::null_policy::INCLUDE);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*res, expect);

  // Test with initial value
  auto const init_scalar = cudf::make_fixed_width_scalar<TypeParam>(2);
  auto const init_expect =
    cudf::test::fixed_width_column_wrapper<TypeParam>{{3, XXX, 2, XXX, XXX, 2}, {1, 0, 1, 0, 0, 1}};

  res = cudf::segmented_reduce(input,
                               d_offsets,
                               *cudf::make_max_aggregation<cudf::segmented_reduce_aggregation>(),
                               cudf::data_type{cudf::type_to_id<TypeParam>()},
                               cudf::null_policy::INCLUDE,
                               *init_scalar);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*res, init_expect);

  // Test with null initial value
  init_scalar->set_valid_async(false);
  auto null_init_expect = cudf::test::fixed_width_column_wrapper<TypeParam>{
    {XXX, XXX, XXX, XXX, XXX, XXX}, {0, 0, 0, 0, 0, 0}};

  res = cudf::segmented_reduce(input,
                               d_offsets,
                               *cudf::make_max_aggregation<cudf::segmented_reduce_aggregation>(),
                               cudf::data_type{cudf::type_to_id<TypeParam>()},
                               cudf::null_policy::INCLUDE,
                               *init_scalar);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*res, null_init_expect);
}

TYPED_TEST(SegmentedReductionTest, MinIncludeNulls)
{
  // [1, 2, 3], [1, null, 3], [1], [null], [null, null], []
  // values:   {1, 2, 3, 1, XXX, 3, 1, XXX, XXX}
  // offsets:  {0, 3, 6, 7, 8, 10, 10}
  // nullmask: {1, 1, 1, 1, 0, 1, 1, 0, 0}
  // outputs:  {1, XXX, 1, XXX, XXX, XXX}
  // output nullmask: {1, 0, 1, 0, 0, 0}
  auto const input = cudf::test::fixed_width_column_wrapper<TypeParam>{
    {1, 2, 3, 1, XXX, 3, 1, XXX, XXX, XXX}, {1, 1, 1, 1, 0, 1, 1, 0, 0}};
  auto const offsets   = std::vector<cudf::size_type>{0, 3, 6, 7, 8, 10, 10};
  auto const d_offsets = cudf::detail::make_device_uvector_async(
    offsets, cudf::get_default_stream(), cudf::get_current_device_resource_ref());
  auto const expect = cudf::test::fixed_width_column_wrapper<TypeParam>{{1, XXX, 1, XXX, XXX, XXX},
                                                                        {1, 0, 1, 0, 0, 0}};

  auto res =
    cudf::segmented_reduce(input,
                           d_offsets,
                           *cudf::make_min_aggregation<cudf::segmented_reduce_aggregation>(),
                           cudf::data_type{cudf::type_to_id<TypeParam>()},
                           cudf::null_policy::INCLUDE);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*res, expect);

  // Test with initial value
  auto const init_scalar = cudf::make_fixed_width_scalar<TypeParam>(2);
  auto const init_expect =
    cudf::test::fixed_width_column_wrapper<TypeParam>{{1, XXX, 1, XXX, XXX, 2}, {1, 0, 1, 0, 0, 1}};

  res = cudf::segmented_reduce(input,
                               d_offsets,
                               *cudf::make_min_aggregation<cudf::segmented_reduce_aggregation>(),
                               cudf::data_type{cudf::type_to_id<TypeParam>()},
                               cudf::null_policy::INCLUDE,
                               *init_scalar);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*res, init_expect);

  // Test with null initial value
  init_scalar->set_valid_async(false);
  auto null_init_expect = cudf::test::fixed_width_column_wrapper<TypeParam>{
    {XXX, XXX, XXX, XXX, XXX, XXX}, {0, 0, 0, 0, 0, 0}};

  res = cudf::segmented_reduce(input,
                               d_offsets,
                               *cudf::make_min_aggregation<cudf::segmented_reduce_aggregation>(),
                               cudf::data_type{cudf::type_to_id<TypeParam>()},
                               cudf::null_policy::INCLUDE,
                               *init_scalar);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*res, null_init_expect);
}

TYPED_TEST(SegmentedReductionTest, AnyIncludeNulls)
{
  // [0, 0, 0], [0, null, 0], [0, 1, 0], [1, null, 0], [], [0], [1], [null], [null, null]
  // values:  {0, 0, 0, 0, XXX, 0, 0, 1, 0, 1, XXX, 0, 0, 1, XXX, XXX, XXX}
  // offsets: {0, 3, 6, 9, 12, 12, 13, 14, 15, 17}
  // nullmask:{1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0}
  // outputs: {0, XXX, 1, XXX, XXX, 0, 1, XXX, XXX}
  // output nullmask: {1, 0, 1, 0, 0, 1, 1, 0, 0}
  auto const input = cudf::test::fixed_width_column_wrapper<TypeParam>{
    {0, 0, 0, 0, XXX, 0, 0, 1, 0, 1, XXX, 0, 0, 1, XXX, XXX, XXX},
    {1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0}};
  auto const offsets   = std::vector<cudf::size_type>{0, 3, 6, 9, 12, 12, 13, 14, 15, 17};
  auto const d_offsets = cudf::detail::make_device_uvector_async(
    offsets, cudf::get_default_stream(), cudf::get_current_device_resource_ref());
  auto const expect = cudf::test::fixed_width_column_wrapper<bool>{
    {false, bool{XXX}, true, bool{XXX}, bool{XXX}, false, true, bool{XXX}, bool{XXX}},
    {true, false, true, false, false, true, true, false, false}};

  auto res =
    cudf::segmented_reduce(input,
                           d_offsets,
                           *cudf::make_any_aggregation<cudf::segmented_reduce_aggregation>(),
                           cudf::data_type{cudf::type_id::BOOL8},
                           cudf::null_policy::INCLUDE);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*res, expect);

  // Test with initial value
  auto const init_scalar = cudf::make_fixed_width_scalar<TypeParam>(1);
  auto const init_expect = cudf::test::fixed_width_column_wrapper<bool>{
    {true, bool{XXX}, true, bool{XXX}, true, true, true, bool{XXX}, bool{XXX}},
    {true, false, true, false, true, true, true, false, false}};

  res = cudf::segmented_reduce(input,
                               d_offsets,
                               *cudf::make_any_aggregation<cudf::segmented_reduce_aggregation>(),
                               cudf::data_type{cudf::type_id::BOOL8},
                               cudf::null_policy::INCLUDE,
                               *init_scalar);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*res, init_expect);

  // Test with null initial value
  init_scalar->set_valid_async(false);
  auto null_init_expect = cudf::test::fixed_width_column_wrapper<bool>{
    {bool{XXX},
     bool{XXX},
     bool{XXX},
     bool{XXX},
     bool{XXX},
     bool{XXX},
     bool{XXX},
     bool{XXX},
     bool{XXX}},
    {false, false, false, false, false, false, false, false, false}};

  res = cudf::segmented_reduce(input,
                               d_offsets,
                               *cudf::make_any_aggregation<cudf::segmented_reduce_aggregation>(),
                               cudf::data_type{cudf::type_id::BOOL8},
                               cudf::null_policy::INCLUDE,
                               *init_scalar);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*res, null_init_expect);
}

TYPED_TEST(SegmentedReductionTest, AllIncludeNulls)
{
  // [1, 2, 3], [1, null, 3], [], [1], [null], [null, null], [1, 0, 3], [1, null, 0], [0]
  // values: {1, 2, 3, 1, XXX, 3, 1, XXX, XXX, XXX, 1, 0, 3, 1, XXX, 0, 0}
  // offsets: {0, 3, 6, 6, 7, 8, 10, 13, 16, 17}
  // nullmask: {1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1}
  // outputs: {true, XXX, XXX, true, XXX, XXX, false, XXX, false}
  // output nullmask: {1, 0, 0, 1, 0, 0, 1, 0, 1}
  auto const input = cudf::test::fixed_width_column_wrapper<TypeParam>{
    {1, 2, 3, 1, XXX, 3, 1, XXX, XXX, XXX, 1, 0, 3, 1, XXX, 0, 0},
    {1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1}};
  auto const offsets   = std::vector<cudf::size_type>{0, 3, 6, 6, 7, 8, 10, 13, 16, 17};
  auto const d_offsets = cudf::detail::make_device_uvector_async(
    offsets, cudf::get_default_stream(), cudf::get_current_device_resource_ref());
  auto const expect = cudf::test::fixed_width_column_wrapper<bool>{
    {true, bool{XXX}, bool{XXX}, true, bool{XXX}, bool{XXX}, false, bool{XXX}, false},
    {true, false, false, true, false, false, true, false, true}};

  auto res =
    cudf::segmented_reduce(input,
                           d_offsets,
                           *cudf::make_all_aggregation<cudf::segmented_reduce_aggregation>(),
                           cudf::data_type{cudf::type_id::BOOL8},
                           cudf::null_policy::INCLUDE);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*res, expect);

  // Test with initial value
  auto const init_scalar = cudf::make_fixed_width_scalar<TypeParam>(1);
  auto const init_expect = cudf::test::fixed_width_column_wrapper<bool>{
    {true, bool{XXX}, true, true, bool{XXX}, bool{XXX}, false, bool{XXX}, false},
    {true, false, true, true, false, false, true, false, true}};

  res = cudf::segmented_reduce(input,
                               d_offsets,
                               *cudf::make_all_aggregation<cudf::segmented_reduce_aggregation>(),
                               cudf::data_type{cudf::type_id::BOOL8},
                               cudf::null_policy::INCLUDE,
                               *init_scalar);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*res, init_expect);

  // Test with null initial value
  init_scalar->set_valid_async(false);
  auto null_init_expect = cudf::test::fixed_width_column_wrapper<bool>{
    {bool{XXX},
     bool{XXX},
     bool{XXX},
     bool{XXX},
     bool{XXX},
     bool{XXX},
     bool{XXX},
     bool{XXX},
     bool{XXX}},
    {false, false, false, false, false, false, false, false, false}};

  res = cudf::segmented_reduce(input,
                               d_offsets,
                               *cudf::make_all_aggregation<cudf::segmented_reduce_aggregation>(),
                               cudf::data_type{cudf::type_id::BOOL8},
                               cudf::null_policy::INCLUDE,
                               *init_scalar);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*res, null_init_expect);
}

TEST_F(SegmentedReductionTestUntyped, PartialSegmentReduction)
{
  // Segmented reduction allows offsets only specify part of the input columns.
  // [1], [2, 3], [4]
  // values: {1, 2, 3, 4, 5, 6, 7}
  // offsets: {0, 1, 3, 4}
  // nullmask: {1, 1, 1, 1, 1, 1, 1}
  // outputs: {1, 5, 4}
  // output nullmask: {1, 1, 1}

  auto const input = cudf::test::fixed_width_column_wrapper<int32_t>{
    {1, 2, 3, 4, 5, 6, 7}, {true, true, true, true, true, true, true}};
  auto const offsets   = std::vector<cudf::size_type>{1, 3, 4};
  auto const d_offsets = cudf::detail::make_device_uvector_async(
    offsets, cudf::get_default_stream(), cudf::get_current_device_resource_ref());
  auto const expect = cudf::test::fixed_width_column_wrapper<int32_t>{{5, 4}, {true, true}};

  auto res =
    cudf::segmented_reduce(input,
                           d_offsets,
                           *cudf::make_sum_aggregation<cudf::segmented_reduce_aggregation>(),
                           cudf::data_type{cudf::type_id::INT32},
                           cudf::null_policy::INCLUDE);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*res, expect);

  // Test with initial value
  auto const init_scalar = cudf::make_fixed_width_scalar<int32_t>(3);
  auto const init_expect = cudf::test::fixed_width_column_wrapper<int32_t>{{8, 7}, {true, true}};

  res = cudf::segmented_reduce(input,
                               d_offsets,
                               *cudf::make_sum_aggregation<cudf::segmented_reduce_aggregation>(),
                               cudf::data_type{cudf::type_id::INT32},
                               cudf::null_policy::INCLUDE,
                               *init_scalar);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*res, init_expect);

  // Test with null initial value
  init_scalar->set_valid_async(false);
  auto null_init_expect =
    cudf::test::fixed_width_column_wrapper<int32_t>{{XXX, XXX}, {false, false}};

  res = cudf::segmented_reduce(input,
                               d_offsets,
                               *cudf::make_sum_aggregation<cudf::segmented_reduce_aggregation>(),
                               cudf::data_type{cudf::type_id::INT32},
                               cudf::null_policy::INCLUDE,
                               *init_scalar);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*res, null_init_expect);
}

TEST_F(SegmentedReductionTestUntyped, NonNullableInput)
{
  // Segmented reduction allows offsets only specify part of the input columns.
  // [1], [], [2, 3], [4, 5, 6, 7]
  // values: {1, 2, 3, 4, 5, 6, 7}
  // offsets: {0, 1, 1, 3, 7}
  // nullmask: nullptr
  // outputs: {1, 5, 4}
  // output nullmask: {1, 1, 1}

  auto const input     = cudf::test::fixed_width_column_wrapper<int32_t>{1, 2, 3, 4, 5, 6, 7};
  auto const offsets   = std::vector<cudf::size_type>{0, 1, 1, 3, 7};
  auto const d_offsets = cudf::detail::make_device_uvector_async(
    offsets, cudf::get_default_stream(), cudf::get_current_device_resource_ref());
  auto const expect =
    cudf::test::fixed_width_column_wrapper<int32_t>{{1, XXX, 5, 22}, {true, false, true, true}};

  auto res =
    cudf::segmented_reduce(input,
                           d_offsets,
                           *cudf::make_sum_aggregation<cudf::segmented_reduce_aggregation>(),
                           cudf::data_type{cudf::type_id::INT32},
                           cudf::null_policy::INCLUDE);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*res, expect);

  // Test with initial value
  auto const init_scalar = cudf::make_fixed_width_scalar<int32_t>(3);
  auto const init_expect =
    cudf::test::fixed_width_column_wrapper<int32_t>{{4, 3, 8, 25}, {true, true, true, true}};

  res = cudf::segmented_reduce(input,
                               d_offsets,
                               *cudf::make_sum_aggregation<cudf::segmented_reduce_aggregation>(),
                               cudf::data_type{cudf::type_id::INT32},
                               cudf::null_policy::INCLUDE,
                               *init_scalar);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*res, init_expect);

  // Test with null initial value
  init_scalar->set_valid_async(false);
  auto null_init_expect = cudf::test::fixed_width_column_wrapper<int32_t>{
    {XXX, XXX, XXX, XXX}, {false, false, false, false}};

  res = cudf::segmented_reduce(input,
                               d_offsets,
                               *cudf::make_sum_aggregation<cudf::segmented_reduce_aggregation>(),
                               cudf::data_type{cudf::type_id::INT32},
                               cudf::null_policy::INCLUDE,
                               *init_scalar);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*res, null_init_expect);
}

TEST_F(SegmentedReductionTestUntyped, Mean)
{
  auto const input =
    cudf::test::fixed_width_column_wrapper<int32_t>{10, 20, 30, 40, 50, 60, 70, 80, 90};
  auto const offsets   = std::vector<cudf::size_type>{0, 1, 1, 4, 9};
  auto const d_offsets = cudf::detail::make_device_uvector_async(
    offsets, cudf::get_default_stream(), cudf::get_current_device_resource_ref());
  auto const agg         = cudf::make_mean_aggregation<cudf::segmented_reduce_aggregation>();
  auto const output_type = cudf::data_type{cudf::type_id::FLOAT32};

  auto const expected =
    cudf::test::fixed_width_column_wrapper<float>{{10, 0, 30, 70}, {true, false, true, true}};
  auto result =
    cudf::segmented_reduce(input, d_offsets, *agg, output_type, cudf::null_policy::INCLUDE);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*result, expected);
  result = cudf::segmented_reduce(input, d_offsets, *agg, output_type, cudf::null_policy::EXCLUDE);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*result, expected);
}

TEST_F(SegmentedReductionTestUntyped, MeanNulls)
{
  auto const input = cudf::test::fixed_width_column_wrapper<int32_t>(
    {10, 20, 30, 40, 50, 60, 0, 80, 90}, {true, true, true, true, true, true, false, true, true});
  auto const offsets   = std::vector<cudf::size_type>{0, 1, 1, 4, 9};
  auto const d_offsets = cudf::detail::make_device_uvector_async(
    offsets, cudf::get_default_stream(), cudf::get_current_device_resource_ref());
  auto const agg         = cudf::make_mean_aggregation<cudf::segmented_reduce_aggregation>();
  auto const output_type = cudf::data_type{cudf::type_id::FLOAT64};

  auto expected =
    cudf::test::fixed_width_column_wrapper<double>{{10, 0, 30, 70}, {true, false, true, true}};
  auto result =
    cudf::segmented_reduce(input, d_offsets, *agg, output_type, cudf::null_policy::EXCLUDE);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*result, expected);

  expected =
    cudf::test::fixed_width_column_wrapper<double>{{10, 0, 30, 0}, {true, false, true, false}};
  result = cudf::segmented_reduce(input, d_offsets, *agg, output_type, cudf::null_policy::INCLUDE);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*result, expected);
}

TEST_F(SegmentedReductionTestUntyped, SumOfSquares)
{
  auto const input =
    cudf::test::fixed_width_column_wrapper<int32_t>{10, 20, 30, 40, 50, 60, 70, 80, 90};
  auto const offsets   = std::vector<cudf::size_type>{0, 1, 1, 4, 9};
  auto const d_offsets = cudf::detail::make_device_uvector_async(
    offsets, cudf::get_default_stream(), cudf::get_current_device_resource_ref());
  auto const agg = cudf::make_sum_of_squares_aggregation<cudf::segmented_reduce_aggregation>();
  auto const output_type = cudf::data_type{cudf::type_id::INT32};

  auto const expected = cudf::test::fixed_width_column_wrapper<int32_t>{{100, 0, 2900, 25500},
                                                                        {true, false, true, true}};

  auto result =
    cudf::segmented_reduce(input, d_offsets, *agg, output_type, cudf::null_policy::INCLUDE);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*result, expected);
  result = cudf::segmented_reduce(input, d_offsets, *agg, output_type, cudf::null_policy::EXCLUDE);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*result, expected);
}

TEST_F(SegmentedReductionTestUntyped, SumOfSquaresNulls)
{
  auto const input = cudf::test::fixed_width_column_wrapper<int32_t>(
    {10, 20, 30, 40, 50, 60, 0, 80, 90}, {true, true, true, true, true, true, false, true, true});
  auto const offsets   = std::vector<cudf::size_type>{0, 1, 1, 4, 9};
  auto const d_offsets = cudf::detail::make_device_uvector_async(
    offsets, cudf::get_default_stream(), cudf::get_current_device_resource_ref());
  auto const agg = cudf::make_sum_of_squares_aggregation<cudf::segmented_reduce_aggregation>();
  auto const output_type = cudf::data_type{cudf::type_id::INT64};

  auto expected = cudf::test::fixed_width_column_wrapper<int64_t>{{100, 0, 2900, 20600},
                                                                  {true, false, true, true}};
  auto result =
    cudf::segmented_reduce(input, d_offsets, *agg, output_type, cudf::null_policy::EXCLUDE);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*result, expected);

  expected =
    cudf::test::fixed_width_column_wrapper<int64_t>{{100, 0, 2900, 0}, {true, false, true, false}};
  result = cudf::segmented_reduce(input, d_offsets, *agg, output_type, cudf::null_policy::INCLUDE);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*result, expected);
}

TEST_F(SegmentedReductionTestUntyped, StandardDeviation)
{
  constexpr float NaN{std::numeric_limits<float>::quiet_NaN()};
  auto const input =
    cudf::test::fixed_width_column_wrapper<int32_t>{10, 20, 30, 40, 50, 60, 70, 80, 90};
  auto const offsets   = std::vector<cudf::size_type>{0, 1, 1, 4, 9};
  auto const d_offsets = cudf::detail::make_device_uvector_async(
    offsets, cudf::get_default_stream(), cudf::get_current_device_resource_ref());
  auto const agg         = cudf::make_std_aggregation<cudf::segmented_reduce_aggregation>();
  auto const output_type = cudf::data_type{cudf::type_id::FLOAT32};

  auto expected = cudf::test::fixed_width_column_wrapper<float>{
    {NaN, 0.f, 10.f, static_cast<float>(std::sqrt(250.))}, {true, false, true, true}};
  auto result =
    cudf::segmented_reduce(input, d_offsets, *agg, output_type, cudf::null_policy::INCLUDE);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*result, expected);
  result = cudf::segmented_reduce(input, d_offsets, *agg, output_type, cudf::null_policy::EXCLUDE);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*result, expected);
}

TEST_F(SegmentedReductionTestUntyped, StandardDeviationNulls)
{
  constexpr double NaN{std::numeric_limits<double>::quiet_NaN()};
  auto const input = cudf::test::fixed_width_column_wrapper<int32_t>(
    {10, 0, 20, 30, 54, 63, 0, 72, 81}, {true, false, true, true, true, true, false, true, true});
  auto const offsets   = std::vector<cudf::size_type>{0, 1, 1, 4, 9};
  auto const d_offsets = cudf::detail::make_device_uvector_async(
    offsets, cudf::get_default_stream(), cudf::get_current_device_resource_ref());
  auto const agg         = cudf::make_std_aggregation<cudf::segmented_reduce_aggregation>();
  auto const output_type = cudf::data_type{cudf::type_id::FLOAT64};

  auto expected = cudf::test::fixed_width_column_wrapper<double>{
    {NaN, 0., std::sqrt(50.), std::sqrt(135.)}, {true, false, true, true}};
  auto result =
    cudf::segmented_reduce(input, d_offsets, *agg, output_type, cudf::null_policy::EXCLUDE);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*result, expected);

  expected =
    cudf::test::fixed_width_column_wrapper<double>{{NaN, 0., 0., 0.}, {true, false, false, false}};
  result = cudf::segmented_reduce(input, d_offsets, *agg, output_type, cudf::null_policy::INCLUDE);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*result, expected);
}

TEST_F(SegmentedReductionTestUntyped, Variance)
{
  constexpr float NaN{std::numeric_limits<float>::quiet_NaN()};
  auto const input =
    cudf::test::fixed_width_column_wrapper<int32_t>{10, 20, 30, 40, 50, 60, 70, 80, 90};
  auto const offsets   = std::vector<cudf::size_type>{0, 1, 1, 4, 9};
  auto const d_offsets = cudf::detail::make_device_uvector_async(
    offsets, cudf::get_default_stream(), cudf::get_current_device_resource_ref());
  auto const agg         = cudf::make_variance_aggregation<cudf::segmented_reduce_aggregation>();
  auto const output_type = cudf::data_type{cudf::type_id::FLOAT32};

  auto expected = cudf::test::fixed_width_column_wrapper<float>{{NaN, 0.f, 100.f, 250.f},
                                                                {true, false, true, true}};
  auto result =
    cudf::segmented_reduce(input, d_offsets, *agg, output_type, cudf::null_policy::INCLUDE);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*result, expected);
  result = cudf::segmented_reduce(input, d_offsets, *agg, output_type, cudf::null_policy::EXCLUDE);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*result, expected);
}

TEST_F(SegmentedReductionTestUntyped, VarianceNulls)
{
  constexpr double NaN{std::numeric_limits<double>::quiet_NaN()};
  auto const input = cudf::test::fixed_width_column_wrapper<int32_t>(
    {10, 0, 20, 30, 54, 63, 0, 72, 81}, {true, false, true, true, true, true, false, true, true});
  auto const offsets   = std::vector<cudf::size_type>{0, 1, 1, 4, 9};
  auto const d_offsets = cudf::detail::make_device_uvector_async(
    offsets, cudf::get_default_stream(), cudf::get_current_device_resource_ref());
  auto const agg         = cudf::make_variance_aggregation<cudf::segmented_reduce_aggregation>();
  auto const output_type = cudf::data_type{cudf::type_id::FLOAT64};

  auto expected =
    cudf::test::fixed_width_column_wrapper<double>{{NaN, 0., 50., 135.}, {true, false, true, true}};
  auto result =
    cudf::segmented_reduce(input, d_offsets, *agg, output_type, cudf::null_policy::EXCLUDE);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*result, expected);

  expected =
    cudf::test::fixed_width_column_wrapper<double>{{NaN, 0., 0., 0.}, {true, false, false, false}};
  result = cudf::segmented_reduce(input, d_offsets, *agg, output_type, cudf::null_policy::INCLUDE);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*result, expected);
}

TEST_F(SegmentedReductionTestUntyped, NUnique)
{
  auto const input =
    cudf::test::fixed_width_column_wrapper<int32_t>({10, 15, 20, 30, 60, 60, 70, 70, 80});
  auto const offsets   = std::vector<cudf::size_type>{0, 1, 1, 2, 4, 9};
  auto const d_offsets = cudf::detail::make_device_uvector_async(
    offsets, cudf::get_default_stream(), cudf::get_current_device_resource_ref());
  auto const agg         = cudf::make_nunique_aggregation<cudf::segmented_reduce_aggregation>();
  auto const output_type = cudf::data_type{cudf::type_id::INT32};

  auto expected = cudf::test::fixed_width_column_wrapper<cudf::size_type>{
    {1, 0, 1, 2, 3}, {true, false, true, true, true}};
  auto result =
    cudf::segmented_reduce(input, d_offsets, *agg, output_type, cudf::null_policy::EXCLUDE);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*result, expected);

  result = cudf::segmented_reduce(input, d_offsets, *agg, output_type, cudf::null_policy::INCLUDE);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*result, expected);
}

TEST_F(SegmentedReductionTestUntyped, NUniqueNulls)
{
  auto const input = cudf::test::fixed_width_column_wrapper<int32_t>(
    {10, 0, 20, 30, 60, 60, 70, 70, 0}, {true, false, true, true, true, true, true, true, false});
  auto const offsets   = std::vector<cudf::size_type>{0, 1, 1, 2, 4, 9};
  auto const d_offsets = cudf::detail::make_device_uvector_async(
    offsets, cudf::get_default_stream(), cudf::get_current_device_resource_ref());
  auto const agg         = cudf::make_nunique_aggregation<cudf::segmented_reduce_aggregation>();
  auto const output_type = cudf::data_type{cudf::type_id::INT32};

  auto expected = cudf::test::fixed_width_column_wrapper<cudf::size_type>{
    {1, 0, 0, 2, 2}, {true, false, false, true, true}};
  auto result =
    cudf::segmented_reduce(input, d_offsets, *agg, output_type, cudf::null_policy::EXCLUDE);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*result, expected);

  expected = cudf::test::fixed_width_column_wrapper<cudf::size_type>{
    {1, 0, 1, 2, 3}, {true, false, true, true, true}};
  result = cudf::segmented_reduce(input, d_offsets, *agg, output_type, cudf::null_policy::INCLUDE);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*result, expected);
}

TEST_F(SegmentedReductionTestUntyped, Errors)
{
  auto const input = cudf::test::fixed_width_column_wrapper<int32_t>(
    {10, 0, 20, 30, 54, 63, 0, 72, 81}, {true, false, true, true, true, true, false, true, true});
  auto const offsets   = std::vector<cudf::size_type>{0, 1, 1, 4, 9};
  auto const d_offsets = cudf::detail::make_device_uvector_async(
    offsets, cudf::get_default_stream(), cudf::get_current_device_resource_ref());
  auto const null_policy = cudf::null_policy::EXCLUDE;
  auto const output_type = cudf::data_type{cudf::type_id::TIMESTAMP_DAYS};
  auto const str_input =
    cudf::test::strings_column_wrapper({"10", "0", "20", "30", "54", "63", "", "72", "81"});

  auto const sum_agg = cudf::make_sum_aggregation<cudf::segmented_reduce_aggregation>();
  EXPECT_THROW(cudf::segmented_reduce(input, d_offsets, *sum_agg, output_type, null_policy),
               cudf::logic_error);
  EXPECT_THROW(cudf::segmented_reduce(str_input, d_offsets, *sum_agg, output_type, null_policy),
               cudf::logic_error);

  auto const product_agg = cudf::make_product_aggregation<cudf::segmented_reduce_aggregation>();
  EXPECT_THROW(cudf::segmented_reduce(input, d_offsets, *product_agg, output_type, null_policy),
               cudf::logic_error);
  EXPECT_THROW(cudf::segmented_reduce(str_input, d_offsets, *product_agg, output_type, null_policy),
               cudf::logic_error);

  auto const min_agg = cudf::make_min_aggregation<cudf::segmented_reduce_aggregation>();
  EXPECT_THROW(cudf::segmented_reduce(input, d_offsets, *min_agg, output_type, null_policy),
               cudf::logic_error);

  auto const max_agg = cudf::make_max_aggregation<cudf::segmented_reduce_aggregation>();
  EXPECT_THROW(cudf::segmented_reduce(input, d_offsets, *max_agg, output_type, null_policy),
               cudf::logic_error);

  auto const any_agg = cudf::make_any_aggregation<cudf::segmented_reduce_aggregation>();
  EXPECT_THROW(cudf::segmented_reduce(input, d_offsets, *any_agg, output_type, null_policy),
               cudf::logic_error);
  EXPECT_THROW(cudf::segmented_reduce(str_input, d_offsets, *any_agg, output_type, null_policy),
               cudf::logic_error);

  auto const all_agg = cudf::make_all_aggregation<cudf::segmented_reduce_aggregation>();
  EXPECT_THROW(cudf::segmented_reduce(input, d_offsets, *all_agg, output_type, null_policy),
               cudf::logic_error);
  EXPECT_THROW(cudf::segmented_reduce(str_input, d_offsets, *all_agg, output_type, null_policy),
               cudf::logic_error);

  auto const mean_agg = cudf::make_mean_aggregation<cudf::segmented_reduce_aggregation>();
  EXPECT_THROW(cudf::segmented_reduce(input, d_offsets, *mean_agg, output_type, null_policy),
               cudf::logic_error);
  EXPECT_THROW(cudf::segmented_reduce(str_input, d_offsets, *mean_agg, output_type, null_policy),
               cudf::logic_error);

  auto const std_agg = cudf::make_std_aggregation<cudf::segmented_reduce_aggregation>();
  EXPECT_THROW(cudf::segmented_reduce(input, d_offsets, *std_agg, output_type, null_policy),
               cudf::logic_error);
  EXPECT_THROW(cudf::segmented_reduce(str_input, d_offsets, *std_agg, output_type, null_policy),
               cudf::logic_error);

  auto const var_agg = cudf::make_variance_aggregation<cudf::segmented_reduce_aggregation>();
  EXPECT_THROW(cudf::segmented_reduce(input, d_offsets, *var_agg, output_type, null_policy),
               cudf::logic_error);
  EXPECT_THROW(cudf::segmented_reduce(str_input, d_offsets, *var_agg, output_type, null_policy),
               cudf::logic_error);

  auto const squares_agg =
    cudf::make_sum_of_squares_aggregation<cudf::segmented_reduce_aggregation>();
  EXPECT_THROW(cudf::segmented_reduce(input, d_offsets, *squares_agg, output_type, null_policy),
               cudf::logic_error);
  EXPECT_THROW(cudf::segmented_reduce(str_input, d_offsets, *squares_agg, output_type, null_policy),
               cudf::logic_error);
}

TEST_F(SegmentedReductionTestUntyped, ReduceEmptyColumn)
{
  auto const input     = cudf::test::fixed_width_column_wrapper<int32_t>{};
  auto const offsets   = std::vector<cudf::size_type>{0};
  auto const d_offsets = cudf::detail::make_device_uvector_async(
    offsets, cudf::get_default_stream(), cudf::get_current_device_resource_ref());
  auto const expect = cudf::test::fixed_width_column_wrapper<int32_t>{};

  auto res =
    cudf::segmented_reduce(input,
                           d_offsets,
                           *cudf::make_sum_aggregation<cudf::segmented_reduce_aggregation>(),
                           cudf::data_type{cudf::type_to_id<int32_t>()},
                           cudf::null_policy::EXCLUDE);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*res, expect);

  // Test with initial value
  auto const init_scalar = cudf::make_fixed_width_scalar<int32_t>(3);
  res                    = cudf::segmented_reduce(input,
                               d_offsets,
                               *cudf::make_sum_aggregation<cudf::segmented_reduce_aggregation>(),
                               cudf::data_type{cudf::type_to_id<int32_t>()},
                               cudf::null_policy::EXCLUDE,
                               *init_scalar);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*res, expect);

  // Test with null initial value
  init_scalar->set_valid_async(false);
  res = cudf::segmented_reduce(input,
                               d_offsets,
                               *cudf::make_sum_aggregation<cudf::segmented_reduce_aggregation>(),
                               cudf::data_type{cudf::type_to_id<int32_t>()},
                               cudf::null_policy::EXCLUDE,
                               *init_scalar);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*res, expect);
}

TEST_F(SegmentedReductionTestUntyped, EmptyInputWithOffsets)
{
  auto const input     = cudf::test::fixed_width_column_wrapper<int32_t>{};
  auto const offsets   = std::vector<cudf::size_type>{0, 0, 0, 0, 0, 0};
  auto const d_offsets = cudf::detail::make_device_uvector_async(
    offsets, cudf::get_default_stream(), cudf::get_current_device_resource_ref());
  auto const expect = cudf::test::fixed_width_column_wrapper<int32_t>{
    {XXX, XXX, XXX, XXX, XXX}, {false, false, false, false, false}};

  auto aggregates =
    std::vector<std::unique_ptr<cudf::segmented_reduce_aggregation,
                                std::default_delete<cudf::segmented_reduce_aggregation>>>();
  aggregates.push_back(cudf::make_max_aggregation<cudf::segmented_reduce_aggregation>());
  aggregates.push_back(cudf::make_min_aggregation<cudf::segmented_reduce_aggregation>());
  aggregates.push_back(cudf::make_sum_aggregation<cudf::segmented_reduce_aggregation>());
  aggregates.push_back(cudf::make_product_aggregation<cudf::segmented_reduce_aggregation>());

  auto output_type = cudf::data_type{cudf::type_to_id<int32_t>()};
  for (auto&& agg : aggregates) {
    auto result =
      cudf::segmented_reduce(input, d_offsets, *agg, output_type, cudf::null_policy::EXCLUDE);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expect);
  }

  auto const expect_bool = cudf::test::fixed_width_column_wrapper<bool>{
    {XXX, XXX, XXX, XXX, XXX}, {false, false, false, false, false}};

  auto result =
    cudf::segmented_reduce(input,
                           d_offsets,
                           *cudf::make_any_aggregation<cudf::segmented_reduce_aggregation>(),
                           cudf::data_type{cudf::type_id::BOOL8},
                           cudf::null_policy::INCLUDE);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expect_bool);
  result = cudf::segmented_reduce(input,
                                  d_offsets,
                                  *cudf::make_all_aggregation<cudf::segmented_reduce_aggregation>(),
                                  cudf::data_type{cudf::type_id::BOOL8},
                                  cudf::null_policy::INCLUDE);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expect_bool);
}

TEST_F(SegmentedReductionTestUntyped, EmptyInputEmptyOffsets)
{
  auto const str_empty = cudf::test::strings_column_wrapper{};
  auto const int_empty = cudf::test::fixed_width_column_wrapper<cudf::size_type>{};
  auto result =
    cudf::segmented_reduce(str_empty,
                           cudf::column_view{int_empty},
                           *cudf::make_max_aggregation<cudf::segmented_reduce_aggregation>(),
                           cudf::data_type{cudf::type_id::STRING},
                           cudf::null_policy::EXCLUDE);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, str_empty);

  result = cudf::segmented_reduce(int_empty,
                                  cudf::column_view{int_empty},
                                  *cudf::make_min_aggregation<cudf::segmented_reduce_aggregation>(),
                                  cudf::data_type{cudf::type_id::INT32},
                                  cudf::null_policy::INCLUDE);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, int_empty);
}

template <typename T>
struct SegmentedReductionFixedPointTest : public cudf::test::BaseFixture {};

TYPED_TEST_SUITE(SegmentedReductionFixedPointTest, cudf::test::FixedPointTypes);

TYPED_TEST(SegmentedReductionFixedPointTest, MaxWithNulls)
{
  using RepType = cudf::device_storage_type_t<TypeParam>;

  auto const offsets   = std::vector<cudf::size_type>{0, 3, 6, 7, 8, 10, 10};
  auto const d_offsets = cudf::detail::make_device_uvector_async(
    offsets, cudf::get_default_stream(), cudf::get_current_device_resource_ref());
  auto const agg = cudf::make_max_aggregation<cudf::segmented_reduce_aggregation>();

  for (auto scale : {-2, 0, 5}) {
    auto const input =
      cudf::test::fixed_point_column_wrapper<RepType>({1, 2, 3, 1, XXX, 3, 1, XXX, XXX, XXX},
                                                      {1, 1, 1, 1, 0, 1, 1, 0, 0, 0},
                                                      numeric::scale_type{scale});
    auto out_type = cudf::column_view(input).type();
    auto expect   = cudf::test::fixed_point_column_wrapper<RepType>(
      {3, XXX, 1, XXX, XXX, XXX}, {1, 0, 1, 0, 0, 0}, numeric::scale_type{scale});
    auto result =
      cudf::segmented_reduce(input, d_offsets, *agg, out_type, cudf::null_policy::INCLUDE);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expect);

    expect = cudf::test::fixed_point_column_wrapper<RepType>(
      {3, 3, 1, XXX, XXX, XXX}, {1, 1, 1, 0, 0, 0}, numeric::scale_type{scale});
    result = cudf::segmented_reduce(input, d_offsets, *agg, out_type, cudf::null_policy::EXCLUDE);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expect);
  }
}

TYPED_TEST(SegmentedReductionFixedPointTest, MinWithNulls)
{
  using RepType = cudf::device_storage_type_t<TypeParam>;

  auto const offsets   = std::vector<cudf::size_type>{0, 3, 6, 7, 8, 10, 10};
  auto const d_offsets = cudf::detail::make_device_uvector_async(
    offsets, cudf::get_default_stream(), cudf::get_current_device_resource_ref());
  auto const agg = cudf::make_min_aggregation<cudf::segmented_reduce_aggregation>();

  for (auto scale : {-2, 0, 5}) {
    auto const input =
      cudf::test::fixed_point_column_wrapper<RepType>({1, 2, 3, 1, XXX, 3, 1, XXX, XXX, XXX},
                                                      {1, 1, 1, 1, 0, 1, 1, 0, 0, 0},
                                                      numeric::scale_type{scale});
    auto out_type = cudf::column_view(input).type();
    auto expect   = cudf::test::fixed_point_column_wrapper<RepType>(
      {1, XXX, 1, XXX, XXX, XXX}, {1, 0, 1, 0, 0, 0}, numeric::scale_type{scale});
    auto result =
      cudf::segmented_reduce(input, d_offsets, *agg, out_type, cudf::null_policy::INCLUDE);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expect);

    expect = cudf::test::fixed_point_column_wrapper<RepType>(
      {1, 1, 1, XXX, XXX, XXX}, {1, 1, 1, 0, 0, 0}, numeric::scale_type{scale});
    result = cudf::segmented_reduce(input, d_offsets, *agg, out_type, cudf::null_policy::EXCLUDE);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expect);
  }
}

TYPED_TEST(SegmentedReductionFixedPointTest, MaxNonNullableInput)
{
  using RepType = cudf::device_storage_type_t<TypeParam>;

  auto const offsets   = std::vector<cudf::size_type>{0, 3, 4, 4};
  auto const d_offsets = cudf::detail::make_device_uvector_async(
    offsets, cudf::get_default_stream(), cudf::get_current_device_resource_ref());
  auto const agg = cudf::make_max_aggregation<cudf::segmented_reduce_aggregation>();

  for (auto scale : {-2, 0, 5}) {
    auto const input =
      cudf::test::fixed_point_column_wrapper<RepType>({1, 2, 3, 1}, numeric::scale_type{scale});
    auto out_type     = cudf::column_view(input).type();
    auto const expect = cudf::test::fixed_point_column_wrapper<RepType>(
      {3, 1, XXX}, {1, 1, 0}, numeric::scale_type{scale});

    auto result =
      cudf::segmented_reduce(input, d_offsets, *agg, out_type, cudf::null_policy::INCLUDE);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expect);

    result = cudf::segmented_reduce(input, d_offsets, *agg, out_type, cudf::null_policy::EXCLUDE);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expect);
  }
}

TYPED_TEST(SegmentedReductionFixedPointTest, MinNonNullableInput)
{
  using RepType = cudf::device_storage_type_t<TypeParam>;

  auto const offsets   = std::vector<cudf::size_type>{0, 3, 4, 4};
  auto const d_offsets = cudf::detail::make_device_uvector_async(
    offsets, cudf::get_default_stream(), cudf::get_current_device_resource_ref());
  auto const agg = cudf::make_min_aggregation<cudf::segmented_reduce_aggregation>();

  for (auto scale : {-2, 0, 5}) {
    auto const input =
      cudf::test::fixed_point_column_wrapper<RepType>({1, 2, 3, 1}, numeric::scale_type{scale});
    auto out_type     = cudf::column_view(input).type();
    auto const expect = cudf::test::fixed_point_column_wrapper<RepType>(
      {1, 1, XXX}, {1, 1, 0}, numeric::scale_type{scale});

    auto result =
      cudf::segmented_reduce(input, d_offsets, *agg, out_type, cudf::null_policy::INCLUDE);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expect);

    result = cudf::segmented_reduce(input, d_offsets, *agg, out_type, cudf::null_policy::EXCLUDE);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expect);
  }
}

TYPED_TEST(SegmentedReductionFixedPointTest, Sum)
{
  using RepType = cudf::device_storage_type_t<TypeParam>;

  auto const offsets   = std::vector<cudf::size_type>{0, 3, 6, 7, 8, 10, 10};
  auto const d_offsets = cudf::detail::make_device_uvector_async(
    offsets, cudf::get_default_stream(), cudf::get_current_device_resource_ref());
  auto const agg = cudf::make_sum_aggregation<cudf::segmented_reduce_aggregation>();

  for (auto scale : {-2, 0, 5}) {
    auto input =
      cudf::test::fixed_point_column_wrapper<RepType>({-10, 0, 33, 100, XXX, 53, 11, XXX, XXX, XXX},
                                                      {1, 1, 1, 1, 0, 1, 1, 0, 0, 0},
                                                      numeric::scale_type{scale});
    auto const out_type = cudf::column_view(input).type();

    auto expect = cudf::test::fixed_point_column_wrapper<RepType>(
      {23, XXX, 11, XXX, XXX, XXX}, {1, 0, 1, 0, 0, 0}, numeric::scale_type{scale});
    auto result =
      cudf::segmented_reduce(input, d_offsets, *agg, out_type, cudf::null_policy::INCLUDE);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expect);

    expect = cudf::test::fixed_point_column_wrapper<RepType>(
      {23, 153, 11, XXX, XXX, XXX}, {1, 1, 1, 0, 0, 0}, numeric::scale_type{scale});
    result = cudf::segmented_reduce(input, d_offsets, *agg, out_type, cudf::null_policy::EXCLUDE);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expect);

    input = cudf::test::fixed_point_column_wrapper<RepType>(
      {-10, 0, 33, 100, 123, 53, 11, 0, -120, 88}, numeric::scale_type{scale});
    expect = cudf::test::fixed_point_column_wrapper<RepType>(
      {23, 276, 11, 0, -32, XXX}, {1, 1, 1, 1, 1, 0}, numeric::scale_type{scale});
    result = cudf::segmented_reduce(input, d_offsets, *agg, out_type, cudf::null_policy::INCLUDE);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expect);
    result = cudf::segmented_reduce(input, d_offsets, *agg, out_type, cudf::null_policy::EXCLUDE);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expect);
  }
}

TYPED_TEST(SegmentedReductionFixedPointTest, Product)
{
  using RepType = cudf::device_storage_type_t<TypeParam>;

  auto const offsets   = std::vector<cudf::size_type>{0, 3, 6, 7, 8, 12, 12};
  auto const d_offsets = cudf::detail::make_device_uvector_async(
    offsets, cudf::get_default_stream(), cudf::get_current_device_resource_ref());
  auto const agg = cudf::make_product_aggregation<cudf::segmented_reduce_aggregation>();

  for (auto scale : {-2, 0, 5}) {
    auto input = cudf::test::fixed_point_column_wrapper<RepType>(
      {-10, 1, 33, 40, XXX, 50, 11000, XXX, XXX, XXX, XXX, XXX},
      {1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0},
      numeric::scale_type{scale});
    auto const out_type = cudf::column_view(input).type();
    auto result =
      cudf::segmented_reduce(input, d_offsets, *agg, out_type, cudf::null_policy::INCLUDE);
    auto expect = cudf::test::fixed_point_column_wrapper<RepType>(
      {-330, XXX, 11000, XXX, XXX, XXX}, {1, 0, 1, 0, 0, 0}, numeric::scale_type{scale * 3});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expect);

    result = cudf::segmented_reduce(input, d_offsets, *agg, out_type, cudf::null_policy::EXCLUDE);
    expect = cudf::test::fixed_point_column_wrapper<RepType>(
      {-330, 2000, 11000, XXX, XXX, XXX}, {1, 1, 1, 0, 0, 0}, numeric::scale_type{scale * 3});
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expect);

    input = cudf::test::fixed_point_column_wrapper<RepType>(
      {-10, 1, 33, 3, 40, 50, 11000, 0, -3, 50, 10, 4}, numeric::scale_type{scale});
    expect = cudf::test::fixed_point_column_wrapper<RepType>(
      {-330, 6000, 11000, 0, -6000, XXX}, {1, 1, 1, 1, 1, 0}, numeric::scale_type{scale * 4});
    result = cudf::segmented_reduce(input, d_offsets, *agg, out_type, cudf::null_policy::INCLUDE);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expect);
    result = cudf::segmented_reduce(input, d_offsets, *agg, out_type, cudf::null_policy::EXCLUDE);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expect);
  }
}

TYPED_TEST(SegmentedReductionFixedPointTest, SumOfSquares)
{
  using RepType = cudf::device_storage_type_t<TypeParam>;

  auto const offsets   = std::vector<cudf::size_type>{0, 3, 6, 7, 8, 10, 10};
  auto const d_offsets = cudf::detail::make_device_uvector_async(
    offsets, cudf::get_default_stream(), cudf::get_current_device_resource_ref());
  auto const agg = cudf::make_sum_of_squares_aggregation<cudf::segmented_reduce_aggregation>();

  for (auto scale : {-2, 0, 5}) {
    auto input =
      cudf::test::fixed_point_column_wrapper<RepType>({-10, 0, 33, 100, XXX, 53, 11, XXX, XXX, XXX},
                                                      {1, 1, 1, 1, 0, 1, 1, 0, 0, 0},
                                                      numeric::scale_type{scale});
    auto const out_type = cudf::column_view(input).type();

    auto expect = cudf::test::fixed_point_column_wrapper<RepType>(
      {1189, XXX, 121, XXX, XXX, XXX}, {1, 0, 1, 0, 0, 0}, numeric::scale_type{scale * 2});
    auto result =
      cudf::segmented_reduce(input, d_offsets, *agg, out_type, cudf::null_policy::INCLUDE);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expect);

    expect = cudf::test::fixed_point_column_wrapper<RepType>(
      {1189, 12809, 121, XXX, XXX, XXX}, {1, 1, 1, 0, 0, 0}, numeric::scale_type{scale * 2});
    result = cudf::segmented_reduce(input, d_offsets, *agg, out_type, cudf::null_policy::EXCLUDE);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expect);

    input = cudf::test::fixed_point_column_wrapper<RepType>(
      {-10, 0, 33, 100, 123, 53, 11, 0, -120, 88}, numeric::scale_type{scale});
    expect = cudf::test::fixed_point_column_wrapper<RepType>(
      {1189, 27938, 121, 0, 22144, XXX}, {1, 1, 1, 1, 1, 0}, numeric::scale_type{scale * 2});
    result = cudf::segmented_reduce(input, d_offsets, *agg, out_type, cudf::null_policy::INCLUDE);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expect);
    result = cudf::segmented_reduce(input, d_offsets, *agg, out_type, cudf::null_policy::EXCLUDE);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expect);
  }
}

// String min/max test grid
// Segment: Length 0, length 1, length 2
// Element nulls: No nulls, all nulls, some nulls
// String: Empty string,
// Position of the min/max: start of segment, end of segment
// Include null, exclude null

#undef XXX
#define XXX ""  // null placeholder

struct SegmentedReductionStringTest : public cudf::test::BaseFixture {
  std::pair<cudf::test::strings_column_wrapper,
            cudf::test::fixed_width_column_wrapper<cudf::size_type>>
  input()
  {
    return std::pair(
      cudf::test::strings_column_wrapper{
        {"world", "cudf", XXX, "", "rapids", "i am", "ai", "apples", "zebras", XXX, XXX, XXX},
        {true, true, false, true, true, true, true, true, true, false, false, false}},
      cudf::test::fixed_width_column_wrapper<cudf::size_type>{0, 1, 4, 7, 9, 9, 10, 12});
  }
};

TEST_F(SegmentedReductionStringTest, MaxIncludeNulls)
{
  // data: ['world'], ['cudf', NULL, ''], ['rapids', 'i am', 'ai'], ['apples', 'zebras'],
  //       [], [NULL], [NULL, NULL]
  // values:  {"world", "cudf", XXX, "", "rapids", "i am", "ai", "apples", "zebras", XXX, XXX, XXX}
  // nullmask:{1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0}
  // offsets: {0, 1, 4, 7, 9, 9, 10, 12}
  // output_dtype: string dtype
  // outputs: {"world", XXX, "rapids", "zebras", XXX, XXX, XXX}
  // output nullmask: {1, 0, 1, 1, 0, 0, 0}

  auto const [input, offsets] = this->input();
  cudf::data_type output_dtype{cudf::type_id::STRING};

  cudf::test::strings_column_wrapper expect{{"world", XXX, "rapids", "zebras", XXX, XXX, XXX},
                                            {true, false, true, true, false, false, false}};

  auto res =
    cudf::segmented_reduce(input,
                           cudf::column_view(offsets),
                           *cudf::make_max_aggregation<cudf::segmented_reduce_aggregation>(),
                           output_dtype,
                           cudf::null_policy::INCLUDE);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*res, expect);
}

TEST_F(SegmentedReductionStringTest, MaxExcludeNulls)
{
  // data: ['world'], ['cudf', NULL, ''], ['rapids', 'i am', 'ai'], ['apples', 'zebras'],
  //       [], [NULL], [NULL, NULL]
  // values:  {"world", "cudf", XXX, "", "rapids", "i am", "ai", "apples", "zebras", XXX, XXX, XXX}
  // nullmask:{1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0}
  // offsets: {0, 1, 4, 7, 9, 9, 10, 12}
  // output_dtype: string dtype
  // outputs: {"world", "cudf", "rapids", "zebras", XXX, XXX, XXX}
  // output nullmask: {1, 1, 1, 1, 0, 0, 0}

  auto const [input, offsets] = this->input();
  cudf::data_type output_dtype{cudf::type_id::STRING};

  cudf::test::strings_column_wrapper expect{{"world", "cudf", "rapids", "zebras", XXX, XXX, XXX},
                                            {true, true, true, true, false, false, false}};

  auto res =
    cudf::segmented_reduce(input,
                           cudf::column_view(offsets),
                           *cudf::make_max_aggregation<cudf::segmented_reduce_aggregation>(),
                           output_dtype,
                           cudf::null_policy::EXCLUDE);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*res, expect);
}

TEST_F(SegmentedReductionStringTest, MinIncludeNulls)
{
  // data: ['world'], ['cudf', NULL, ''], ['rapids', 'i am', 'ai'], ['apples', 'zebras'],
  //       [], [NULL], [NULL, NULL]
  // values:  {"world", "cudf", XXX, "", "rapids", "i am", "ai", "apples", "zebras", XXX, XXX, XXX}
  // nullmask:{1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0}
  // offsets: {0, 1, 4, 7, 9, 9, 10, 12}
  // output_dtype: string dtype
  // outputs: {"world", XXX, "ai", "apples", XXX, XXX, XXX}
  // output nullmask: {1, 0, 1, 1, 0, 0, 0}

  auto const [input, offsets] = this->input();
  cudf::data_type output_dtype{cudf::type_id::STRING};

  cudf::test::strings_column_wrapper expect{{"world", XXX, "ai", "apples", XXX, XXX, XXX},
                                            {true, false, true, true, false, false, false}};

  auto res =
    cudf::segmented_reduce(input,
                           cudf::column_view(offsets),
                           *cudf::make_min_aggregation<cudf::segmented_reduce_aggregation>(),
                           output_dtype,
                           cudf::null_policy::INCLUDE);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*res, expect);
}

TEST_F(SegmentedReductionStringTest, MinExcludeNulls)
{
  // data: ['world'], ['cudf', NULL, ''], ['rapids', 'i am', 'ai'], ['apples', 'zebras'],
  //       [], [NULL], [NULL, NULL]
  // values:  {"world", "cudf", XXX, "", "rapids", "i am", "ai", "apples", "zebras", XXX, XXX, XXX}
  // nullmask:{1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0}
  // offsets: {0, 1, 4, 7, 9, 9, 10, 12}
  // output_dtype: string dtype
  // outputs: {"world", "", "ai", "apples", XXX, XXX, XXX}
  // output nullmask: {1, 1, 1, 1, 0, 0, 0}

  auto const [input, offsets] = this->input();
  cudf::data_type output_dtype{cudf::type_id::STRING};

  cudf::test::strings_column_wrapper expect{{"world", "", "ai", "apples", XXX, XXX, XXX},
                                            {true, true, true, true, false, false, false}};

  auto res =
    cudf::segmented_reduce(input,
                           cudf::column_view(offsets),
                           *cudf::make_min_aggregation<cudf::segmented_reduce_aggregation>(),
                           output_dtype,
                           cudf::null_policy::EXCLUDE);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*res, expect);
}

TEST_F(SegmentedReductionStringTest, EmptyInputWithOffsets)
{
  auto const input     = cudf::test::strings_column_wrapper{};
  auto const offsets   = std::vector<cudf::size_type>{0, 0, 0, 0};
  auto const d_offsets = cudf::detail::make_device_uvector_async(
    offsets, cudf::get_default_stream(), cudf::get_current_device_resource_ref());
  auto const expect = cudf::test::strings_column_wrapper({XXX, XXX, XXX}, {false, false, false});

  auto result =
    cudf::segmented_reduce(input,
                           d_offsets,
                           *cudf::make_max_aggregation<cudf::segmented_reduce_aggregation>(),
                           cudf::data_type{cudf::type_id::STRING},
                           cudf::null_policy::EXCLUDE);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expect);
  result = cudf::segmented_reduce(input,
                                  d_offsets,
                                  *cudf::make_min_aggregation<cudf::segmented_reduce_aggregation>(),
                                  cudf::data_type{cudf::type_id::STRING},
                                  cudf::null_policy::INCLUDE);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expect);
}

#undef XXX
