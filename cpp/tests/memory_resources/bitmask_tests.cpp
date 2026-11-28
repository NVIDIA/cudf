/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "memory_resources_test_utilities.hpp"

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/default_stream.hpp>

#include <cudf/copying.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/span.hpp>

#include <cuda/buffer>

#include <utility>
#include <vector>

using cudf::test::temporary_allocation_expectation;
using cudf::test::memory_resources_tests::expect_api_routes_memory_resources;

namespace {

constexpr cudf::size_type num_rows = 300;

auto make_nullable_column(int modulus)
{
  auto values   = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i; });
  auto validity = cudf::detail::make_counting_transform_iterator(
    0, [modulus](auto i) { return i % modulus != 0; });
  return cudf::test::fixed_width_column_wrapper<int32_t>(values, values + num_rows, validity);
}

void expect_bitmasks_equal(cudf::bitmask_type const* expected,
                           cudf::bitmask_type const* actual,
                           cudf::size_type size)
{
  auto const stream     = cudf::test::get_default_stream();
  auto const words      = static_cast<std::size_t>(cudf::num_bitmask_words(size));
  auto const h_expected = cudf::detail::make_std_vector(
    cudf::device_span<cudf::bitmask_type const>{expected, words}, stream);
  auto const h_actual = cudf::detail::make_std_vector(
    cudf::device_span<cudf::bitmask_type const>{actual, words}, stream);
  for (cudf::size_type i = 0; i < size; ++i) {
    EXPECT_EQ(cudf::bit_is_set(h_expected.data(), i), cudf::bit_is_set(h_actual.data(), i))
      << "bit " << i;
  }
}

std::size_t buffer_bytes(cuda::device_buffer<cudf::bitmask_type> const& buffer)
{
  return buffer.size() * sizeof(cudf::bitmask_type);
}

}  // namespace

struct BitmaskMemoryResourcesTest : public cudf::test::BaseFixture {};

TEST_F(BitmaskMemoryResourcesTest, CopyBitmaskSliced)
{
  auto input        = make_nullable_column(3);
  auto const sliced = cudf::slice(input, {7, num_rows - 5}).front();
  expect_api_routes_memory_resources(
    [&](cudf::memory_resources mr) {
      return cudf::copy_bitmask(sliced, cudf::test::get_default_stream(), mr);
    },
    buffer_bytes,
    [&](auto const& expected, auto const& result) {
      expect_bitmasks_equal(expected.data(), result.data(), sliced.size());
    },
    temporary_allocation_expectation::NONE);
}

TEST_F(BitmaskMemoryResourcesTest, CreateNullMask)
{
  expect_api_routes_memory_resources(
    [&](cudf::memory_resources mr) {
      return cudf::create_null_mask(
        num_rows, cudf::mask_state::ALL_NULL, cudf::test::get_default_stream(), mr);
    },
    buffer_bytes,
    [&](auto const& expected, auto const& result) {
      expect_bitmasks_equal(expected.data(), result.data(), num_rows);
    },
    temporary_allocation_expectation::NONE);
}

TEST_F(BitmaskMemoryResourcesTest, BitmaskAnd)
{
  auto lhs         = make_nullable_column(3);
  auto rhs         = make_nullable_column(5);
  auto const input = cudf::table_view{{lhs, rhs}};
  expect_api_routes_memory_resources(
    [&](cudf::memory_resources mr) {
      return cudf::bitmask_and(input, cudf::test::get_default_stream(), mr);
    },
    [](auto const& result) { return buffer_bytes(result.first); },
    [&](auto const& expected, auto const& result) {
      EXPECT_EQ(expected.second, result.second);
      expect_bitmasks_equal(expected.first.data(), result.first.data(), num_rows);
    },
    temporary_allocation_expectation::SOME);
}

TEST_F(BitmaskMemoryResourcesTest, BitmaskOr)
{
  auto lhs         = make_nullable_column(3);
  auto rhs         = make_nullable_column(2);
  auto const input = cudf::table_view{{lhs, rhs}};
  expect_api_routes_memory_resources(
    [&](cudf::memory_resources mr) {
      return cudf::bitmask_or(input, cudf::test::get_default_stream(), mr);
    },
    [](auto const& result) { return buffer_bytes(result.first); },
    [&](auto const& expected, auto const& result) {
      EXPECT_EQ(expected.second, result.second);
      expect_bitmasks_equal(expected.first.data(), result.first.data(), num_rows);
    },
    temporary_allocation_expectation::SOME);
}

TEST_F(BitmaskMemoryResourcesTest, NullCount)
{
  auto input      = make_nullable_column(3);
  auto const mask = cudf::column_view{input}.null_mask();
  expect_api_routes_memory_resources(
    [&](cudf::memory_resources mr) {
      return cudf::null_count(mask, 3, num_rows - 7, cudf::test::get_default_stream(), mr);
    },
    [](cudf::size_type) { return std::size_t{0}; },
    [](cudf::size_type expected, cudf::size_type result) { EXPECT_EQ(expected, result); },
    temporary_allocation_expectation::SOME);
}

TEST_F(BitmaskMemoryResourcesTest, BatchNullCount)
{
  auto lhs         = make_nullable_column(3);
  auto rhs         = make_nullable_column(4);
  auto const masks = std::vector<cudf::bitmask_type const*>{cudf::column_view{lhs}.null_mask(),
                                                            cudf::column_view{rhs}.null_mask()};
  expect_api_routes_memory_resources(
    [&](cudf::memory_resources mr) {
      return cudf::batch_null_count(masks, 1, num_rows - 1, cudf::test::get_default_stream(), mr);
    },
    [](auto const&) { return std::size_t{0}; },
    [](auto const& expected, auto const& result) { EXPECT_EQ(expected, result); },
    temporary_allocation_expectation::SOME);
}

TEST_F(BitmaskMemoryResourcesTest, IndexOfFirstSetBit)
{
  auto input      = make_nullable_column(3);
  auto const mask = cudf::column_view{input}.null_mask();
  expect_api_routes_memory_resources(
    [&](cudf::memory_resources mr) {
      return cudf::index_of_first_set_bit(mask, 3, num_rows, cudf::test::get_default_stream(), mr);
    },
    [](cudf::size_type) { return std::size_t{0}; },
    [](cudf::size_type expected, cudf::size_type result) { EXPECT_EQ(expected, result); },
    temporary_allocation_expectation::SOME);
}
