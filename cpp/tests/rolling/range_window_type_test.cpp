/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/random.hpp>
#include <cudf_test/testing_main.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/column/column_view.hpp>
#include <cudf/rolling.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/sorting.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/wrappers/durations.hpp>
#include <cudf/wrappers/timestamps.hpp>

#include <src/rolling/detail/rolling.hpp>

#include <limits>
#include <set>
#include <vector>

using ints_column      = cudf::test::fixed_width_column_wrapper<int32_t>;
using size_type_column = cudf::test::fixed_width_column_wrapper<cudf::size_type>;

void expect_range_windows_equal(
  std::pair<std::unique_ptr<cudf::column>, std::unique_ptr<cudf::column>> const& result,
  cudf::column_view const& expect_preceding,
  cudf::column_view const& expect_following)
{
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    std::get<0>(result)->view(), expect_preceding, cudf::test::debug_output_level::ALL_ERRORS);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    std::get<1>(result)->view(), expect_following, cudf::test::debug_output_level::ALL_ERRORS);
}

TEST(MultiOrderByRangeWindows, UngroupedPeerBounds)
{
  auto const orderby0 = ints_column{1, 1, 1, 2, 2, 3};
  auto const orderby1 = ints_column{1, 1, 2, 1, 1, 1};

  std::vector<cudf::order> orders{cudf::order::ASCENDING, cudf::order::ASCENDING};
  std::vector<cudf::null_order> null_orders{cudf::null_order::BEFORE, cudf::null_order::BEFORE};
  auto const orderby = cudf::table_view{{orderby0, orderby1}};

  expect_range_windows_equal(
    cudf::detail::make_range_windows(cudf::table_view{},
                                     orderby,
                                     orders,
                                     null_orders,
                                     cudf::current_row{},
                                     cudf::current_row{},
                                     cudf::get_default_stream(),
                                     cudf::get_current_device_resource_ref()),
    size_type_column{1, 2, 1, 1, 2, 1},
    size_type_column{1, 0, 0, 1, 0, 0});

  expect_range_windows_equal(
    cudf::detail::make_range_windows(cudf::table_view{},
                                     orderby,
                                     orders,
                                     null_orders,
                                     cudf::unbounded{},
                                     cudf::current_row{},
                                     cudf::get_default_stream(),
                                     cudf::get_current_device_resource_ref()),
    size_type_column{1, 2, 3, 4, 5, 6},
    size_type_column{1, 0, 0, 1, 0, 0});
}

TEST(MultiOrderByRangeWindows, GroupedPeerBounds)
{
  auto const group_keys = ints_column{0, 0, 0, 1, 1, 1, 1, 1};
  auto const orderby0   = ints_column{1, 1, 2, 1, 1, 1, 2, 2};
  auto const orderby1   = ints_column{1, 2, 1, 1, 1, 2, 1, 1};

  std::vector<cudf::order> orders{cudf::order::ASCENDING, cudf::order::ASCENDING};
  std::vector<cudf::null_order> null_orders{cudf::null_order::BEFORE, cudf::null_order::BEFORE};

  expect_range_windows_equal(
    cudf::detail::make_range_windows(cudf::table_view{{group_keys}},
                                     cudf::table_view{{orderby0, orderby1}},
                                     orders,
                                     null_orders,
                                     cudf::unbounded{},
                                     cudf::current_row{},
                                     cudf::get_default_stream(),
                                     cudf::get_current_device_resource_ref()),
    size_type_column{1, 2, 3, 1, 2, 3, 4, 5},
    size_type_column{0, 0, 0, 1, 0, 0, 1, 0});

  expect_range_windows_equal(
    cudf::detail::make_range_windows(cudf::table_view{{group_keys}},
                                     cudf::table_view{{orderby0, orderby1}},
                                     orders,
                                     null_orders,
                                     cudf::current_row{},
                                     cudf::unbounded{},
                                     cudf::get_default_stream(),
                                     cudf::get_current_device_resource_ref()),
    size_type_column{1, 1, 1, 1, 2, 1, 1, 2},
    size_type_column{2, 1, 0, 4, 3, 2, 1, 0});
}

TEST(MultiOrderByRangeWindows, GroupedPeerBoundsAscDesc)
{
  auto const group_keys = ints_column{0, 0, 0, 1, 1, 1, 1, 1};
  auto const orderby0   = ints_column{1, 1, 2, 1, 1, 1, 2, 2};
  auto const orderby1   = ints_column{2, 1, 1, 2, 1, 1, 1, 1};

  std::vector<cudf::order> orders{cudf::order::ASCENDING, cudf::order::DESCENDING};
  std::vector<cudf::null_order> null_orders{cudf::null_order::BEFORE, cudf::null_order::BEFORE};

  expect_range_windows_equal(
    cudf::detail::make_range_windows(cudf::table_view{{group_keys}},
                                     cudf::table_view{{orderby0, orderby1}},
                                     orders,
                                     null_orders,
                                     cudf::unbounded{},
                                     cudf::current_row{},
                                     cudf::get_default_stream(),
                                     cudf::get_current_device_resource_ref()),
    size_type_column{1, 2, 3, 1, 2, 3, 4, 5},
    size_type_column{0, 0, 0, 0, 1, 0, 1, 0});

  expect_range_windows_equal(
    cudf::detail::make_range_windows(cudf::table_view{{group_keys}},
                                     cudf::table_view{{orderby0, orderby1}},
                                     orders,
                                     null_orders,
                                     cudf::current_row{},
                                     cudf::unbounded{},
                                     cudf::get_default_stream(),
                                     cudf::get_current_device_resource_ref()),
    size_type_column{1, 1, 1, 1, 1, 2, 1, 2},
    size_type_column{2, 1, 0, 4, 3, 2, 1, 0});
}

TEST(MultiOrderByRangeWindows, GroupedPeerBoundsDescDesc)
{
  auto const group_keys = ints_column{0, 0, 0, 1, 1, 1, 1, 1};
  auto const orderby0   = ints_column{2, 1, 1, 2, 2, 1, 1, 1};
  auto const orderby1   = ints_column{1, 2, 1, 1, 1, 2, 1, 1};

  std::vector<cudf::order> orders{cudf::order::DESCENDING, cudf::order::DESCENDING};
  std::vector<cudf::null_order> null_orders{cudf::null_order::BEFORE, cudf::null_order::BEFORE};

  expect_range_windows_equal(
    cudf::detail::make_range_windows(cudf::table_view{{group_keys}},
                                     cudf::table_view{{orderby0, orderby1}},
                                     orders,
                                     null_orders,
                                     cudf::unbounded{},
                                     cudf::current_row{},
                                     cudf::get_default_stream(),
                                     cudf::get_current_device_resource_ref()),
    size_type_column{1, 2, 3, 1, 2, 3, 4, 5},
    size_type_column{0, 0, 0, 1, 0, 0, 1, 0});

  expect_range_windows_equal(
    cudf::detail::make_range_windows(cudf::table_view{{group_keys}},
                                     cudf::table_view{{orderby0, orderby1}},
                                     orders,
                                     null_orders,
                                     cudf::current_row{},
                                     cudf::unbounded{},
                                     cudf::get_default_stream(),
                                     cudf::get_current_device_resource_ref()),
    size_type_column{1, 1, 1, 1, 2, 1, 1, 2},
    size_type_column{2, 1, 0, 4, 3, 2, 1, 0});
}

TEST(MultiOrderByRangeWindows, CurrentRowIncludesNullPeers)
{
  auto const orderby0 = ints_column{{0, 0, 1, 1}, {false, false, true, true}};
  auto const orderby1 = ints_column{1, 1, 1, 2};

  std::vector<cudf::order> orders{cudf::order::ASCENDING, cudf::order::ASCENDING};
  std::vector<cudf::null_order> null_orders{cudf::null_order::BEFORE, cudf::null_order::BEFORE};

  expect_range_windows_equal(
    cudf::detail::make_range_windows(cudf::table_view{},
                                     cudf::table_view{{orderby0, orderby1}},
                                     orders,
                                     null_orders,
                                     cudf::current_row{},
                                     cudf::current_row{},
                                     cudf::get_default_stream(),
                                     cudf::get_current_device_resource_ref()),
    size_type_column{1, 2, 1, 1},
    size_type_column{1, 0, 0, 0});
}

TEST(MultiOrderByRangeWindows, CurrentRowPeerDetectionAcrossAllNullPositions)
{
  // Unsorted canonical tuple-set: (NULL,NULL), (NULL,NULL), (1,NULL), (1,1), (NULL,1).
  // Peer groups: {(NULL,NULL) x 2}, {(1,NULL)}, {(1,1)}, {(NULL,1)}.
  std::vector<int32_t> const col0_data{0, 0, 1, 1, 0};
  std::vector<bool> const col0_valid{false, false, true, true, false};
  std::vector<int32_t> const col1_data{0, 0, 0, 1, 1};
  std::vector<bool> const col1_valid{false, false, false, true, true};
  auto const col0     = ints_column{col0_data.begin(), col0_data.end(), col0_valid.begin()};
  auto const col1     = ints_column{col1_data.begin(), col1_data.end(), col1_valid.begin()};
  auto const unsorted = cudf::table_view{{col0, col1}};

  // Expected multiset of (preceding, following) for the fixed peer-group structure:
  //   one peer of size 2 -> (1,1) and (2,0); three singletons -> (1,0) x 3.
  std::multiset<std::pair<cudf::size_type, cudf::size_type>> const expected_pairs{
    {1, 1}, {2, 0}, {1, 0}, {1, 0}, {1, 0}};

  for (auto const order0 : {cudf::order::ASCENDING, cudf::order::DESCENDING}) {
    for (auto const null_order0 : {cudf::null_order::BEFORE, cudf::null_order::AFTER}) {
      for (auto const order1 : {cudf::order::ASCENDING, cudf::order::DESCENDING}) {
        for (auto const null_order1 : {cudf::null_order::BEFORE, cudf::null_order::AFTER}) {
          std::vector<cudf::order> const orders{order0, order1};
          std::vector<cudf::null_order> const null_orders{null_order0, null_order1};
          auto const sorted = cudf::sort(unsorted, orders, null_orders);

          auto const [preceding, following] =
            cudf::detail::make_range_windows(cudf::table_view{},
                                             sorted->view(),
                                             orders,
                                             null_orders,
                                             cudf::current_row{},
                                             cudf::current_row{},
                                             cudf::get_default_stream(),
                                             cudf::get_current_device_resource_ref());

          auto const [preceding_host, _p_valid] = cudf::test::to_host<cudf::size_type>(*preceding);
          auto const [following_host, _f_valid] = cudf::test::to_host<cudf::size_type>(*following);
          std::multiset<std::pair<cudf::size_type, cudf::size_type>> actual_pairs;
          for (std::size_t i = 0; i < preceding_host.size(); ++i) {
            actual_pairs.emplace(preceding_host[i], following_host[i]);
          }
          EXPECT_EQ(actual_pairs, expected_pairs)
            << "Failed for orders=(" << static_cast<int>(order0) << ", " << static_cast<int>(order1)
            << "), null_orders=(" << static_cast<int>(null_order0) << ", "
            << static_cast<int>(null_order1) << ")";
        }
      }
    }
  }
}

TEST(MultiOrderByRangeWindows, BoundedRangesAreUnsupported)
{
  auto const orderby0 = ints_column{1, 1, 2};
  auto const orderby1 = ints_column{1, 2, 1};
  auto const delta    = cudf::make_fixed_width_scalar<int32_t>(1);

  std::vector<cudf::order> orders{cudf::order::ASCENDING, cudf::order::ASCENDING};
  std::vector<cudf::null_order> null_orders{cudf::null_order::BEFORE, cudf::null_order::BEFORE};

  EXPECT_THROW(
    static_cast<void>(cudf::detail::make_range_windows(cudf::table_view{},
                                                       cudf::table_view{{orderby0, orderby1}},
                                                       orders,
                                                       null_orders,
                                                       cudf::bounded_closed{*delta},
                                                       cudf::current_row{},
                                                       cudf::get_default_stream(),
                                                       cudf::get_current_device_resource_ref())),
    cudf::logic_error);
}

template <typename T>
struct UngroupedBase : cudf::test::BaseFixture {
  static constexpr T min{std::numeric_limits<T>::lowest()};
  static constexpr T max{std::numeric_limits<T>::max()};
  using cw_t = cudf::test::fixed_width_column_wrapper<T>;
  cw_t ascending_no_nulls{};
  cw_t ascending_nulls_before{};
  cw_t ascending_nulls_after{};
  cw_t descending_no_nulls{};
  cw_t descending_nulls_before{};
  cw_t descending_nulls_after{};
  UngroupedBase()
    : ascending_no_nulls{
        // clang-format off
        {min, T{5}, T{5}, T{6}, T{7}, T{9}, T{9}, T{12}, T{13}, T{17}, T{22}, T{22}, max}},
        // clang-format on
      ascending_nulls_before{
        {min, T{5}, T{5}, T{6}, T{7}, T{9}, T{9}, T{12}, T{13}, T{17}, T{22}, T{22}, max},
        {false, false, false, false, true, true, true, true, true, true, true, true, true}},
      ascending_nulls_after{
        {min, T{5}, T{5}, T{6}, T{7}, T{9}, T{9}, T{12}, T{13}, T{17}, T{22}, T{22}, max},
        {true, true, true, true, true, true, false, false, false, false, false, false, false}},
      descending_no_nulls{
        {max, T{22}, T{22}, T{17}, T{13}, T{12}, T{9}, T{9}, T{7}, T{6}, T{5}, T{5}, min}},
      descending_nulls_before{
        {max, T{22}, T{22}, T{17}, T{13}, T{12}, T{9}, T{9}, T{7}, T{6}, T{5}, T{5}, min},
        {true, true, true, true, true, true, true, false, false, false, false, false, false}},
      descending_nulls_after{
        {max, T{22}, T{22}, T{17}, T{13}, T{12}, T{9}, T{9}, T{7}, T{6}, T{5}, T{5}, min},
        {false, false, false, false, false, true, true, true, true, true, true, true, true}}
  {
  }

  void run_test(cudf::column_view const& orderby,
                cudf::order order,
                cudf::null_order null_order,
                cudf::range_window_type preceding,
                cudf::range_window_type following,
                cudf::column_view const& expect_preceding,
                cudf::column_view const& expect_following)
  {
    auto result = cudf::make_range_windows(
      cudf::table_view{}, orderby, order, null_order, preceding, following);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(
      std::get<0>(result)->view(), expect_preceding, cudf::test::debug_output_level::ALL_ERRORS);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(
      std::get<1>(result)->view(), expect_following, cudf::test::debug_output_level::ALL_ERRORS);
  }
};
template <typename T>
struct UngroupedIntegralRangeWindows : UngroupedBase<T> {};
TYPED_TEST_SUITE(UngroupedIntegralRangeWindows, cudf::test::IntegralTypesNotBool);

template <typename T>
struct UngroupedSignedIntegralRangeWindows : UngroupedBase<T> {};
using SignedIntegralTypes = cudf::test::Types<int8_t, int16_t, int32_t, int64_t>;
TYPED_TEST_SUITE(UngroupedSignedIntegralRangeWindows, SignedIntegralTypes);

TYPED_TEST(UngroupedIntegralRangeWindows, AscendingNoNullsZeroPrecedingFollowing)
{
  using T   = TypeParam;
  auto prec = cudf::make_fixed_width_scalar<T>(0);
  auto foll = cudf::make_fixed_width_scalar<T>(0);

  this->run_test(this->ascending_no_nulls,
                 cudf::order::ASCENDING,
                 cudf::null_order::BEFORE,
                 cudf::bounded_closed{*prec},
                 cudf::bounded_closed{*foll},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{
                   {1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1}},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{
                   {0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0}});
  this->run_test(this->ascending_no_nulls,
                 cudf::order::ASCENDING,
                 cudf::null_order::BEFORE,
                 cudf::bounded_open{*prec},
                 cudf::bounded_open{*foll},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{
                   {0, -1, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0}},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{
                   {-1, -1, -2, -1, -1, -1, -2, -1, -1, -1, -1, -2, -1}});
}

TYPED_TEST(UngroupedIntegralRangeWindows, AscendingNoNullsPositivePrecedingFollowing)
{
  using T   = TypeParam;
  auto prec = cudf::make_fixed_width_scalar<T>(2);
  auto foll = cudf::make_fixed_width_scalar<T>(1);

  this->run_test(this->ascending_no_nulls,
                 cudf::order::ASCENDING,
                 cudf::null_order::BEFORE,
                 cudf::bounded_closed{*prec},
                 cudf::bounded_closed{*foll},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{
                   {1, 1, 2, 3, 4, 2, 3, 1, 2, 1, 1, 2, 1}},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{
                   {0, 2, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0}});
  this->run_test(this->ascending_no_nulls,
                 cudf::order::ASCENDING,
                 cudf::null_order::BEFORE,
                 cudf::bounded_open{*prec},
                 cudf::bounded_open{*foll},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{
                   {1, 1, 2, 3, 2, 1, 2, 1, 2, 1, 1, 2, 1}},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{
                   {0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0}});
}

TYPED_TEST(UngroupedSignedIntegralRangeWindows, AscendingNoNullsNegativePreceding)
{
  using T   = TypeParam;
  auto prec = cudf::make_fixed_width_scalar<T>(-1);
  auto foll = cudf::make_fixed_width_scalar<T>(2);

  this->run_test(this->ascending_no_nulls,
                 cudf::order::ASCENDING,
                 cudf::null_order::BEFORE,
                 cudf::bounded_closed{*prec},
                 cudf::bounded_closed{*foll},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{
                   {0, -1, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0}},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{
                   {0, 3, 2, 1, 2, 1, 0, 1, 0, 0, 1, 0, 0}});
  this->run_test(this->ascending_no_nulls,
                 cudf::order::ASCENDING,
                 cudf::null_order::BEFORE,
                 cudf::bounded_open{*prec},
                 cudf::bounded_open{*foll},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{
                   {0, -2, -1, -1, 0, -1, 0, -1, 0, 0, -1, 0, 0}},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{
                   {0, 2, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0}});
}

TYPED_TEST(UngroupedSignedIntegralRangeWindows, AscendingNoNullsNegativeFollowing)
{
  using T   = TypeParam;
  auto prec = cudf::make_fixed_width_scalar<T>(4);
  auto foll = cudf::make_fixed_width_scalar<T>(-2);

  this->run_test(this->ascending_no_nulls,
                 cudf::order::ASCENDING,
                 cudf::null_order::BEFORE,
                 cudf::bounded_closed{*prec},
                 cudf::bounded_closed{*foll},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{
                   {1, 1, 2, 3, 4, 5, 6, 3, 4, 2, 1, 2, 1}},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{
                   {-1, -1, -2, -3, -2, -1, -2, -1, -2, -1, -1, -2, -1}});
  this->run_test(this->ascending_no_nulls,
                 cudf::order::ASCENDING,
                 cudf::null_order::BEFORE,
                 cudf::bounded_open{*prec},
                 cudf::bounded_open{*foll},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{
                   {1, 1, 2, 3, 4, 3, 4, 3, 2, 1, 1, 2, 1}},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{
                   {-1, -1, -2, -3, -4, -2, -3, -1, -2, -1, -1, -2, -1}});
}

TYPED_TEST(UngroupedIntegralRangeWindows, DescendingNoNullsZeroPrecedingFollowing)
{
  using T   = TypeParam;
  auto prec = cudf::make_fixed_width_scalar<T>(0);
  auto foll = cudf::make_fixed_width_scalar<T>(0);

  this->run_test(this->descending_no_nulls,
                 cudf::order::DESCENDING,
                 cudf::null_order::BEFORE,
                 cudf::bounded_closed{*prec},
                 cudf::bounded_closed{*foll},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{
                   {1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1}},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{
                   {0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0}});
  this->run_test(this->descending_no_nulls,
                 cudf::order::DESCENDING,
                 cudf::null_order::BEFORE,
                 cudf::bounded_open{*prec},
                 cudf::bounded_open{*foll},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{
                   {0, -1, 0, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0}},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{
                   {-1, -1, -2, -1, -1, -1, -1, -2, -1, -1, -1, -2, -1}});
}

TYPED_TEST(UngroupedIntegralRangeWindows, DescendingNoNullsPositivePrecedingFollowing)
{
  using T   = TypeParam;
  auto prec = cudf::make_fixed_width_scalar<T>(2);
  auto foll = cudf::make_fixed_width_scalar<T>(1);

  this->run_test(this->descending_no_nulls,
                 cudf::order::DESCENDING,
                 cudf::null_order::BEFORE,
                 cudf::bounded_closed{*prec},
                 cudf::bounded_closed{*foll},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{
                   {1, 1, 2, 1, 1, 2, 1, 2, 3, 2, 3, 4, 1}},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{
                   {0, 1, 0, 0, 1, 0, 1, 0, 1, 2, 1, 0, 0}});
  this->run_test(this->descending_no_nulls,
                 cudf::order::DESCENDING,
                 cudf::null_order::BEFORE,
                 cudf::bounded_open{*prec},
                 cudf::bounded_open{*foll},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{
                   {1, 1, 2, 1, 1, 2, 1, 2, 1, 2, 2, 3, 1}},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{
                   {0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0}});
}

TYPED_TEST(UngroupedSignedIntegralRangeWindows, DescendingNoNullsNegativePreceding)
{
  using T   = TypeParam;
  auto prec = cudf::make_fixed_width_scalar<T>(-1);
  auto foll = cudf::make_fixed_width_scalar<T>(2);

  this->run_test(this->descending_no_nulls,
                 cudf::order::DESCENDING,
                 cudf::null_order::BEFORE,
                 cudf::bounded_closed{*prec},
                 cudf::bounded_closed{*foll},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{
                   {0, -1, 0, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0}},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{
                   {0, 1, 0, 0, 1, 0, 2, 1, 3, 2, 1, 0, 0}});
  this->run_test(this->descending_no_nulls,
                 cudf::order::DESCENDING,
                 cudf::null_order::BEFORE,
                 cudf::bounded_open{*prec},
                 cudf::bounded_open{*foll},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{
                   {0, -1, 0, 0, -1, 0, -1, 0, -1, -2, -1, 0, 0}},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{
                   {0, 1, 0, 0, 1, 0, 1, 0, 1, 2, 1, 0, 0}});
}

TYPED_TEST(UngroupedSignedIntegralRangeWindows, DescendingNoNullsNegativeFollowing)
{
  using T   = TypeParam;
  auto prec = cudf::make_fixed_width_scalar<T>(4);
  auto foll = cudf::make_fixed_width_scalar<T>(-2);

  this->run_test(this->descending_no_nulls,
                 cudf::order::DESCENDING,
                 cudf::null_order::BEFORE,
                 cudf::bounded_closed{*prec},
                 cudf::bounded_closed{*foll},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{
                   {1, 1, 2, 1, 2, 2, 3, 4, 3, 4, 5, 6, 1}},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{
                   {-1, -1, -2, -1, -1, -2, -1, -2, -1, -2, -2, -3, -1}});
  this->run_test(this->descending_no_nulls,
                 cudf::order::DESCENDING,
                 cudf::null_order::BEFORE,
                 cudf::bounded_open{*prec},
                 cudf::bounded_open{*foll},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{
                   {1, 1, 2, 1, 1, 2, 2, 3, 3, 4, 3, 4, 1}},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{
                   {-1, -1, -2, -1, -1, -2, -1, -2, -3, -2, -3, -4, -1}});
}

TYPED_TEST(UngroupedIntegralRangeWindows, AscendingNullsZeroPrecedingFollowing)
{
  using T   = TypeParam;
  auto prec = cudf::make_fixed_width_scalar<T>(0);
  auto foll = cudf::make_fixed_width_scalar<T>(0);

  this->run_test(this->ascending_nulls_before,
                 cudf::order::ASCENDING,
                 cudf::null_order::BEFORE,
                 cudf::bounded_closed{*prec},
                 cudf::bounded_closed{*foll},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{
                   {1, 2, 3, 4, 1, 1, 2, 1, 1, 1, 1, 2, 1}},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{
                   {3, 2, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0}});
  this->run_test(this->ascending_nulls_before,
                 cudf::order::ASCENDING,
                 cudf::null_order::BEFORE,
                 cudf::bounded_open{*prec},
                 cudf::bounded_open{*foll},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{
                   {1, 2, 3, 4, 0, -1, 0, 0, 0, 0, -1, 0, 0}},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{
                   {3, 2, 1, 0, -1, -1, -2, -1, -1, -1, -1, -2, -1}});
}

TYPED_TEST(UngroupedIntegralRangeWindows, AscendingNullsPositivePrecedingFollowing)
{
  using T   = TypeParam;
  auto prec = cudf::make_fixed_width_scalar<T>(2);
  auto foll = cudf::make_fixed_width_scalar<T>(1);

  this->run_test(this->ascending_nulls_before,
                 cudf::order::ASCENDING,
                 cudf::null_order::BEFORE,
                 cudf::bounded_closed{*prec},
                 cudf::bounded_closed{*foll},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{
                   {1, 2, 3, 4, 1, 2, 3, 1, 2, 1, 1, 2, 1}},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{
                   {3, 2, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0}});
  this->run_test(this->ascending_nulls_before,
                 cudf::order::ASCENDING,
                 cudf::null_order::BEFORE,
                 cudf::bounded_open{*prec},
                 cudf::bounded_open{*foll},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{
                   {1, 2, 3, 4, 1, 1, 2, 1, 2, 1, 1, 2, 1}},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{
                   {3, 2, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0}});
}

TYPED_TEST(UngroupedSignedIntegralRangeWindows, AscendingNullsNegativePreceding)
{
  using T   = TypeParam;
  auto prec = cudf::make_fixed_width_scalar<T>(-1);
  auto foll = cudf::make_fixed_width_scalar<T>(2);

  this->run_test(this->ascending_nulls_after,
                 cudf::order::ASCENDING,
                 cudf::null_order::AFTER,
                 cudf::bounded_closed{*prec},
                 cudf::bounded_closed{*foll},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{
                   {0, -1, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7}},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{
                   {0, 3, 2, 1, 1, 0, 6, 5, 4, 3, 2, 1, 0}});
  this->run_test(this->ascending_nulls_after,
                 cudf::order::ASCENDING,
                 cudf::null_order::AFTER,
                 cudf::bounded_open{*prec},
                 cudf::bounded_open{*foll},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{
                   {0, -2, -1, -1, 0, 0, 1, 2, 3, 4, 5, 6, 7}},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{
                   {0, 2, 1, 1, 0, 0, 6, 5, 4, 3, 2, 1, 0}});
}

TYPED_TEST(UngroupedSignedIntegralRangeWindows, AscendingNullsNegativeFollowing)
{
  using T   = TypeParam;
  auto prec = cudf::make_fixed_width_scalar<T>(4);
  auto foll = cudf::make_fixed_width_scalar<T>(-2);

  this->run_test(this->ascending_nulls_after,
                 cudf::order::ASCENDING,
                 cudf::null_order::AFTER,
                 cudf::bounded_closed{*prec},
                 cudf::bounded_closed{*foll},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{
                   {1, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 6, 7}},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{
                   {-1, -1, -2, -3, -2, -1, 6, 5, 4, 3, 2, 1, 0}});
  this->run_test(this->ascending_nulls_after,
                 cudf::order::ASCENDING,
                 cudf::null_order::AFTER,
                 cudf::bounded_open{*prec},
                 cudf::bounded_open{*foll},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{
                   {1, 1, 2, 3, 4, 3, 1, 2, 3, 4, 5, 6, 7}},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{
                   {-1, -1, -2, -3, -4, -2, 6, 5, 4, 3, 2, 1, 0}});
}

TYPED_TEST(UngroupedIntegralRangeWindows, DescendingNullsZeroPrecedingFollowing)
{
  using T   = TypeParam;
  auto prec = cudf::make_fixed_width_scalar<T>(0);
  auto foll = cudf::make_fixed_width_scalar<T>(0);

  this->run_test(this->descending_nulls_before,
                 cudf::order::DESCENDING,
                 cudf::null_order::BEFORE,
                 cudf::bounded_closed{*prec},
                 cudf::bounded_closed{*foll},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{
                   {1, 1, 2, 1, 1, 1, 1, 1, 2, 3, 4, 5, 6}},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{
                   {0, 1, 0, 0, 0, 0, 0, 5, 4, 3, 2, 1, 0}});
  this->run_test(this->descending_nulls_before,
                 cudf::order::DESCENDING,
                 cudf::null_order::BEFORE,
                 cudf::bounded_open{*prec},
                 cudf::bounded_open{*foll},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{
                   {0, -1, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6}},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{
                   {-1, -1, -2, -1, -1, -1, -1, 5, 4, 3, 2, 1, 0}});
}

TYPED_TEST(UngroupedIntegralRangeWindows, DescendingNullsPositivePrecedingFollowing)
{
  using T   = TypeParam;
  auto prec = cudf::make_fixed_width_scalar<T>(2);
  auto foll = cudf::make_fixed_width_scalar<T>(1);

  this->run_test(this->descending_nulls_before,
                 cudf::order::DESCENDING,
                 cudf::null_order::BEFORE,
                 cudf::bounded_closed{*prec},
                 cudf::bounded_closed{*foll},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{
                   {1, 1, 2, 1, 1, 2, 1, 1, 2, 3, 4, 5, 6}},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{
                   {0, 1, 0, 0, 1, 0, 0, 5, 4, 3, 2, 1, 0}});
  this->run_test(this->descending_nulls_before,
                 cudf::order::DESCENDING,
                 cudf::null_order::BEFORE,
                 cudf::bounded_open{*prec},
                 cudf::bounded_open{*foll},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{
                   {1, 1, 2, 1, 1, 2, 1, 1, 2, 3, 4, 5, 6}},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{
                   {0, 1, 0, 0, 0, 0, 0, 5, 4, 3, 2, 1, 0}});
}

TYPED_TEST(UngroupedSignedIntegralRangeWindows, DescendingNullsNegativePreceding)
{
  using T   = TypeParam;
  auto prec = cudf::make_fixed_width_scalar<T>(-1);
  auto foll = cudf::make_fixed_width_scalar<T>(2);

  this->run_test(this->descending_nulls_after,
                 cudf::order::DESCENDING,
                 cudf::null_order::AFTER,
                 cudf::bounded_closed{*prec},
                 cudf::bounded_closed{*foll},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{
                   {1, 2, 3, 4, 5, 0, -1, 0, 0, 0, -1, 0, 0}},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{
                   {4, 3, 2, 1, 0, 0, 2, 1, 3, 2, 1, 0, 0}});
  this->run_test(this->descending_nulls_after,
                 cudf::order::DESCENDING,
                 cudf::null_order::AFTER,
                 cudf::bounded_open{*prec},
                 cudf::bounded_open{*foll},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{
                   {1, 2, 3, 4, 5, 0, -1, 0, -1, -2, -1, 0, 0}},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{
                   {4, 3, 2, 1, 0, 0, 1, 0, 1, 2, 1, 0, 0}});
}

TYPED_TEST(UngroupedSignedIntegralRangeWindows, DescendingNullsNegativeFollowing)
{
  using T   = TypeParam;
  auto prec = cudf::make_fixed_width_scalar<T>(4);
  auto foll = cudf::make_fixed_width_scalar<T>(-2);

  this->run_test(this->descending_nulls_after,
                 cudf::order::DESCENDING,
                 cudf::null_order::AFTER,
                 cudf::bounded_closed{*prec},
                 cudf::bounded_closed{*foll},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{
                   {1, 2, 3, 4, 5, 1, 2, 3, 3, 4, 5, 6, 1}},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{
                   {4, 3, 2, 1, 0, -1, -1, -2, -1, -2, -2, -3, -1}});
  this->run_test(this->descending_nulls_after,
                 cudf::order::DESCENDING,
                 cudf::null_order::AFTER,
                 cudf::bounded_open{*prec},
                 cudf::bounded_open{*foll},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{
                   {1, 2, 3, 4, 5, 1, 2, 3, 3, 4, 3, 4, 1}},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{
                   {4, 3, 2, 1, 0, -1, -1, -2, -3, -2, -3, -4, -1}});
}

template <typename T>
struct GroupedBase : cudf::test::BaseFixture {
  static constexpr T min{std::numeric_limits<T>::lowest()};
  static constexpr T max{std::numeric_limits<T>::max()};
  using cw_t = cudf::test::fixed_width_column_wrapper<T>;
  cw_t group_keys{};
  cw_t ascending_no_nulls{};
  cw_t ascending_nulls_before{};
  cw_t ascending_nulls_after{};
  cw_t descending_no_nulls{};
  cw_t descending_nulls_before{};
  cw_t descending_nulls_after{};
  GroupedBase()
    // clang-format off
    : group_keys{{
        // Group-1
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        // Group-2
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        // Group-3
        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
      }},
      ascending_no_nulls{{
        // Group-1
        min, T{5}, T{5}, T{6}, T{7}, T{9}, T{9}, T{12}, T{13}, T{17}, T{22}, T{22}, max,
        // Group-2
        min, T{5}, T{5}, T{6}, T{7}, T{9}, T{9}, T{12}, T{13}, T{17}, T{22}, T{22}, max,
        // Group-3
        min, T{5}, T{5}, T{6}, T{7}, T{9}, T{9}, T{12}, T{13}, T{17}, T{22}, T{22}, max,
      }},
      ascending_nulls_before{{
        // Group-1
        min, T{5}, T{5}, T{6}, T{7}, T{9}, T{9}, T{12}, T{13}, T{17}, T{22}, T{22}, max,
        // Group-2
        min, T{5}, T{5}, T{6}, T{7}, T{9}, T{9}, T{12}, T{13}, T{17}, T{22}, T{22}, max,
        // Group-3,
        min, T{5}, T{5}, T{6}, T{7}, T{9}, T{9}, T{12}, T{13}, T{17}, T{22}, T{22}, max,
      }, {
        // Group-1
        false, false, false, false, true, true, true, true, true, true, true, true, true,
        // Group-2
        true, true, true, true, true, true, true, true, true, true, true, true, true,
        // Group-3
        false, false, true, true, true, true, true, true, true, true, true, true, true,
      }},
      ascending_nulls_after{{
        // Group-1
        min, T{5}, T{5}, T{6}, T{7}, T{9}, T{9}, T{12}, T{13}, T{17}, T{22}, T{22}, max,
        // Group-2
        min, T{5}, T{5}, T{6}, T{7}, T{9}, T{9}, T{12}, T{13}, T{17}, T{22}, T{22}, max,
        // Group-3
        min, T{5}, T{5}, T{6}, T{7}, T{9}, T{9}, T{12}, T{13}, T{17}, T{22}, T{22}, max,
      }, {
        // Group-1
        true, true, true, true, true, true, false, false, false, false, false, false, false,
        // Group-2
        true, true, true, true, true, true, true, true, true, true, true, true, true,
        // Group-3
        true, true, true, true, true, true, true, true, true, true, false, false, false,
      }},
      descending_no_nulls{{
        // Group-1
        max, T{22}, T{22}, T{17}, T{13}, T{12}, T{9}, T{9}, T{7}, T{6}, T{5}, T{5}, min,
        // Group-2
        max, T{22}, T{22}, T{17}, T{13}, T{12}, T{9}, T{9}, T{7}, T{6}, T{5}, T{5}, min,
        // Group-3
        max, T{22}, T{22}, T{17}, T{13}, T{12}, T{9}, T{9}, T{7}, T{6}, T{5}, T{5}, min,
      }},
      descending_nulls_before{{
        // Group-1
        max, T{22}, T{22}, T{17}, T{13}, T{12}, T{9}, T{9}, T{7}, T{6}, T{5}, T{5}, min,
        // Group-2
        max, T{22}, T{22}, T{17}, T{13}, T{12}, T{9}, T{9}, T{7}, T{6}, T{5}, T{5}, min,
        // Group-3
        max, T{22}, T{22}, T{17}, T{13}, T{12}, T{9}, T{9}, T{7}, T{6}, T{5}, T{5}, min,
      }, {
        // Group-1
        true, true, true, true, true, true, true, false, false, false, false, false, false,
        // Group-2
        true, true, true, true, true, true, true, true, true, true, true, true, true,
        // Group-3
        true, true, true, true, true, true, true, true, true, true, false, false, false,
      }},
      descending_nulls_after{{
        // Group-1
        max, T{22}, T{22}, T{17}, T{13}, T{12}, T{9}, T{9}, T{7}, T{6}, T{5}, T{5}, min,
        // Group-2
        max, T{22}, T{22}, T{17}, T{13}, T{12}, T{9}, T{9}, T{7}, T{6}, T{5}, T{5}, min,
        // Group-3
        max, T{22}, T{22}, T{17}, T{13}, T{12}, T{9}, T{9}, T{7}, T{6}, T{5}, T{5}, min,
      }, {
        // Group-1
        false, false, false, false, false, true, true, true, true, true, true, true, true,
        // Group-2
        true, true, true, true, true, true, true, true, true, true, true, true, true,
        // Group-3
        false, false, false, false, true, true, true, true, true, true, true, true, true,
      }}
  // clang-format on
  {
  }

  void run_test(cudf::column_view const& orderby,
                cudf::order order,
                cudf::null_order null_order,
                cudf::range_window_type preceding,
                cudf::range_window_type following,
                cudf::column_view const& expect_preceding,
                cudf::column_view const& expect_following)
  {
    auto result = cudf::make_range_windows(
      cudf::table_view{{this->group_keys}}, orderby, order, null_order, preceding, following);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(
      std::get<0>(result)->view(), expect_preceding, cudf::test::debug_output_level::ALL_ERRORS);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(
      std::get<1>(result)->view(), expect_following, cudf::test::debug_output_level::ALL_ERRORS);
  }
};

template <typename T>
struct GroupedIntegralRangeWindows : GroupedBase<T> {};
TYPED_TEST_SUITE(GroupedIntegralRangeWindows, cudf::test::IntegralTypesNotBool);

template <typename T>
struct GroupedSignedIntegralRangeWindows : GroupedBase<T> {};
TYPED_TEST_SUITE(GroupedSignedIntegralRangeWindows, SignedIntegralTypes);

TYPED_TEST(GroupedIntegralRangeWindows, AscendingNoNullsZeroPrecedingFollowing)
{
  using T   = TypeParam;
  auto prec = cudf::make_fixed_width_scalar<T>(0);
  auto foll = cudf::make_fixed_width_scalar<T>(0);

  this->run_test(this->ascending_no_nulls,
                 cudf::order::ASCENDING,
                 cudf::null_order::BEFORE,
                 cudf::bounded_closed{*prec},
                 cudf::bounded_closed{*foll},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
                   // clang-format off
                   // Group-1
                   1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1,
                   // Group-2
                   1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1,
                   // Group-3
                   1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1,
                   // clang-format on
                 }},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
                   // clang-format off
                   // Group-1
                   0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0,
                   // Group-2
                   0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0,
                   // Group-3
                   0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0,
                   // clang-format on
                 }});
  this->run_test(this->ascending_no_nulls,
                 cudf::order::ASCENDING,
                 cudf::null_order::BEFORE,
                 cudf::bounded_open{*prec},
                 cudf::bounded_open{*foll},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
                   // clang-format off
                   // Group-1
                   0, -1, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0,
                   // Group-2
                   0, -1, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0,
                   // Group-3
                   0, -1, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0,
                   // clang-format on
                 }},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
                   // clang-format off
                   // Group-1
                   -1, -1, -2, -1, -1, -1, -2, -1, -1, -1, -1, -2, -1,
                   // Group-2
                   -1, -1, -2, -1, -1, -1, -2, -1, -1, -1, -1, -2, -1,
                   // Group-3
                   -1, -1, -2, -1, -1, -1, -2, -1, -1, -1, -1, -2, -1,
                   // clang-format on
                 }});
}

TYPED_TEST(GroupedIntegralRangeWindows, AscendingNoNullsPositivePrecedingFollowing)
{
  using T   = TypeParam;
  auto prec = cudf::make_fixed_width_scalar<T>(2);
  auto foll = cudf::make_fixed_width_scalar<T>(1);

  this->run_test(this->ascending_no_nulls,
                 cudf::order::ASCENDING,
                 cudf::null_order::BEFORE,
                 cudf::bounded_closed{*prec},
                 cudf::bounded_closed{*foll},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
                   // clang-format off
                   // Group-1
                   1, 1, 2, 3, 4, 2, 3, 1, 2, 1, 1, 2, 1,
                   // Group-2
                   1, 1, 2, 3, 4, 2, 3, 1, 2, 1, 1, 2, 1,
                   // Group-3
                   1, 1, 2, 3, 4, 2, 3, 1, 2, 1, 1, 2, 1,
                   // clang-format on
                 }},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
                   // clang-format off
                   // Group-1
                   0, 2, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0,
                   // Group-2
                   0, 2, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0,
                   // Group-3
                   0, 2, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0,
                   // clang-format on
                 }});
  this->run_test(this->ascending_no_nulls,
                 cudf::order::ASCENDING,
                 cudf::null_order::BEFORE,
                 cudf::bounded_open{*prec},
                 cudf::bounded_open{*foll},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
                   // clang-format off
                   // Group-1
                   1, 1, 2, 3, 2, 1, 2, 1, 2, 1, 1, 2, 1,
                   // Group-2
                   1, 1, 2, 3, 2, 1, 2, 1, 2, 1, 1, 2, 1,
                   // Group-3
                   1, 1, 2, 3, 2, 1, 2, 1, 2, 1, 1, 2, 1,
                   // clang-format on
                 }},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
                   // clang-format off
                   // Group-1
                   0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0,
                   // Group-2
                   0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0,
                   // Group-3
                   0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0,
                   // clang-format on
                 }});
}

TYPED_TEST(GroupedSignedIntegralRangeWindows, AscendingNoNullsNegativePreceding)
{
  using T   = TypeParam;
  auto prec = cudf::make_fixed_width_scalar<T>(-1);
  auto foll = cudf::make_fixed_width_scalar<T>(2);

  this->run_test(this->ascending_no_nulls,
                 cudf::order::ASCENDING,
                 cudf::null_order::BEFORE,
                 cudf::bounded_closed{*prec},
                 cudf::bounded_closed{*foll},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
                   // clang-format off
                   // Group-1
                   0, -1, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0,
                   // Group-2
                   0, -1, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0,
                   // Group-3
                   0, -1, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0,
                   // clang-format on
                 }},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
                   // clang-format off
                   // Group-1
                   0, 3, 2, 1, 2, 1, 0, 1, 0, 0, 1, 0, 0,
                   // Group-2
                   0, 3, 2, 1, 2, 1, 0, 1, 0, 0, 1, 0, 0,
                   // Group-3
                   0, 3, 2, 1, 2, 1, 0, 1, 0, 0, 1, 0, 0,
                   // clang-format on
                 }});
  this->run_test(this->ascending_no_nulls,
                 cudf::order::ASCENDING,
                 cudf::null_order::BEFORE,
                 cudf::bounded_open{*prec},
                 cudf::bounded_open{*foll},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
                   // clang-format off
                   // Group-1
                   0, -2, -1, -1, 0, -1, 0, -1, 0, 0, -1, 0, 0,
                   // Group-2
                   0, -2, -1, -1, 0, -1, 0, -1, 0, 0, -1, 0, 0,
                   // Group-3
                   0, -2, -1, -1, 0, -1, 0, -1, 0, 0, -1, 0, 0,
                   // clang-format on
                 }},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
                   // clang-format off
                   // Group-1
                   0, 2, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0,
                   // Group-2
                   0, 2, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0,
                   // Group-3
                   0, 2, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0,
                   // clang-format on
                 }});
}

TYPED_TEST(GroupedSignedIntegralRangeWindows, AscendingNoNullsNegativeFollowing)
{
  using T   = TypeParam;
  auto prec = cudf::make_fixed_width_scalar<T>(4);
  auto foll = cudf::make_fixed_width_scalar<T>(-2);

  this->run_test(this->ascending_no_nulls,
                 cudf::order::ASCENDING,
                 cudf::null_order::BEFORE,
                 cudf::bounded_closed{*prec},
                 cudf::bounded_closed{*foll},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
                   // clang-format off
                   // Group-1
                   1, 1, 2, 3, 4, 5, 6, 3, 4, 2, 1, 2, 1,
                   // Group-2
                   1, 1, 2, 3, 4, 5, 6, 3, 4, 2, 1, 2, 1,
                   // Group-3
                   1, 1, 2, 3, 4, 5, 6, 3, 4, 2, 1, 2, 1,
                   // clang-format on
                 }},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
                   // clang-format off
                   // Group-1
                   -1, -1, -2, -3, -2, -1, -2, -1, -2, -1, -1, -2, -1,
                   // Group-2
                   -1, -1, -2, -3, -2, -1, -2, -1, -2, -1, -1, -2, -1,
                   // Group-3
                   -1, -1, -2, -3, -2, -1, -2, -1, -2, -1, -1, -2, -1,
                   // clang-format on
                 }});
  this->run_test(this->ascending_no_nulls,
                 cudf::order::ASCENDING,
                 cudf::null_order::BEFORE,
                 cudf::bounded_open{*prec},
                 cudf::bounded_open{*foll},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
                   // clang-format off
                   // Group-1
                   1, 1, 2, 3, 4, 3, 4, 3, 2, 1, 1, 2, 1,
                   // Group-2
                   1, 1, 2, 3, 4, 3, 4, 3, 2, 1, 1, 2, 1,
                   // Group-3
                   1, 1, 2, 3, 4, 3, 4, 3, 2, 1, 1, 2, 1,
                   // clang-format on
                 }},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
                   // clang-format off
                   // Group-1
                   -1, -1, -2, -3, -4, -2, -3, -1, -2, -1, -1, -2, -1,
                   // Group-2
                   -1, -1, -2, -3, -4, -2, -3, -1, -2, -1, -1, -2, -1,
                   // Group-3
                   -1, -1, -2, -3, -4, -2, -3, -1, -2, -1, -1, -2, -1,
                   // clang-format on
                 }});
}

TYPED_TEST(GroupedIntegralRangeWindows, DescendingNoNullsZeroPrecedingFollowing)
{
  using T   = TypeParam;
  auto prec = cudf::make_fixed_width_scalar<T>(0);
  auto foll = cudf::make_fixed_width_scalar<T>(0);

  this->run_test(this->descending_no_nulls,
                 cudf::order::DESCENDING,
                 cudf::null_order::BEFORE,
                 cudf::bounded_closed{*prec},
                 cudf::bounded_closed{*foll},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
                   // clang-format off
                   // Group-1
                   1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1,
                   // Group-2
                   1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1,
                   // Group-3
                   1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1,
                   // clang-format on
                 }},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
                   // clang-format off
                   // Group-1
                   0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0,
                   // Group-2
                   0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0,
                   // Group-3
                   0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0,
                   // clang-format on
                 }});
  this->run_test(this->descending_no_nulls,
                 cudf::order::DESCENDING,
                 cudf::null_order::BEFORE,
                 cudf::bounded_open{*prec},
                 cudf::bounded_open{*foll},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
                   // clang-format off
                   // Group-1
                   0, -1, 0, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0,
                   // Group-2
                   0, -1, 0, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0,
                   // Group-3
                   0, -1, 0, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0,
                   // clang-format on
                 }},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
                   // clang-format off
                   // Group-1
                   -1, -1, -2, -1, -1, -1, -1, -2, -1, -1, -1, -2, -1,
                   // Group-2
                   -1, -1, -2, -1, -1, -1, -1, -2, -1, -1, -1, -2, -1,
                   // Group-3
                   -1, -1, -2, -1, -1, -1, -1, -2, -1, -1, -1, -2, -1,
                   // clang-format on
                 }});
}

TYPED_TEST(GroupedIntegralRangeWindows, DescendingNoNullsPositivePrecedingFollowing)
{
  using T   = TypeParam;
  auto prec = cudf::make_fixed_width_scalar<T>(2);
  auto foll = cudf::make_fixed_width_scalar<T>(1);

  this->run_test(this->descending_no_nulls,
                 cudf::order::DESCENDING,
                 cudf::null_order::BEFORE,
                 cudf::bounded_closed{*prec},
                 cudf::bounded_closed{*foll},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
                   // clang-format off
                   // Group-1
                   1, 1, 2, 1, 1, 2, 1, 2, 3, 2, 3, 4, 1,
                   // Group-2
                   1, 1, 2, 1, 1, 2, 1, 2, 3, 2, 3, 4, 1,
                   // Group-3
                   1, 1, 2, 1, 1, 2, 1, 2, 3, 2, 3, 4, 1,
                   // clang-format on
                 }},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
                   // clang-format off
                   // Group-1
                   0, 1, 0, 0, 1, 0, 1, 0, 1, 2, 1, 0, 0,
                   // Group-2
                   0, 1, 0, 0, 1, 0, 1, 0, 1, 2, 1, 0, 0,
                   // Group-3
                   0, 1, 0, 0, 1, 0, 1, 0, 1, 2, 1, 0, 0,
                   // clang-format on
                 }});
  this->run_test(this->descending_no_nulls,
                 cudf::order::DESCENDING,
                 cudf::null_order::BEFORE,
                 cudf::bounded_open{*prec},
                 cudf::bounded_open{*foll},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
                   // clang-format off
                   // Group-1
                   1, 1, 2, 1, 1, 2, 1, 2, 1, 2, 2, 3, 1,
                   // Group-2
                   1, 1, 2, 1, 1, 2, 1, 2, 1, 2, 2, 3, 1,
                   // Group-3
                   1, 1, 2, 1, 1, 2, 1, 2, 1, 2, 2, 3, 1,
                   // clang-format on
                 }},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
                   // clang-format off
                   // Group-1
                   0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0,
                   // Group-2
                   0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0,
                   // Group-3
                   0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0,
                   // clang-format on
                 }});
}

TYPED_TEST(GroupedSignedIntegralRangeWindows, DescendingNoNullsNegativePreceding)
{
  using T   = TypeParam;
  auto prec = cudf::make_fixed_width_scalar<T>(-1);
  auto foll = cudf::make_fixed_width_scalar<T>(2);

  this->run_test(this->descending_no_nulls,
                 cudf::order::DESCENDING,
                 cudf::null_order::BEFORE,
                 cudf::bounded_closed{*prec},
                 cudf::bounded_closed{*foll},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
                   // clang-format off
                   // Group-1
                   0, -1, 0, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0,
                   // Group-2
                   0, -1, 0, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0,
                   // Group-3
                   0, -1, 0, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0,
                   // clang-format on
                 }},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
                   // clang-format off
                   // Group-1
                   0, 1, 0, 0, 1, 0, 2, 1, 3, 2, 1, 0, 0,
                   // Group-2
                   0, 1, 0, 0, 1, 0, 2, 1, 3, 2, 1, 0, 0,
                   // Group-3
                   0, 1, 0, 0, 1, 0, 2, 1, 3, 2, 1, 0, 0,
                   // clang-format on
                 }});
  this->run_test(this->descending_no_nulls,
                 cudf::order::DESCENDING,
                 cudf::null_order::BEFORE,
                 cudf::bounded_open{*prec},
                 cudf::bounded_open{*foll},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
                   // clang-format off
                   // Group-1
                   0, -1, 0, 0, -1, 0, -1, 0, -1, -2, -1, 0, 0,
                   // Group-2
                   0, -1, 0, 0, -1, 0, -1, 0, -1, -2, -1, 0, 0,
                   // Group-3
                   0, -1, 0, 0, -1, 0, -1, 0, -1, -2, -1, 0, 0,
                   // clang-format on
                 }},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
                   // clang-format off
                   // Group-1
                   0, 1, 0, 0, 1, 0, 1, 0, 1, 2, 1, 0, 0,
                   // Group-2
                   0, 1, 0, 0, 1, 0, 1, 0, 1, 2, 1, 0, 0,
                   // Group-3
                   0, 1, 0, 0, 1, 0, 1, 0, 1, 2, 1, 0, 0,
                   // clang-format on
                 }});
}

TYPED_TEST(GroupedSignedIntegralRangeWindows, DescendingNoNullsNegativeFollowing)
{
  using T   = TypeParam;
  auto prec = cudf::make_fixed_width_scalar<T>(4);
  auto foll = cudf::make_fixed_width_scalar<T>(-2);

  this->run_test(this->descending_no_nulls,
                 cudf::order::DESCENDING,
                 cudf::null_order::BEFORE,
                 cudf::bounded_closed{*prec},
                 cudf::bounded_closed{*foll},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
                   // clang-format off
                   // Group-1
                   1, 1, 2, 1, 2, 2, 3, 4, 3, 4, 5, 6, 1,
                   // Group-2
                   1, 1, 2, 1, 2, 2, 3, 4, 3, 4, 5, 6, 1,
                   // Group-3
                   1, 1, 2, 1, 2, 2, 3, 4, 3, 4, 5, 6, 1,
                   // clang-format on
                 }},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
                   // clang-format off
                   // Group-1
                   -1, -1, -2, -1, -1, -2, -1, -2, -1, -2, -2, -3, -1,
                   // Group-2
                   -1, -1, -2, -1, -1, -2, -1, -2, -1, -2, -2, -3, -1,
                   // Group-3
                   -1, -1, -2, -1, -1, -2, -1, -2, -1, -2, -2, -3, -1,
                   // clang-format on
                 }});
  this->run_test(this->descending_no_nulls,
                 cudf::order::DESCENDING,
                 cudf::null_order::BEFORE,
                 cudf::bounded_open{*prec},
                 cudf::bounded_open{*foll},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
                   // clang-format off
                   // Group-1
                   1, 1, 2, 1, 1, 2, 2, 3, 3, 4, 3, 4, 1,
                   // Group-2
                   1, 1, 2, 1, 1, 2, 2, 3, 3, 4, 3, 4, 1,
                   // Group-3
                   1, 1, 2, 1, 1, 2, 2, 3, 3, 4, 3, 4, 1,
                   // clang-format on
                 }},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
                   // clang-format off
                   // Group-1
                   -1, -1, -2, -1, -1, -2, -1, -2, -3, -2, -3, -4, -1,
                   // Group-2
                   -1, -1, -2, -1, -1, -2, -1, -2, -3, -2, -3, -4, -1,
                   // Group-3
                   -1, -1, -2, -1, -1, -2, -1, -2, -3, -2, -3, -4, -1,
                   // clang-format on
                 }});
}

TYPED_TEST(GroupedIntegralRangeWindows, AscendingNullsZeroPrecedingFollowing)
{
  using T   = TypeParam;
  auto prec = cudf::make_fixed_width_scalar<T>(0);
  auto foll = cudf::make_fixed_width_scalar<T>(0);

  this->run_test(this->ascending_nulls_before,
                 cudf::order::ASCENDING,
                 cudf::null_order::BEFORE,
                 cudf::bounded_closed{*prec},
                 cudf::bounded_closed{*foll},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
                   // clang-format off
                   // Group-1
                   1, 2, 3, 4, 1, 1, 2, 1, 1, 1, 1, 2, 1,
                   // Group-2
                   1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1,
                   // Group-3
                   1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1,
                   // clang-format on
                 }},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
                   // clang-format off
                   // Group-1
                   3, 2, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0,
                   // Group-2
                   0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0,
                   // Group-3
                   1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0,
                   // clang-format on
                 }});
  this->run_test(this->ascending_nulls_before,
                 cudf::order::ASCENDING,
                 cudf::null_order::BEFORE,
                 cudf::bounded_open{*prec},
                 cudf::bounded_open{*foll},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
                   // clang-format off
                   // Group-1
                   1, 2, 3, 4, 0, -1, 0, 0, 0, 0, -1, 0, 0,
                   // Group-2
                   0, -1, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0,
                   // Group-3
                   1, 2, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0,
                   // clang-format on
                 }},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
                   // clang-format off
                   // Group-1
                   3, 2, 1, 0, -1, -1, -2, -1, -1, -1, -1, -2, -1,
                   // Group-2
                   -1, -1, -2, -1, -1, -1, -2, -1, -1, -1, -1, -2, -1,
                   // Group-3
                   1, 0, -1, -1, -1, -1, -2, -1, -1, -1, -1, -2, -1,
                   // clang-format on
                 }});
}

TYPED_TEST(GroupedIntegralRangeWindows, AscendingNullsPositivePrecedingFollowing)
{
  using T   = TypeParam;
  auto prec = cudf::make_fixed_width_scalar<T>(2);
  auto foll = cudf::make_fixed_width_scalar<T>(1);

  this->run_test(this->ascending_nulls_before,
                 cudf::order::ASCENDING,
                 cudf::null_order::BEFORE,
                 cudf::bounded_closed{*prec},
                 cudf::bounded_closed{*foll},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
                   // clang-format off
                   // Group-1
                   1, 2, 3, 4, 1, 2, 3, 1, 2, 1, 1, 2, 1,
                   // Group-2
                   1, 1, 2, 3, 4, 2, 3, 1, 2, 1, 1, 2, 1,
                   // Group-3
                   1, 2, 1, 2, 3, 2, 3, 1, 2, 1, 1, 2, 1,
                   // clang-format on
                 }},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
                   // clang-format off
                   // Group-1
                   3, 2, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0,
                   // Group-2
                   0, 2, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0,
                   // Group-3
                   1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0,
                   // clang-format on
                 }});
  this->run_test(this->ascending_nulls_before,
                 cudf::order::ASCENDING,
                 cudf::null_order::BEFORE,
                 cudf::bounded_open{*prec},
                 cudf::bounded_open{*foll},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
                   // clang-format off
                   // Group-1
                   1, 2, 3, 4, 1, 1, 2, 1, 2, 1, 1, 2, 1,
                   // Group-2
                   1, 1, 2, 3, 2, 1, 2, 1, 2, 1, 1, 2, 1,
                   // Group-3
                   1, 2, 1, 2, 2, 1, 2, 1, 2, 1, 1, 2, 1
                   // clang-format on
                 }},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
                   // clang-format off
                   // Group-1
                   3, 2, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0,
                   // Group-2
                   0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0,
                   // Group-3
                   1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0,
                   // clang-format on
                 }});
}

TYPED_TEST(GroupedSignedIntegralRangeWindows, AscendingNullsNegativePreceding)
{
  using T   = TypeParam;
  auto prec = cudf::make_fixed_width_scalar<T>(-1);
  auto foll = cudf::make_fixed_width_scalar<T>(2);

  this->run_test(this->ascending_nulls_after,
                 cudf::order::ASCENDING,
                 cudf::null_order::AFTER,
                 cudf::bounded_closed{*prec},
                 cudf::bounded_closed{*foll},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
                   // clang-format off
                   // Group-1
                   0, -1, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7,
                   // Group-2
                   0, -1, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0,
                   // Group-3
                   0, -1, 0, 0, 0, -1, 0, 0, 0, 0, 1, 2, 3,
                   // clang-format on
                 }},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
                   // clang-format off
                   // Group-1
                   0, 3, 2, 1, 1, 0, 6, 5, 4, 3, 2, 1, 0,
                   // Group-2
                   0, 3, 2, 1, 2, 1, 0, 1, 0, 0, 1, 0, 0,
                   // Group-3
                   0, 3, 2, 1, 2, 1, 0, 1, 0, 0, 2, 1, 0,
                   // clang-format on
                 }});
  this->run_test(this->ascending_nulls_after,
                 cudf::order::ASCENDING,
                 cudf::null_order::AFTER,
                 cudf::bounded_open{*prec},
                 cudf::bounded_open{*foll},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
                   // clang-format off
                   // Group-1
                   0, -2, -1, -1, 0, 0, 1, 2, 3, 4, 5, 6, 7,
                   // Group-2
                   0, -2, -1, -1, 0, -1, 0, -1, 0, 0, -1, 0, 0,
                   // Group-3
                   0, -2, -1, -1, 0, -1, 0, -1, 0, 0, 1, 2, 3,
                   // clang-format on
                 }},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
                   // clang-format off
                   // Group-1
                   0, 2, 1, 1, 0, 0, 6, 5, 4, 3, 2, 1, 0,
                   // Group-2
                   0, 2, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0,
                   // Group-3
                   0, 2, 1, 1, 0, 1, 0, 1, 0, 0, 2, 1, 0,
                   // clang-format on
                 }});
}

TYPED_TEST(GroupedSignedIntegralRangeWindows, AscendingNullsNegativeFollowing)
{
  using T   = TypeParam;
  auto prec = cudf::make_fixed_width_scalar<T>(4);
  auto foll = cudf::make_fixed_width_scalar<T>(-2);

  this->run_test(this->ascending_nulls_after,
                 cudf::order::ASCENDING,
                 cudf::null_order::AFTER,
                 cudf::bounded_closed{*prec},
                 cudf::bounded_closed{*foll},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
                   // clang-format off
                   // Group-1
                   1, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 6, 7,
                   // Group-2
                   1, 1, 2, 3, 4, 5, 6, 3, 4, 2, 1, 2, 1,
                   // Group-3
                   1, 1, 2, 3, 4, 5, 6, 3, 4, 2, 1, 2, 3,
                   // clang-format on
                 }},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
                   // clang-format off
                   // Group-1
                   -1, -1, -2, -3, -2, -1, 6, 5, 4, 3, 2, 1, 0,
                   // Group-2
                   -1, -1, -2, -3, -2, -1, -2, -1, -2, -1, -1, -2, -1,
                   // Group-3
                   -1, -1, -2, -3, -2, -1, -2, -1, -2, -1, 2, 1, 0,
                   // clang-format on
                 }});
  this->run_test(this->ascending_nulls_after,
                 cudf::order::ASCENDING,
                 cudf::null_order::AFTER,
                 cudf::bounded_open{*prec},
                 cudf::bounded_open{*foll},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
                   // clang-format off
                   // Group-1
                   1, 1, 2, 3, 4, 3, 1, 2, 3, 4, 5, 6, 7,
                   // Group-2
                   1, 1, 2, 3, 4, 3, 4, 3, 2, 1, 1, 2, 1,
                   // Group-3
                   1, 1, 2, 3, 4, 3, 4, 3, 2, 1, 1, 2, 3,
                   // clang-format on
                 }},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
                   // clang-format off
                   // Group-1
                   -1, -1, -2, -3, -4, -2, 6, 5, 4, 3, 2, 1, 0,
                   // Group-2
                   -1, -1, -2, -3, -4, -2, -3, -1, -2, -1, -1, -2, -1,
                   // Group-3
                   -1, -1, -2, -3, -4, -2, -3, -1, -2, -1, 2, 1, 0,
                   // clang-format on
                 }});
}

TYPED_TEST(GroupedIntegralRangeWindows, DescendingNullsZeroPrecedingFollowing)
{
  using T   = TypeParam;
  auto prec = cudf::make_fixed_width_scalar<T>(0);
  auto foll = cudf::make_fixed_width_scalar<T>(0);

  this->run_test(this->descending_nulls_before,
                 cudf::order::DESCENDING,
                 cudf::null_order::BEFORE,
                 cudf::bounded_closed{*prec},
                 cudf::bounded_closed{*foll},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
                   // clang-format off
                   // Group-1
                   1, 1, 2, 1, 1, 1, 1, 1, 2, 3, 4, 5, 6,
                   // Group-2
                   1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1,
                   // Group-3
                   1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 2, 3,
                   // clang-format on
                 }},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
                   // clang-format off
                   // Group-1
                   0, 1, 0, 0, 0, 0, 0, 5, 4, 3, 2, 1, 0,
                   // Group-2
                   0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0,
                   // Group-3
                   0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 2, 1, 0,
                   // clang-format on
                 }});
  this->run_test(this->descending_nulls_before,
                 cudf::order::DESCENDING,
                 cudf::null_order::BEFORE,
                 cudf::bounded_open{*prec},
                 cudf::bounded_open{*foll},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
                   // clang-format off
                   // Group-1
                   0, -1, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6,
                   // Group-2
                   0, -1, 0, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0,
                   // Group-3
                   0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 1, 2, 3,
                   // clang-format on
                 }},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
                   // clang-format off
                   // Group-1
                   -1, -1, -2, -1, -1, -1, -1, 5, 4, 3, 2, 1, 0,
                   // Group-2
                   -1, -1, -2, -1, -1, -1, -1, -2, -1, -1, -1, -2, -1,
                    // Group-3
                   -1, -1, -2, -1, -1, -1, -1, -2, -1, -1, 2, 1, 0,
                   // clang-format on
                 }});
}

TYPED_TEST(GroupedIntegralRangeWindows, DescendingNullsPositivePrecedingFollowing)
{
  using T   = TypeParam;
  auto prec = cudf::make_fixed_width_scalar<T>(2);
  auto foll = cudf::make_fixed_width_scalar<T>(1);

  this->run_test(this->descending_nulls_before,
                 cudf::order::DESCENDING,
                 cudf::null_order::BEFORE,
                 cudf::bounded_closed{*prec},
                 cudf::bounded_closed{*foll},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
                   // clang-format off
                   // Group-1
                   1, 1, 2, 1, 1, 2, 1, 1, 2, 3, 4, 5, 6,
                   // Group-2
                   1, 1, 2, 1, 1, 2, 1, 2, 3, 2, 3, 4, 1,
                   // Group-3
                   1, 1, 2, 1, 1, 2, 1, 2, 3, 2, 1, 2, 3,
                   // clang-format on
                 }},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
                   // clang-format off
                   // Group-1
                   0, 1, 0, 0, 1, 0, 0, 5, 4, 3, 2, 1, 0,
                   // Group-2
                   0, 1, 0, 0, 1, 0, 1, 0, 1, 2, 1, 0, 0,
                   // Group-3
                   0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 2, 1, 0,
                   // clang-format on
                 }});
  this->run_test(this->descending_nulls_before,
                 cudf::order::DESCENDING,
                 cudf::null_order::BEFORE,
                 cudf::bounded_open{*prec},
                 cudf::bounded_open{*foll},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
                   // clang-format off
                   // Group-1
                   1, 1, 2, 1, 1, 2, 1, 1, 2, 3, 4, 5, 6,
                   // Group-2
                   1, 1, 2, 1, 1, 2, 1, 2, 1, 2, 2, 3, 1,
                   // Group-3
                   1, 1, 2, 1, 1, 2, 1, 2, 1, 2, 1, 2, 3,
                   // clang-format on
                 }},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
                   // clang-format off
                   // Group-1
                   0, 1, 0, 0, 0, 0, 0, 5, 4, 3, 2, 1, 0,
                   // Group-2
                   0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0,
                   // Group-3
                   0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 2, 1, 0,
                   // clang-format on
                 }});
}

TYPED_TEST(GroupedSignedIntegralRangeWindows, DescendingNullsNegativePreceding)
{
  using T   = TypeParam;
  auto prec = cudf::make_fixed_width_scalar<T>(-1);
  auto foll = cudf::make_fixed_width_scalar<T>(2);

  this->run_test(this->descending_nulls_after,
                 cudf::order::DESCENDING,
                 cudf::null_order::AFTER,
                 cudf::bounded_closed{*prec},
                 cudf::bounded_closed{*foll},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
                   // clang-format off
                   // Group-1
                   1, 2, 3, 4, 5, 0, -1, 0, 0, 0, -1, 0, 0,
                   // Group-2
                   0, -1, 0, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0,
                   // Group-3
                   1, 2, 3, 4, 0, 0, -1, 0, 0, 0, -1, 0, 0,
                   // clang-format on
                 }},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
                   // clang-format off
                   // Group-1
                   4, 3, 2, 1, 0, 0, 2, 1, 3, 2, 1, 0, 0,
                   // Group-2
                   0, 1, 0, 0, 1, 0, 2, 1, 3, 2, 1, 0, 0,
                   // Group-3
                   3, 2, 1, 0, 1, 0, 2, 1, 3, 2, 1, 0, 0,
                   // clang-format on
                 }});
  this->run_test(this->descending_nulls_after,
                 cudf::order::DESCENDING,
                 cudf::null_order::AFTER,
                 cudf::bounded_open{*prec},
                 cudf::bounded_open{*foll},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
                   // clang-format off
                   // Group-1
                   1, 2, 3, 4, 5, 0, -1, 0, -1, -2, -1, 0, 0,
                   // Group-2
                   0, -1, 0, 0, -1, 0, -1, 0, -1, -2, -1, 0, 0,
                   // Group-3
                   1, 2, 3, 4, -1, 0, -1, 0, -1, -2, -1, 0, 0,
                   // clang-format on
                 }},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
                   // clang-format off
                   // Group-1
                   4, 3, 2, 1, 0, 0, 1, 0, 1, 2, 1, 0, 0,
                   // Group-2
                   0, 1, 0, 0, 1, 0, 1, 0, 1, 2, 1, 0, 0,
                   // Group-3
                   3, 2, 1, 0, 1, 0, 1, 0, 1, 2, 1, 0, 0,
                   // clang-format on
                 }});
}

TYPED_TEST(GroupedSignedIntegralRangeWindows, DescendingNullsNegativeFollowing)
{
  using T   = TypeParam;
  auto prec = cudf::make_fixed_width_scalar<T>(4);
  auto foll = cudf::make_fixed_width_scalar<T>(-2);

  this->run_test(this->descending_nulls_after,
                 cudf::order::DESCENDING,
                 cudf::null_order::AFTER,
                 cudf::bounded_closed{*prec},
                 cudf::bounded_closed{*foll},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
                   // clang-format off
                   // Group-1
                   1, 2, 3, 4, 5, 1, 2, 3, 3, 4, 5, 6, 1,
                   // Group-2
                   1, 1, 2, 1, 2, 2, 3, 4, 3, 4, 5, 6, 1,
                   // Group-3
                   1, 2, 3, 4, 1, 2, 3, 4, 3, 4, 5, 6, 1,
                   // clang-format on
                 }},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
                   // clang-format off
                   // Group-1
                   4, 3, 2, 1, 0, -1, -1, -2, -1, -2, -2, -3, -1,
                   // Group-2
                   -1, -1, -2, -1, -1, -2, -1, -2, -1, -2, -2, -3, -1,
                   // Group-3
                   3, 2, 1, 0, -1, -2, -1, -2, -1, -2, -2, -3, -1,
                   // clang-format on
                 }});
  this->run_test(this->descending_nulls_after,
                 cudf::order::DESCENDING,
                 cudf::null_order::AFTER,
                 cudf::bounded_open{*prec},
                 cudf::bounded_open{*foll},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
                   // clang-format off
                   // Group-1
                   1, 2, 3, 4, 5, 1, 2, 3, 3, 4, 3, 4, 1,
                   // Group-2
                   1, 1, 2, 1, 1, 2, 2, 3, 3, 4, 3, 4, 1,
                   // Group-3
                   1, 2, 3, 4, 1, 2, 2, 3, 3, 4, 3, 4, 1,
                   // clang-format on
                 }},
                 cudf::test::fixed_width_column_wrapper<cudf::size_type>{{
                   // clang-format off
                   // Group-1
                   4, 3, 2, 1, 0, -1, -1, -2, -3, -2, -3, -4, -1,
                   // Group-2
                   -1, -1, -2, -1, -1, -2, -1, -2, -3, -2, -3, -4, -1,
                   // Group-3
                   3, 2, 1, 0, -1, -2, -1, -2, -3, -2, -3, -4, -1,
                   // clang-format on
                 }});
}

// ===========================================================================================
// Column-valued (per-row) RANGE bounds: `bounded_closed_column` / `bounded_open_column`.
//
// The strongest correctness guarantee is parity: filling a delta column with a single constant
// must produce exactly the same preceding/following window-extent columns as the equivalent scalar
// `bounded_closed`/`bounded_open` bound, because the scalar path is the trusted reference.
// ===========================================================================================
namespace {

// Build a non-nullable column of `size` copies of `value`.
template <typename T>
[[nodiscard]] cudf::test::fixed_width_column_wrapper<T> constant_column(cudf::size_type size,
                                                                        T value)
{
  std::vector<T> const data(static_cast<std::size_t>(size), value);
  return cudf::test::fixed_width_column_wrapper<T>(data.begin(), data.end());
}

// Assert that a constant delta column reproduces the scalar-delta result, for both closed and open
// endpoints applied to preceding and following.
template <typename T>
void expect_column_delta_matches_scalar(cudf::table_view const& group_keys,
                                        cudf::column_view const& orderby,
                                        cudf::order order,
                                        cudf::null_order null_order,
                                        T delta_value)
{
  auto const delta_wrapper          = constant_column<T>(orderby.size(), delta_value);
  cudf::column_view const delta_col = delta_wrapper;
  auto const scalar                 = cudf::make_fixed_width_scalar<T>(delta_value);

  auto const closed_column = cudf::make_range_windows(group_keys,
                                                      orderby,
                                                      order,
                                                      null_order,
                                                      cudf::bounded_closed_column{delta_col},
                                                      cudf::bounded_closed_column{delta_col});
  auto const closed_scalar = cudf::make_range_windows(group_keys,
                                                      orderby,
                                                      order,
                                                      null_order,
                                                      cudf::bounded_closed{*scalar},
                                                      cudf::bounded_closed{*scalar});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(closed_column.first->view(),
                                 closed_scalar.first->view(),
                                 cudf::test::debug_output_level::ALL_ERRORS);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(closed_column.second->view(),
                                 closed_scalar.second->view(),
                                 cudf::test::debug_output_level::ALL_ERRORS);

  auto const open_column = cudf::make_range_windows(group_keys,
                                                    orderby,
                                                    order,
                                                    null_order,
                                                    cudf::bounded_open_column{delta_col},
                                                    cudf::bounded_open_column{delta_col});
  auto const open_scalar = cudf::make_range_windows(group_keys,
                                                    orderby,
                                                    order,
                                                    null_order,
                                                    cudf::bounded_open{*scalar},
                                                    cudf::bounded_open{*scalar});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(open_column.first->view(),
                                 open_scalar.first->view(),
                                 cudf::test::debug_output_level::ALL_ERRORS);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(open_column.second->view(),
                                 open_scalar.second->view(),
                                 cudf::test::debug_output_level::ALL_ERRORS);
}

}  // namespace

TYPED_TEST(UngroupedIntegralRangeWindows, ColumnDeltaMatchesScalarNoNulls)
{
  using T = TypeParam;
  for (T const delta : {T{0}, T{1}, T{2}, T{5}}) {
    expect_column_delta_matches_scalar<T>(cudf::table_view{},
                                          this->ascending_no_nulls,
                                          cudf::order::ASCENDING,
                                          cudf::null_order::BEFORE,
                                          delta);
    expect_column_delta_matches_scalar<T>(cudf::table_view{},
                                          this->descending_no_nulls,
                                          cudf::order::DESCENDING,
                                          cudf::null_order::BEFORE,
                                          delta);
  }
}

TYPED_TEST(UngroupedIntegralRangeWindows, ColumnDeltaMatchesScalarWithNulls)
{
  using T = TypeParam;
  for (T const delta : {T{0}, T{2}}) {
    expect_column_delta_matches_scalar<T>(cudf::table_view{},
                                          this->ascending_nulls_before,
                                          cudf::order::ASCENDING,
                                          cudf::null_order::BEFORE,
                                          delta);
    expect_column_delta_matches_scalar<T>(cudf::table_view{},
                                          this->descending_nulls_before,
                                          cudf::order::DESCENDING,
                                          cudf::null_order::BEFORE,
                                          delta);
  }
}

TYPED_TEST(UngroupedSignedIntegralRangeWindows, ColumnDeltaMatchesScalarNegative)
{
  using T = TypeParam;
  for (T const delta : {T{-1}, T{-2}}) {
    expect_column_delta_matches_scalar<T>(cudf::table_view{},
                                          this->ascending_no_nulls,
                                          cudf::order::ASCENDING,
                                          cudf::null_order::BEFORE,
                                          delta);
    expect_column_delta_matches_scalar<T>(cudf::table_view{},
                                          this->descending_no_nulls,
                                          cudf::order::DESCENDING,
                                          cudf::null_order::BEFORE,
                                          delta);
  }
}

TYPED_TEST(GroupedIntegralRangeWindows, ColumnDeltaMatchesScalar)
{
  using T               = TypeParam;
  auto const group_keys = cudf::table_view{{this->group_keys}};
  for (T const delta : {T{0}, T{2}}) {
    expect_column_delta_matches_scalar<T>(group_keys,
                                          this->ascending_no_nulls,
                                          cudf::order::ASCENDING,
                                          cudf::null_order::BEFORE,
                                          delta);
    expect_column_delta_matches_scalar<T>(group_keys,
                                          this->descending_no_nulls,
                                          cudf::order::DESCENDING,
                                          cudf::null_order::BEFORE,
                                          delta);
    expect_column_delta_matches_scalar<T>(group_keys,
                                          this->ascending_nulls_before,
                                          cudf::order::ASCENDING,
                                          cudf::null_order::BEFORE,
                                          delta);
  }
}

TEST(UngroupedColumnDeltaRangeWindows, VaryingDeltaMatchesPerRowScalar)
{
  // A genuinely per-row delta. Oracle: row i's expected window equals the window a *scalar* delta
  // of deltas[i] produces at row i, computed by running the trusted scalar path once per distinct
  // delta and gathering row i from the result whose delta matches deltas[i].
  auto const orderby = ints_column{0, 2, 4, 6, 8, 10, 12, 14, 16, 18};
  std::vector<int32_t> const deltas{0, 1, 2, 3, 4, 5, 4, 3, 2, 1};
  auto const n                = static_cast<cudf::size_type>(deltas.size());
  auto const delta_col        = ints_column(deltas.begin(), deltas.end());
  cudf::column_view const dcv = delta_col;

  auto const result         = cudf::make_range_windows(cudf::table_view{},
                                               orderby,
                                               cudf::order::ASCENDING,
                                               cudf::null_order::BEFORE,
                                               cudf::bounded_closed_column{dcv},
                                               cudf::bounded_closed_column{dcv});
  auto const preceding_host = cudf::test::to_host<cudf::size_type>(result.first->view()).first;
  auto const following_host = cudf::test::to_host<cudf::size_type>(result.second->view()).first;

  std::vector<cudf::size_type> expected_preceding(n);
  std::vector<cudf::size_type> expected_following(n);
  std::set<int32_t> const unique_deltas(deltas.begin(), deltas.end());
  for (int32_t const d : unique_deltas) {
    auto const scalar = cudf::make_fixed_width_scalar<int32_t>(d);
    auto const sc     = cudf::make_range_windows(cudf::table_view{},
                                             orderby,
                                             cudf::order::ASCENDING,
                                             cudf::null_order::BEFORE,
                                             cudf::bounded_closed{*scalar},
                                             cudf::bounded_closed{*scalar});
    auto const sp     = cudf::test::to_host<cudf::size_type>(sc.first->view()).first;
    auto const sf     = cudf::test::to_host<cudf::size_type>(sc.second->view()).first;
    for (cudf::size_type i = 0; i < n; ++i) {
      if (deltas[static_cast<std::size_t>(i)] == d) {
        expected_preceding[static_cast<std::size_t>(i)] = sp[static_cast<std::size_t>(i)];
        expected_following[static_cast<std::size_t>(i)] = sf[static_cast<std::size_t>(i)];
      }
    }
  }
  EXPECT_EQ(preceding_host, expected_preceding);
  EXPECT_EQ(following_host, expected_following);
}

TEST(UngroupedColumnDeltaRangeWindows, ValidationThrows)
{
  auto const orderby = ints_column{1, 2, 3, 4, 5};

  // Delta column size does not match the orderby size.
  {
    auto const bad              = ints_column{1, 1, 1};
    cudf::column_view const dcv = bad;
    EXPECT_THROW(static_cast<void>(cudf::make_range_windows(cudf::table_view{},
                                                            orderby,
                                                            cudf::order::ASCENDING,
                                                            cudf::null_order::BEFORE,
                                                            cudf::bounded_closed_column{dcv},
                                                            cudf::current_row{})),
                 cudf::logic_error);
  }
  // Delta column contains nulls.
  {
    auto const bad              = ints_column{{1, 1, 1, 1, 1}, cudf::test::iterators::null_at(2)};
    cudf::column_view const dcv = bad;
    EXPECT_THROW(static_cast<void>(cudf::make_range_windows(cudf::table_view{},
                                                            orderby,
                                                            cudf::order::ASCENDING,
                                                            cudf::null_order::BEFORE,
                                                            cudf::bounded_closed_column{dcv},
                                                            cudf::current_row{})),
                 cudf::logic_error);
  }
  // Delta column type differs from the orderby type.
  {
    auto const bad              = cudf::test::fixed_width_column_wrapper<int64_t>{1, 1, 1, 1, 1};
    cudf::column_view const dcv = bad;
    EXPECT_THROW(static_cast<void>(cudf::make_range_windows(cudf::table_view{},
                                                            orderby,
                                                            cudf::order::ASCENDING,
                                                            cudf::null_order::BEFORE,
                                                            cudf::bounded_closed_column{dcv},
                                                            cudf::current_row{})),
                 cudf::data_type_error);
  }
}

TEST(UngroupedColumnDeltaRangeWindows, FixedPointOrderByRejectsColumnDelta)
{
  auto const orderby =
    cudf::test::fixed_point_column_wrapper<int32_t>({1, 2, 3, 4, 5}, numeric::scale_type{0});
  auto const delta =
    cudf::test::fixed_point_column_wrapper<int32_t>({1, 1, 1, 1, 1}, numeric::scale_type{0});
  cudf::column_view const ocv = orderby;
  cudf::column_view const dcv = delta;
  EXPECT_THROW(static_cast<void>(cudf::make_range_windows(cudf::table_view{},
                                                          ocv,
                                                          cudf::order::ASCENDING,
                                                          cudf::null_order::BEFORE,
                                                          cudf::bounded_closed_column{dcv},
                                                          cudf::current_row{})),
               cudf::data_type_error);
}

TEST(MultiOrderByColumnDeltaRangeWindows, ColumnBoundsAreUnsupported)
{
  // A column-valued delta is no better defined than a scalar delta across multiple order-by keys.
  auto const orderby0         = ints_column{1, 1, 2};
  auto const orderby1         = ints_column{1, 2, 1};
  auto const delta            = ints_column{1, 1, 1};
  cudf::column_view const dcv = delta;
  std::vector<cudf::order> orders{cudf::order::ASCENDING, cudf::order::ASCENDING};
  std::vector<cudf::null_order> null_orders{cudf::null_order::BEFORE, cudf::null_order::BEFORE};

  EXPECT_THROW(
    static_cast<void>(cudf::detail::make_range_windows(cudf::table_view{},
                                                       cudf::table_view{{orderby0, orderby1}},
                                                       orders,
                                                       null_orders,
                                                       cudf::bounded_closed_column{dcv},
                                                       cudf::current_row{},
                                                       cudf::get_default_stream(),
                                                       cudf::get_current_device_resource_ref())),
    cudf::logic_error);
}

TEST(UngroupedColumnDeltaRangeWindows, TimestampColumnDeltaMatchesScalar)
{
  using Timestamp = cudf::timestamp_ms;
  using Duration  = cudf::duration_ms;
  using ts_col    = cudf::test::fixed_width_column_wrapper<Timestamp, Timestamp::rep>;
  using dur_col   = cudf::test::fixed_width_column_wrapper<Duration, Duration::rep>;

  auto const orderby = ts_col{Timestamp::rep{0},
                              Timestamp::rep{1000},
                              Timestamp::rep{2000},
                              Timestamp::rep{2000},
                              Timestamp::rep{5000},
                              Timestamp::rep{9000}};
  std::vector<Duration::rep> const deltas(6, Duration::rep{2000});
  auto const delta_wrapper          = dur_col(deltas.begin(), deltas.end());
  cudf::column_view const delta_col = delta_wrapper;
  auto const scalar                 = cudf::duration_scalar<Duration>{Duration{2000}, true};

  auto const column_result = cudf::make_range_windows(cudf::table_view{},
                                                      orderby,
                                                      cudf::order::ASCENDING,
                                                      cudf::null_order::BEFORE,
                                                      cudf::bounded_closed_column{delta_col},
                                                      cudf::bounded_closed_column{delta_col});
  auto const scalar_result = cudf::make_range_windows(cudf::table_view{},
                                                      orderby,
                                                      cudf::order::ASCENDING,
                                                      cudf::null_order::BEFORE,
                                                      cudf::bounded_closed{scalar},
                                                      cudf::bounded_closed{scalar});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(column_result.first->view(),
                                 scalar_result.first->view(),
                                 cudf::test::debug_output_level::ALL_ERRORS);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(column_result.second->view(),
                                 scalar_result.second->view(),
                                 cudf::test::debug_output_level::ALL_ERRORS);
}
