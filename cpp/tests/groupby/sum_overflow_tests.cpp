/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <tests/groupby/groupby_test_util.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/aggregation.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/groupby.hpp>
#include <cudf/sorting.hpp>
#include <cudf/structs/structs_column_view.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <cuda/iterator>

#include <algorithm>
#include <vector>

using namespace cudf::test::iterators;
using namespace numeric;

// SUM_OVERFLOW tests - supports signed integer and decimal types
template <typename V>
struct groupby_sum_overflow_test : public cudf::test::BaseFixture {};

using sum_overflow_supported_types =
  cudf::test::Concat<cudf::test::Types<int8_t, int16_t, int32_t, int64_t>,
                     cudf::test::FixedPointTypes>;

TYPED_TEST_SUITE(groupby_sum_overflow_test, sum_overflow_supported_types);

TYPED_TEST(groupby_sum_overflow_test, Basic)
{
  using K = int32_t;
  using V = TypeParam;

  cudf::test::fixed_width_column_wrapper<K> keys{1, 2, 3, 1, 2, 2, 1, 3, 3, 2};
  cudf::test::fixed_width_column_wrapper<K> expect_keys{1, 2, 3};

  if constexpr (cudf::is_fixed_point<V>()) {
    // For decimal types
    using RepType    = cudf::device_storage_type_t<V>;
    auto const scale = scale_type{0};
    auto vals =
      cudf::test::fixed_point_column_wrapper<RepType>{{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, scale};

    // Create expected struct column with sum and overflow children
    auto sum_col      = cudf::test::fixed_point_column_wrapper<RepType>{{9, 19, 17}, scale};
    auto overflow_col = cudf::test::fixed_width_column_wrapper<bool>{false, false, false};
    std::vector<std::unique_ptr<cudf::column>> children;
    children.push_back(sum_col.release());
    children.push_back(overflow_col.release());
    auto expect_vals = cudf::create_structs_hierarchy(3, std::move(children), 0, {});

    auto agg = cudf::make_sum_overflow_aggregation<cudf::groupby_aggregation>();
    test_single_agg(keys, vals, expect_keys, *expect_vals, std::move(agg));

    auto agg_sort = cudf::make_sum_overflow_aggregation<cudf::groupby_aggregation>();
    test_single_agg(
      keys, vals, expect_keys, *expect_vals, std::move(agg_sort), force_use_sort_impl::YES);
  } else {
    // For integer types
    cudf::test::fixed_width_column_wrapper<V> vals{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

    // Create expected struct column with sum and overflow children
    // Sum column type matches input type V
    auto sum_col      = cudf::test::fixed_width_column_wrapper<V>{9, 19, 17};
    auto overflow_col = cudf::test::fixed_width_column_wrapper<bool>{false, false, false};
    std::vector<std::unique_ptr<cudf::column>> children;
    children.push_back(sum_col.release());
    children.push_back(overflow_col.release());
    auto expect_vals = cudf::create_structs_hierarchy(3, std::move(children), 0, {});

    auto agg = cudf::make_sum_overflow_aggregation<cudf::groupby_aggregation>();
    test_single_agg(keys, vals, expect_keys, *expect_vals, std::move(agg));

    auto agg_sort = cudf::make_sum_overflow_aggregation<cudf::groupby_aggregation>();
    test_single_agg(
      keys, vals, expect_keys, *expect_vals, std::move(agg_sort), force_use_sort_impl::YES);
  }
}

TYPED_TEST(groupby_sum_overflow_test, SortPathWithTdigest)
{
  using K = int32_t;
  using V = TypeParam;

  cudf::test::fixed_width_column_wrapper<K> keys{1, 2, 3, 1, 2, 2, 1, 3, 3, 2};
  cudf::test::fixed_width_column_wrapper<K> expect_keys{1, 2, 3};

  // Co-request TDIGEST (a sort-only aggregation) so the whole groupby takes the sort-based path,
  // then verify the SUM_OVERFLOW struct matches the hash result and TDIGEST also runs.
  auto run_and_check = [&](cudf::column_view const& vals, cudf::column_view const& expect_vals) {
    std::vector<cudf::groupby::aggregation_request> requests;
    requests.emplace_back();
    requests[0].values = vals;
    requests[0].aggregations.push_back(
      cudf::make_sum_overflow_aggregation<cudf::groupby_aggregation>());
    requests[0].aggregations.push_back(
      cudf::make_tdigest_aggregation<cudf::groupby_aggregation>(1000));

    auto result = cudf::groupby::groupby(cudf::table_view{{keys}}).aggregate(requests);

    // Sort-based groupby returns keys in sorted order, aligning with expect_keys/expect_vals.
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.first->get_column(0).view(), expect_keys);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.second[0].results[0]->view(), expect_vals);
    // TDIGEST produces one tdigest per group.
    EXPECT_EQ(result.second[0].results[1]->size(), 3);
  };

  if constexpr (cudf::is_fixed_point<V>()) {
    using RepType    = cudf::device_storage_type_t<V>;
    auto const scale = scale_type{0};
    auto vals =
      cudf::test::fixed_point_column_wrapper<RepType>{{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, scale};
    auto sum_col      = cudf::test::fixed_point_column_wrapper<RepType>{{9, 19, 17}, scale};
    auto overflow_col = cudf::test::fixed_width_column_wrapper<bool>{false, false, false};
    std::vector<std::unique_ptr<cudf::column>> children;
    children.push_back(sum_col.release());
    children.push_back(overflow_col.release());
    auto expect_vals = cudf::create_structs_hierarchy(3, std::move(children), 0, {});
    run_and_check(vals, *expect_vals);
  } else {
    cudf::test::fixed_width_column_wrapper<V> vals{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    auto sum_col      = cudf::test::fixed_width_column_wrapper<V>{9, 19, 17};
    auto overflow_col = cudf::test::fixed_width_column_wrapper<bool>{false, false, false};
    std::vector<std::unique_ptr<cudf::column>> children;
    children.push_back(sum_col.release());
    children.push_back(overflow_col.release());
    auto expect_vals = cudf::create_structs_hierarchy(3, std::move(children), 0, {});
    run_and_check(vals, *expect_vals);
  }
}

TYPED_TEST(groupby_sum_overflow_test, EmptyCols)
{
  using K = int32_t;
  using V = TypeParam;

  cudf::test::fixed_width_column_wrapper<K> keys{};
  cudf::test::fixed_width_column_wrapper<K> expect_keys{};

  // For integer types
  cudf::test::fixed_width_column_wrapper<V> vals{};

  // Create expected empty struct column with sum and overflow children
  auto sum_col      = cudf::test::fixed_width_column_wrapper<V>{};
  auto overflow_col = cudf::test::fixed_width_column_wrapper<bool>{};
  std::vector<std::unique_ptr<cudf::column>> children;
  children.push_back(sum_col.release());
  children.push_back(overflow_col.release());
  auto expect_vals = cudf::create_structs_hierarchy(0, std::move(children), 0, {});

  auto agg = cudf::make_sum_overflow_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, *expect_vals, std::move(agg));
}

TYPED_TEST(groupby_sum_overflow_test, ZeroValidKeys)
{
  using K = int32_t;
  using V = TypeParam;

  cudf::test::fixed_width_column_wrapper<K> keys({1, 2, 3}, cudf::test::iterators::all_nulls());
  cudf::test::fixed_width_column_wrapper<K> expect_keys{};

  // For integer types
  cudf::test::fixed_width_column_wrapper<V> vals{3, 4, 5};

  // Create expected empty struct column with sum and overflow children
  auto sum_col      = cudf::test::fixed_width_column_wrapper<V>{};
  auto overflow_col = cudf::test::fixed_width_column_wrapper<bool>{};
  std::vector<std::unique_ptr<cudf::column>> children;
  children.push_back(sum_col.release());
  children.push_back(overflow_col.release());
  auto expect_vals = cudf::create_structs_hierarchy(0, std::move(children), 0, {});

  auto agg = cudf::make_sum_overflow_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, *expect_vals, std::move(agg));
}

TYPED_TEST(groupby_sum_overflow_test, ZeroValidValues)
{
  using K = int32_t;
  using V = TypeParam;

  cudf::test::fixed_width_column_wrapper<K> keys{1, 1, 1};
  cudf::test::fixed_width_column_wrapper<V> vals({3, 4, 5}, cudf::test::iterators::all_nulls());

  cudf::test::fixed_width_column_wrapper<K> expect_keys{1};

  // Create expected struct column with sum and overflow children (null result)
  // Child columns have no null masks, only struct-level null mask matters
  auto sum_col      = cudf::test::fixed_width_column_wrapper<V>({0});
  auto overflow_col = cudf::test::fixed_width_column_wrapper<bool>({false});
  std::vector<std::unique_ptr<cudf::column>> children;
  children.push_back(sum_col.release());
  children.push_back(overflow_col.release());
  std::vector<int> validity{0};  // null struct
  auto [validity_mask, null_count] =
    cudf::test::detail::make_null_mask(validity.begin(), validity.end());
  auto expect_vals =
    cudf::create_structs_hierarchy(1, std::move(children), null_count, std::move(validity_mask));

  auto agg = cudf::make_sum_overflow_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, *expect_vals, std::move(agg));

  // Exercise the sort-based path for an all-null group.
  auto agg_sort = cudf::make_sum_overflow_aggregation<cudf::groupby_aggregation>();
  test_single_agg(
    keys, vals, expect_keys, *expect_vals, std::move(agg_sort), force_use_sort_impl::YES);
}

TYPED_TEST(groupby_sum_overflow_test, NullKeysAndValues)
{
  using K = int32_t;
  using V = TypeParam;

  cudf::test::fixed_width_column_wrapper<K> keys(
    {1, 2, 3, 1, 2, 2, 1, 3, 3, 2, 4},
    {true, true, true, true, true, true, true, false, true, true, true});
  cudf::test::fixed_width_column_wrapper<V> vals({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 4},
                                                 {0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0});

  //  { 1, 1,     2, 2, 2,   3, 3,    4}
  cudf::test::fixed_width_column_wrapper<K> expect_keys({1, 2, 3, 4},
                                                        cudf::test::iterators::no_nulls());

  // Create expected struct column with sum and overflow children
  //  { 3, 6,     1, 4, 9,   2, 8,    -}
  // Child columns have no null masks, only struct-level null mask matters
  auto sum_col      = cudf::test::fixed_width_column_wrapper<V>({9, 14, 10, 0});
  auto overflow_col = cudf::test::fixed_width_column_wrapper<bool>({false, false, false, false});
  std::vector<std::unique_ptr<cudf::column>> children;
  children.push_back(sum_col.release());
  children.push_back(overflow_col.release());
  std::vector<int> validity{1, 1, 1, 0};
  auto [validity_mask, null_count] =
    cudf::test::detail::make_null_mask(validity.begin(), validity.end());
  auto expect_vals =
    cudf::create_structs_hierarchy(4, std::move(children), null_count, std::move(validity_mask));

  auto agg = cudf::make_sum_overflow_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, *expect_vals, std::move(agg));

  // Exercise the sort-based path with null keys and null values.
  auto agg_sort = cudf::make_sum_overflow_aggregation<cudf::groupby_aggregation>();
  test_single_agg(
    keys, vals, expect_keys, *expect_vals, std::move(agg_sort), force_use_sort_impl::YES);
}

TYPED_TEST(groupby_sum_overflow_test, OverflowDetection)
{
  using K = int32_t;
  using V = TypeParam;

  auto check_overflow_flags = [](cudf::column_view const& keys,
                                 cudf::column_view const& vals,
                                 cudf::column_view const& expect_keys,
                                 cudf::column_view const& expect_overflow) {
    std::vector<cudf::groupby::aggregation_request> requests;
    requests.emplace_back();
    requests[0].values = vals;
    requests[0].aggregations.push_back(
      cudf::make_sum_overflow_aggregation<cudf::groupby_aggregation>());

    auto result = cudf::groupby::groupby(cudf::table_view{{keys}}).aggregate(requests);
    auto const overflow_child =
      cudf::structs_column_view{result.second[0].results[0]->view()}.get_sliced_child(1);

    auto sorted = cudf::sort_by_key(
      cudf::table_view{{result.first->get_column(0).view(), overflow_child}}, result.first->view());
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted->view().column(0), expect_keys);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted->view().column(1), expect_overflow);
  };

  // Same check, but a co-requested sort-only aggregation (TDIGEST) forces the sort path.
  auto check_overflow_flags_sort = [](cudf::column_view const& keys,
                                      cudf::column_view const& vals,
                                      cudf::column_view const& expect_keys,
                                      cudf::column_view const& expect_overflow) {
    std::vector<cudf::groupby::aggregation_request> requests;
    requests.emplace_back();
    requests[0].values = vals;
    requests[0].aggregations.push_back(
      cudf::make_sum_overflow_aggregation<cudf::groupby_aggregation>());
    requests[0].aggregations.push_back(
      cudf::make_tdigest_aggregation<cudf::groupby_aggregation>(1000));

    auto result = cudf::groupby::groupby(cudf::table_view{{keys}}).aggregate(requests);
    auto const overflow_child =
      cudf::structs_column_view{result.second[0].results[0]->view()}.get_sliced_child(1);

    auto sorted = cudf::sort_by_key(
      cudf::table_view{{result.first->get_column(0).view(), overflow_child}}, result.first->view());
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted->view().column(0), expect_keys);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted->view().column(1), expect_overflow);
  };

  cudf::test::fixed_width_column_wrapper<K> keys{1, 2, 3, 4, 1, 2, 2, 1, 3, 3, 2, 4, 4};
  cudf::test::fixed_width_column_wrapper<K> expect_keys{1, 2, 3, 4};
  cudf::test::fixed_width_column_wrapper<bool> expect_overflow{true, false, true, true};

  if constexpr (cudf::is_fixed_point<V>()) {
    using RepType        = cudf::device_storage_type_t<V>;
    auto constexpr scale = scale_type{0};

    auto constexpr type_max       = cuda::std::numeric_limits<RepType>::max();
    auto constexpr type_min       = cuda::std::numeric_limits<RepType>::min();
    RepType const large_positive  = type_max - 5;
    RepType const small_increment = 10;
    RepType const large_negative  = type_min + 5;
    RepType const small_decrement = -10;
    RepType const small_val1      = 10;
    RepType const small_val2      = 20;
    RepType const small_val3      = 30;
    RepType const small_val4      = 40;

    cudf::test::fixed_point_column_wrapper<RepType> vals{
      {large_positive,   // Group 1
       small_val1,       // Group 2
       small_val2,       // Group 3
       large_negative,   // Group 4
       small_increment,  // Group 1: positive overflow
       small_val2,       // Group 2
       small_val3,       // Group 2
       large_positive,   // Group 1
       large_positive,   // Group 3
       1,                // Group 3
       small_val4,       // Group 2
       small_decrement,  // Group 4: negative overflow
       large_negative},  // Group 4
      scale};

    check_overflow_flags(keys, vals, expect_keys, expect_overflow);

    // Adding TDIGEST forces sort-based groupby; the overflow flags must match the hash path.
    check_overflow_flags_sort(keys, vals, expect_keys, expect_overflow);
  } else {
    using DeviceType = cudf::device_storage_type_t<V>;

    auto constexpr type_max          = cuda::std::numeric_limits<DeviceType>::max();
    auto constexpr type_min          = cuda::std::numeric_limits<DeviceType>::min();
    DeviceType const large_positive  = type_max - 5;
    DeviceType const small_increment = 10;
    DeviceType const large_negative  = type_min + 5;
    DeviceType const small_decrement = -10;
    DeviceType const small_val1      = 10;
    DeviceType const small_val2      = 20;
    DeviceType const small_val3      = 30;
    DeviceType const small_val4      = 40;

    cudf::test::fixed_width_column_wrapper<V> vals{static_cast<V>(large_positive),
                                                   static_cast<V>(small_val1),
                                                   static_cast<V>(small_val2),
                                                   static_cast<V>(large_negative),
                                                   static_cast<V>(small_increment),
                                                   static_cast<V>(small_val2),
                                                   static_cast<V>(small_val3),
                                                   static_cast<V>(large_positive),
                                                   static_cast<V>(large_positive),
                                                   static_cast<V>(1),
                                                   static_cast<V>(small_val4),
                                                   static_cast<V>(small_decrement),
                                                   static_cast<V>(large_negative)};

    check_overflow_flags(keys, vals, expect_keys, expect_overflow);
    check_overflow_flags_sort(keys, vals, expect_keys, expect_overflow);
  }
}

TYPED_TEST(groupby_sum_overflow_test, SlicedInput)
{
  using K = int32_t;
  using V = TypeParam;

  // Prepend one row to each column and slice it off, so the groupby sees a non-zero offset. The
  // sliced view matches the `basic` test's data, so the expected result is the same.
  cudf::test::fixed_width_column_wrapper<K> keys_full{0, 1, 2, 3, 1, 2, 2, 1, 3, 3, 2};
  cudf::test::fixed_width_column_wrapper<K> expect_keys{1, 2, 3};

  auto overflow_col = cudf::test::fixed_width_column_wrapper<bool>{false, false, false};

  if constexpr (cudf::is_fixed_point<V>()) {
    using RepType    = cudf::device_storage_type_t<V>;
    auto const scale = scale_type{0};
    auto vals_full =
      cudf::test::fixed_point_column_wrapper<RepType>{{99, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, scale};
    auto sum_col = cudf::test::fixed_point_column_wrapper<RepType>{{9, 19, 17}, scale};
    std::vector<std::unique_ptr<cudf::column>> children;
    children.push_back(sum_col.release());
    children.push_back(overflow_col.release());
    auto expect_vals = cudf::create_structs_hierarchy(3, std::move(children), 0, {});

    auto const keys = cudf::slice(keys_full, {1, 11}).front();
    auto const vals = cudf::slice(vals_full, {1, 11}).front();
    test_single_agg(keys,
                    vals,
                    expect_keys,
                    *expect_vals,
                    cudf::make_sum_overflow_aggregation<cudf::groupby_aggregation>());
    test_single_agg(keys,
                    vals,
                    expect_keys,
                    *expect_vals,
                    cudf::make_sum_overflow_aggregation<cudf::groupby_aggregation>(),
                    force_use_sort_impl::YES);
  } else {
    cudf::test::fixed_width_column_wrapper<V> vals_full{99, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    auto sum_col = cudf::test::fixed_width_column_wrapper<V>{9, 19, 17};
    std::vector<std::unique_ptr<cudf::column>> children;
    children.push_back(sum_col.release());
    children.push_back(overflow_col.release());
    auto expect_vals = cudf::create_structs_hierarchy(3, std::move(children), 0, {});

    auto const keys = cudf::slice(keys_full, {1, 11}).front();
    auto const vals = cudf::slice(vals_full, {1, 11}).front();
    test_single_agg(keys,
                    vals,
                    expect_keys,
                    *expect_vals,
                    cudf::make_sum_overflow_aggregation<cudf::groupby_aggregation>());
    test_single_agg(keys,
                    vals,
                    expect_keys,
                    *expect_vals,
                    cudf::make_sum_overflow_aggregation<cudf::groupby_aggregation>(),
                    force_use_sort_impl::YES);
  }
}

TYPED_TEST(groupby_sum_overflow_test, MultiBlock)
{
  using K = int32_t;
  using V = TypeParam;

  constexpr cudf::size_type num_groups     = 5000;
  constexpr cudf::size_type rows_per_group = 10;
  constexpr cudf::size_type num_rows       = num_groups * rows_per_group;

  std::vector<K> h_keys(num_rows);
  std::generate(h_keys.begin(), h_keys.end(), [n = cudf::size_type{0}]() mutable {
    return static_cast<K>(n++ % num_groups);
  });
  cudf::test::fixed_width_column_wrapper<K> keys(h_keys.begin(), h_keys.end());

  auto const expect_key_it = cuda::counting_iterator<K>{0};
  cudf::test::fixed_width_column_wrapper<K> expect_keys(expect_key_it, expect_key_it + num_groups);

  auto const overflow_it = cuda::make_constant_iterator(false);
  auto overflow_col =
    cudf::test::fixed_width_column_wrapper<bool>(overflow_it, overflow_it + num_groups);

  std::vector<std::unique_ptr<cudf::column>> children;
  if constexpr (cudf::is_fixed_point<V>()) {
    using RepType     = cudf::device_storage_type_t<V>;
    auto const scale  = scale_type{0};
    auto const val_it = cuda::make_constant_iterator(RepType{1});
    auto vals = cudf::test::fixed_point_column_wrapper<RepType>(val_it, val_it + num_rows, scale);
    auto const sum_it = cuda::make_constant_iterator(static_cast<RepType>(rows_per_group));
    auto sum_col =
      cudf::test::fixed_point_column_wrapper<RepType>(sum_it, sum_it + num_groups, scale);
    children.push_back(sum_col.release());
    children.push_back(overflow_col.release());
    auto expect_vals = cudf::create_structs_hierarchy(num_groups, std::move(children), 0, {});
    test_single_agg(keys,
                    vals,
                    expect_keys,
                    *expect_vals,
                    cudf::make_sum_overflow_aggregation<cudf::groupby_aggregation>());
  } else {
    auto const val_it = cuda::make_constant_iterator(V{1});
    auto vals         = cudf::test::fixed_width_column_wrapper<V>(val_it, val_it + num_rows);
    auto const sum_it = cuda::make_constant_iterator(static_cast<V>(rows_per_group));
    auto sum_col      = cudf::test::fixed_width_column_wrapper<V>(sum_it, sum_it + num_groups);
    children.push_back(sum_col.release());
    children.push_back(overflow_col.release());
    auto expect_vals = cudf::create_structs_hierarchy(num_groups, std::move(children), 0, {});
    test_single_agg(keys,
                    vals,
                    expect_keys,
                    *expect_vals,
                    cudf::make_sum_overflow_aggregation<cudf::groupby_aggregation>());
  }
}

// Test that SUM_OVERFLOW throws an error for bool type (which is not supported)
TEST(groupby_sum_overflow_error_test, BoolTypeNotSupported)
{
  using K = int32_t;
  using V = bool;  // bool type should not be supported by SUM_OVERFLOW

  cudf::test::fixed_width_column_wrapper<K> keys{1, 1, 1};
  cudf::test::fixed_width_column_wrapper<V> vals{true, false, true};

  auto agg = cudf::make_sum_overflow_aggregation<cudf::groupby_aggregation>();

  std::vector<cudf::groupby::aggregation_request> requests;
  requests.emplace_back();
  requests[0].values = vals;
  requests[0].aggregations.push_back(std::move(agg));

  auto gb_obj = cudf::groupby::groupby(cudf::table_view({keys}));

  // This should throw an exception since bool is not supported
  EXPECT_THROW(gb_obj.aggregate(requests), cudf::logic_error);
}

// Test that SUM_OVERFLOW throws an error for unsigned integer types (which are not supported)
TEST(groupby_sum_overflow_error_test, UnsignedTypesNotSupported)
{
  using K = int32_t;

  // Test uint32_t
  {
    using V = uint32_t;  // unsigned types should not be supported by SUM_OVERFLOW

    cudf::test::fixed_width_column_wrapper<K> keys{1, 1, 1};
    cudf::test::fixed_width_column_wrapper<V> vals{1, 2, 3};

    auto agg = cudf::make_sum_overflow_aggregation<cudf::groupby_aggregation>();

    std::vector<cudf::groupby::aggregation_request> requests;
    requests.emplace_back();
    requests[0].values = vals;
    requests[0].aggregations.push_back(std::move(agg));

    auto gb_obj = cudf::groupby::groupby(cudf::table_view({keys}));

    // This should throw an exception since unsigned types are not supported
    EXPECT_THROW(gb_obj.aggregate(requests), cudf::logic_error);
  }

  // Test uint64_t
  {
    using V = uint64_t;  // unsigned types should not be supported by SUM_OVERFLOW

    cudf::test::fixed_width_column_wrapper<K> keys{1, 1, 1};
    cudf::test::fixed_width_column_wrapper<V> vals{1, 2, 3};

    auto agg = cudf::make_sum_overflow_aggregation<cudf::groupby_aggregation>();

    std::vector<cudf::groupby::aggregation_request> requests;
    requests.emplace_back();
    requests[0].values = vals;
    requests[0].aggregations.push_back(std::move(agg));

    auto gb_obj = cudf::groupby::groupby(cudf::table_view({keys}));

    // This should throw an exception since unsigned types are not supported
    EXPECT_THROW(gb_obj.aggregate(requests), cudf::logic_error);
  }
}
