/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <tests/groupby/groupby_test_util.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/type_list_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/aggregation.hpp>

#include <initializer_list>
#include <vector>

using namespace cudf::test::iterators;

template <typename V>
struct groupby_mean_test : public cudf::test::BaseFixture {};

template <typename Target, typename Source>
std::vector<Target> convert(std::initializer_list<Source> in)
{
  std::vector<Target> out(std::cbegin(in), std::cend(in));
  return out;
}

using supported_types =
  cudf::test::Concat<cudf::test::Types<int8_t, int16_t, int32_t, int64_t, float, double>,
                     cudf::test::DurationTypes>;
TYPED_TEST_SUITE(groupby_mean_test, supported_types);
using K = int32_t;

TYPED_TEST(groupby_mean_test, basic)
{
  using V  = TypeParam;
  using R  = std::conditional_t<cudf::is_duration<V>(), V, double>;
  using RT = std::conditional_t<cudf::is_duration<R>(), int, double>;

  cudf::test::fixed_width_column_wrapper<K> keys{1, 2, 3, 1, 2, 2, 1, 3, 3, 2};
  cudf::test::fixed_width_column_wrapper<V> vals{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

  // clang-format off
  cudf::test::fixed_width_column_wrapper<K> expect_keys{1,       2,          3};
  //                                       {0, 3, 6, 1, 4, 5, 9, 2, 7, 8}
  std::vector<RT> expect_v = convert<RT>(  {3.,      19. / 4,    17. / 3});
  cudf::test::fixed_width_column_wrapper<R, RT> expect_vals(expect_v.cbegin(), expect_v.cend());
  // clang-format on

  auto agg = cudf::make_mean_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));
}

TYPED_TEST(groupby_mean_test, empty_cols)
{
  using V = TypeParam;
  using R = std::conditional_t<cudf::is_duration<V>(), V, double>;

  cudf::test::fixed_width_column_wrapper<K> keys{};
  cudf::test::fixed_width_column_wrapper<V> vals{};

  cudf::test::fixed_width_column_wrapper<K> expect_keys{};
  cudf::test::fixed_width_column_wrapper<R> expect_vals{};

  auto agg = cudf::make_mean_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));
}

TYPED_TEST(groupby_mean_test, zero_valid_keys)
{
  using V = TypeParam;
  using R = std::conditional_t<cudf::is_duration<V>(), V, double>;

  cudf::test::fixed_width_column_wrapper<K> keys({1, 2, 3}, all_nulls());
  cudf::test::fixed_width_column_wrapper<V> vals{3, 4, 5};

  cudf::test::fixed_width_column_wrapper<K> expect_keys{};
  cudf::test::fixed_width_column_wrapper<R> expect_vals{};

  auto agg = cudf::make_mean_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));
}

TYPED_TEST(groupby_mean_test, zero_valid_values)
{
  using V = TypeParam;
  using R = std::conditional_t<cudf::is_duration<V>(), V, double>;

  cudf::test::fixed_width_column_wrapper<K> keys{1, 1, 1};
  cudf::test::fixed_width_column_wrapper<V> vals({3, 4, 5}, all_nulls());

  cudf::test::fixed_width_column_wrapper<K> expect_keys{1};
  cudf::test::fixed_width_column_wrapper<R> expect_vals({0}, all_nulls());

  auto agg = cudf::make_mean_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));
}

TYPED_TEST(groupby_mean_test, null_keys_and_values)
{
  using V  = TypeParam;
  using R  = std::conditional_t<cudf::is_duration<V>(), V, double>;
  using RT = std::conditional_t<cudf::is_duration<R>(), int, double>;

  cudf::test::fixed_width_column_wrapper<K> keys(
    {1, 2, 3, 1, 2, 2, 1, 3, 3, 2, 4},
    {true, true, true, true, true, true, true, false, true, true, true});
  cudf::test::fixed_width_column_wrapper<V> vals({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 4},
                                                 {0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0});

  // clang-format off
  //                                                    {1, 1,     2, 2, 2,   3, 3,    4}
  cudf::test::fixed_width_column_wrapper<K> expect_keys({1,        2,         3,       4}, no_nulls());
  //                                                    {3, 6,     1, 4, 9,   2, 8,    -}
  std::vector<RT> expect_v = convert<RT>(               {4.5,      14. / 3,   5.,      0.});
  // clang-format on
  cudf::test::fixed_width_column_wrapper<R, RT> expect_vals(
    expect_v.cbegin(), expect_v.cend(), {1, 1, 1, 0});

  auto agg = cudf::make_mean_aggregation<cudf::groupby_aggregation>();
  test_single_agg(keys, vals, expect_keys, expect_vals, std::move(agg));
}
// clang-format on

struct groupby_dictionary_mean_test : public cudf::test::BaseFixture {};

TEST_F(groupby_dictionary_mean_test, basic)
{
  using V = int16_t;
  using R = double;

  // clang-format off
  cudf::test::fixed_width_column_wrapper<K> keys{1, 2, 3, 1, 2, 2, 1, 3, 3, 2};
  cudf::test::dictionary_column_wrapper<V>  vals{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

  cudf::test::fixed_width_column_wrapper<K> expect_keys(        {1,      2,       3});
  cudf::test::fixed_width_column_wrapper<R, double> expect_vals({9. / 3, 19. / 4, 17. / 3});
  // clang-format on

  test_single_agg(
    keys, vals, expect_keys, expect_vals, cudf::make_mean_aggregation<cudf::groupby_aggregation>());
}

template <typename T>
struct FixedPointTestBothReps : public cudf::test::BaseFixture {};

TYPED_TEST_SUITE(FixedPointTestBothReps, cudf::test::FixedPointTypes);

TYPED_TEST(FixedPointTestBothReps, GroupBySortMeanDecimalAsValue)
{
  using namespace numeric;
  using decimalXX  = TypeParam;
  using RepType    = cudf::device_storage_type_t<decimalXX>;
  using fp_wrapper = cudf::test::fixed_point_column_wrapper<RepType>;

  for (auto const i : {2, 1, 0, -1, -2}) {
    auto const scale = scale_type{i};
    // clang-format off
    auto const keys  = cudf::test::fixed_width_column_wrapper<K>{1, 2, 3, 1, 2, 2, 1, 3, 3, 2};
    auto const vals  = fp_wrapper{                              {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, scale};
    // clang-format on

    auto const expect_keys     = cudf::test::fixed_width_column_wrapper<K>{1, 2, 3};
    auto const expect_vals_min = fp_wrapper{{3, 4, 5}, scale};

    auto agg = cudf::make_mean_aggregation<cudf::groupby_aggregation>();
    test_single_agg(
      keys, vals, expect_keys, expect_vals_min, std::move(agg), force_use_sort_impl::YES);
  }
}

TYPED_TEST(FixedPointTestBothReps, GroupByHashMeanDecimalAsValue)
{
  using namespace numeric;
  using decimalXX  = TypeParam;
  using RepType    = cudf::device_storage_type_t<decimalXX>;
  using fp_wrapper = cudf::test::fixed_point_column_wrapper<RepType>;
  using K          = int32_t;

  for (auto const i : {2, 1, 0, -1, -2}) {
    auto const scale = scale_type{i};
    // clang-format off
    auto const keys  = cudf::test::fixed_width_column_wrapper<K>{1, 2, 3, 1, 2, 2, 1, 3, 3, 2};
    auto const vals  = fp_wrapper{                              {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, scale};
    // clang-format on

    auto const expect_keys     = cudf::test::fixed_width_column_wrapper<K>{1, 2, 3};
    auto const expect_vals_min = fp_wrapper{{3, 4, 5}, scale};

    auto agg = cudf::make_mean_aggregation<cudf::groupby_aggregation>();
    test_single_agg(keys, vals, expect_keys, expect_vals_min, std::move(agg));
  }
}
