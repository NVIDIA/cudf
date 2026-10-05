/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <tests/groupby/groupby_test_util.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/testing_main.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/aggregation.hpp>
#include <cudf/groupby.hpp>
#include <cudf/scalar/scalar_factories.hpp>

using K = int32_t;  // Key type.

template <typename V>
struct groupby_stream_test : public cudf::test::BaseFixture {
  cudf::test::fixed_width_column_wrapper<K> keys{1, 2, 3, 1, 2, 2, 1, 3, 3, 2};
  cudf::test::fixed_width_column_wrapper<V> vals{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

  void test_groupby(std::unique_ptr<cudf::groupby_aggregation>&& agg,
                    force_use_sort_impl use_sort        = force_use_sort_impl::NO,
                    cudf::null_policy include_null_keys = cudf::null_policy::INCLUDE,
                    cudf::sorted keys_are_sorted        = cudf::sorted::NO)
  {
    auto requests = [&] {
      auto requests = std::vector<cudf::groupby::aggregation_request>{};
      requests.push_back(cudf::groupby::aggregation_request{});
      requests.front().values = vals;
      if (use_sort == force_use_sort_impl::YES) {
        requests.front().aggregations.push_back(
          cudf::make_nth_element_aggregation<cudf::groupby_aggregation>(0));
      }
      requests.front().aggregations.push_back(std::move(agg));
      return requests;
    }();

    auto gby =
      cudf::groupby::groupby{cudf::table_view{{keys}}, include_null_keys, keys_are_sorted, {}, {}};
    gby.aggregate(requests, cudf::test::get_default_stream());
    // No need to verify results, for stream test.
  }
};

TYPED_TEST_SUITE(groupby_stream_test, cudf::test::AllTypes);

TYPED_TEST(groupby_stream_test, test_count)
{
  auto const make_count_agg = [&](cudf::null_policy include_nulls = cudf::null_policy::EXCLUDE) {
    return cudf::make_count_aggregation<cudf::groupby_aggregation>(include_nulls);
  };

  this->test_groupby(make_count_agg());
  this->test_groupby(make_count_agg(), force_use_sort_impl::YES);
  this->test_groupby(make_count_agg(cudf::null_policy::INCLUDE));
}

struct GroupbyTest : public cudf::test::BaseFixture {};

TEST_F(GroupbyTest, Scan)
{
  using key_wrapper   = cudf::test::fixed_width_column_wrapper<int32_t>;
  using value_wrapper = cudf::test::fixed_width_column_wrapper<int32_t>;

  key_wrapper keys{1, 2, 3, 1, 2, 2, 1, 3, 3, 2};
  value_wrapper vals({5, 6, 7, 8, 9, 0, 1, 2, 3, 4});

  auto agg = cudf::make_min_aggregation<cudf::groupby_scan_aggregation>();
  std::vector<cudf::groupby::scan_request> requests;
  requests.emplace_back();
  requests[0].values = vals;
  requests[0].aggregations.push_back(std::move(agg));

  cudf::groupby::groupby gb_obj(cudf::table_view({keys}));
  // cudf::groupby scan uses sort implementation
  auto result = gb_obj.scan(requests, cudf::test::get_default_stream());
}

TEST_F(GroupbyTest, Shift)
{
  cudf::test::fixed_width_column_wrapper<int32_t> key{1, 2, 1, 2, 2, 1, 1};
  cudf::test::fixed_width_column_wrapper<int32_t> val{3, 4, 5, 6, 7, 8, 9};
  cudf::size_type offset = 2;
  auto slr               = cudf::make_default_constructed_scalar(cudf::column_view(val).type(),
                                                                 cudf::test::get_default_stream());

  cudf::groupby::groupby gb_obj(cudf::table_view({key}));
  std::vector<cudf::size_type> offsets{offset};
  auto got =
    gb_obj.shift(cudf::table_view{{val}}, offsets, {*slr}, cudf::test::get_default_stream());
}

TEST_F(GroupbyTest, GetGroups)
{
  cudf::test::fixed_width_column_wrapper<int32_t> keys{1, 1, 2, 1, 2, 3};
  cudf::test::fixed_width_column_wrapper<int32_t> values({0, 0, 1, 1, 2, 2});
  cudf::groupby::groupby gb(cudf::table_view({keys}));
  auto gb_groups = gb.get_groups(cudf::table_view({values}), cudf::test::get_default_stream());
}

TEST_F(GroupbyTest, ReplaceNullsTest)
{
  cudf::test::fixed_width_column_wrapper<int32_t> key{0, 1, 0, 1, 0, 1};
  cudf::test::fixed_width_column_wrapper<int32_t> val({42, 7, 24, 10, 1, 1000}, {1, 1, 1, 0, 0, 0});
  cudf::groupby::groupby gb_obj(cudf::table_view({key}));
  std::vector<cudf::replace_policy> policies{cudf::replace_policy::PRECEDING};
  auto p =
    gb_obj.replace_nulls(cudf::table_view({val}), policies, cudf::test::get_default_stream());
}

CUDF_TEST_PROGRAM_MAIN()
