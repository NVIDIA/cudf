/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/tdigest_utilities.hpp>
#include <cudf_test/type_list_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/detail/tdigest/tdigest.hpp>
#include <cudf/groupby.hpp>
#include <cudf/quantiles.hpp>
#include <cudf/reduction.hpp>
#include <cudf/tdigest/tdigest_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cuda/iterator>

namespace {
struct percentile_approx_dispatch {
  template <typename T, typename Func>
  std::unique_ptr<cudf::column> operator()(Func op,
                                           cudf::column_view const& values,
                                           int delta,
                                           std::vector<double> const& percentages,
                                           [[maybe_unused]] cudf::size_type ulps)
    requires(cudf::is_numeric<T>() || cudf::is_fixed_point<T>())
  {
    // gpu implementation.
    auto agg_result = op(values, delta);

    cudf::test::fixed_width_column_wrapper<double> g_percentages(percentages.begin(),
                                                                 percentages.end());
    cudf::tdigest::tdigest_column_view tdv(*agg_result);
    auto result = cudf::percentile_approx(tdv, g_percentages);

    return result;
  }

  template <typename T, typename Func>
  std::unique_ptr<cudf::column> operator()(Func op,
                                           cudf::column_view const& values,
                                           int delta,
                                           std::vector<double> const& percentages,
                                           cudf::size_type ulps)
    requires(!cudf::is_numeric<T>() && !cudf::is_fixed_point<T>())
  {
    CUDF_FAIL("Invalid input type for percentile_approx test");
  }
};

void percentile_approx_test(cudf::column_view const& _keys,
                            cudf::column_view const& _values,
                            int delta,
                            std::vector<double> const& percentages,
                            cudf::size_type ulps)
{
  auto stream                                 = cudf::get_default_stream();
  bool is_cpu_cluster_computation_disabled[2] = {true, false};
  for (int idx = 0; idx < 2; idx++) {
    cudf::tdigest::detail::is_cpu_cluster_computation_disabled =
      is_cpu_cluster_computation_disabled[idx];

    // first pass:  validate the actual percentages we get per group.

    // produce the groups.
    cudf::table_view k({_keys});
    cudf::groupby::groupby pass1_gb(k);
    cudf::table_view v({_values});
    auto groups = pass1_gb.get_groups(v, stream);
    // slice it all up so we have keys/columns for everything.
    std::vector<cudf::column_view> keys;
    std::vector<cudf::column_view> values;
    for (size_t g_idx = 0; g_idx < groups.offsets.size() - 1; g_idx++) {
      auto k = cudf::slice(
        groups.keys->get_column(0), {groups.offsets[g_idx], groups.offsets[g_idx + 1]}, stream);
      keys.push_back(k[0]);

      auto v = cudf::slice(
        groups.values->get_column(0), {groups.offsets[g_idx], groups.offsets[g_idx + 1]}, stream);
      values.push_back(v[0]);
    }

    std::vector<std::unique_ptr<cudf::column>> groupby_parts;
    std::vector<std::unique_ptr<cudf::column>> reduce_parts;
    for (size_t v_idx = 0; v_idx < values.size(); v_idx++) {
      // via groupby
      auto groupby = [&](cudf::column_view const& values, int delta) {
        cudf::table_view t({keys[v_idx]});
        cudf::groupby::groupby gb(t);
        std::vector<cudf::groupby::aggregation_request> requests;
        std::vector<std::unique_ptr<cudf::groupby_aggregation>> aggregations;
        aggregations.push_back(cudf::make_tdigest_aggregation<cudf::groupby_aggregation>(delta));
        requests.push_back({values, std::move(aggregations)});
        auto result = std::move(gb.aggregate(requests, stream).second[0].results[0]);
        stream.synchronize();
        return result;
      };
      groupby_parts.push_back(cudf::type_dispatcher(values[v_idx].type(),
                                                    percentile_approx_dispatch{},
                                                    groupby,
                                                    values[v_idx],
                                                    delta,
                                                    percentages,
                                                    ulps));

      // via reduce
      auto reduce = [stream](cudf::column_view const& values, int delta) {
        // result is a scalar, but we want to extract out the underlying column
        auto scalar_result =
          cudf::reduce(values,
                       *cudf::make_tdigest_aggregation<cudf::reduce_aggregation>(delta),
                       cudf::data_type{cudf::type_id::STRUCT},
                       stream);
        auto tbl = static_cast<cudf::struct_scalar const*>(scalar_result.get())->view();
        stream.synchronize();
        std::vector<std::unique_ptr<cudf::column>> cols;
        std::transform(
          tbl.begin(), tbl.end(), std::back_inserter(cols), [](cudf::column_view const& col) {
            return std::make_unique<cudf::column>(col);
          });
        return cudf::make_structs_column(tbl.num_rows(), std::move(cols), 0, rmm::device_buffer());
      };
      // groupby path
      reduce_parts.push_back(cudf::type_dispatcher(values[v_idx].type(),
                                                   percentile_approx_dispatch{},
                                                   reduce,
                                                   values[v_idx],
                                                   delta,
                                                   percentages,
                                                   ulps));
      stream.synchronize();
    }

    // second pass. run the percentile_approx with all the keys in one pass and make sure we get the
    // same results as the concatenated by-key results.
    std::vector<cudf::column_view> part_views;
    std::transform(groupby_parts.begin(),
                   groupby_parts.end(),
                   std::back_inserter(part_views),
                   [](std::unique_ptr<cudf::column> const& c) { return c->view(); });
    auto expected = cudf::concatenate(part_views);

    std::vector<cudf::column_view> reduce_part_views;
    std::transform(reduce_parts.begin(),
                   reduce_parts.end(),
                   std::back_inserter(reduce_part_views),
                   [](std::unique_ptr<cudf::column> const& c) { return c->view(); });
    auto reduce_expected = cudf::concatenate(reduce_part_views);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*expected, *reduce_expected);

    cudf::groupby::groupby gb(k);
    std::vector<cudf::groupby::aggregation_request> requests;
    std::vector<std::unique_ptr<cudf::groupby_aggregation>> aggregations;
    aggregations.push_back(cudf::make_tdigest_aggregation<cudf::groupby_aggregation>(delta));
    requests.push_back({_values, std::move(aggregations)});
    auto gb_result = gb.aggregate(requests, stream);

    cudf::test::fixed_width_column_wrapper<double> g_percentages(percentages.begin(),
                                                                 percentages.end());
    cudf::tdigest::tdigest_column_view tdv(*(gb_result.second[0].results[0]));
    auto result = cudf::percentile_approx(tdv, g_percentages, stream);
    stream.synchronize();

    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*expected, *result);
  }
}

void simple_test(cudf::data_type input_type, std::vector<std::pair<int, int>> params)
{
  auto stream = cudf::get_default_stream();
  auto values = cudf::test::generate_standardized_percentile_distribution(input_type);
  // all in the same group
  auto keys = cudf::make_fixed_width_column(
    cudf::data_type{cudf::type_id::INT32}, values->size(), cudf::mask_state::UNALLOCATED);
  CUDF_CUDA_TRY(cudaMemsetAsync(
    keys->mutable_view().data<int32_t>(), 0, values->size() * sizeof(int32_t), stream.value()));

  // runs both groupby and reduce paths
  std::for_each(params.begin(), params.end(), [&](std::pair<int, int> const& params) {
    percentile_approx_test(
      *keys, *values, params.first, {0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0}, params.second);
  });
}

struct group_index {
  int32_t operator()(int32_t i) { return i / 150000; }
};

void grouped_test(cudf::data_type input_type, std::vector<std::pair<int, int>> params)
{
  auto stream = cudf::get_default_stream();
  auto values = cudf::test::generate_standardized_percentile_distribution(input_type);
  // all in the same group
  auto keys = cudf::make_fixed_width_column(
    cudf::data_type{cudf::type_id::INT32}, values->size(), cudf::mask_state::UNALLOCATED);
  auto i      = cuda::counting_iterator<int>{0};
  auto h_keys = std::vector<int32_t>(values->size());
  std::transform(i, i + values->size(), h_keys.begin(), group_index{});
  CUDF_CUDA_TRY(cudaMemcpyAsync(keys->mutable_view().data<int32_t>(),
                                h_keys.data(),
                                h_keys.size() * sizeof(int32_t),
                                cudaMemcpyDefault,
                                stream.value()));

  std::for_each(params.begin(), params.end(), [&](std::pair<int, int> const& params) {
    percentile_approx_test(
      *keys, *values, params.first, {0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0}, params.second);
  });
}

std::pair<rmm::device_buffer, cudf::size_type> make_null_mask(cudf::column_view const& col)
{
  auto itr = cudf::test::iterators::valids_at_multiples_of(2);
  return cudf::test::detail::make_null_mask(itr, itr + col.size());
}

void simple_with_nulls_test(cudf::data_type input_type, std::vector<std::pair<int, int>> params)
{
  auto values = cudf::test::generate_standardized_percentile_distribution(input_type);
  // all in the same group
  auto keys = cudf::make_fixed_width_column(
    cudf::data_type{cudf::type_id::INT32}, values->size(), cudf::mask_state::UNALLOCATED);
  CUDF_CUDA_TRY(
    cudaMemset(keys->mutable_view().data<int32_t>(), 0, values->size() * sizeof(int32_t)));

  // add a null mask
  auto mask = make_null_mask(*values);
  values->set_null_mask(mask.first, mask.second);

  std::for_each(params.begin(), params.end(), [&](std::pair<int, int> const& params) {
    percentile_approx_test(
      *keys, *values, params.first, {0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0}, params.second);
  });
}

void grouped_with_nulls_test(cudf::data_type input_type, std::vector<std::pair<int, int>> params)
{
  auto stream = cudf::get_default_stream();
  auto values = cudf::test::generate_standardized_percentile_distribution(input_type);
  // all in the same group
  auto keys = cudf::make_fixed_width_column(
    cudf::data_type{cudf::type_id::INT32}, values->size(), cudf::mask_state::UNALLOCATED);
  auto i      = cuda::counting_iterator<int>{0};
  auto h_keys = std::vector<int32_t>(values->size());
  std::transform(i, i + values->size(), h_keys.begin(), group_index{});
  CUDF_CUDA_TRY(cudaMemcpyAsync(keys->mutable_view().data<int32_t>(),
                                h_keys.data(),
                                h_keys.size() * sizeof(int32_t),
                                cudaMemcpyDefault,
                                stream.value()));

  // add a null mask
  auto mask = make_null_mask(*values);
  values->set_null_mask(mask.first, mask.second);

  std::for_each(params.begin(), params.end(), [&](std::pair<int, int> const& params) {
    percentile_approx_test(
      *keys, *values, params.first, {0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0}, params.second);
  });
}

template <typename T>
cudf::data_type get_appropriate_type()
{
  if constexpr (cudf::is_fixed_point<T>()) { return cudf::data_type{cudf::type_to_id<T>(), -7}; }
  return cudf::data_type{cudf::type_to_id<T>()};
}
}  // namespace

using PercentileApproxTypes =
  cudf::test::Concat<cudf::test::NumericTypes, cudf::test::FixedPointTypes>;

template <typename T>
struct PercentileApproxInputTypesTest : public cudf::test::BaseFixture {};
TYPED_TEST_SUITE(PercentileApproxInputTypesTest, PercentileApproxTypes);

TYPED_TEST(PercentileApproxInputTypesTest, Simple)
{
  using T               = TypeParam;
  auto const input_type = get_appropriate_type<T>();

  simple_test(input_type,
              {{1000, cudf::test::default_ulp},
               {100, cudf::test::default_ulp * 4},
               {10, cudf::test::default_ulp * 11}});
}

TYPED_TEST(PercentileApproxInputTypesTest, Grouped)
{
  using T               = TypeParam;
  auto const input_type = get_appropriate_type<T>();

  grouped_test(input_type,
               {{1000, cudf::test::default_ulp},
                {100, cudf::test::default_ulp * 2},
                {10, cudf::test::default_ulp * 10}});
}

using PercentileApproxMinimalTypes = testing::Types<int32_t, double, numeric::decimal64>;

// no need to recheck all types to verify null handling
template <typename T>
struct PercentileApproxInputMinimalTypesTest : public cudf::test::BaseFixture {};
TYPED_TEST_SUITE(PercentileApproxInputMinimalTypesTest, PercentileApproxMinimalTypes);

TYPED_TEST(PercentileApproxInputMinimalTypesTest, SimpleWithNulls)
{
  using T               = TypeParam;
  auto const input_type = get_appropriate_type<T>();

  simple_with_nulls_test(input_type,
                         {{1000, cudf::test::default_ulp},
                          {100, cudf::test::default_ulp * 2},
                          {10, cudf::test::default_ulp * 11}});
}

TYPED_TEST(PercentileApproxInputMinimalTypesTest, GroupedWithNulls)
{
  using T               = TypeParam;
  auto const input_type = get_appropriate_type<T>();

  grouped_with_nulls_test(input_type,
                          {{1000, cudf::test::default_ulp},
                           {100, cudf::test::default_ulp * 2},
                           {10, cudf::test::default_ulp * 6}});
}

struct PercentileApproxTest : public cudf::test::BaseFixture {};

TEST_F(PercentileApproxTest, EmptyInput)
{
  auto empty_ = cudf::tdigest::detail::make_empty_tdigests_column(
    1, cudf::get_default_stream(), cudf::get_current_device_resource_ref());
  cudf::test::fixed_width_column_wrapper<double> percentiles{0.0, 0.25, 0.3};

  std::vector<cudf::column_view> input;
  input.push_back(*empty_);
  input.push_back(*empty_);
  input.push_back(*empty_);
  auto empty = cudf::concatenate(input);

  cudf::tdigest::tdigest_column_view tdv(*empty);
  auto result = cudf::percentile_approx(tdv, percentiles);

  cudf::test::fixed_width_column_wrapper<cudf::size_type> offsets{0, 0, 0, 0};
  std::vector<bool> nulls{false, false, false};
  auto [null_mask, null_count] = cudf::test::detail::make_null_mask(nulls.begin(), nulls.end());

  auto expected = cudf::make_lists_column(3,
                                          offsets.release(),
                                          cudf::make_empty_column(cudf::type_id::FLOAT64),
                                          null_count,
                                          std::move(null_mask));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, *expected);
}

TEST_F(PercentileApproxTest, EmptyPercentiles)
{
  auto const delta = 1000;

  cudf::test::fixed_width_column_wrapper<double> values{0, 1, 2, 3, 4, 5};
  cudf::test::fixed_width_column_wrapper<int> keys{0, 0, 0, 1, 1, 1};
  cudf::table_view t({keys});
  cudf::groupby::groupby gb(t);
  std::vector<cudf::groupby::aggregation_request> requests;
  std::vector<std::unique_ptr<cudf::groupby_aggregation>> aggregations;
  aggregations.push_back(cudf::make_tdigest_aggregation<cudf::groupby_aggregation>(delta));
  requests.push_back({values, std::move(aggregations)});
  auto tdigest_column = gb.aggregate(requests);

  cudf::test::fixed_width_column_wrapper<double> percentiles{};

  cudf::tdigest::tdigest_column_view tdv(*tdigest_column.second[0].results[0]);
  auto result = cudf::percentile_approx(tdv, percentiles);

  cudf::test::fixed_width_column_wrapper<cudf::size_type> offsets{0, 0, 0};
  std::vector<bool> nulls{false, false};
  auto [null_mask, null_count] = cudf::test::detail::make_null_mask(nulls.begin(), nulls.end());

  auto expected = cudf::make_lists_column(2,
                                          offsets.release(),
                                          cudf::make_empty_column(cudf::type_id::FLOAT64),
                                          null_count,
                                          std::move(null_mask));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, *expected);
}

TEST_F(PercentileApproxTest, NullPercentiles)
{
  auto const delta = 1000;

  cudf::test::fixed_width_column_wrapper<double> values{1, 1, 2, 3, 4, 5, 6, 7, 8};
  cudf::test::fixed_width_column_wrapper<int> keys{0, 0, 0, 0, 0, 1, 1, 1, 1};
  cudf::table_view t({keys});
  cudf::groupby::groupby gb(t);
  std::vector<cudf::groupby::aggregation_request> requests;
  std::vector<std::unique_ptr<cudf::groupby_aggregation>> aggregations;
  aggregations.push_back(cudf::make_tdigest_aggregation<cudf::groupby_aggregation>(delta));
  requests.push_back({values, std::move(aggregations)});
  auto tdigest_column = gb.aggregate(requests);

  cudf::tdigest::tdigest_column_view tdv(*tdigest_column.second[0].results[0]);

  cudf::test::fixed_width_column_wrapper<double> npercentiles{{0.5, 0.5, 1.0, 1.0},
                                                              {false, false, true, true}};
  auto result = cudf::percentile_approx(tdv, npercentiles);

  std::vector<bool> valids{false, false, true, true};
  cudf::test::lists_column_wrapper<double> expected{{{99, 99, 4, 4}, valids.begin()},
                                                    {{99, 99, 8, 8}, valids.begin()}};

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expected);
}

TEST_F(PercentileApproxTest, ReductionGold)
{
  auto const delta = 1000;

  auto const values = cudf::test::fixed_width_column_wrapper<double>{9, 1, 7, 5, 2};

  auto const tdigest =
    cudf::reduce(values,
                 *cudf::make_tdigest_aggregation<cudf::reduce_aggregation>(delta),
                 cudf::data_type{cudf::type_id::STRUCT});
  auto const tdigest_col = cudf::make_column_from_scalar(*tdigest, 1);

  auto const percentiles =
    cudf::test::fixed_width_column_wrapper<double>{0.0, 0.25, 0.5, 0.75, 1.0};
  auto const result = cudf::percentile_approx(tdigest_col->view(), percentiles);

  auto const expected = cudf::test::lists_column_wrapper<double>{{1, 2, 5, 7, 9}};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expected);
}

TEST_F(PercentileApproxTest, GroupByGold)
{
  auto const delta = 1000;

  auto const values =
    cudf::test::fixed_width_column_wrapper<double>{9, 1, 7, 5, 2, 50, 10, 40, 20, 30};
  auto const keys = cudf::test::fixed_width_column_wrapper<int32_t>{0, 0, 0, 0, 0, 1, 1, 1, 1, 1};
  auto const percentiles =
    cudf::test::fixed_width_column_wrapper<double>{0.0, 0.25, 0.5, 0.75, 1.0};

  cudf::groupby::groupby gb(cudf::table_view{{keys}});
  std::vector<cudf::groupby::aggregation_request> requests;
  std::vector<std::unique_ptr<cudf::groupby_aggregation>> aggregations;
  aggregations.push_back(cudf::make_tdigest_aggregation<cudf::groupby_aggregation>(delta));
  requests.push_back({values, std::move(aggregations)});
  auto const tdigest_column = gb.aggregate(requests);

  cudf::tdigest::tdigest_column_view tdv(*tdigest_column.second[0].results[0]);
  auto const result = cudf::percentile_approx(tdv, percentiles);

  auto const expected =
    cudf::test::lists_column_wrapper<double>{{1, 2, 5, 7, 9}, {10, 20, 30, 40, 50}};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expected);
}

TEST_F(PercentileApproxTest, GroupByWithNullsGold)
{
  auto const delta = 1000;

  auto const values = cudf::test::fixed_width_column_wrapper<double>{
    {9, 99, 7, 5, 1, 50, 10, 40, 20, 30}, {1, 0, 1, 1, 1, 1, 1, 1, 1, 1}};
  auto const keys = cudf::test::fixed_width_column_wrapper<int32_t>{0, 0, 0, 0, 0, 1, 1, 1, 1, 1};
  auto const percentiles =
    cudf::test::fixed_width_column_wrapper<double>{0.0, 0.25, 0.5, 0.75, 1.0};

  cudf::groupby::groupby gb(cudf::table_view{{keys}});
  std::vector<cudf::groupby::aggregation_request> requests;
  std::vector<std::unique_ptr<cudf::groupby_aggregation>> aggregations;
  aggregations.push_back(cudf::make_tdigest_aggregation<cudf::groupby_aggregation>(delta));
  requests.push_back({values, std::move(aggregations)});
  auto const tdigest_column = gb.aggregate(requests);

  cudf::tdigest::tdigest_column_view tdv(*tdigest_column.second[0].results[0]);
  auto const result = cudf::percentile_approx(tdv, percentiles);

  auto const expected =
    cudf::test::lists_column_wrapper<double>{{1, 1, 5, 7, 9}, {10, 20, 30, 40, 50}};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expected);
}

TEST_F(PercentileApproxTest, ReductionWithLowRowCount)
{
  // Test that the tdigest reduction with a low row count still produces the correct results.
  // With 10 rows, the tdigest will have 10 centroids, where each row corresponds exactly to a
  // single decile. In this case, there should be no interpolation required.
  auto const max_centroids = 1000;

  auto const values = cudf::test::fixed_width_column_wrapper<double>{
    708, 717, 769, 1022, 1097, 1108, 1400, 1460, 2469, 2761};

  auto const tdigest =
    cudf::reduce(values,
                 *cudf::make_tdigest_aggregation<cudf::reduce_aggregation>(max_centroids),
                 cudf::data_type{cudf::type_id::STRUCT});

  auto const tdigest_col = cudf::make_column_from_scalar(*tdigest, 1);

  auto const percentiles = cudf::test::fixed_width_column_wrapper<double>{
    0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};
  auto const result = cudf::percentile_approx(tdigest_col->view(), percentiles);

  auto const expected = cudf::test::lists_column_wrapper<double>{
    {708, 708, 717, 769, 1022, 1097, 1108, 1400, 1460, 2469, 2761}};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expected);
}
