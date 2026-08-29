/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/tdigest_utilities.hpp>
#include <cudf_test/type_list_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/detail/tdigest/tdigest.hpp>
#include <cudf/groupby.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/quantiles.hpp>
#include <cudf/reduction.hpp>
#include <cudf/tdigest/tdigest_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cuda/iterator>
#include <cuda/stream>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iterator>
#include <utility>
#include <vector>

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
  cuda::stream_ref stream                     = cudf::get_default_stream();
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
        stream.sync();
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
        stream.sync();
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
      stream.sync();
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
    stream.sync();

    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*expected, *result);
  }
}

void simple_test(cudf::data_type input_type, std::vector<std::pair<int, int>> params)
{
  cuda::stream_ref stream = cudf::get_default_stream();
  auto values             = cudf::test::generate_standardized_percentile_distribution(input_type);
  // all in the same group
  auto keys = cudf::make_fixed_width_column(
    cudf::data_type{cudf::type_id::INT32}, values->size(), cudf::mask_state::UNALLOCATED);
  CUDF_CUDA_TRY(cudaMemsetAsync(
    keys->mutable_view().data<int32_t>(), 0, values->size() * sizeof(int32_t), stream.get()));

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
  cuda::stream_ref stream = cudf::get_default_stream();
  auto values             = cudf::test::generate_standardized_percentile_distribution(input_type);
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
                                stream.get()));

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

std::vector<double> make_compressed_test_values(cudf::size_type group, cudf::size_type row_count)
{
  std::vector<double> values(row_count);
  for (cudf::size_type i = 0; i < row_count; ++i) {
    auto const permutation =
      static_cast<double>((i * 37 + group * 101) % row_count) - (row_count / 2.0);
    auto const trend = static_cast<double>(i) * (0.015 + 0.003 * group);
    auto const wave  = std::sin(static_cast<double>(i) * 0.037 + group) * (9.0 + group);
    values[i]        = permutation * (1.5 + 0.2 * group) + trend + wave + group * 500.0;
  }
  return values;
}

double exact_host_quantile(std::vector<double> const& sorted_values, double percentile)
{
  auto const rank       = percentile * static_cast<double>(sorted_values.size() - 1);
  auto const lower_rank = static_cast<std::size_t>(std::floor(rank));
  auto const upper_rank = static_cast<std::size_t>(std::ceil(rank));
  if (lower_rank == upper_rank) { return sorted_values[lower_rank]; }

  auto const weight = rank - static_cast<double>(lower_rank);
  return sorted_values[lower_rank] +
         (sorted_values[upper_rank] - sorted_values[lower_rank]) * weight;
}

std::pair<double, double> empirical_rank_interval(std::vector<double> const& sorted_values,
                                                  double value)
{
  auto const denominator = static_cast<double>(sorted_values.size() - 1);
  auto const first       = std::lower_bound(sorted_values.begin(), sorted_values.end(), value);

  if (first != sorted_values.end() && *first == value) {
    auto const last = std::upper_bound(first, sorted_values.end(), value);
    return {static_cast<double>(std::distance(sorted_values.begin(), first)) / denominator,
            static_cast<double>(std::distance(sorted_values.begin(), last) - 1) / denominator};
  }

  if (first == sorted_values.begin()) { return {0.0, 0.0}; }
  if (first == sorted_values.end()) { return {1.0, 1.0}; }

  auto const upper_index = static_cast<std::size_t>(std::distance(sorted_values.begin(), first));
  auto const lower_index = upper_index - 1;
  auto const lower_value = sorted_values[lower_index];
  auto const upper_value = sorted_values[upper_index];
  auto const fraction    = (value - lower_value) / (upper_value - lower_value);
  auto const rank        = (static_cast<double>(lower_index) + fraction) / denominator;
  return {rank, rank};
}

std::vector<std::vector<double>> lists_column_to_host(cudf::column_view const& column)
{
  auto const lists   = cudf::lists_column_view{column};
  auto const offsets = cudf::test::to_host<cudf::size_type>(lists.offsets()).first;
  auto const child   = cudf::test::to_host<double>(lists.child()).first;

  std::vector<std::vector<double>> result(lists.size());
  for (cudf::size_type row = 0; row < lists.size(); ++row) {
    result[row].assign(child.begin() + offsets[row], child.begin() + offsets[row + 1]);
  }
  return result;
}

void expect_approx_percentiles_near_exact(std::vector<double> const& values,
                                          std::vector<double> const& percentiles,
                                          std::vector<double> const& actual,
                                          double max_rank_error)
{
  ASSERT_EQ(actual.size(), percentiles.size());

  auto sorted_values = values;
  std::sort(sorted_values.begin(), sorted_values.end());
  for (std::size_t idx = 0; idx < percentiles.size(); ++idx) {
    ASSERT_TRUE(std::isfinite(actual[idx]));
    ASSERT_GE(actual[idx], sorted_values.front());
    ASSERT_LE(actual[idx], sorted_values.back());

    auto const [rank_low, rank_high] = empirical_rank_interval(sorted_values, actual[idx]);
    auto const nearest_rank          = std::clamp(percentiles[idx], rank_low, rank_high);
    auto const rank_error            = std::abs(percentiles[idx] - nearest_rank);
    auto const exact                 = exact_host_quantile(sorted_values, percentiles[idx]);

    EXPECT_LE(rank_error, max_rank_error)
      << "percentile=" << percentiles[idx] << " actual_rank=[" << rank_low << ", " << rank_high
      << "] exact_value=" << exact << " actual_value=" << actual[idx];
  }
}

struct scoped_cpu_clustering_setting {
  bool const previous = cudf::tdigest::detail::is_cpu_cluster_computation_disabled;
  ~scoped_cpu_clustering_setting()
  {
    cudf::tdigest::detail::is_cpu_cluster_computation_disabled = previous;
  }
};

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
  cuda::stream_ref stream = cudf::get_default_stream();
  auto values             = cudf::test::generate_standardized_percentile_distribution(input_type);
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
                                stream.get()));

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

namespace {

TEST_F(PercentileApproxTest, CompressedTdigestsAgainstHostQuantiles)
{
  auto const max_centroids  = 20;
  auto const rows_per_group = 4096;
  // Regression budget validated for max_centroids=20 with both clustering paths.
  auto const max_rank_error = 0.025;
  auto const percentiles    = std::vector<double>{0.01, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99};
  auto const percentiles_column =
    cudf::test::fixed_width_column_wrapper<double>(percentiles.begin(), percentiles.end());
  auto const group_values = std::vector<std::vector<double>>{
    make_compressed_test_values(0, rows_per_group),
    make_compressed_test_values(1, rows_per_group),
  };

  auto const restore_cpu_clustering_setting = scoped_cpu_clustering_setting{};
  // Small group counts use host-side cluster compression by default. Exercise both cluster
  // compression paths because the compressed centroid layout is what percentile_approx consumes.
  for (auto const cpu_clustering_disabled : {true, false}) {
    SCOPED_TRACE(::testing::Message() << "cpu_clustering_disabled=" << cpu_clustering_disabled);
    cudf::tdigest::detail::is_cpu_cluster_computation_disabled = cpu_clustering_disabled;

    for (auto const& values : group_values) {
      auto const values_column =
        cudf::test::fixed_width_column_wrapper<double>(values.begin(), values.end());
      auto const tdigest =
        cudf::reduce(values_column,
                     *cudf::make_tdigest_aggregation<cudf::reduce_aggregation>(max_centroids),
                     cudf::data_type{cudf::type_id::STRUCT});
      auto const tdigest_col = cudf::make_column_from_scalar(*tdigest, 1);
      auto const result      = cudf::percentile_approx(tdigest_col->view(), percentiles_column);

      auto const actual = lists_column_to_host(result->view());
      ASSERT_EQ(actual.size(), 1);
      expect_approx_percentiles_near_exact(values, percentiles, actual.front(), max_rank_error);
    }

    std::vector<int32_t> keys;
    std::vector<double> values;
    keys.reserve(rows_per_group * group_values.size());
    values.reserve(rows_per_group * group_values.size());
    for (cudf::size_type row = 0; row < rows_per_group; ++row) {
      for (std::size_t group = 0; group < group_values.size(); ++group) {
        keys.push_back(static_cast<int32_t>(group));
        values.push_back(group_values[group][row]);
      }
    }

    auto const keys_column =
      cudf::test::fixed_width_column_wrapper<int32_t>(keys.begin(), keys.end());
    auto const values_column =
      cudf::test::fixed_width_column_wrapper<double>(values.begin(), values.end());

    cudf::groupby::groupby gb(cudf::table_view{{keys_column}});
    std::vector<cudf::groupby::aggregation_request> requests;
    std::vector<std::unique_ptr<cudf::groupby_aggregation>> aggregations;
    aggregations.push_back(
      cudf::make_tdigest_aggregation<cudf::groupby_aggregation>(max_centroids));
    requests.push_back({values_column, std::move(aggregations)});
    auto const groupby_result = gb.aggregate(requests);

    cudf::tdigest::tdigest_column_view tdv(*groupby_result.second[0].results[0]);
    auto const result = cudf::percentile_approx(tdv, percentiles_column);

    auto const actual = lists_column_to_host(result->view());
    auto const result_keys =
      cudf::test::to_host<int32_t>(groupby_result.first->get_column(0)).first;
    ASSERT_EQ(actual.size(), result_keys.size());
    ASSERT_EQ(actual.size(), group_values.size());
    std::vector<bool> saw_group(group_values.size(), false);
    for (std::size_t row = 0; row < actual.size(); ++row) {
      auto const key = result_keys[row];
      ASSERT_GE(key, 0);
      ASSERT_LT(key, static_cast<int32_t>(group_values.size()));
      ASSERT_FALSE(saw_group[key]);
      saw_group[key] = true;
      expect_approx_percentiles_near_exact(
        group_values[key], percentiles, actual[row], max_rank_error);
    }
    EXPECT_TRUE(std::all_of(
      saw_group.cbegin(), saw_group.cend(), [](bool const was_seen) { return was_seen; }));
  }
}

}  // namespace

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
