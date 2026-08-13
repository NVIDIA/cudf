/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>

#include <cudf/aggregation/host_udf.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/groupby.hpp>
#include <cudf/reduction.hpp>

#include <random>
#include <vector>

namespace {

/**
 * @brief Generate a random aggregation object from {min, max, sum, product}.
 */
std::unique_ptr<cudf::aggregation> get_random_agg()
{
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> distr(1, 4);
  switch (distr(gen)) {
    case 1: return cudf::make_min_aggregation();
    case 2: return cudf::make_max_aggregation();
    case 3: return cudf::make_sum_aggregation();
    case 4: return cudf::make_product_aggregation();
    default: CUDF_UNREACHABLE("This should not be reached.");
  }
  return nullptr;
}

/**
 * @brief A host-based UDF implementation used for unit tests for groupby aggregation.
 */
struct host_udf_groupby_test : cudf::groupby_host_udf {
  int test_location_line;  // the location where testing is called
  bool* test_run;          // to check if the test is accidentally skipped
  bool test_other_agg;     // test calling other aggregation

  host_udf_groupby_test(int test_location_line_, bool* test_run_, bool test_other_agg_)
    : test_location_line{test_location_line_}, test_run{test_run_}, test_other_agg{test_other_agg_}
  {
  }

  [[nodiscard]] std::size_t do_hash() const override { return 0; }
  [[nodiscard]] bool is_equal(host_udf_base const& other) const override
  {
    // Just check if the other object is also instance of this class.
    return dynamic_cast<host_udf_groupby_test const*>(&other) != nullptr;
  }
  [[nodiscard]] std::unique_ptr<host_udf_base> clone() const override
  {
    return std::make_unique<host_udf_groupby_test>(test_location_line, test_run, test_other_agg);
  }

  [[nodiscard]] std::unique_ptr<cudf::column> get_empty_output(
    [[maybe_unused]] rmm::cuda_stream_view stream,
    [[maybe_unused]] rmm::device_async_resource_ref mr) const override
  {
    // Dummy output.
    return cudf::make_empty_column(cudf::data_type{cudf::type_id::INT32});
  }

  [[nodiscard]] std::unique_ptr<cudf::column> operator()(
    rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr) const override
  {
    SCOPED_TRACE("Test instance created at line: " + std::to_string(test_location_line));

    // Perform tests on types for the groupby data: we must ensure the data corresponding to each
    // `groupby_data` enum having the correct type.

    {
      auto const inp_data = get_input_values();
      EXPECT_TRUE((std::is_same_v<cudf::column_view, std::decay_t<decltype(inp_data)>>));
    }

    {
      auto const inp_data = get_grouped_values();
      EXPECT_TRUE((std::is_same_v<cudf::column_view, std::decay_t<decltype(inp_data)>>));
    }

    {
      auto const inp_data = get_sorted_grouped_values();
      EXPECT_TRUE((std::is_same_v<cudf::column_view, std::decay_t<decltype(inp_data)>>));
    }

    {
      auto const inp_data = get_num_groups();
      EXPECT_TRUE((std::is_same_v<cudf::size_type, std::decay_t<decltype(inp_data)>>));
    }

    {
      auto const inp_data = get_group_offsets();
      EXPECT_TRUE((std::is_same_v<cudf::device_span<cudf::size_type const>,
                                  std::decay_t<decltype(inp_data)>>));
    }

    {
      auto const inp_data = get_group_labels();
      EXPECT_TRUE((std::is_same_v<cudf::device_span<cudf::size_type const>,
                                  std::decay_t<decltype(inp_data)>>));
    }

    // Perform tests on type of the result from computing other aggregations.
    if (test_other_agg) {
      auto const inp_data = compute_aggregation(get_random_agg());
      EXPECT_TRUE((std::is_same_v<cudf::column_view, std::decay_t<decltype(inp_data)>>));
    }

    *test_run = true;  // test is run successfully
    return get_empty_output(stream, mr);
  }
};

}  // namespace

using int32s_col = cudf::test::fixed_width_column_wrapper<int32_t>;

struct HostUDFTest : cudf::test::BaseFixture {};

TEST_F(HostUDFTest, GroupbyBuiltinInput)
{
  bool test_run   = false;
  auto const keys = int32s_col{0, 1, 2};
  auto const vals = int32s_col{0, 1, 2};
  auto agg        = cudf::make_host_udf_aggregation<cudf::groupby_aggregation>(
    std::make_unique<host_udf_groupby_test>(__LINE__, &test_run, /*test_other_agg*/ false));

  std::vector<cudf::groupby::aggregation_request> requests;
  requests.emplace_back();
  requests[0].values = vals;
  requests[0].aggregations.push_back(std::move(agg));
  cudf::groupby::groupby gb_obj(
    cudf::table_view({keys}), cudf::null_policy::INCLUDE, cudf::sorted::NO, {}, {});
  [[maybe_unused]] auto const grp_result = gb_obj.aggregate(
    requests, cudf::test::get_default_stream(), cudf::get_current_device_resource_ref());
  EXPECT_TRUE(test_run);
}

TEST_F(HostUDFTest, GroupbyWithCallingOtherAggregations)
{
  auto const keys = int32s_col{0, 1, 2};
  auto const vals = int32s_col{0, 1, 2};

  constexpr int NUM_RANDOM_TESTS = 20;

  for (int i = 0; i < NUM_RANDOM_TESTS; ++i) {
    bool test_run = false;
    auto agg      = cudf::make_host_udf_aggregation<cudf::groupby_aggregation>(
      std::make_unique<host_udf_groupby_test>(__LINE__, &test_run, /*test_other_agg*/ true));

    std::vector<cudf::groupby::aggregation_request> requests;
    requests.emplace_back();
    requests[0].values = vals;
    requests[0].aggregations.push_back(std::move(agg));
    cudf::groupby::groupby gb_obj(
      cudf::table_view({keys}), cudf::null_policy::INCLUDE, cudf::sorted::NO, {}, {});
    [[maybe_unused]] auto const grp_result = gb_obj.aggregate(
      requests, cudf::test::get_default_stream(), cudf::get_current_device_resource_ref());
    EXPECT_TRUE(test_run);
  }
}
