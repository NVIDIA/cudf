/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/memory_resource_utilities.hpp>
#include <cudf_test/tdigest_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/reduction.hpp>

template <typename T>
struct ReductionTDigestAllTypes : public cudf::test::BaseFixture {};
TYPED_TEST_SUITE(ReductionTDigestAllTypes, cudf::test::NumericTypes);

namespace {
struct reduce_op {
  std::unique_ptr<cudf::column> operator()(cudf::column_view const& values, int delta) const
  {
    // result is a scalar, but we want to extract out the underlying column
    auto scalar_result =
      cudf::reduce(values,
                   *cudf::make_tdigest_aggregation<cudf::reduce_aggregation>(delta),
                   cudf::data_type{cudf::type_id::STRUCT});
    auto tbl = static_cast<cudf::struct_scalar const*>(scalar_result.get())->view();
    std::vector<std::unique_ptr<cudf::column>> cols;
    std::transform(
      tbl.begin(), tbl.end(), std::back_inserter(cols), [](cudf::column_view const& col) {
        return std::make_unique<cudf::column>(col);
      });
    return cudf::make_structs_column(tbl.num_rows(), std::move(cols), 0, rmm::device_buffer());
  }
};

struct reduce_merge_op {
  std::unique_ptr<cudf::column> operator()(cudf::column_view const& values, int delta) const
  {
    // result is a scalar, but we want to extract out the underlying column
    auto scalar_result =
      cudf::reduce(values,
                   *cudf::make_merge_tdigest_aggregation<cudf::reduce_aggregation>(delta),
                   cudf::data_type{cudf::type_id::STRUCT});
    auto tbl = static_cast<cudf::struct_scalar const*>(scalar_result.get())->view();
    std::vector<std::unique_ptr<cudf::column>> cols;
    std::transform(
      tbl.begin(), tbl.end(), std::back_inserter(cols), [](cudf::column_view const& col) {
        return std::make_unique<cudf::column>(col);
      });
    return cudf::make_structs_column(tbl.num_rows(), std::move(cols), 0, rmm::device_buffer());
  }
};
}  // namespace

TYPED_TEST(ReductionTDigestAllTypes, Simple)
{
  using T = TypeParam;
  cudf::test::tdigest_simple_aggregation<T>(reduce_op{});
}

TYPED_TEST(ReductionTDigestAllTypes, SimpleWithNulls)
{
  using T = TypeParam;
  cudf::test::tdigest_simple_with_nulls_aggregation<T>(reduce_op{});
}

TYPED_TEST(ReductionTDigestAllTypes, AllNull)
{
  using T = TypeParam;
  cudf::test::tdigest_simple_all_nulls_aggregation<T>(reduce_op{});
}

struct ReductionTDigestMerge : public cudf::test::BaseFixtureWithHarness {};

TEST_F(ReductionTDigestMerge, Simple)
{
  cudf::test::tdigest_merge_simple(reduce_op{}, reduce_merge_op{});
}

TEST_F(ReductionTDigestMerge, TestUtilityMemoryResourceControl)
{
  auto stream   = this->stream();
  auto& harness = this->harness();

  // generate_typed_percentile_distribution: output lives on output MR, temps are released.
  // Note: do not install a failing current resource here; cast still routes Thrust scratch
  // through get_current_device_resource_ref().
  {
    auto distribution = cudf::test::generate_typed_percentile_distribution(
      {10.0}, {4}, cudf::data_type{cudf::type_id::FLOAT64}, false, stream, harness.resources());
    harness.synchronize(stream);
    harness.expect_output_allocations_live(stream);
    harness.expect_temporary_allocation_activity(stream);
    harness.expect_temporary_allocations_released(stream);
  }
  harness.expect_no_live_allocations(stream);

  // Inputs stay on setup_mr so they do not affect output/temporary live-byte checks
  cudf::test::fixed_width_column_wrapper<double> means({1.0, 2.0}, stream, harness.setup_mr());
  cudf::test::fixed_width_column_wrapper<double> weights({1.0, 1.0}, stream, harness.setup_mr());

  {
    auto expected = cudf::test::make_expected_tdigest_column(
      {{means, weights, 1.0, 2.0}}, stream, harness.resources());
    auto const output_bytes_before = harness.expect_output_allocations_live(stream);
    harness.expect_temporary_allocation_activity(stream);
    auto const temporary_bytes_before = harness.expect_temporary_allocations_released(stream);

    cudf::tdigest::tdigest_column_view tdv(*expected);
    cudf::test::tdigest_sample_compare(
      tdv, {{0, 1.0, 1.0}, {1, 2.0, 1.0}}, stream, harness.resources());
    cudf::test::tdigest_minmax_compare<double>(tdv, means, stream, harness.resources());

    // Compare helpers must not allocate output; only temporary traffic
    harness.synchronize(stream);
    EXPECT_EQ(harness.output_mr().get_bytes_counter().value, output_bytes_before.value);
    EXPECT_EQ(harness.output_mr().get_bytes_counter().total, output_bytes_before.total);
    EXPECT_GT(harness.temporary_mr().get_bytes_counter().total, temporary_bytes_before.total);
    harness.expect_temporary_allocations_released(stream);
  }
  EXPECT_GT(harness.setup_mr().get_bytes_counter().value, 0);
}

// tests an issue with the cluster generating code with a small number of centroids that have large
// weights
TEST_F(ReductionTDigestMerge, FewHeavyCentroids)
{
  auto stream   = this->stream();
  auto mr       = this->resources();
  auto setup_mr = cudf::memory_resources{this->harness().setup_mr()};

  // digest 1
  cudf::test::fixed_width_column_wrapper<double> c0c({1.0, 2.0}, stream, setup_mr);
  cudf::test::fixed_width_column_wrapper<double> c0w({100.0, 50.0}, stream, setup_mr);
  cudf::test::structs_column_wrapper c0s({c0c, c0w}, {}, stream, setup_mr);
  cudf::test::fixed_width_column_wrapper<cudf::size_type> c0_offsets({0, 2}, stream, setup_mr);
  auto c0l =
    cudf::make_lists_column(1, c0_offsets.release(), c0s.release(), 0, rmm::device_buffer{});
  cudf::test::fixed_width_column_wrapper<double> c0min({1.0}, stream, setup_mr);
  cudf::test::fixed_width_column_wrapper<double> c0max({2.0}, stream, setup_mr);
  std::vector<std::unique_ptr<cudf::column>> c0_children;
  c0_children.push_back(std::move(c0l));
  c0_children.push_back(c0min.release());
  c0_children.push_back(c0max.release());
  // tdigest struct
  auto c0 =
    cudf::make_structs_column(1, std::move(c0_children), 0, {}, stream, setup_mr.get_output_mr());
  cudf::tdigest::tdigest_column_view tdv0(*c0);

  // digest 2
  cudf::test::fixed_width_column_wrapper<double> c1c({3.0, 4.0}, stream, setup_mr);
  cudf::test::fixed_width_column_wrapper<double> c1w({200.0, 50.0}, stream, setup_mr);
  cudf::test::structs_column_wrapper c1s({c1c, c1w}, {}, stream, setup_mr);
  cudf::test::fixed_width_column_wrapper<cudf::size_type> c1_offsets({0, 2}, stream, setup_mr);
  auto c1l =
    cudf::make_lists_column(1, c1_offsets.release(), c1s.release(), 0, rmm::device_buffer{});
  cudf::test::fixed_width_column_wrapper<double> c1min({3.0}, stream, setup_mr);
  cudf::test::fixed_width_column_wrapper<double> c1max({4.0}, stream, setup_mr);
  std::vector<std::unique_ptr<cudf::column>> c1_children;
  c1_children.push_back(std::move(c1l));
  c1_children.push_back(c1min.release());
  c1_children.push_back(c1max.release());
  // tdigest struct
  auto c1 =
    cudf::make_structs_column(1, std::move(c1_children), 0, {}, stream, setup_mr.get_output_mr());

  std::vector<cudf::column_view> views;
  views.push_back(*c0);
  views.push_back(*c1);
  auto values = cudf::concatenate(views, stream, setup_mr.get_output_mr());

  // merge
  auto scalar_result =
    cudf::reduce(*values,
                 *cudf::make_merge_tdigest_aggregation<cudf::reduce_aggregation>(1000),
                 cudf::data_type{cudf::type_id::STRUCT},
                 stream,
                 mr.get_output_mr());

  // convert to a table
  auto tbl = static_cast<cudf::struct_scalar const*>(scalar_result.get())->view();
  std::vector<std::unique_ptr<cudf::column>> cols;
  std::transform(
    tbl.begin(), tbl.end(), std::back_inserter(cols), [&](cudf::column_view const& col) {
      return std::make_unique<cudf::column>(col, stream, mr.get_output_mr());
    });
  auto result = cudf::make_structs_column(
    tbl.num_rows(), std::move(cols), 0, rmm::device_buffer(), stream, mr.get_output_mr());

  // we expect to see exactly 4 centroids (the same inputs) with properly computed min/max.
  cudf::test::fixed_width_column_wrapper<double> ec({1.0, 2.0, 3.0, 4.0}, stream, setup_mr);
  cudf::test::fixed_width_column_wrapper<double> ew({100.0, 50.0, 200.0, 50.0}, stream, setup_mr);
  cudf::test::structs_column_wrapper es({ec, ew}, {}, stream, setup_mr);
  cudf::test::fixed_width_column_wrapper<cudf::size_type> e_offsets({0, 4}, stream, setup_mr);
  auto el = cudf::make_lists_column(1, e_offsets.release(), es.release(), 0, rmm::device_buffer{});
  cudf::test::fixed_width_column_wrapper<double> emin({1.0}, stream, setup_mr);
  cudf::test::fixed_width_column_wrapper<double> emax({4.0}, stream, setup_mr);
  std::vector<std::unique_ptr<cudf::column>> e_children;
  e_children.push_back(std::move(el));
  e_children.push_back(emin.release());
  e_children.push_back(emax.release());
  // tdigest struct
  auto expected =
    cudf::make_structs_column(1, std::move(e_children), 0, {}, stream, setup_mr.get_output_mr());

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(
    *result, *expected, cudf::test::debug_output_level::FIRST_ERROR, stream, mr);
}
