/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/column_utilities.hpp>
#include <cudf_test/cudf_gtest.hpp>
#include <cudf_test/tdigest_utilities.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/tdigest/tdigest.hpp>
#include <cudf/tdigest/tdigest_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/exec_policy.hpp>

#include <cuda/std/tuple>

// for use with groupby and reduction aggregation tests.

namespace cudf {
namespace test {

void tdigest_sample_compare(cudf::tdigest::tdigest_column_view const& tdv,
                            std::vector<expected_value> const& h_expected)
{
  column_view result_mean   = tdv.means();
  column_view result_weight = tdv.weights();

  auto h_expected_src = std::vector<size_type>(h_expected.size());
  std::transform(h_expected.begin(), h_expected.end(), h_expected_src.begin(), [](auto const& ex) {
    return cuda::std::get<0>(ex);
  });
  auto h_expected_mean = std::vector<double>(h_expected.size());
  std::transform(h_expected.begin(), h_expected.end(), h_expected_mean.begin(), [](auto const& ex) {
    return cuda::std::get<1>(ex);
  });
  auto h_expected_weight = std::vector<double>(h_expected.size());
  std::transform(
    h_expected.begin(), h_expected.end(), h_expected_weight.begin(), [](auto const& ex) {
      return cuda::std::get<2>(ex);
    });

  auto d_expected_src = cudf::detail::make_device_uvector_async(
    h_expected_src, cudf::get_default_stream(), cudf::get_current_device_resource_ref());
  auto d_expected_mean = cudf::detail::make_device_uvector_async(
    h_expected_mean, cudf::get_default_stream(), cudf::get_current_device_resource_ref());
  auto d_expected_weight = cudf::detail::make_device_uvector_async(
    h_expected_weight, cudf::get_default_stream(), cudf::get_current_device_resource_ref());

  auto map = cudf::device_span<cudf::size_type const>(d_expected_src);
  auto sampled_result_mean =
    std::move(cudf::gather(cudf::table_view({result_mean}), map)->release().front());
  auto sampled_result_weight =
    std::move(cudf::gather(cudf::table_view({result_weight}), map)->release().front());

  auto expected_mean   = cudf::device_span<double const>(d_expected_mean);
  auto expected_weight = cudf::device_span<double const>(d_expected_weight);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(expected_mean, *sampled_result_mean);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_weight, *sampled_result_weight);
}

std::unique_ptr<column> make_expected_tdigest_column(std::vector<expected_tdigest> const& groups)
{
  std::vector<std::unique_ptr<column>> tdigests;

  // make an individual digest
  auto make_digest = [&](expected_tdigest const& tdigest) {
    std::vector<std::unique_ptr<column>> inner_children;
    inner_children.push_back(std::make_unique<cudf::column>(tdigest.mean));
    inner_children.push_back(std::make_unique<cudf::column>(tdigest.weight));
    // tdigest struct
    auto tdigests =
      cudf::make_structs_column(tdigest.mean.size(), std::move(inner_children), 0, {});

    auto offsets =
      cudf::test::fixed_width_column_wrapper<cudf::size_type>({0, tdigest.mean.size()});
    auto list    = cudf::make_lists_column(1, offsets.release(), std::move(tdigests), 0, {});

    auto min_col = cudf::test::fixed_width_column_wrapper<double>({tdigest.min});
    auto max_col = cudf::test::fixed_width_column_wrapper<double>({tdigest.max});

    std::vector<std::unique_ptr<column>> children;
    children.push_back(std::move(list));
    children.push_back(min_col.release());
    children.push_back(max_col.release());
    return make_structs_column(1, std::move(children), 0, {});
  };

  // build the individual digests
  std::transform(groups.begin(), groups.end(), std::back_inserter(tdigests), make_digest);

  // concatenate them
  std::vector<column_view> views;
  std::transform(tdigests.begin(),
                 tdigests.end(),
                 std::back_inserter(views),
                 [](std::unique_ptr<column> const& c) { return c->view(); });

  return cudf::concatenate(views);
}

}  // namespace test
}  // namespace cudf
