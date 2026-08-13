/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>

#include <cudf/aggregation/host_udf.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/groupby.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/iterator>
#include <cuda/std/limits>
#include <cuda/std/tuple>
#include <thrust/transform.h>

using doubles_col = cudf::test::fixed_width_column_wrapper<double>;
using int32s_col  = cudf::test::fixed_width_column_wrapper<int32_t>;

namespace {
/**
 * @brief A host-based UDF implementation for groupby.
 *
 * For each group of values, the aggregation computes
 * `(group_idx + 1) * group_sum_of_squares - group_max * group_sum`.
 */
struct host_udf_groupby_example : cudf::groupby_host_udf {
  host_udf_groupby_example() = default;

  [[nodiscard]] std::unique_ptr<cudf::column> get_empty_output(
    rmm::cuda_stream_view, rmm::device_async_resource_ref) const override
  {
    return cudf::make_empty_column(
      cudf::data_type{cudf::type_to_id<typename groupby_fn::OutputType>()});
  }

  [[nodiscard]] std::unique_ptr<cudf::column> operator()(
    rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr) const override
  {
    auto const values = get_grouped_values();
    return cudf::type_dispatcher(values.type(), groupby_fn{*this}, stream, mr);
  }

  [[nodiscard]] bool is_equal(host_udf_base const& other) const override
  {
    // Just check if the other object is also instance of this class.
    return dynamic_cast<host_udf_groupby_example const*>(&other) != nullptr;
  }

  [[nodiscard]] std::unique_ptr<host_udf_base> clone() const override
  {
    return std::make_unique<host_udf_groupby_example>();
  }

  struct groupby_fn {
    // Store pointer to the parent class so we can call its functions.
    host_udf_groupby_example const& parent;

    // For simplicity, this example only accepts a single type input and output.
    using InputType  = double;
    using OutputType = double;

    template <typename T, typename... Args, CUDF_ENABLE_IF(!std::is_same_v<InputType, T>)>
    std::unique_ptr<cudf::column> operator()(Args...) const
    {
      CUDF_FAIL("Unsupported input type.");
    }

    template <typename T, CUDF_ENABLE_IF(std::is_same_v<InputType, T>)>
    std::unique_ptr<cudf::column> operator()(rmm::cuda_stream_view stream,
                                             rmm::device_async_resource_ref mr) const
    {
      auto const values = parent.get_grouped_values();
      if (values.size() == 0) { return parent.get_empty_output(stream, mr); }

      auto const offsets = parent.get_group_offsets();
      CUDF_EXPECTS(offsets.size() > 0, "Invalid offsets.");
      auto const num_groups    = static_cast<int>(offsets.size()) - 1;
      auto const group_indices = parent.get_group_labels();
      auto const group_max =
        parent.compute_aggregation(cudf::make_max_aggregation<cudf::groupby_aggregation>());
      auto const group_sum =
        parent.compute_aggregation(cudf::make_sum_aggregation<cudf::groupby_aggregation>());

      auto const values_dv_ptr = cudf::column_device_view::create(values, stream);
      auto const output = cudf::make_numeric_column(cudf::data_type{cudf::type_to_id<OutputType>()},
                                                    num_groups,
                                                    cudf::mask_state::UNALLOCATED,
                                                    stream,
                                                    mr);

      // Store row index if it is valid, otherwise store a negative value denoting a null row.
      rmm::device_uvector<cudf::size_type> valid_idx(num_groups, stream);

      thrust::transform(
        rmm::exec_policy_nosync(stream),
        cuda::counting_iterator<cudf::size_type>{0},
        cuda::counting_iterator{num_groups},
        cuda::make_zip_iterator(output->mutable_view().begin<OutputType>(), valid_idx.begin()),
        transform_fn{*values_dv_ptr,
                     offsets,
                     group_indices,
                     group_max.begin<InputType>(),
                     group_sum.begin<InputType>()});

      auto const valid_idx_cv = cudf::column_view{
        cudf::data_type{cudf::type_id::INT32}, num_groups, valid_idx.begin(), nullptr, 0};
      return std::move(cudf::gather(cudf::table_view{{output->view()}},
                                    valid_idx_cv,
                                    cudf::out_of_bounds_policy::NULLIFY,
                                    stream,
                                    mr)
                         ->release()
                         .front());
    }

    struct transform_fn {
      cudf::column_device_view values;
      cudf::device_span<cudf::size_type const> offsets;
      cudf::device_span<cudf::size_type const> group_indices;
      InputType const* group_max;
      InputType const* group_sum;

      cuda::std::tuple<OutputType, cudf::size_type> __device__ operator()(cudf::size_type idx) const
      {
        auto const start = offsets[idx];
        auto const end   = offsets[idx + 1];

        auto constexpr invalid_idx = cuda::std::numeric_limits<cudf::size_type>::lowest();
        if (start == end) { return {OutputType{0}, invalid_idx}; }

        auto sum_sqr = OutputType{0};
        bool has_valid{false};
        for (auto i = start; i < end; ++i) {
          if (values.is_null(i)) { continue; }
          has_valid      = true;
          auto const val = static_cast<OutputType>(values.element<InputType>(i));
          sum_sqr += val * val;
        }

        if (!has_valid) { return {OutputType{0}, invalid_idx}; }
        return {static_cast<OutputType>(group_indices[start] + 1) * sum_sqr -
                  static_cast<OutputType>(group_max[idx]) * static_cast<OutputType>(group_sum[idx]),
                idx};
      }
    };
  };
};

}  // namespace

struct HostUDFGroupbyExampleTest : cudf::test::BaseFixture {};

TEST_F(HostUDFGroupbyExampleTest, SimpleInput)
{
  double constexpr null = 0.0;
  auto const keys       = int32s_col{0, 1, 2, 0, 1, 2, 0, 1, 2, 0};
  auto const vals       = doubles_col{{0.0, null, 2.0, 3.0, null, 5.0, null, null, 8.0, 9.0},
                                      {true, false, true, true, false, true, false, false, true, true}};
  auto agg              = cudf::make_host_udf_aggregation<cudf::groupby_aggregation>(
    std::make_unique<host_udf_groupby_example>());

  std::vector<cudf::groupby::aggregation_request> requests;
  requests.emplace_back();
  requests[0].values = vals;
  requests[0].aggregations.push_back(std::move(agg));
  cudf::groupby::groupby gb_obj(
    cudf::table_view({keys}), cudf::null_policy::INCLUDE, cudf::sorted::NO, {}, {});

  auto const grp_result = gb_obj.aggregate(requests, cudf::test::get_default_stream());
  auto const& result    = grp_result.second[0].results[0];

  // Output type of groupby is double.
  // Values grouped by keys: [ {0, 3, null, 9}, {null, null, null}, {2, 5, 8} ]
  // Group sum_sqr: [ 90, null, 93 ]
  // Group max: [ 9, null, 8 ]
  // Group sum: [ 12, null, 15 ]
  // Output: [ 1 * 90 - 9 * 12, null, 3 * 93 - 8 * 15 ]
  auto const expected = doubles_col{{-18.0, null, 159.0}, {true, false, true}};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
}

TEST_F(HostUDFGroupbyExampleTest, EmptyInput)
{
  auto const keys = int32s_col{};
  auto const vals = doubles_col{};
  auto agg        = cudf::make_host_udf_aggregation<cudf::groupby_aggregation>(
    std::make_unique<host_udf_groupby_example>());

  std::vector<cudf::groupby::aggregation_request> requests;
  requests.emplace_back();
  requests[0].values = vals;
  requests[0].aggregations.push_back(std::move(agg));
  cudf::groupby::groupby gb_obj(
    cudf::table_view({keys}), cudf::null_policy::INCLUDE, cudf::sorted::NO, {}, {});

  auto const grp_result = gb_obj.aggregate(requests, cudf::test::get_default_stream());
  auto const& result    = grp_result.second[0].results[0];
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(vals, *result);
}
