/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>

#include <cudf/aggregation/host_udf.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/reduction.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/iterator>
#include <cuda/std/limits>
#include <cuda/std/tuple>
#include <thrust/transform.h>

using doubles_col = cudf::test::fixed_width_column_wrapper<double>;
using int32s_col  = cudf::test::fixed_width_column_wrapper<int32_t>;
using int64s_col  = cudf::test::fixed_width_column_wrapper<int64_t>;

namespace {
/**
 * @brief A host-based UDF implementation for reduction.
 *
 * The aggregation computes `sum(value^2, for value in group)` (this is sum of squared).
 */
struct host_udf_reduction_example : cudf::reduce_host_udf {
  host_udf_reduction_example() = default;

  [[nodiscard]] std::unique_ptr<cudf::scalar> operator()(
    cudf::column_view const& input,
    cudf::data_type output_dtype,
    std::optional<std::reference_wrapper<cudf::scalar const>> init,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) const override
  {
    return cudf::double_type_dispatcher(
      input.type(), output_dtype, reduce_fn{}, input, output_dtype, init, stream, mr);
  }

  [[nodiscard]] bool is_equal(host_udf_base const& other) const override
  {
    // Just check if the other object is also instance of this class.
    return dynamic_cast<host_udf_reduction_example const*>(&other) != nullptr;
  }

  [[nodiscard]] std::unique_ptr<host_udf_base> clone() const override
  {
    return std::make_unique<host_udf_reduction_example>();
  }

  struct reduce_fn {
    // For simplicity, this example only accepts a single type input and output.
    using InputType  = double;
    using OutputType = int64_t;

    template <typename T,
              typename U,
              typename... Args,
              CUDF_ENABLE_IF(!std::is_same_v<InputType, T> || !std::is_same_v<OutputType, U>)>
    std::unique_ptr<cudf::scalar> operator()(Args...) const
    {
      CUDF_FAIL("Unsupported input/output type.");
    }

    template <typename T,
              typename U,
              CUDF_ENABLE_IF(std::is_same_v<InputType, T>&& std::is_same_v<OutputType, U>)>
    [[nodiscard]] std::unique_ptr<cudf::scalar> operator()(
      cudf::column_view const& input,
      cudf::data_type output_dtype,
      std::optional<std::reference_wrapper<cudf::scalar const>> init,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const
    {
      CUDF_EXPECTS(output_dtype == cudf::data_type{cudf::type_to_id<OutputType>()},
                   "Invalid output type.");
      if (input.size() == 0) {
        return cudf::make_default_constructed_scalar(output_dtype, stream, mr);
      }

      auto const init_value = [&]() -> InputType {
        if (init.has_value() && init.value().get().is_valid(stream)) {
          auto const numeric_init_scalar =
            dynamic_cast<cudf::numeric_scalar<InputType> const*>(&init.value().get());
          CUDF_EXPECTS(numeric_init_scalar != nullptr, "Invalid init scalar for reduction.");
          return numeric_init_scalar->value(stream);
        }
        return InputType{0};
      }();

      auto const input_dv_ptr = cudf::column_device_view::create(input, stream);
      auto const result       = thrust::transform_reduce(rmm::exec_policy_nosync(stream),
                                                   cuda::counting_iterator<cudf::size_type>{0},
                                                   cuda::counting_iterator{input.size()},
                                                   transform_fn{*input_dv_ptr},
                                                   static_cast<OutputType>(init_value),
                                                   cuda::std::plus<>{});

      auto output = cudf::make_numeric_scalar(output_dtype, stream, mr);
      static_cast<cudf::scalar_type_t<OutputType>*>(output.get())->set_value(result, stream);
      return output;
    }

    struct transform_fn {
      cudf::column_device_view values;
      OutputType __device__ operator()(cudf::size_type idx) const
      {
        if (values.is_null(idx)) { return OutputType{0}; }
        auto const val = static_cast<OutputType>(values.element<InputType>(idx));
        return val * val;
      }
    };
  };
};

}  // namespace

struct HostUDFReductionExampleTest : cudf::test::BaseFixture {};

TEST_F(HostUDFReductionExampleTest, SimpleInput)
{
  auto const vals = doubles_col{0.0, 1.0, 2.0, 3.0, 4.0, 5.0};
  auto const agg  = cudf::make_host_udf_aggregation<cudf::reduce_aggregation>(
    std::make_unique<host_udf_reduction_example>());
  auto const reduced = cudf::reduce(vals,
                                    *agg,
                                    cudf::data_type{cudf::type_id::INT64},
                                    cudf::get_default_stream(),
                                    cudf::get_current_device_resource_ref());
  EXPECT_TRUE(reduced->is_valid());
  EXPECT_EQ(cudf::type_id::INT64, reduced->type().id());
  auto const result =
    static_cast<cudf::scalar_type_t<int64_t>*>(reduced.get())->value(cudf::get_default_stream());
  auto constexpr expected = 55;  // 0^2 + 1^2 + 2^2 + 3^2 + 4^2 + 5^2 = 55
  EXPECT_EQ(expected, result);
}

TEST_F(HostUDFReductionExampleTest, EmptyInput)
{
  auto const vals = doubles_col{};
  auto const agg  = cudf::make_host_udf_aggregation<cudf::reduce_aggregation>(
    std::make_unique<host_udf_reduction_example>());
  auto const reduced = cudf::reduce(vals,
                                    *agg,
                                    cudf::data_type{cudf::type_id::INT64},
                                    cudf::get_default_stream(),
                                    cudf::get_current_device_resource_ref());
  EXPECT_FALSE(reduced->is_valid());
  EXPECT_EQ(cudf::type_id::INT64, reduced->type().id());
}

namespace {

/**
 * @brief A host-based UDF implementation for segmented reduction.
 *
 * The aggregation computes `sum(value^2, for value in group)` (this is sum of squared).
 */
struct host_udf_segmented_reduction_example : cudf::segmented_reduce_host_udf {
  host_udf_segmented_reduction_example() = default;

  [[nodiscard]] std::unique_ptr<cudf::column> operator()(
    cudf::column_view const& input,
    cudf::device_span<cudf::size_type const> offsets,
    cudf::data_type output_dtype,
    cudf::null_policy null_handling,
    std::optional<std::reference_wrapper<cudf::scalar const>> init,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) const override
  {
    return cudf::double_type_dispatcher(input.type(),
                                        output_dtype,
                                        segmented_reduce_fn{},
                                        input,
                                        offsets,
                                        output_dtype,
                                        null_handling,
                                        init,
                                        stream,
                                        mr);
  }

  [[nodiscard]] bool is_equal(host_udf_base const& other) const override
  {
    // Just check if the other object is also instance of this class.
    return dynamic_cast<host_udf_segmented_reduction_example const*>(&other) != nullptr;
  }

  [[nodiscard]] std::unique_ptr<host_udf_base> clone() const override
  {
    return std::make_unique<host_udf_segmented_reduction_example>();
  }

  struct segmented_reduce_fn {
    // For simplicity, this example only accepts a single type input and output.
    using InputType  = double;
    using OutputType = int64_t;

    template <typename T,
              typename U,
              typename... Args,
              CUDF_ENABLE_IF(!std::is_same_v<InputType, T> || !std::is_same_v<OutputType, U>)>
    std::unique_ptr<cudf::column> operator()(Args...) const
    {
      CUDF_FAIL("Unsupported input/output type.");
    }

    template <typename T,
              typename U,
              CUDF_ENABLE_IF(std::is_same_v<InputType, T>&& std::is_same_v<OutputType, U>)>
    std::unique_ptr<cudf::column> operator()(
      cudf::column_view const& input,
      cudf::device_span<cudf::size_type const> offsets,
      cudf::data_type output_dtype,
      cudf::null_policy null_handling,
      std::optional<std::reference_wrapper<cudf::scalar const>> init,
      rmm::cuda_stream_view stream,
      rmm::device_async_resource_ref mr) const
    {
      CUDF_EXPECTS(output_dtype == cudf::data_type{cudf::type_to_id<OutputType>()},
                   "Invalid output type.");
      CUDF_EXPECTS(offsets.size() > 0, "Invalid offsets.");
      auto const num_segments = static_cast<cudf::size_type>(offsets.size()) - 1;

      if (input.size() == 0) {
        if (num_segments <= 0) { return cudf::make_empty_column(output_dtype); }
        return cudf::make_numeric_column(
          output_dtype, num_segments, cudf::mask_state::ALL_NULL, stream, mr);
      }

      auto const init_value = [&]() -> InputType {
        if (init.has_value() && init.value().get().is_valid(stream)) {
          auto const numeric_init_scalar =
            dynamic_cast<cudf::numeric_scalar<InputType> const*>(&init.value().get());
          CUDF_EXPECTS(numeric_init_scalar != nullptr, "Invalid init scalar for reduction.");
          return numeric_init_scalar->value(stream);
        }
        return InputType{0};
      }();

      auto const input_dv_ptr = cudf::column_device_view::create(input, stream);
      auto output             = cudf::make_numeric_column(
        output_dtype, num_segments, cudf::mask_state::UNALLOCATED, stream);

      // Store row index if it is valid, otherwise store a negative value denoting a null row.
      rmm::device_uvector<cudf::size_type> valid_idx(num_segments, stream);

      thrust::transform(
        rmm::exec_policy_nosync(stream),
        cuda::counting_iterator<cudf::size_type>{0},
        cuda::counting_iterator{num_segments},
        cuda::make_zip_iterator(output->mutable_view().begin<OutputType>(), valid_idx.begin()),
        transform_fn{*input_dv_ptr, offsets, static_cast<OutputType>(init_value), null_handling});

      auto const valid_idx_cv = cudf::column_view{
        cudf::data_type{cudf::type_id::INT32}, num_segments, valid_idx.begin(), nullptr, 0};
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
      OutputType init_value;
      cudf::null_policy null_handling;

      cuda::std::tuple<OutputType, cudf::size_type> __device__ operator()(cudf::size_type idx) const
      {
        auto const start = offsets[idx];
        auto const end   = offsets[idx + 1];

        auto constexpr invalid_idx = cuda::std::numeric_limits<cudf::size_type>::lowest();
        if (start == end) { return {OutputType{0}, invalid_idx}; }

        auto sum = init_value;
        for (auto i = start; i < end; ++i) {
          if (values.is_null(i)) {
            if (null_handling == cudf::null_policy::INCLUDE) { sum += init_value * init_value; }
            continue;
          }
          auto const val = static_cast<OutputType>(values.element<InputType>(i));
          sum += val * val;
        }
        auto const segment_size = end - start;
        return {static_cast<OutputType>(segment_size) * sum, idx};
      }
    };
  };
};

}  // namespace

struct HostUDFSegmentedReductionExampleTest : cudf::test::BaseFixture {};

TEST_F(HostUDFSegmentedReductionExampleTest, SimpleInput)
{
  double constexpr null = 0.0;
  auto const vals       = doubles_col{{0.0, null, 2.0, 3.0, null, 5.0, null, null, 8.0, 9.0},
                                      {true, false, true, true, false, true, false, false, true, true}};
  auto const offsets    = int32s_col{0, 3, 5, 10}.release();
  auto const agg        = cudf::make_host_udf_aggregation<cudf::segmented_reduce_aggregation>(
    std::make_unique<host_udf_segmented_reduction_example>());

  // Test without init value.
  {
    auto const result = cudf::segmented_reduce(
      vals,
      cudf::device_span<int const>(offsets->view().begin<int>(), offsets->size()),
      *agg,
      cudf::data_type{cudf::type_id::INT64},
      cudf::null_policy::INCLUDE,
      std::nullopt,  // init value
      cudf::get_default_stream(),
      cudf::get_current_device_resource_ref());

    // When null_policy is set to `INCLUDE`, the null values are replaced with the init value.
    // Since init value is not given, it is set to 0.
    // [ 3 * (0^2 + init^2 + 2^2), 2 * (3^2 + init^2), 5 * (5^2 + init^2 + init^2 + 8^2 + 9^2) ]
    auto const expected = int64s_col{{12, 18, 850}, {true, true, true}};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
  }

  // Test with init value, and include nulls.
  {
    auto const init_scalar = cudf::make_fixed_width_scalar<double>(3.0);
    auto const result      = cudf::segmented_reduce(
      vals,
      cudf::device_span<int const>(offsets->view().begin<int>(), offsets->size()),
      *agg,
      cudf::data_type{cudf::type_id::INT64},
      cudf::null_policy::INCLUDE,
      *init_scalar,
      cudf::get_default_stream(),
      cudf::get_current_device_resource_ref());

    // When null_policy is set to `INCLUDE`, the null values are replaced with the init value.
    // [ 3 * (3 + 0^2 + 3^2 + 2^2), 2 * (3 + 3^2 + 3^2), 5 * (3 + 5^2 + 3^2 + 3^2 + 8^2 + 9^2) ]
    auto const expected = int64s_col{{48, 42, 955}, {true, true, true}};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
  }

  // Test with init value, and exclude nulls.
  {
    auto const init_scalar = cudf::make_fixed_width_scalar<double>(3.0);
    auto const result      = cudf::segmented_reduce(
      vals,
      cudf::device_span<int const>(offsets->view().begin<int>(), offsets->size()),
      *agg,
      cudf::data_type{cudf::type_id::INT64},
      cudf::null_policy::EXCLUDE,
      *init_scalar,
      cudf::get_default_stream(),
      cudf::get_current_device_resource_ref());

    // [ 3 * (3 + 0^2 + 2^2), 2 * (3 + 3^2), 5 * (3 + 5^2 + 8^2 + 9^2) ]
    auto const expected = int64s_col{{21, 24, 865}, {true, true, true}};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
  }
}

TEST_F(HostUDFSegmentedReductionExampleTest, EmptySegments)
{
  auto const vals    = doubles_col{};
  auto const offsets = int32s_col{0, 0, 0, 0}.release();
  auto const agg     = cudf::make_host_udf_aggregation<cudf::segmented_reduce_aggregation>(
    std::make_unique<host_udf_segmented_reduction_example>());
  auto const result = cudf::segmented_reduce(
    vals,
    cudf::device_span<int const>(offsets->view().begin<int>(), offsets->size()),
    *agg,
    cudf::data_type{cudf::type_id::INT64},
    cudf::null_policy::INCLUDE,
    std::nullopt,  // init value
    cudf::get_default_stream(),
    cudf::get_current_device_resource_ref());
  auto const expected = int64s_col{{0, 0, 0}, {false, false, false}};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
}

TEST_F(HostUDFSegmentedReductionExampleTest, EmptyInput)
{
  auto const vals    = doubles_col{};
  auto const offsets = int32s_col{}.release();
  auto const agg     = cudf::make_host_udf_aggregation<cudf::segmented_reduce_aggregation>(
    std::make_unique<host_udf_segmented_reduction_example>());
  auto const result = cudf::segmented_reduce(
    vals,
    cudf::device_span<int const>(offsets->view().begin<int>(), offsets->size()),
    *agg,
    cudf::data_type{cudf::type_id::INT64},
    cudf::null_policy::INCLUDE,
    std::nullopt,  // init value
    cudf::get_default_stream(),
    cudf::get_current_device_resource_ref());
  auto const expected = int64s_col{};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
}
