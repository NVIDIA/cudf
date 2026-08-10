/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/detail/binaryop.hpp>
#include <cudf/detail/fill.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/unary.hpp>
#include <cudf/detail/valid_if.cuh>
#include <cudf/fixed_point/conv.hpp>
#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/iterator>
#include <thrust/transform.h>

namespace cudf {
namespace detail {
namespace {  // anonymous namespace
template <typename _TargetT>
struct unary_cast {
  template <typename SourceT, typename TargetT = _TargetT>
  __device__ inline TargetT operator()(SourceT const element)
    requires(cudf::is_numeric<SourceT>() && cudf::is_numeric<TargetT>())
  {
    return static_cast<TargetT>(element);
  }

  template <typename SourceT, typename TargetT = _TargetT>
  __device__ inline TargetT operator()(SourceT const element)
    requires(cudf::is_timestamp<SourceT>() && cudf::is_timestamp<TargetT>())
  {
    // Convert source tick counts into target tick counts without blindly truncating them
    // by dividing the respective duration time periods (which may not work for time before
    // UNIX epoch)
    return TargetT{cuda::std::chrono::floor<TargetT::duration>(element.time_since_epoch())};
  }

  template <typename SourceT, typename TargetT = _TargetT>
  __device__ inline TargetT operator()(SourceT const element)
    requires(cudf::is_duration<SourceT>() && cudf::is_duration<TargetT>())
  {
    return TargetT{cuda::std::chrono::floor<TargetT>(element)};
  }

  template <typename SourceT, typename TargetT = _TargetT>
  __device__ inline TargetT operator()(SourceT const element)
    requires(cudf::is_numeric<SourceT>() && cudf::is_duration<TargetT>())
  {
    return TargetT{static_cast<typename TargetT::rep>(element)};
  }

  template <typename SourceT, typename TargetT = _TargetT>
  __device__ inline TargetT operator()(SourceT const element)
    requires(cudf::is_timestamp<SourceT>() && cudf::is_duration<TargetT>())
  {
    return TargetT{cuda::std::chrono::floor<TargetT>(element.time_since_epoch())};
  }

  template <typename SourceT, typename TargetT = _TargetT>
  __device__ inline TargetT operator()(SourceT const element)
    requires(cudf::is_duration<SourceT>() && cudf::is_numeric<TargetT>())
  {
    return static_cast<TargetT>(element.count());
  }

  template <typename SourceT, typename TargetT = _TargetT>
  __device__ inline TargetT operator()(SourceT const element)
    requires(cudf::is_duration<SourceT>() && cudf::is_timestamp<TargetT>())
  {
    return TargetT{cuda::std::chrono::floor<TargetT::duration>(element)};
  }
};

template <typename _SourceT, typename _TargetT>
struct fixed_point_unary_cast {
  numeric::scale_type scale;
  using FixedPointT = std::conditional_t<cudf::is_fixed_point<_SourceT>(), _SourceT, _TargetT>;
  using DeviceT     = device_storage_type_t<FixedPointT>;

  template <typename SourceT = _SourceT, typename TargetT = _TargetT>
  __device__ inline TargetT operator()(DeviceT const element)
    requires(cudf::is_fixed_point<_SourceT>() && cudf::is_numeric<TargetT>())
  {
    auto const fixed_point = SourceT{numeric::scaled_integer<DeviceT>{element, scale}};
    if constexpr (cuda::std::is_floating_point_v<TargetT>) {
      return convert_fixed_to_floating<TargetT>(fixed_point);
    } else {
      return static_cast<TargetT>(fixed_point);
    }
  }

  template <typename SourceT = _SourceT, typename TargetT = _TargetT>
  __device__ inline DeviceT operator()(SourceT const element)
    requires(cudf::is_numeric<_SourceT>() && cudf::is_fixed_point<TargetT>())
  {
    if constexpr (cuda::std::is_floating_point_v<SourceT>) {
      return convert_floating_to_fixed<TargetT>(element, scale).value();
    } else {
      return TargetT{element, scale}.value();
    }
  }
};

template <typename From, typename To>
constexpr inline auto is_supported_non_fixed_point_cast()
{
  return cudf::is_fixed_width<From>() && cudf::is_fixed_width<To>() &&
         // Disallow fixed_point here (requires different specialization)
         !(cudf::is_fixed_point<From>() || cudf::is_fixed_point<To>()) &&
         // Disallow conversions between timestamps and numeric
         !(cudf::is_timestamp<From>() && is_numeric<To>()) &&
         !(cudf::is_timestamp<To>() && is_numeric<From>());
}

template <typename From, typename To>
constexpr inline auto is_supported_fixed_point_cast()
{
  return (cudf::is_fixed_point<From>() && cudf::is_numeric<To>()) ||
         (cudf::is_numeric<From>() && cudf::is_fixed_point<To>()) ||
         (cudf::is_fixed_point<From>() && cudf::is_fixed_point<To>());
}

template <typename From, typename To>
constexpr inline auto is_supported_cast()
{
  return is_supported_non_fixed_point_cast<From, To>() || is_supported_fixed_point_cast<From, To>();
}

template <typename From, typename To>
struct device_cast {
  __device__ To operator()(From element) { return static_cast<To>(element); }
};

/**
 * @brief Takes a `fixed_point` column_view as @p input and returns a `fixed_point` column with new
 * @p scale
 *
 * @tparam T     Type of the `fixed_point` column_view (`decimal32`, `decimal64` or `decimal128`)
 * @param input  Input `column_view`
 * @param scale  `scale` of the returned `column`
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr     Device memory resource used to allocate the returned column's device memory
 *
 * @return std::unique_ptr<column> Returned column with new @p scale
 */
template <typename T>
std::unique_ptr<column> rescale(column_view input,
                                numeric::scale_type scale,
                                rmm::cuda_stream_view stream,
                                rmm::device_async_resource_ref mr)
  requires(is_fixed_point<T>())
{
  using namespace numeric;
  using RepType = device_storage_type_t<T>;

  auto const type = cudf::data_type{cudf::type_to_id<T>(), scale};
  if (input.type().scale() >= scale) {
    auto const scalar = make_fixed_point_scalar<T>(0, scale_type{scale}, stream);
    return detail::binary_operation(input, *scalar, binary_operator::ADD, type, stream, mr);
  } else {
    auto const diff = input.type().scale() - scale;
    // The value of fixed point scalar will overflow if the scale difference is larger than the
    // max digits of underlying integral type. Under this condition, the output values can be
    // nothing other than zero value. Therefore, we simply return a zero column.
    if (-diff > cuda::std::numeric_limits<RepType>::digits10) {
      auto const scalar  = make_fixed_point_scalar<T>(0, scale_type{scale}, stream);
      auto output_column = make_column_from_scalar(*scalar, input.size(), stream, mr);
      if (input.nullable()) {
        auto null_mask = detail::copy_bitmask(input, stream, mr);
        output_column->set_null_mask(std::move(null_mask), input.null_count());
      }
      return output_column;
    }

    RepType scalar_value = 10;
    for (int i = 1; i < -diff; ++i) {
      scalar_value *= 10;
    }

    auto const scalar = make_fixed_point_scalar<T>(scalar_value, scale_type{diff}, stream);
    return detail::binary_operation(input, *scalar, binary_operator::DIV, type, stream, mr);
  }
};

/**
 * @brief Check if a floating point value is convertible to fixed point type.
 *
 * A floating point value is convertible if it is not null, not `NaN`, and not `inf`.
 *
 * Note that convertible input values may be out of the representable range of the target fixed
 * point type. Values out of the representable range need to be checked separately.
 */
template <typename FloatType>
struct is_convertible_floating_point {
  column_device_view d_input;

  bool __device__ operator()(size_type idx) const
  {
    static_assert(std::is_floating_point_v<FloatType>);

    if (d_input.is_null(idx)) { return false; }
    auto const value = d_input.element<FloatType>(idx);
    return std::isfinite(value);
  }
};

template <typename _SourceT>
struct dispatch_unary_cast_to {
  column_view input;

  dispatch_unary_cast_to(column_view inp) : input(inp) {}

  template <typename TargetT, typename SourceT = _SourceT>
  std::unique_ptr<column> operator()(data_type type,
                                     rmm::cuda_stream_view stream,
                                     rmm::device_async_resource_ref mr)
    requires(is_supported_non_fixed_point_cast<SourceT, TargetT>())
  {
    auto const size = input.size();
    auto output     = std::make_unique<column>(type,
                                           size,
                                           rmm::device_buffer{size * sizeof(TargetT), stream, mr},
                                           detail::copy_bitmask(input, stream, mr),
                                           input.null_count());

    mutable_column_view output_mutable = *output;

    thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                      input.begin<SourceT>(),
                      input.end<SourceT>(),
                      output_mutable.begin<TargetT>(),
                      unary_cast<TargetT>{});

    return output;
  }

  template <typename TargetT, typename SourceT = _SourceT>
  std::unique_ptr<column> operator()(data_type type,
                                     rmm::cuda_stream_view stream,
                                     rmm::device_async_resource_ref mr)
    requires(cudf::is_fixed_point<SourceT>() && cudf::is_numeric<TargetT>())
  {
    auto const size = input.size();
    auto output     = std::make_unique<column>(type,
                                           size,
                                           rmm::device_buffer{size * sizeof(TargetT), stream, mr},
                                           detail::copy_bitmask(input, stream, mr),
                                           input.null_count());

    mutable_column_view output_mutable = *output;

    using DeviceT    = device_storage_type_t<SourceT>;
    auto const scale = numeric::scale_type{input.type().scale()};

    thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                      input.begin<DeviceT>(),
                      input.end<DeviceT>(),
                      output_mutable.begin<TargetT>(),
                      fixed_point_unary_cast<SourceT, TargetT>{scale});

    return output;
  }

  template <typename TargetT, typename SourceT = _SourceT>
  std::unique_ptr<column> operator()(data_type type,
                                     rmm::cuda_stream_view stream,
                                     rmm::device_async_resource_ref mr)
    requires(cudf::is_numeric<SourceT>() && cudf::is_fixed_point<TargetT>())
  {
    using DeviceT = device_storage_type_t<TargetT>;

    auto const size = input.size();
    auto output     = std::make_unique<column>(
      type, size, rmm::device_buffer{size * sizeof(DeviceT), stream, mr}, rmm::device_buffer{}, 0);

    mutable_column_view output_mutable = *output;

    auto const scale = numeric::scale_type{type.scale()};

    thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                      input.begin<SourceT>(),
                      input.end<SourceT>(),
                      output_mutable.begin<DeviceT>(),
                      fixed_point_unary_cast<SourceT, TargetT>{scale});

    if constexpr (cudf::is_floating_point<SourceT>()) {
      // For floating-point values, beside input nulls, we also need to set nulls for the output
      // rows corresponding to NaN and inf in the input.
      auto const d_input_ptr = column_device_view::create(input, stream);
      auto [null_mask, null_count] =
        cudf::detail::valid_if(cuda::counting_iterator<cudf::size_type>{0},
                               cuda::counting_iterator{size},
                               is_convertible_floating_point<SourceT>{*d_input_ptr},
                               stream,
                               mr);
      if (null_count > 0) { output->set_null_mask(std::move(null_mask), null_count); }
    } else {
      output->set_null_mask(detail::copy_bitmask(input, stream, mr), input.null_count());
    }

    return output;
  }

  template <typename TargetT, typename SourceT = _SourceT>
  std::unique_ptr<column> operator()(data_type type,
                                     rmm::cuda_stream_view stream,
                                     rmm::device_async_resource_ref mr)
    requires(cudf::is_fixed_point<SourceT>() && cudf::is_fixed_point<TargetT>() &&
             std::is_same_v<SourceT, TargetT>)
  {
    if (input.type() == type) {
      return std::make_unique<column>(input, stream, mr);  // TODO add test for this
    }

    return detail::rescale<TargetT>(input, numeric::scale_type{type.scale()}, stream, mr);
  }

  template <typename TargetT, typename SourceT = _SourceT>
  std::unique_ptr<column> operator()(data_type type,
                                     rmm::cuda_stream_view stream,
                                     rmm::device_async_resource_ref mr)
    requires(cudf::is_fixed_point<SourceT>() && cudf::is_fixed_point<TargetT>() &&
             not std::is_same_v<SourceT, TargetT>)
  {
    using namespace numeric;
    using SourceDeviceT = device_storage_type_t<SourceT>;
    using TargetDeviceT = device_storage_type_t<TargetT>;

    auto casted = [&]() {
      auto const size = input.size();
      auto output =
        std::make_unique<column>(cudf::data_type{type.id(), input.type().scale()},
                                 size,
                                 rmm::device_buffer{size * sizeof(TargetDeviceT), stream},
                                 detail::copy_bitmask(input, stream, mr),
                                 input.null_count());

      mutable_column_view output_mutable = *output;

      thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                        input.begin<SourceDeviceT>(),
                        input.end<SourceDeviceT>(),
                        output_mutable.begin<TargetDeviceT>(),
                        device_cast<SourceDeviceT, TargetDeviceT>{});

      return output;
    };

    if (input.type().scale() == type.scale()) return casted();

    if constexpr (sizeof(SourceDeviceT) < sizeof(TargetDeviceT)) {
      // device_cast BEFORE rescale when SourceDeviceT is < TargetDeviceT
      auto temporary = casted();
      return detail::rescale<TargetT>(*temporary, scale_type{type.scale()}, stream, mr);
    } else {
      // device_cast AFTER rescale when SourceDeviceT is > TargetDeviceT to avoid overflow
      auto temporary = detail::rescale<SourceT>(input, scale_type{type.scale()}, stream, mr);
      return detail::cast(*temporary, type, stream, mr);
    }
  }

  template <typename TargetT, typename SourceT = _SourceT>
  std::unique_ptr<column> operator()(data_type,
                                     rmm::cuda_stream_view,
                                     rmm::device_async_resource_ref)

    requires(not is_supported_cast<SourceT, TargetT>())
  {
    if (!cudf::is_fixed_width<TargetT>())
      CUDF_FAIL("Column type must be numeric or chrono or decimal32/64/128");
    else if (cudf::is_fixed_point<SourceT>())
      CUDF_FAIL("Currently only decimal32/64/128 to floating point/integral is supported");
    else if (cudf::is_timestamp<SourceT>() && is_numeric<TargetT>())
      CUDF_FAIL("Timestamps can be created only from duration");
    else
      CUDF_FAIL("Timestamps cannot be converted to numeric without converting it to a duration");
  }
};

struct dispatch_unary_cast_from {
  column_view input;

  dispatch_unary_cast_from(column_view inp) : input(inp) {}

  template <typename T>
  std::unique_ptr<column> operator()(data_type type,
                                     rmm::cuda_stream_view stream,
                                     rmm::device_async_resource_ref mr)
    requires(cudf::is_fixed_width<T>())
  {
    return type_dispatcher(type, dispatch_unary_cast_to<T>{input}, type, stream, mr);
  }

  template <typename T, typename... Args>
  std::unique_ptr<column> operator()(Args&&...)
    requires(!cudf::is_fixed_width<T>())
  {
    CUDF_FAIL("Column type must be numeric or chrono or decimal32/64/128");
  }
};
}  // anonymous namespace

std::unique_ptr<column> cast(column_view const& input,
                             data_type type,
                             rmm::cuda_stream_view stream,
                             rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(is_fixed_width(type), "Unary cast type must be fixed-width.");

  return type_dispatcher(input.type(), detail::dispatch_unary_cast_from{input}, type, stream, mr);
}

struct is_supported_cast_impl {
  template <typename From, typename To>
  bool operator()() const
  {
    return is_supported_cast<From, To>();
  }
};

}  // namespace detail

std::unique_ptr<column> cast(column_view const& input,
                             data_type type,
                             rmm::cuda_stream_view stream,
                             rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::cast(input, type, stream, mr);
}

bool is_supported_cast(data_type from, data_type to) noexcept
{
  // No matching detail API call/nvtx annotation, since this doesn't
  // launch a kernel.
  return double_type_dispatcher(from, to, detail::is_supported_cast_impl{});
}

}  // namespace cudf
