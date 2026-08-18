/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column_device_view.cuh>
#include <cudf/copying.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/unary.hpp>
#include <cudf/dictionary/detail/encode.hpp>
#include <cudf/dictionary/detail/iterator.cuh>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/exec_policy.hpp>

#include <cuda/std/bit>
#include <cuda/std/cmath>
#include <cuda/std/type_traits>
#include <cuda/stream>
#include <thrust/transform.h>

namespace cudf {
namespace detail {
namespace {

// trig functions

struct DeviceSin {
  template <typename T>
  __device__ T operator()(T data)
  {
    return cuda::std::sin(data);
  }
};

struct DeviceCos {
  template <typename T>
  __device__ T operator()(T data)
  {
    return cuda::std::cos(data);
  }
};

struct DeviceTan {
  template <typename T>
  __device__ T operator()(T data)
  {
    return cuda::std::tan(data);
  }
};

struct DeviceArcSin {
  template <typename T>
  __device__ T operator()(T data)
  {
    return cuda::std::asin(data);
  }
};

struct DeviceArcCos {
  template <typename T>
  __device__ T operator()(T data)
  {
    return cuda::std::acos(data);
  }
};

struct DeviceArcTan {
  template <typename T>
  __device__ T operator()(T data)
  {
    return cuda::std::atan(data);
  }
};

struct DeviceSinH {
  template <typename T>
  __device__ T operator()(T data)
  {
    return cuda::std::sinh(data);
  }
};

struct DeviceCosH {
  template <typename T>
  __device__ T operator()(T data)
  {
    return cuda::std::cosh(data);
  }
};

struct DeviceTanH {
  template <typename T>
  __device__ T operator()(T data)
  {
    return cuda::std::tanh(data);
  }
};

struct DeviceArcSinH {
  template <typename T>
  __device__ T operator()(T data)
  {
    return cuda::std::asinh(data);
  }
};

struct DeviceArcCosH {
  template <typename T>
  __device__ T operator()(T data)
  {
    return cuda::std::acosh(data);
  }
};

struct DeviceArcTanH {
  template <typename T>
  __device__ T operator()(T data)
  {
    return cuda::std::atanh(data);
  }
};

// exponential functions

struct DeviceExp {
  template <typename T>
  __device__ T operator()(T data)
  {
    return cuda::std::exp(data);
  }
};

struct DeviceLog {
  template <typename T>
  __device__ T operator()(T data)
  {
    return cuda::std::log(data);
  }
};

struct DeviceSqrt {
  template <typename T>
  __device__ T operator()(T data)
  {
    return cuda::std::sqrt(data);
  }
};

struct DeviceCbrt {
  template <typename T>
  __device__ T operator()(T data)
  {
    return cuda::std::cbrt(data);
  }
};

// rounding functions

struct DeviceCeil {
  template <typename T>
  __device__ T operator()(T data)
  {
    return cuda::std::ceil(data);
  }
};

struct DeviceFloor {
  template <typename T>
  __device__ T operator()(T data)
  {
    return cuda::std::floor(data);
  }
};

struct DeviceAbs {
  template <typename T>
  T __device__ operator()(T data)
    requires(cuda::std::is_signed_v<T>)
  {
    return cuda::std::abs(data);
  }
  template <typename T>
  T __device__ operator()(T data)
    requires(!cuda::std::is_signed_v<T>)
  {
    return data;
  }
};

// round float to int

struct DeviceRInt {
  template <typename T>
  __device__ T operator()(T data)
  {
    return cuda::std::rint(data);
  }
};

struct DeviceBitCount {
  template <typename T>
  int32_t __device__ operator()(T data)
  {
    if constexpr (cuda::std::is_same_v<T, bool>) {
      return static_cast<int32_t>(data);
    } else {
      using UnsignedT = cuda::std::make_unsigned_t<T>;
      return cuda::std::popcount(static_cast<UnsignedT>(data));
    }
  }
};

struct DeviceBitInvert {
  template <typename T>
  __device__ T operator()(T data)
  {
    return ~data;
  }
};

// logical op

struct DeviceNot {
  template <typename T>
  __device__ bool operator()(T data)
  {
    return !data;
  }
};

// negation

struct DeviceNegate {
  template <typename T>
  __device__ T operator()(T data)
  {
    return -data;
  }
};

// fixed_point ops

/*
 * Ceiling is calculated using integer division. When we divide by `n`, we get the integer part of
 * the `fixed_point` number. For a negative number, this is all that is needed since the ceiling
 * operation is defined as the least integer greater than the value. For a positive number, we may
 * need to round up if the `fixed_point` number has a fractional part. This is handled by comparing
 * the truncated value to the original value and if they are not equal, the result needs to be
 * incremented by `n`.
 */
template <typename T>
struct fixed_point_ceil {
  T n;  // 10^-scale (value required to determine integer part of fixed_point number)
  __device__ T operator()(T data)
  {
    T const a = (data / n) * n;                  // result of integer division
    return a + (data > 0 && a != data ? n : 0);  // add 1 if positive and not round number
  }
};

/*
 * Floor is calculated using integer division. When we divide by `n`, we get the integer part of
 * the `fixed_point` number. For a positive number, this is all that is needed since the floor
 * operation is defined as the greatest integer less than the value. For a negative number, we may
 * need to round down if the `fixed_point` number has a fractional part. This is handled by
 * comparing the truncated value to the original value and if they are not equal, the result needs
 * to be decremented by `n`.
 */
template <typename T>
struct fixed_point_floor {
  T n;  // 10^-scale (value required to determine integer part of fixed_point number)
  __device__ T operator()(T data)
  {
    T const a = (data / n) * n;                  // result of integer division
    return a - (data < 0 && a != data ? n : 0);  // subtract 1 if negative and not round number
  }
};

template <typename T>
struct fixed_point_abs {
  T n;
  __device__ T operator()(T data) { return numeric::detail::abs(data); }
};

template <typename T>
struct fixed_point_negate {
  T n;
  __device__ T operator()(T data) { return -data; }
};

template <typename T, template <typename> typename FixedPointFunctor>
std::unique_ptr<column> unary_op_with(column_view const& input,
                                      cuda::stream_ref stream,
                                      rmm::device_async_resource_ref mr)
{
  using Type                     = device_storage_type_t<T>;
  using FixedPointUnaryOpFunctor = FixedPointFunctor<Type>;

  // When scale is >= 0 and unary_operator is CEIL or FLOOR, the unary_operation is a no-op
  if (input.type().scale() >= 0 &&
      (std::is_same_v<FixedPointUnaryOpFunctor, fixed_point_ceil<Type>> ||
       std::is_same_v<FixedPointUnaryOpFunctor, fixed_point_floor<Type>>))
    return std::make_unique<cudf::column>(input, stream, mr);

  auto result = cudf::make_fixed_width_column(input.type(),
                                              input.size(),
                                              detail::copy_bitmask(input, stream, mr),
                                              input.null_count(),
                                              stream,
                                              mr);

  auto out_view = result->mutable_view();

  Type n = 10;
  for (int i = 1; i < -input.type().scale(); ++i) {
    n *= 10;
  }

  thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                    input.begin<Type>(),
                    input.end<Type>(),
                    out_view.begin<Type>(),
                    FixedPointUnaryOpFunctor{n});

  result->set_null_count(input.null_count());

  return result;
}

template <typename OutputType, typename UFN, typename InputIterator>
std::unique_ptr<cudf::column> transform_fn(InputIterator begin,
                                           InputIterator end,
                                           rmm::device_buffer&& null_mask,
                                           size_type null_count,
                                           cuda::stream_ref stream,
                                           rmm::device_async_resource_ref mr)
{
  auto const size = cudf::distance(begin, end);

  std::unique_ptr<cudf::column> output =
    make_fixed_width_column(data_type{type_to_id<OutputType>()},
                            size,
                            std::forward<rmm::device_buffer>(null_mask),
                            null_count,
                            stream,
                            mr);

  auto output_view = output->mutable_view();
  thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                    begin,
                    end,
                    output_view.begin<OutputType>(),
                    UFN{});
  output->set_null_count(null_count);
  return output;
}

template <typename T, typename UFN>
std::unique_ptr<cudf::column> transform_fn(cudf::column_view const& input,
                                           cuda::stream_ref stream,
                                           rmm::device_async_resource_ref mr)
{
  return transform_fn<T, UFN>(input.begin<T>(),
                              input.end<T>(),
                              detail::copy_bitmask(input, stream, mr),
                              input.null_count(),
                              stream,
                              mr);
}

template <typename T, typename UFN>
std::unique_ptr<cudf::column> transform_fn(cudf::dictionary_column_view const& input,
                                           cuda::stream_ref stream,
                                           rmm::device_async_resource_ref mr)
{
  auto dictionary_view = cudf::column_device_view::create(input.parent(), stream);
  auto dictionary_itr  = dictionary::detail::make_dictionary_iterator<T>(*dictionary_view);
  auto default_mr      = cudf::get_current_device_resource_ref();
  // call unary-op using temporary output buffer
  auto output = transform_fn<T, UFN>(dictionary_itr,
                                     dictionary_itr + input.size(),
                                     detail::copy_bitmask(input.parent(), stream, default_mr),
                                     input.null_count(),
                                     stream,
                                     default_mr);
  return cudf::dictionary::detail::encode(
    output->view(), dictionary::detail::get_indices_type_for_size(output->size()), stream, mr);
}

template <typename T>
struct ArithmeticOps {
  static constexpr bool is_supported() { return std::is_arithmetic_v<T>; }
};

template <typename T>
struct NegateOps {
  static constexpr bool is_supported() { return std::is_signed_v<T> || cudf::is_duration<T>(); }
};

template <typename T>
struct BitWiseOps {
  static constexpr bool is_supported() { return std::is_integral_v<T>; }
};

template <typename T>
struct FloatOnlyOps {
  static constexpr bool is_supported() { return std::is_floating_point_v<T>; }
};

/**
 * @brief Generic math-ops dispatcher
 *
 * Performs a transform on the input data using the operator defined by UFN.
 * The Supported type determines which types are allowed by the operator.
 *
 * @tparam UFN The actual operator to perform on the input data
 * @tparam Supported Contains the 'is_supported()' function
 */
template <typename UFN, template <typename> typename Supported>
struct MathOpDispatcher {
  template <typename T>
  std::unique_ptr<cudf::column> operator()(cudf::column_view const& input,
                                           cuda::stream_ref stream,
                                           rmm::device_async_resource_ref mr)
    requires(Supported<T>::is_supported())
  {
    return (input.type().id() == type_id::DICTIONARY32)
             ? transform_fn<T, UFN>(cudf::dictionary_column_view(input), stream, mr)
             : transform_fn<T, UFN>(input, stream, mr);
  }

  template <typename T, typename... Args>
  std::unique_ptr<cudf::column> operator()(Args&&...)
    requires(!Supported<T>::is_supported())
  {
    CUDF_FAIL("Unsupported data type for this operation");
  }
};

template <typename UFN>
struct BitwiseCountDispatcher {
 private:
  template <typename T>
  static constexpr bool is_supported()
  {
    return std::is_integral_v<T>;
  }

  // Always use int32_t as output type for bit count.
  using OutputType = int32_t;

 public:
  template <typename T>
  std::unique_ptr<cudf::column> operator()(cudf::column_view const& input,
                                           cuda::stream_ref stream,
                                           rmm::device_async_resource_ref mr)
    requires(is_supported<T>())
  {
    if (input.type().id() == type_id::DICTIONARY32) {
      auto dictionary_view = cudf::column_device_view::create(input, stream);
      auto dictionary_itr  = dictionary::detail::make_dictionary_iterator<T>(*dictionary_view);
      return transform_fn<OutputType, UFN>(dictionary_itr,
                                           dictionary_itr + input.size(),
                                           cudf::detail::copy_bitmask(input, stream, mr),
                                           input.null_count(),
                                           stream,
                                           mr);
    }
    return transform_fn<OutputType, UFN>(input.begin<T>(),
                                         input.end<T>(),
                                         cudf::detail::copy_bitmask(input, stream, mr),
                                         input.null_count(),
                                         stream,
                                         mr);
  }

  template <typename T, typename... Args>
  std::unique_ptr<cudf::column> operator()(Args&&...)
    requires(!is_supported<T>())
  {
    CUDF_FAIL("Unsupported datatype for operation");
  }
};

template <typename UFN>
struct LogicalOpDispatcher {
 private:
  template <typename T>
  static constexpr bool is_supported()
  {
    return std::is_arithmetic_v<T> || std::is_same_v<T, bool>;
  }

 public:
  template <typename T>
  std::unique_ptr<cudf::column> operator()(cudf::column_view const& input,
                                           cuda::stream_ref stream,
                                           rmm::device_async_resource_ref mr)
    requires(is_supported<T>())
  {
    if (input.type().id() == type_id::DICTIONARY32) {
      auto dictionary_view = cudf::column_device_view::create(input, stream);
      auto dictionary_itr  = dictionary::detail::make_dictionary_iterator<T>(*dictionary_view);
      return transform_fn<bool, UFN>(dictionary_itr,
                                     dictionary_itr + input.size(),
                                     cudf::detail::copy_bitmask(input, stream, mr),
                                     input.null_count(),
                                     stream,
                                     mr);
    }
    return transform_fn<bool, UFN>(input.begin<T>(),
                                   input.end<T>(),
                                   cudf::detail::copy_bitmask(input, stream, mr),
                                   input.null_count(),
                                   stream,
                                   mr);
  }

  template <typename T, typename... Args>
  std::unique_ptr<cudf::column> operator()(Args&&...)
    requires(!is_supported<T>())
  {
    CUDF_FAIL("Unsupported datatype for operation");
  }
};

struct FixedPointOpDispatcher {
  template <typename T, typename... Args>
  std::unique_ptr<column> operator()(Args&&...)
    requires(not cudf::is_fixed_point<T>())
  {
    CUDF_FAIL("FixedPointOpDispatcher only for fixed_point");
  }

  template <typename T>
  std::unique_ptr<column> operator()(column_view const& input,
                                     cudf::unary_operator op,
                                     cuda::stream_ref stream,
                                     rmm::device_async_resource_ref mr)
    requires(cudf::is_fixed_point<T>())
  {
    // clang-format off
    switch (op) {
      case cudf::unary_operator::CEIL:   return unary_op_with<T, fixed_point_ceil>(input, stream, mr);
      case cudf::unary_operator::FLOOR:  return unary_op_with<T, fixed_point_floor>(input, stream, mr);
      case cudf::unary_operator::ABS:    return unary_op_with<T, fixed_point_abs>(input, stream, mr);
      case cudf::unary_operator::NEGATE: return unary_op_with<T, fixed_point_negate>(input, stream, mr);
      default: CUDF_FAIL("Unsupported fixed_point unary operation");
    }
    // clang-format on
  }
};

}  // namespace

std::unique_ptr<cudf::column> unary_operation(cudf::column_view const& input,
                                              cudf::unary_operator op,
                                              cuda::stream_ref stream,
                                              rmm::device_async_resource_ref mr)
{
  if (cudf::is_fixed_point(input.type()))
    return type_dispatcher(input.type(), detail::FixedPointOpDispatcher{}, input, op, stream, mr);

  if (input.is_empty()) {
    if (op == cudf::unary_operator::NOT) { return make_empty_column(type_id::BOOL8); }
    if (op == cudf::unary_operator::BIT_COUNT) { return make_empty_column(type_id::INT32); }
    return empty_like(input);
  }

  // dispatch on the keys if dictionary saves a 2nd dispatch later
  auto dispatch_type = input.type().id() == type_id::DICTIONARY32
                         ? dictionary_column_view(input).keys().type()
                         : input.type();

  switch (op) {
    case cudf::unary_operator::SIN:
      return cudf::type_dispatcher(
        dispatch_type, MathOpDispatcher<DeviceSin, ArithmeticOps>{}, input, stream, mr);
    case cudf::unary_operator::COS:
      return cudf::type_dispatcher(
        dispatch_type, MathOpDispatcher<DeviceCos, ArithmeticOps>{}, input, stream, mr);
    case cudf::unary_operator::TAN:
      return cudf::type_dispatcher(
        dispatch_type, MathOpDispatcher<DeviceTan, ArithmeticOps>{}, input, stream, mr);
    case cudf::unary_operator::ARCSIN:
      return cudf::type_dispatcher(
        dispatch_type, MathOpDispatcher<DeviceArcSin, ArithmeticOps>{}, input, stream, mr);
    case cudf::unary_operator::ARCCOS:
      return cudf::type_dispatcher(
        dispatch_type, MathOpDispatcher<DeviceArcCos, ArithmeticOps>{}, input, stream, mr);
    case cudf::unary_operator::ARCTAN:
      return cudf::type_dispatcher(
        dispatch_type, MathOpDispatcher<DeviceArcTan, ArithmeticOps>{}, input, stream, mr);
    case cudf::unary_operator::SINH:
      return cudf::type_dispatcher(
        dispatch_type, MathOpDispatcher<DeviceSinH, ArithmeticOps>{}, input, stream, mr);
    case cudf::unary_operator::COSH:
      return cudf::type_dispatcher(
        dispatch_type, MathOpDispatcher<DeviceCosH, ArithmeticOps>{}, input, stream, mr);
    case cudf::unary_operator::TANH:
      return cudf::type_dispatcher(
        dispatch_type, MathOpDispatcher<DeviceTanH, ArithmeticOps>{}, input, stream, mr);
    case cudf::unary_operator::ARCSINH:
      return cudf::type_dispatcher(
        dispatch_type, MathOpDispatcher<DeviceArcSinH, ArithmeticOps>{}, input, stream, mr);
    case cudf::unary_operator::ARCCOSH:
      return cudf::type_dispatcher(
        dispatch_type, MathOpDispatcher<DeviceArcCosH, ArithmeticOps>{}, input, stream, mr);
    case cudf::unary_operator::ARCTANH:
      return cudf::type_dispatcher(
        dispatch_type, MathOpDispatcher<DeviceArcTanH, ArithmeticOps>{}, input, stream, mr);
    case cudf::unary_operator::EXP:
      return cudf::type_dispatcher(
        dispatch_type, MathOpDispatcher<DeviceExp, ArithmeticOps>{}, input, stream, mr);
    case cudf::unary_operator::LOG:
      return cudf::type_dispatcher(
        dispatch_type, MathOpDispatcher<DeviceLog, ArithmeticOps>{}, input, stream, mr);
    case cudf::unary_operator::SQRT:
      return cudf::type_dispatcher(
        dispatch_type, MathOpDispatcher<DeviceSqrt, ArithmeticOps>{}, input, stream, mr);
    case cudf::unary_operator::CBRT:
      return cudf::type_dispatcher(
        dispatch_type, MathOpDispatcher<DeviceCbrt, ArithmeticOps>{}, input, stream, mr);
    case cudf::unary_operator::CEIL:
      return cudf::type_dispatcher(
        dispatch_type, MathOpDispatcher<DeviceCeil, ArithmeticOps>{}, input, stream, mr);
    case cudf::unary_operator::FLOOR:
      return cudf::type_dispatcher(
        dispatch_type, MathOpDispatcher<DeviceFloor, ArithmeticOps>{}, input, stream, mr);
    case cudf::unary_operator::ABS:
      return cudf::type_dispatcher(
        dispatch_type, MathOpDispatcher<DeviceAbs, ArithmeticOps>{}, input, stream, mr);
    case cudf::unary_operator::RINT:
      return cudf::type_dispatcher(
        dispatch_type, MathOpDispatcher<DeviceRInt, FloatOnlyOps>{}, input, stream, mr);
    case cudf::unary_operator::BIT_COUNT:
      return cudf::type_dispatcher(
        dispatch_type, detail::BitwiseCountDispatcher<DeviceBitCount>{}, input, stream, mr);
    case cudf::unary_operator::BIT_INVERT:
      return cudf::type_dispatcher(
        dispatch_type, MathOpDispatcher<DeviceBitInvert, BitWiseOps>{}, input, stream, mr);
    case cudf::unary_operator::NOT:
      return cudf::type_dispatcher(
        dispatch_type, detail::LogicalOpDispatcher<DeviceNot>{}, input, stream, mr);
    case cudf::unary_operator::NEGATE:
      return cudf::type_dispatcher(
        dispatch_type, MathOpDispatcher<DeviceNegate, NegateOps>{}, input, stream, mr);
    default: CUDF_FAIL("Undefined unary operation");
  }
}

}  // namespace detail

std::unique_ptr<cudf::column> unary_operation(cudf::column_view const& input,
                                              cudf::unary_operator op,
                                              cuda::stream_ref stream,
                                              rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::unary_operation(input, op, stream, mr);
}

}  // namespace cudf
