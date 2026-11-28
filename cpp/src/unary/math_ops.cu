/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column_device_view.cuh>
#include <cudf/copying.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/operators/arithmetic.cuh>
#include <cudf/detail/operators/bitwise.cuh>
#include <cudf/detail/operators/logic.cuh>
#include <cudf/detail/operators/math.cuh>
#include <cudf/detail/operators/trigonometric.cuh>
#include <cudf/detail/unary.hpp>
#include <cudf/dictionary/detail/encode.hpp>
#include <cudf/dictionary/detail/iterator.cuh>
#include <cudf/utilities/concepts.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/exec_policy.hpp>

#include <cuda/std/type_traits>
#include <cuda/stream>
#include <thrust/transform.h>

#include <transform/checked_arithmetic.hpp>

namespace cudf {
namespace detail {
namespace {

// Column unary operations retain the input type, including for integral inputs to scalar math.
#define CUDF_UNARY_MATH_OPERATOR(Name, Function)         \
  struct Name {                                          \
    template <typename T>                                \
    __device__ T operator()(T data) const                \
    {                                                    \
      if constexpr (cuda::std::is_integral_v<T>) {       \
        return ops::Function(static_cast<double>(data)); \
      } else {                                           \
        return ops::Function(data);                      \
      }                                                  \
    }                                                    \
  };

CUDF_UNARY_MATH_OPERATOR(DeviceSin, sin)
CUDF_UNARY_MATH_OPERATOR(DeviceCos, cos)
CUDF_UNARY_MATH_OPERATOR(DeviceTan, tan)
CUDF_UNARY_MATH_OPERATOR(DeviceArcSin, arcsin)
CUDF_UNARY_MATH_OPERATOR(DeviceArcCos, arccos)
CUDF_UNARY_MATH_OPERATOR(DeviceArcTan, arctan)
CUDF_UNARY_MATH_OPERATOR(DeviceSinH, sinh)
CUDF_UNARY_MATH_OPERATOR(DeviceCosH, cosh)
CUDF_UNARY_MATH_OPERATOR(DeviceTanH, tanh)
CUDF_UNARY_MATH_OPERATOR(DeviceArcSinH, arcsinh)
CUDF_UNARY_MATH_OPERATOR(DeviceArcCosH, arccosh)
CUDF_UNARY_MATH_OPERATOR(DeviceArcTanH, arctanh)
CUDF_UNARY_MATH_OPERATOR(DeviceExp, exp)
CUDF_UNARY_MATH_OPERATOR(DeviceLog, log)
CUDF_UNARY_MATH_OPERATOR(DeviceSqrt, sqrt)
CUDF_UNARY_MATH_OPERATOR(DeviceCbrt, cbrt)
CUDF_UNARY_MATH_OPERATOR(DeviceCeil, ceil)
CUDF_UNARY_MATH_OPERATOR(DeviceFloor, floor)
CUDF_UNARY_MATH_OPERATOR(DeviceRInt, rint)

#undef CUDF_UNARY_MATH_OPERATOR

struct DeviceAbs {
  template <typename T>
  __device__ T operator()(T data) const
  {
    if constexpr (cuda::std::is_same_v<T, bool>) {
      return data;
    } else {
      return ops::abs(data);
    }
  }
};

struct DeviceBitCount {
  template <typename T>
  __device__ int32_t operator()(T data) const
  {
    return ops::bit_count(data);
  }
};

struct DeviceBitInvert {
  template <typename T>
  __device__ T operator()(T data) const
  {
    return ops::bitwise_invert(+data);
  }
};

struct DeviceNot {
  template <typename T>
  __device__ bool operator()(T data) const
  {
    return ops::logical_not(data);
  }
};

struct DeviceNegate {
  template <typename T>
  __device__ T operator()(T data) const
  {
    return ops::neg(data);
  }
};

// Fixed-point rounding operates on the stored representation with a precomputed power of ten.
template <typename T>
struct fixed_point_ceil {
  T n;
  __device__ T operator()(T data) const { return ops::ceil_to_multiple(data, n); }
};

template <typename T>
struct fixed_point_floor {
  T n;
  __device__ T operator()(T data) const { return ops::floor_to_multiple(data, n); }
};

template <typename T>
struct fixed_point_abs {
  __device__ T operator()(T data) const { return numeric::detail::abs(data); }
};

template <typename T>
struct fixed_point_negate {
  __device__ T operator()(T data) const { return ops::neg(data); }
};

template <typename T, template <typename> typename FixedPointFunctor>
std::unique_ptr<column> unary_op_with(column_view const& input,
                                      cuda::stream_ref stream,
                                      cudf::memory_resources mr)
{
  using Type                     = device_storage_type_t<T>;
  using FixedPointUnaryOpFunctor = FixedPointFunctor<Type>;
  constexpr bool is_rounding =
    cuda::std::is_same_v<FixedPointUnaryOpFunctor, fixed_point_ceil<Type>> ||
    cuda::std::is_same_v<FixedPointUnaryOpFunctor, fixed_point_floor<Type>>;

  // When scale is >= 0 and unary_operator is CEIL or FLOOR, the unary_operation is a no-op
  if (is_rounding && input.type().scale() >= 0)
    return std::make_unique<cudf::column>(input, stream, mr);

  auto result = cudf::make_fixed_width_column(input.type(),
                                              input.size(),
                                              detail::copy_bitmask(input, stream, mr),
                                              input.null_count(),
                                              stream,
                                              mr);

  auto out_view = result->mutable_view();

  FixedPointUnaryOpFunctor op{};
  if constexpr (is_rounding) {
    op.n = numeric::detail::ipow<Type, numeric::Radix::BASE_10>(-input.type().scale());
  }

  thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                    input.begin<Type>(),
                    input.end<Type>(),
                    out_view.begin<Type>(),
                    op);

  result->set_null_count(input.null_count());

  return result;
}

template <typename OutputType, typename UFN, typename InputIterator>
std::unique_ptr<cudf::column> transform_fn(InputIterator begin,
                                           InputIterator end,
                                           cuda::device_buffer<cudf::bitmask_type>&& null_mask,
                                           size_type null_count,
                                           cuda::stream_ref stream,
                                           cudf::memory_resources mr)
{
  auto const size = cudf::distance(begin, end);

  std::unique_ptr<cudf::column> output =
    make_fixed_width_column(data_type{type_to_id<OutputType>()},
                            size,
                            std::forward<cuda::device_buffer<cudf::bitmask_type>>(null_mask),
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
                                           cudf::memory_resources mr)
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
                                           cudf::memory_resources mr)
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
  static constexpr bool is_supported() { return cuda::std::is_arithmetic_v<T>; }
};

template <typename T>
struct NegateOps {
  static constexpr bool is_supported()
  {
    return cuda::std::is_signed_v<T> || cudf::is_duration<T>();
  }
};

template <typename T>
struct BitWiseOps {
  static constexpr bool is_supported() { return cuda::std::is_integral_v<T>; }
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
                                           cudf::memory_resources mr)
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
    return cuda::std::is_integral_v<T>;
  }

  // Always use int32_t as output type for bit count.
  using OutputType = int32_t;

 public:
  template <typename T>
  std::unique_ptr<cudf::column> operator()(cudf::column_view const& input,
                                           cuda::stream_ref stream,
                                           cudf::memory_resources mr)
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
    return cuda::std::is_arithmetic_v<T> || std::is_same_v<T, bool>;
  }

 public:
  template <typename T>
  std::unique_ptr<cudf::column> operator()(cudf::column_view const& input,
                                           cuda::stream_ref stream,
                                           cudf::memory_resources mr)
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
    requires(not cudf::concepts::fixed_point<T>)
  {
    CUDF_FAIL("FixedPointOpDispatcher only for fixed_point");
  }

  template <cudf::concepts::fixed_point T>
  std::unique_ptr<column> operator()(column_view const& input,
                                     cudf::unary_operator op,
                                     cuda::stream_ref stream,
                                     cudf::memory_resources mr)
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
                                              cudf::memory_resources mr)
{
  if (checked_arithmetic::is_checked(op)) {
    // Omitting an error policy for a checked operator means propagate any row error.
    return checked_arithmetic::unary_operation(input, op, error_policy::PROPAGATE, stream, mr);
  }

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
                                              cudf::memory_resources mr)
{
  CUDF_FUNC_RANGE();
  return detail::unary_operation(input, op, stream, mr);
}

std::unique_ptr<cudf::column> unary_operation(cudf::column_view const& input,
                                              cudf::unary_operator op,
                                              cudf::error_policy policy,
                                              cuda::stream_ref stream,
                                              cudf::memory_resources mr)
{
  CUDF_FUNC_RANGE();
  return detail::checked_arithmetic::unary_operation(input, op, policy, stream, mr);
}

}  // namespace cudf
