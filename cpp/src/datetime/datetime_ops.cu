/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/datetime.hpp>
#include <cudf/detail/datetime.hpp>
#include <cudf/detail/indexalator.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/operators/datetime.cuh>
#include <cudf/scalar/scalar.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/concepts.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>
#include <cudf/wrappers/durations.hpp>

#include <rmm/exec_policy.hpp>

#include <cuda/iterator>
#include <cuda/stream>
#include <thrust/transform.h>

namespace cudf {
namespace datetime {
namespace detail {

using rounding_function = cudf::detail::ops::datetime_rounding;

template <datetime_component Component>
struct extract_component_operator {
  template <typename Timestamp>
  __device__ int16_t operator()(Timestamp ts) const
  {
    return cudf::detail::ops::extract_datetime_component<Component>(ts);
  }
};

struct RoundingDispatcher {
  rounding_function round_kind;
  rounding_frequency component;

  template <typename Timestamp>
  __device__ Timestamp operator()(Timestamp ts) const
  {
    return cudf::detail::ops::round_datetime(ts, round_kind, component);
  }
};

#define CUDF_DATETIME_OPERATOR(Name, Function, Result) \
  struct Name {                                        \
    template <typename Timestamp>                      \
    __device__ Result operator()(Timestamp ts) const   \
    {                                                  \
      return cudf::detail::ops::Function(ts);          \
    }                                                  \
  };

CUDF_DATETIME_OPERATOR(extract_last_day_of_month, last_day_of_month, timestamp_D)
CUDF_DATETIME_OPERATOR(days_in_month_op, days_in_month, int16_t)
CUDF_DATETIME_OPERATOR(extract_day_num_of_year, day_of_year, int16_t)
CUDF_DATETIME_OPERATOR(extract_quarter_op, extract_quarter, int16_t)
CUDF_DATETIME_OPERATOR(is_leap_year_op, is_leap_year, bool)

#undef CUDF_DATETIME_OPERATOR

// Specific function for applying ceil/floor/round date ops
struct dispatch_round {
  template <cudf::concepts::timestamp Timestamp>
  std::unique_ptr<cudf::column> operator()(rounding_function round_kind,
                                           rounding_frequency component,
                                           cudf::column_view const& column,
                                           cuda::stream_ref stream,
                                           cudf::memory_resources mr) const
  {
    auto size            = column.size();
    auto output_col_type = data_type{cudf::type_to_id<Timestamp>()};

    // Return an empty column if source column is empty
    if (size == 0) return make_empty_column(output_col_type);

    auto output = make_fixed_width_column(output_col_type,
                                          size,
                                          cudf::detail::copy_bitmask(column, stream, mr),
                                          column.null_count(),
                                          stream,
                                          mr);

    thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                      column.begin<Timestamp>(),
                      column.end<Timestamp>(),
                      output->mutable_view().begin<Timestamp>(),
                      RoundingDispatcher{round_kind, component});

    output->set_null_count(column.null_count());

    return output;
  }

  template <typename Timestamp, typename... Args>
  std::unique_ptr<cudf::column> operator()(Args&&...)
    requires(not cudf::concepts::timestamp<Timestamp>)
  {
    CUDF_FAIL("Must be cudf::timestamp");
  }
};

// Apply the functor for every element/row in the input column to create the output column
template <typename TransformFunctor, typename OutputColT>
struct launch_functor {
  column_view input;
  mutable_column_view output;

  launch_functor(column_view inp, mutable_column_view out) : input(inp), output(out) {}

  template <typename Element>
  void operator()(cuda::stream_ref stream) const
  {
    CUDF_FAIL("Cannot extract datetime component from non-timestamp column.");
  }

  template <cudf::concepts::timestamp Timestamp>
  void operator()(cuda::stream_ref stream) const
  {
    thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                      input.begin<Timestamp>(),
                      input.end<Timestamp>(),
                      output.begin<OutputColT>(),
                      TransformFunctor{});
  }
};

// Create an output column by applying the functor to every element from the input column
template <typename TransformFunctor, cudf::type_id OutputColCudfT>
std::unique_ptr<column> apply_datetime_op(column_view const& column,
                                          cuda::stream_ref stream,
                                          cudf::memory_resources mr)
{
  CUDF_EXPECTS(is_timestamp(column.type()), "Column type should be timestamp");
  auto size            = column.size();
  auto output_col_type = data_type{OutputColCudfT};

  // Return an empty column if source column is empty
  if (size == 0) return make_empty_column(output_col_type);

  auto output = make_fixed_width_column(output_col_type,
                                        size,
                                        cudf::detail::copy_bitmask(column, stream, mr),
                                        column.null_count(),
                                        stream,
                                        mr);
  auto launch = launch_functor<TransformFunctor, cudf::id_to_type<OutputColCudfT>>{
    column, static_cast<mutable_column_view>(*output)};

  type_dispatcher(column.type(), launch, stream);

  return output;
}

struct add_calendrical_months_functor {
  template <typename Element, typename... Args>
  std::unique_ptr<column> operator()(Args&&...) const
    requires(not cudf::concepts::timestamp<Element>)
  {
    CUDF_FAIL("Cannot extract datetime component from non-timestamp column.");
  }

  template <cudf::concepts::timestamp Timestamp, typename MonthIterator>
  std::unique_ptr<column> operator()(column_view timestamp_column,
                                     MonthIterator months_begin,
                                     cuda::stream_ref stream,
                                     cudf::memory_resources mr) const
  {
    auto size            = timestamp_column.size();
    auto output_col_type = timestamp_column.type();

    // Return an empty column if source column is empty
    if (size == 0) return make_empty_column(output_col_type);

    // The nullmask of `output` cannot be determined without information from
    // the `months` type (column or scalar). Therefore, it is initialized as
    // `UNALLOCATED` and assigned at a later stage.
    auto output =
      make_fixed_width_column(output_col_type, size, mask_state::UNALLOCATED, stream, mr);
    auto output_mview = output->mutable_view();

    thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                      timestamp_column.begin<Timestamp>(),
                      timestamp_column.end<Timestamp>(),
                      months_begin,
                      output->mutable_view().begin<Timestamp>(),
                      [] __device__(auto& timestamp, auto& months) {
                        return cudf::detail::ops::add_calendrical_months(
                          timestamp, cuda::std::chrono::months{months});
                      });
    return output;
  }
};

std::unique_ptr<column> add_calendrical_months(column_view const& timestamp_column,
                                               column_view const& months_column,
                                               cuda::stream_ref stream,
                                               cudf::memory_resources mr)
{
  CUDF_EXPECTS(is_timestamp(timestamp_column.type()), "Column type should be timestamp");
  CUDF_EXPECTS(
    months_column.type().id() == type_id::INT16 or months_column.type().id() == type_id::INT32,
    "Months column type should be INT16 or INT32.");
  CUDF_EXPECTS(timestamp_column.size() == months_column.size(),
               "Timestamp and months column should be of the same size");

  auto const months_begin_iter =
    cudf::detail::indexalator_factory::make_input_iterator(months_column);
  auto output = type_dispatcher(timestamp_column.type(),
                                add_calendrical_months_functor{},
                                timestamp_column,
                                months_begin_iter,
                                stream,
                                mr);

  auto [output_null_mask, null_count] =
    cudf::detail::bitmask_and(table_view{{timestamp_column, months_column}}, stream, mr);
  output->set_null_mask(std::move(output_null_mask), null_count);
  return output;
}

std::unique_ptr<column> add_calendrical_months(column_view const& timestamp_column,
                                               scalar const& months,
                                               cuda::stream_ref stream,
                                               cudf::memory_resources mr)
{
  CUDF_EXPECTS(is_timestamp(timestamp_column.type()), "Column type should be timestamp");
  CUDF_EXPECTS(months.type().id() == type_id::INT16 or months.type().id() == type_id::INT32,
               "Months type should be INT16 or INT32");

  if (months.is_valid(stream)) {
    auto const months_begin_iter = cuda::make_permutation_iterator(
      cudf::detail::indexalator_factory::make_input_iterator(months),
      cuda::make_constant_iterator(0));
    auto output = type_dispatcher(timestamp_column.type(),
                                  add_calendrical_months_functor{},
                                  timestamp_column,
                                  months_begin_iter,
                                  stream,
                                  mr);
    output->set_null_mask(cudf::detail::copy_bitmask(timestamp_column, stream, mr),
                          timestamp_column.null_count());
    return output;
  } else {
    return make_timestamp_column(
      timestamp_column.type(), timestamp_column.size(), mask_state::ALL_NULL, stream, mr);
  }
}

std::unique_ptr<column> round_general(rounding_function round_kind,
                                      rounding_frequency component,
                                      column_view const& column,
                                      cuda::stream_ref stream,
                                      cudf::memory_resources mr)
{
  return cudf::type_dispatcher(
    column.type(), dispatch_round{}, round_kind, component, column, stream, mr);
}

std::unique_ptr<column> last_day_of_month(column_view const& column,
                                          cuda::stream_ref stream,
                                          cudf::memory_resources mr)
{
  return detail::apply_datetime_op<detail::extract_last_day_of_month,
                                   cudf::type_id::TIMESTAMP_DAYS>(column, stream, mr);
}

std::unique_ptr<column> day_of_year(column_view const& column,
                                    cuda::stream_ref stream,
                                    cudf::memory_resources mr)
{
  return detail::apply_datetime_op<detail::extract_day_num_of_year, cudf::type_id::INT16>(
    column, stream, mr);
}

std::unique_ptr<column> is_leap_year(column_view const& column,
                                     cuda::stream_ref stream,
                                     cudf::memory_resources mr)
{
  return apply_datetime_op<is_leap_year_op, type_id::BOOL8>(column, stream, mr);
}

std::unique_ptr<column> days_in_month(column_view const& column,
                                      cuda::stream_ref stream,
                                      cudf::memory_resources mr)
{
  return apply_datetime_op<days_in_month_op, type_id::INT16>(column, stream, mr);
}

std::unique_ptr<column> extract_quarter(column_view const& column,
                                        cuda::stream_ref stream,
                                        cudf::memory_resources mr)
{
  return apply_datetime_op<extract_quarter_op, type_id::INT16>(column, stream, mr);
}

std::unique_ptr<cudf::column> extract_datetime_component(cudf::column_view const& column,
                                                         datetime_component component,
                                                         cuda::stream_ref stream,
                                                         cudf::memory_resources mr)
{
#define extract(field)                                                                 \
  case field:                                                                          \
    return apply_datetime_op<extract_component_operator<field>, cudf::type_id::INT16>( \
      column, stream, mr)

  switch (component) {
    extract(datetime_component::YEAR);
    extract(datetime_component::MONTH);
    extract(datetime_component::DAY);
    extract(datetime_component::WEEKDAY);
    extract(datetime_component::HOUR);
    extract(datetime_component::MINUTE);
    extract(datetime_component::SECOND);
    extract(datetime_component::MILLISECOND);
    extract(datetime_component::MICROSECOND);
    extract(datetime_component::NANOSECOND);
    default: CUDF_FAIL("Unsupported datetime component.");
  }
#undef extract
}

}  // namespace detail

std::unique_ptr<column> ceil_datetimes(column_view const& column,
                                       rounding_frequency freq,
                                       cuda::stream_ref stream,
                                       cudf::memory_resources mr)
{
  CUDF_FUNC_RANGE();
  return detail::round_general(detail::rounding_function::CEIL, freq, column, stream, mr);
}

std::unique_ptr<column> floor_datetimes(column_view const& column,
                                        rounding_frequency freq,
                                        cuda::stream_ref stream,
                                        cudf::memory_resources mr)
{
  CUDF_FUNC_RANGE();
  return detail::round_general(detail::rounding_function::FLOOR, freq, column, stream, mr);
}

std::unique_ptr<column> round_datetimes(column_view const& column,
                                        rounding_frequency freq,
                                        cuda::stream_ref stream,
                                        cudf::memory_resources mr)
{
  CUDF_FUNC_RANGE();
  return detail::round_general(detail::rounding_function::ROUND, freq, column, stream, mr);
}

std::unique_ptr<cudf::column> extract_datetime_component(cudf::column_view const& column,
                                                         datetime_component component,
                                                         cuda::stream_ref stream,
                                                         cudf::memory_resources mr)
{
  CUDF_FUNC_RANGE();
  return detail::extract_datetime_component(column, component, stream, mr);
}

std::unique_ptr<column> last_day_of_month(column_view const& column,
                                          cuda::stream_ref stream,
                                          cudf::memory_resources mr)
{
  CUDF_FUNC_RANGE();
  return detail::last_day_of_month(column, stream, mr);
}

std::unique_ptr<column> day_of_year(column_view const& column,
                                    cuda::stream_ref stream,
                                    cudf::memory_resources mr)
{
  CUDF_FUNC_RANGE();
  return detail::day_of_year(column, stream, mr);
}

std::unique_ptr<cudf::column> add_calendrical_months(cudf::column_view const& timestamp_column,
                                                     cudf::column_view const& months_column,
                                                     cuda::stream_ref stream,
                                                     cudf::memory_resources mr)
{
  CUDF_FUNC_RANGE();
  return detail::add_calendrical_months(timestamp_column, months_column, stream, mr);
}

std::unique_ptr<cudf::column> add_calendrical_months(cudf::column_view const& timestamp_column,
                                                     cudf::scalar const& months,
                                                     cuda::stream_ref stream,
                                                     cudf::memory_resources mr)
{
  CUDF_FUNC_RANGE();
  return detail::add_calendrical_months(timestamp_column, months, stream, mr);
}

std::unique_ptr<column> is_leap_year(column_view const& column,
                                     cuda::stream_ref stream,
                                     cudf::memory_resources mr)
{
  CUDF_FUNC_RANGE();
  return detail::is_leap_year(column, stream, mr);
}

std::unique_ptr<column> days_in_month(column_view const& column,
                                      cuda::stream_ref stream,
                                      cudf::memory_resources mr)
{
  CUDF_FUNC_RANGE();
  return detail::days_in_month(column, stream, mr);
}

std::unique_ptr<column> extract_quarter(column_view const& column,
                                        cuda::stream_ref stream,
                                        cudf::memory_resources mr)
{
  CUDF_FUNC_RANGE();
  return detail::extract_quarter(column, stream, mr);
}

}  // namespace datetime
}  // namespace cudf
