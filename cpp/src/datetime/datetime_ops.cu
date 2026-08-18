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
#include <cudf/detail/datetime_ops.cuh>
#include <cudf/detail/indexalator.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>
#include <cudf/wrappers/durations.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/iterator>
#include <thrust/transform.h>

namespace cudf {
namespace datetime {
namespace detail {

enum class rounding_function {
  CEIL,   ///< Rounds up to the next integer multiple of the provided frequency
  FLOOR,  ///< Rounds down to the next integer multiple of the provided frequency
  ROUND   ///< Rounds to the nearest integer multiple of the provided frequency
};

template <datetime_component Component>
struct extract_component_operator {
  template <typename Timestamp>
  __device__ inline int16_t operator()(Timestamp const ts) const
  {
    using namespace cuda::std::chrono;

    auto days_since_epoch = floor<days>(ts);

    auto time_since_midnight = ts - days_since_epoch;

    if (time_since_midnight.count() < 0) { time_since_midnight += days(1); }

    auto const hrs_  = [&] { return duration_cast<hours>(time_since_midnight); };
    auto const mins_ = [&] { return duration_cast<minutes>(time_since_midnight) - hrs_(); };
    auto const secs_ = [&] {
      return duration_cast<seconds>(time_since_midnight) - hrs_() - mins_();
    };
    auto const millisecs_ = [&] {
      return duration_cast<milliseconds>(time_since_midnight) - hrs_() - mins_() - secs_();
    };
    auto const microsecs_ = [&] {
      return duration_cast<microseconds>(time_since_midnight) - hrs_() - mins_() - secs_() -
             millisecs_();
    };
    auto const nanosecs_ = [&] {
      return duration_cast<nanoseconds>(time_since_midnight) - hrs_() - mins_() - secs_() -
             millisecs_() - microsecs_();
    };

    switch (Component) {
      case datetime_component::YEAR:
        return static_cast<int>(year_month_day(days_since_epoch).year());
      case datetime_component::MONTH:
        return static_cast<unsigned>(year_month_day(days_since_epoch).month());
      case datetime_component::DAY:
        return static_cast<unsigned>(year_month_day(days_since_epoch).day());
      case datetime_component::WEEKDAY:
        return year_month_weekday(days_since_epoch).weekday().iso_encoding();
      case datetime_component::HOUR: return hrs_().count();
      case datetime_component::MINUTE: return mins_().count();
      case datetime_component::SECOND: return secs_().count();
      case datetime_component::MILLISECOND: return millisecs_().count();
      case datetime_component::MICROSECOND: return microsecs_().count();
      case datetime_component::NANOSECOND: return nanosecs_().count();
      default: return 0;
    }
  }
};

// This functor takes the rounding type as runtime info and dispatches to the ceil/floor/round
// function.
template <typename DurationType>
struct RoundFunctor {
  template <typename Timestamp>
  __device__ inline auto operator()(rounding_function round_kind, Timestamp dt)
  {
    switch (round_kind) {
      case rounding_function::CEIL: return cuda::std::chrono::ceil<DurationType>(dt);
      case rounding_function::FLOOR: return cuda::std::chrono::floor<DurationType>(dt);
      case rounding_function::ROUND: return cuda::std::chrono::round<DurationType>(dt);
      default: CUDF_UNREACHABLE("Unsupported rounding kind.");
    }
  }
};

struct RoundingDispatcher {
  rounding_function round_kind;
  rounding_frequency component;

  RoundingDispatcher(rounding_function round_kind, rounding_frequency component)
    : round_kind(round_kind), component(component)
  {
  }

  template <typename Timestamp>
  __device__ inline Timestamp operator()(Timestamp const ts) const
  {
    switch (component) {
      case rounding_frequency::DAY:
        return time_point_cast<typename Timestamp::duration>(
          RoundFunctor<duration_D>{}(round_kind, ts));
      case rounding_frequency::HOUR:
        return time_point_cast<typename Timestamp::duration>(
          RoundFunctor<duration_h>{}(round_kind, ts));
      case rounding_frequency::MINUTE:
        return time_point_cast<typename Timestamp::duration>(
          RoundFunctor<duration_m>{}(round_kind, ts));
      case rounding_frequency::SECOND:
        return time_point_cast<typename Timestamp::duration>(
          RoundFunctor<duration_s>{}(round_kind, ts));
      case rounding_frequency::MILLISECOND:
        return time_point_cast<typename Timestamp::duration>(
          RoundFunctor<duration_ms>{}(round_kind, ts));
      case rounding_frequency::MICROSECOND:
        return time_point_cast<typename Timestamp::duration>(
          RoundFunctor<duration_us>{}(round_kind, ts));
      case rounding_frequency::NANOSECOND:
        return time_point_cast<typename Timestamp::duration>(
          RoundFunctor<duration_ns>{}(round_kind, ts));
      default: CUDF_UNREACHABLE("Unsupported datetime rounding resolution.");
    }
  }
};

// Number of days until month indexed by leap year and month (0-based index)
static __device__ int16_t const days_until_month[2][13] = {
  {0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365},  // For non leap years
  {0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335, 366}   // For leap years
};

// Round up the date to the last day of the month and return the
// date only (without the time component)
struct extract_last_day_of_month {
  template <typename Timestamp>
  __device__ inline timestamp_D operator()(Timestamp const ts) const
  {
    using namespace cuda::std::chrono;
    year_month_day const ymd(floor<days>(ts));
    auto const ymdl = year_month_day_last{ymd.year() / ymd.month() / last};
    return timestamp_D{sys_days{ymdl}};
  }
};

// Extract the number of days of the month
// A similar operator to `extract_last_day_of_month`, except this returns
// an integer while the other returns a timestamp.
struct days_in_month_op {
  template <typename Timestamp>
  __device__ inline int16_t operator()(Timestamp const ts) const
  {
    using namespace cuda::std::chrono;
    auto const date = year_month_day(floor<days>(ts));
    auto const ymdl = year_month_day_last(date.year() / date.month() / last);
    return static_cast<int16_t>(unsigned{ymdl.day()});
  }
};

// Extract the day number of the year present in the timestamp
struct extract_day_num_of_year {
  template <typename Timestamp>
  __device__ inline int16_t operator()(Timestamp const ts) const
  {
    using namespace cuda::std::chrono;

    // Only has the days - time component is chopped off, which is what we want
    auto const days_since_epoch = floor<days>(ts);
    auto const date             = year_month_day(days_since_epoch);

    return days_until_month[date.year().is_leap()][unsigned{date.month()} - 1] +
           unsigned{date.day()};
  }
};

// Extract the quarter to which the timestamp belongs to
struct extract_quarter_op {
  template <typename Timestamp>
  __device__ inline int16_t operator()(Timestamp const ts) const
  {
    using namespace cuda::std::chrono;

    // Only has the days - time component is chopped off, which is what we want
    auto const days_since_epoch = floor<days>(ts);
    auto const date             = year_month_day(days_since_epoch);
    auto const month            = unsigned{date.month()};

    // (x + y - 1) / y = ceil(x/y), where x and y are unsigned. x = month, y = 3
    return (month + 2) / 3;
  }
};

// Returns true if the year is a leap year
struct is_leap_year_op {
  template <typename Timestamp>
  __device__ inline bool operator()(Timestamp const ts) const
  {
    using namespace cuda::std::chrono;
    auto const days_since_epoch = floor<days>(ts);
    auto const date             = year_month_day(days_since_epoch);
    return date.year().is_leap();
  }
};

// Specific function for applying ceil/floor/round date ops
struct dispatch_round {
  template <typename Timestamp>
  std::unique_ptr<cudf::column> operator()(rounding_function round_kind,
                                           rounding_frequency component,
                                           cudf::column_view const& column,
                                           rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr) const
    requires(cudf::is_timestamp<Timestamp>())
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
    requires(!cudf::is_timestamp<Timestamp>())
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
  void operator()(rmm::cuda_stream_view stream) const
    requires(!cudf::is_timestamp_t<Element>::value)
  {
    CUDF_FAIL("Cannot extract datetime component from non-timestamp column.");
  }

  template <typename Timestamp>
  void operator()(rmm::cuda_stream_view stream) const
    requires(cudf::is_timestamp_t<Timestamp>::value)
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
                                          rmm::cuda_stream_view stream,
                                          rmm::device_async_resource_ref mr)
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
    requires(!cudf::is_timestamp_t<Element>::value)
  {
    CUDF_FAIL("Cannot extract datetime component from non-timestamp column.");
  }

  template <typename Timestamp, typename MonthIterator>
  std::unique_ptr<column> operator()(column_view timestamp_column,
                                     MonthIterator months_begin,
                                     rmm::cuda_stream_view stream,
                                     rmm::device_async_resource_ref mr) const
    requires(cudf::is_timestamp_t<Timestamp>::value)
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
                        return add_calendrical_months_with_scale_back(
                          timestamp, cuda::std::chrono::months{months});
                      });
    return output;
  }
};

std::unique_ptr<column> add_calendrical_months(column_view const& timestamp_column,
                                               column_view const& months_column,
                                               rmm::cuda_stream_view stream,
                                               rmm::device_async_resource_ref mr)
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
                                               rmm::cuda_stream_view stream,
                                               rmm::device_async_resource_ref mr)
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
                                      rmm::cuda_stream_view stream,
                                      rmm::device_async_resource_ref mr)
{
  return cudf::type_dispatcher(
    column.type(), dispatch_round{}, round_kind, component, column, stream, mr);
}

std::unique_ptr<column> last_day_of_month(column_view const& column,
                                          rmm::cuda_stream_view stream,
                                          rmm::device_async_resource_ref mr)
{
  return detail::apply_datetime_op<detail::extract_last_day_of_month,
                                   cudf::type_id::TIMESTAMP_DAYS>(column, stream, mr);
}

std::unique_ptr<column> day_of_year(column_view const& column,
                                    rmm::cuda_stream_view stream,
                                    rmm::device_async_resource_ref mr)
{
  return detail::apply_datetime_op<detail::extract_day_num_of_year, cudf::type_id::INT16>(
    column, stream, mr);
}

std::unique_ptr<column> is_leap_year(column_view const& column,
                                     rmm::cuda_stream_view stream,
                                     rmm::device_async_resource_ref mr)
{
  return apply_datetime_op<is_leap_year_op, type_id::BOOL8>(column, stream, mr);
}

std::unique_ptr<column> days_in_month(column_view const& column,
                                      rmm::cuda_stream_view stream,
                                      rmm::device_async_resource_ref mr)
{
  return apply_datetime_op<days_in_month_op, type_id::INT16>(column, stream, mr);
}

std::unique_ptr<column> extract_quarter(column_view const& column,
                                        rmm::cuda_stream_view stream,
                                        rmm::device_async_resource_ref mr)
{
  return apply_datetime_op<extract_quarter_op, type_id::INT16>(column, stream, mr);
}

std::unique_ptr<cudf::column> extract_datetime_component(cudf::column_view const& column,
                                                         datetime_component component,
                                                         rmm::cuda_stream_view stream,
                                                         rmm::device_async_resource_ref mr)
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
                                       rmm::cuda_stream_view stream,
                                       rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::round_general(detail::rounding_function::CEIL, freq, column, stream, mr);
}

std::unique_ptr<column> floor_datetimes(column_view const& column,
                                        rounding_frequency freq,
                                        rmm::cuda_stream_view stream,
                                        rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::round_general(detail::rounding_function::FLOOR, freq, column, stream, mr);
}

std::unique_ptr<column> round_datetimes(column_view const& column,
                                        rounding_frequency freq,
                                        rmm::cuda_stream_view stream,
                                        rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::round_general(detail::rounding_function::ROUND, freq, column, stream, mr);
}

std::unique_ptr<cudf::column> extract_datetime_component(cudf::column_view const& column,
                                                         datetime_component component,
                                                         rmm::cuda_stream_view stream,
                                                         rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::extract_datetime_component(column, component, stream, mr);
}

std::unique_ptr<column> last_day_of_month(column_view const& column,
                                          rmm::cuda_stream_view stream,
                                          rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::last_day_of_month(column, stream, mr);
}

std::unique_ptr<column> day_of_year(column_view const& column,
                                    rmm::cuda_stream_view stream,
                                    rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::day_of_year(column, stream, mr);
}

std::unique_ptr<cudf::column> add_calendrical_months(cudf::column_view const& timestamp_column,
                                                     cudf::column_view const& months_column,
                                                     rmm::cuda_stream_view stream,
                                                     rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::add_calendrical_months(timestamp_column, months_column, stream, mr);
}

std::unique_ptr<cudf::column> add_calendrical_months(cudf::column_view const& timestamp_column,
                                                     cudf::scalar const& months,
                                                     rmm::cuda_stream_view stream,
                                                     rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::add_calendrical_months(timestamp_column, months, stream, mr);
}

std::unique_ptr<column> is_leap_year(column_view const& column,
                                     rmm::cuda_stream_view stream,
                                     rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::is_leap_year(column, stream, mr);
}

std::unique_ptr<column> days_in_month(column_view const& column,
                                      rmm::cuda_stream_view stream,
                                      rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::days_in_month(column, stream, mr);
}

std::unique_ptr<column> extract_quarter(column_view const& column,
                                        rmm::cuda_stream_view stream,
                                        rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::extract_quarter(column, stream, mr);
}

}  // namespace datetime
}  // namespace cudf
