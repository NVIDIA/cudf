/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/offsets_iterator_factory.cuh>
#include <cudf/strings/detail/convert/int_to_string.cuh>
#include <cudf/strings/detail/strings_children.cuh>
#include <cudf/strings/detail/utilities.cuh>
#include <cudf/types.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cuda/iterator>
#include <cuda/std/cmath>
#include <cuda/stream>
#include <thrust/for_each.h>

namespace cudf {
namespace io {
namespace detail {
namespace csv {

namespace {

// duration components timeparts structure
struct alignas(4) duration_component {
  int32_t day;         //-2,147,483,648 to 2,147,483,647
  int32_t nanosecond;  // 000000000 to 999999999
  int8_t hour;         // 00 to 23
  int8_t minute;       // 00 to 59
  int8_t second;       // 00 to 59
  bool is_negative;    // true/false
};

template <typename T>
__device__ void dissect_duration(T duration, duration_component* timeparts)
{
  timeparts->is_negative = (duration < T{0});
  timeparts->day         = cuda::std::chrono::floor<duration_D>(duration).count();

  if (cuda::std::is_same_v<T, duration_D>) return;

  // adjust for pandas format
  if (timeparts->is_negative) {
    duration =
      cuda::std::chrono::duration_cast<T>(duration % duration_D(1) + cuda::std::chrono::hours(24));
  }
  duration_s seconds = cuda::std::chrono::duration_cast<duration_s>(duration);
  timeparts->hour =
    (cuda::std::chrono::duration_cast<cuda::std::chrono::hours>(seconds) % duration_D(1)).count();
  timeparts->minute = (cuda::std::chrono::duration_cast<cuda::std::chrono::minutes>(seconds) %
                       cuda::std::chrono::hours(1))
                        .count();
  timeparts->second = (seconds % cuda::std::chrono::minutes(1)).count();
  if (not cuda::std::is_same_v<T, duration_s>) {
    timeparts->nanosecond =
      (cuda::std::chrono::duration_cast<duration_ns>(duration) % duration_s(1)).count();
  }
}

template <typename T>
struct duration_to_string_size_fn {
  column_device_view const d_durations;

  __device__ size_type operator()(size_type idx)
  {
    if (d_durations.is_null(idx)) return 0;
    auto duration                = d_durations.element<T>(idx);
    duration_component timeparts = {0};  // days, hours, minutes, seconds, nanoseconds(9)
    dissect_duration(duration, &timeparts);
    // [-] %d days [+]HH:MM:SS.mmmuuunnn
    return cudf::strings::detail::count_digits(timeparts.day) + 6 + timeparts.is_negative + 18;
  }
};

template <typename T>
struct duration_to_string_fn : public duration_to_string_size_fn<T> {
  cudf::detail::input_offsetalator d_offsets;
  char* d_chars;
  using duration_to_string_size_fn<T>::d_durations;

  duration_to_string_fn(column_device_view const d_durations,
                        cudf::detail::input_offsetalator d_offsets,
                        char* d_chars)
    : duration_to_string_size_fn<T>{d_durations}, d_offsets(d_offsets), d_chars(d_chars)
  {
  }

  __device__ char* int_to_2digitstr(int8_t value, char* str)
  {
    assert(value >= -99 && value <= 99);
    value  = cuda::std::abs(value);
    str[0] = '0' + value / 10;
    str[1] = '0' + value % 10;
    return str + 2;
  }

  inline __device__ char* day(char* ptr, duration_component const* timeparts)
  {
    cudf::strings::detail::integer_to_string(timeparts->day, ptr);
    return (ptr + cudf::strings::detail::count_digits(timeparts->day));
  }

  inline __device__ char* hour_24(char* ptr, duration_component const* timeparts)
  {
    return int_to_2digitstr(timeparts->hour, ptr);
  }

  inline __device__ char* minute(char* ptr, duration_component const* timeparts)
  {
    return int_to_2digitstr(timeparts->minute, ptr);
  }
  inline __device__ char* second(char* ptr, duration_component const* timeparts)
  {
    return int_to_2digitstr(timeparts->second, ptr);
  }

  inline __device__ char* nanosecond(char* ptr, duration_component const* timeparts)
  {
    auto value = timeparts->nanosecond;
    *ptr       = '.';
    for (int idx = 9; idx > 0; idx--) {
      *(ptr + idx) = '0' + cuda::std::abs(value % 10);
      value /= 10;
    }
    return ptr + 10;
  }

  inline __device__ char* pandas_format(duration_component const* timeparts, char* ptr)
  {
    // if (timeparts->is_negative) *ptr++ = '-';
    ptr = day(ptr, timeparts);
    ptr = cudf::strings::detail::copy_and_increment(ptr, " days ", 6);
    if (timeparts->is_negative) *ptr++ = '+';
    ptr    = hour_24(ptr, timeparts);
    *ptr++ = ':';
    ptr    = minute(ptr, timeparts);
    *ptr++ = ':';
    ptr    = second(ptr, timeparts);
    return nanosecond(ptr, timeparts);
  }

  __device__ void operator()(size_type idx)
  {
    if (d_durations.is_null(idx)) return;
    auto duration                = d_durations.template element<T>(idx);
    duration_component timeparts = {0};  // days, hours, minutes, seconds, nanoseconds(9)
    dissect_duration(duration, &timeparts);
    // convert to characters
    pandas_format(&timeparts, d_chars + d_offsets[idx]);
  }
};

/**
 * @brief This dispatch method is for converting durations into strings.
 *
 * The template function declaration ensures only duration types are used.
 */
struct dispatch_from_durations_fn {
  template <typename T>
  std::unique_ptr<column> operator()(column_view const& durations,
                                     cuda::stream_ref stream,
                                     rmm::device_async_resource_ref mr) const
    requires(cudf::is_duration<T>())
  {
    size_type strings_count = durations.size();
    auto column             = column_device_view::create(durations, stream);
    auto d_column           = *column;

    // copy null mask
    rmm::device_buffer null_mask = cudf::detail::copy_bitmask(durations, stream, mr);

    // build offsets column
    auto offsets_transformer_itr =
      cudf::detail::make_counting_transform_iterator(0, duration_to_string_size_fn<T>{d_column});
    auto [offsets_column, chars_bytes] = cudf::strings::detail::make_offsets_child_column(
      offsets_transformer_itr, offsets_transformer_itr + strings_count, stream, mr);
    auto d_new_offsets =
      cudf::detail::offsetalator_factory::make_input_iterator(offsets_column->view());

    // build chars column
    auto chars_data = rmm::device_uvector<char>(chars_bytes, stream, mr);
    auto d_chars    = chars_data.data();

    thrust::for_each_n(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                       cuda::counting_iterator<size_type>{0},
                       strings_count,
                       duration_to_string_fn<T>{d_column, d_new_offsets, d_chars});

    return make_strings_column(strings_count,
                               std::move(offsets_column),
                               chars_data.release(),
                               durations.null_count(),
                               std::move(null_mask));
  }

  // non-duration types throw an exception
  template <typename T>
  std::unique_ptr<column> operator()(column_view const&,
                                     cuda::stream_ref,
                                     rmm::device_async_resource_ref) const
    requires(not cudf::is_duration<T>())
  {
    CUDF_FAIL("Values for from_durations function must be a duration type.");
  }
};

}  // namespace

std::unique_ptr<column> pandas_format_durations(column_view const& durations,
                                                cuda::stream_ref stream,
                                                rmm::device_async_resource_ref mr)
{
  size_type strings_count = durations.size();
  if (strings_count == 0) return make_empty_column(type_id::STRING);

  return type_dispatcher(durations.type(), dispatch_from_durations_fn{}, durations, stream, mr);
}

}  // namespace csv
}  // namespace detail
}  // namespace io
}  // namespace cudf
