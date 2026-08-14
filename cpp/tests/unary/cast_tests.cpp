/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/detail/iterator.cuh>
#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/wrappers/timestamps.hpp>

#include <cuda/iterator>
#include <thrust/host_vector.h>

#include <vector>

static auto const test_timestamps_D = std::vector<int32_t>{
  -1528,  // 1965-10-26 GMT
  17716,  // 2018-07-04 GMT
  19382,  // 2023-01-25 GMT
};

static auto const test_timestamps_s = std::vector<int64_t>{
  -131968728,  // 1965-10-26 14:01:12 GMT
  1530705600,  // 2018-07-04 12:00:00 GMT
  1674631932,  // 2023-01-25 07:32:12 GMT
};

static auto const test_timestamps_ms = std::vector<int64_t>{
  -131968727238,  // 1965-10-26 14:01:12.762 GMT
  1530705600000,  // 2018-07-04 12:00:00.000 GMT
  1674631932929,  // 2023-01-25 07:32:12.929 GMT
};

static auto const test_timestamps_us = std::vector<int64_t>{
  -131968727238000,  // 1965-10-26 14:01:12.762000000 GMT
  1530705600000000,  // 2018-07-04 12:00:00.000000000 GMT
  1674631932929000,  // 2023-01-25 07:32:12.929000000 GMT
};

static auto const test_timestamps_ns = std::vector<int64_t>{
  -131968727238000000,  // 1965-10-26 14:01:12.762000000 GMT
  1530705600000000000,  // 2018-07-04 12:00:00.000000000 GMT
  1674631932929000000,  // 2023-01-25 07:32:12.929000000 GMT
};

static auto const test_durations_D  = test_timestamps_D;
static auto const test_durations_s  = test_timestamps_s;
static auto const test_durations_ms = test_timestamps_ms;
static auto const test_durations_us = test_timestamps_us;
static auto const test_durations_ns = test_timestamps_ns;

template <typename T, typename R>
inline auto make_column(std::vector<R> data)
{
  return cudf::test::fixed_width_column_wrapper<T, R>(data.begin(), data.end());
}

template <typename T, typename R>
inline auto make_column(std::vector<R> data, std::vector<bool> mask)
{
  return cudf::test::fixed_width_column_wrapper<T, R>(data.begin(), data.end(), mask.begin());
}

inline cudf::column make_exp_chrono_column(cudf::type_id type_id)
{
  switch (type_id) {
    case cudf::type_id::TIMESTAMP_DAYS:
      return cudf::column(
        cudf::data_type{type_id},
        test_timestamps_D.size(),
        rmm::device_buffer{test_timestamps_D.data(),
                           test_timestamps_D.size() * sizeof(test_timestamps_D.front()),
                           cudf::get_default_stream()},

        rmm::device_buffer{},
        0);
    case cudf::type_id::TIMESTAMP_SECONDS:
      return cudf::column(
        cudf::data_type{type_id},
        test_timestamps_s.size(),
        rmm::device_buffer{test_timestamps_s.data(),
                           test_timestamps_s.size() * sizeof(test_timestamps_s.front()),
                           cudf::get_default_stream()},
        rmm::device_buffer{},
        0);
    case cudf::type_id::TIMESTAMP_MILLISECONDS:
      return cudf::column(
        cudf::data_type{type_id},
        test_timestamps_ms.size(),
        rmm::device_buffer{test_timestamps_ms.data(),
                           test_timestamps_ms.size() * sizeof(test_timestamps_ms.front()),
                           cudf::get_default_stream()},
        rmm::device_buffer{},
        0);
    case cudf::type_id::TIMESTAMP_MICROSECONDS:
      return cudf::column(
        cudf::data_type{type_id},
        test_timestamps_us.size(),
        rmm::device_buffer{test_timestamps_us.data(),
                           test_timestamps_us.size() * sizeof(test_timestamps_us.front()),
                           cudf::get_default_stream()},
        rmm::device_buffer{},
        0);
    case cudf::type_id::TIMESTAMP_NANOSECONDS:
      return cudf::column(
        cudf::data_type{type_id},
        test_timestamps_ns.size(),
        rmm::device_buffer{test_timestamps_ns.data(),
                           test_timestamps_ns.size() * sizeof(test_timestamps_ns.front()),
                           cudf::get_default_stream()},
        rmm::device_buffer{},
        0);
    case cudf::type_id::DURATION_DAYS:
      return cudf::column(
        cudf::data_type{type_id},
        test_durations_D.size(),
        rmm::device_buffer{test_durations_D.data(),
                           test_durations_D.size() * sizeof(test_durations_D.front()),
                           cudf::get_default_stream()},
        rmm::device_buffer{},
        0);
    case cudf::type_id::DURATION_SECONDS:
      return cudf::column(
        cudf::data_type{type_id},
        test_durations_s.size(),
        rmm::device_buffer{test_durations_s.data(),
                           test_durations_s.size() * sizeof(test_durations_s.front()),
                           cudf::get_default_stream()},
        rmm::device_buffer{},
        0);
    case cudf::type_id::DURATION_MILLISECONDS:
      return cudf::column(
        cudf::data_type{type_id},
        test_durations_ms.size(),
        rmm::device_buffer{test_durations_ms.data(),
                           test_durations_ms.size() * sizeof(test_durations_ms.front()),
                           cudf::get_default_stream()},
        rmm::device_buffer{},
        0);
    case cudf::type_id::DURATION_MICROSECONDS:
      return cudf::column(
        cudf::data_type{type_id},
        test_durations_us.size(),
        rmm::device_buffer{test_durations_us.data(),
                           test_durations_us.size() * sizeof(test_durations_us.front()),
                           cudf::get_default_stream()},
        rmm::device_buffer{},
        0);
    case cudf::type_id::DURATION_NANOSECONDS:
      return cudf::column(
        cudf::data_type{type_id},
        test_durations_ns.size(),
        rmm::device_buffer{test_durations_ns.data(),
                           test_durations_ns.size() * sizeof(test_durations_ns.front()),
                           cudf::get_default_stream()},
        rmm::device_buffer{},
        0);
    default: CUDF_FAIL("Unsupported type_id");
  }
};

template <typename T, typename R>
inline auto make_column(thrust::host_vector<R> data)
{
  return cudf::test::fixed_width_column_wrapper<T, R>(data.begin(), data.end());
}

template <typename T, typename R>
inline auto make_column(thrust::host_vector<R> data, thrust::host_vector<bool> mask)
{
  return cudf::test::fixed_width_column_wrapper<T, R>(data.begin(), data.end(), mask.begin());
}

template <typename T, typename R>
void validate_cast_result(cudf::column_view expected, cudf::column_view actual)
{
  using namespace cudf::test;
  // round-trip through the host because sizeof(T) may not equal sizeof(R)
  auto [h_data, null_mask] = to_host<T>(expected);
  if (null_mask.empty()) {
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(make_column<R, T>(h_data), actual);
  } else {
    thrust::host_vector<bool> h_null_mask(expected.size());
    for (cudf::size_type i = 0; i < expected.size(); ++i) {
      h_null_mask[i] = cudf::bit_is_set(null_mask.data(), i);
    }
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(make_column<R, T>(h_data, h_null_mask), actual);
  }
}

template <typename T>
inline auto make_data_type()
{
  return cudf::data_type{cudf::type_to_id<T>()};
}

TEST(IsSupportedCast, UnsupportedTypes)
{
  auto const to_int32 = cudf::data_type{cudf::type_id::INT32};
  EXPECT_FALSE(cudf::is_supported_cast(cudf::data_type{cudf::type_id::STRING}, to_int32));
  EXPECT_FALSE(cudf::is_supported_cast(cudf::data_type{cudf::type_id::LIST}, to_int32));
  EXPECT_FALSE(cudf::is_supported_cast(cudf::data_type{cudf::type_id::STRUCT}, to_int32));
  EXPECT_FALSE(cudf::is_supported_cast(cudf::data_type{cudf::type_id::DICTIONARY32}, to_int32));
}

struct CastTimestampsSimple : public cudf::test::BaseFixture {};

TEST_F(CastTimestampsSimple, IsIdempotent)
{
  using namespace cudf::test;

  auto timestamps_D  = make_column<cudf::timestamp_D>(test_timestamps_D);
  auto timestamps_s  = make_column<cudf::timestamp_s>(test_timestamps_s);
  auto timestamps_ms = make_column<cudf::timestamp_ms>(test_timestamps_ms);
  auto timestamps_us = make_column<cudf::timestamp_us>(test_timestamps_us);
  auto timestamps_ns = make_column<cudf::timestamp_ns>(test_timestamps_ns);

  // Timestamps to duration
  auto timestamps_D_dur = cudf::cast(timestamps_D, make_data_type<cudf::timestamp_D::duration>());
  auto timestamps_s_dur = cudf::cast(timestamps_s, make_data_type<cudf::timestamp_s::duration>());
  auto timestamps_ms_dur =
    cudf::cast(timestamps_ms, make_data_type<cudf::timestamp_ms::duration>());
  auto timestamps_us_dur =
    cudf::cast(timestamps_us, make_data_type<cudf::timestamp_us::duration>());
  auto timestamps_ns_dur =
    cudf::cast(timestamps_ns, make_data_type<cudf::timestamp_ns::duration>());

  // Duration back to timestamp
  auto timestamps_D_got =
    cudf::cast(*timestamps_D_dur, cudf::data_type{cudf::type_id::TIMESTAMP_DAYS});
  auto timestamps_s_got =
    cudf::cast(*timestamps_s_dur, cudf::data_type{cudf::type_id::TIMESTAMP_SECONDS});
  auto timestamps_ms_got =
    cudf::cast(*timestamps_ms_dur, cudf::data_type{cudf::type_id::TIMESTAMP_MILLISECONDS});
  auto timestamps_us_got =
    cudf::cast(*timestamps_us_dur, cudf::data_type{cudf::type_id::TIMESTAMP_MICROSECONDS});
  auto timestamps_ns_got =
    cudf::cast(*timestamps_ns_dur, cudf::data_type{cudf::type_id::TIMESTAMP_NANOSECONDS});

  validate_cast_result<cudf::timestamp_D, cudf::timestamp_D>(timestamps_D, *timestamps_D_got);
  validate_cast_result<cudf::timestamp_s, cudf::timestamp_s>(timestamps_s, *timestamps_s_got);
  validate_cast_result<cudf::timestamp_ms, cudf::timestamp_ms>(timestamps_ms, *timestamps_ms_got);
  validate_cast_result<cudf::timestamp_us, cudf::timestamp_us>(timestamps_us, *timestamps_us_got);
  validate_cast_result<cudf::timestamp_ns, cudf::timestamp_ns>(timestamps_ns, *timestamps_ns_got);
}

struct CastDurationsSimple : public cudf::test::BaseFixture {};

TEST_F(CastDurationsSimple, IsIdempotent)
{
  using namespace cudf::test;

  auto durations_D  = make_column<cudf::duration_D>(test_durations_D);
  auto durations_s  = make_column<cudf::duration_s>(test_durations_s);
  auto durations_ms = make_column<cudf::duration_ms>(test_durations_ms);
  auto durations_us = make_column<cudf::duration_us>(test_durations_us);
  auto durations_ns = make_column<cudf::duration_ns>(test_durations_ns);

  auto durations_D_rep  = cudf::cast(durations_D, make_data_type<cudf::duration_D::rep>());
  auto durations_s_rep  = cudf::cast(durations_s, make_data_type<cudf::duration_s::rep>());
  auto durations_ms_rep = cudf::cast(durations_ms, make_data_type<cudf::duration_ms::rep>());
  auto durations_us_rep = cudf::cast(durations_us, make_data_type<cudf::duration_us::rep>());
  auto durations_ns_rep = cudf::cast(durations_ns, make_data_type<cudf::duration_ns::rep>());

  auto durations_D_got =
    cudf::cast(*durations_D_rep, cudf::data_type{cudf::type_id::DURATION_DAYS});
  auto durations_s_got =
    cudf::cast(*durations_s_rep, cudf::data_type{cudf::type_id::DURATION_SECONDS});
  auto durations_ms_got =
    cudf::cast(*durations_ms_rep, cudf::data_type{cudf::type_id::DURATION_MILLISECONDS});
  auto durations_us_got =
    cudf::cast(*durations_us_rep, cudf::data_type{cudf::type_id::DURATION_MICROSECONDS});
  auto durations_ns_got =
    cudf::cast(*durations_ns_rep, cudf::data_type{cudf::type_id::DURATION_NANOSECONDS});

  validate_cast_result<cudf::duration_D, cudf::duration_D>(durations_D, *durations_D_got);
  validate_cast_result<cudf::duration_s, cudf::duration_s>(durations_s, *durations_s_got);
  validate_cast_result<cudf::duration_ms, cudf::duration_ms>(durations_ms, *durations_ms_got);
  validate_cast_result<cudf::duration_us, cudf::duration_us>(durations_us, *durations_us_got);
  validate_cast_result<cudf::duration_ns, cudf::duration_ns>(durations_ns, *durations_ns_got);
}

template <typename T>
struct CastChronosTyped : public cudf::test::BaseFixture {};

TYPED_TEST_SUITE(CastChronosTyped, cudf::test::ChronoTypes);

// Return a list of chrono type ids whose precision is greater than or equal
// to the input type id
std::vector<cudf::type_id> get_higher_precision_chrono_type_ids(cudf::type_id search)
{
  size_t idx = 0;
  std::vector<cudf::type_id> gte_ids{};
  // Arranged such that for every pair of types, the types that precede them have a lower precision
  std::vector<cudf::type_id> timestamp_ids{cudf::type_id::TIMESTAMP_DAYS,
                                           cudf::type_id::DURATION_DAYS,
                                           cudf::type_id::TIMESTAMP_SECONDS,
                                           cudf::type_id::DURATION_SECONDS,
                                           cudf::type_id::TIMESTAMP_MILLISECONDS,
                                           cudf::type_id::DURATION_MILLISECONDS,
                                           cudf::type_id::TIMESTAMP_MICROSECONDS,
                                           cudf::type_id::DURATION_MICROSECONDS,
                                           cudf::type_id::TIMESTAMP_NANOSECONDS,
                                           cudf::type_id::DURATION_NANOSECONDS};
  for (cudf::type_id type_id : timestamp_ids) {
    if (type_id == search) break;
    idx++;
  }

  for (auto i = idx - idx % 2; i < timestamp_ids.size(); ++i)
    gte_ids.emplace_back(timestamp_ids[i]);
  return gte_ids;
}

// Test that all chrono types whose precision is >= to the TypeParam
// down-casts appropriately to the lower-precision TypeParam
TYPED_TEST(CastChronosTyped, DownCastingFloorsValues)
{
  using T = TypeParam;
  using namespace cudf::test;
  auto dtype_exp  = make_data_type<T>();
  auto chrono_exp = make_exp_chrono_column(dtype_exp.id());
  // Construct a list of the chrono type_ids whose precision is
  // greater than or equal to the precision of TypeParam's, e.g:
  // timestamp_ms -> {timestamp_ms, duration_ms, timestamp_us, duration_us, timestamp_ns,
  // duration_ns}; duration_us -> {timestamp_us, duration_us, timestamp_ns, duration_ns}; etc.
  auto higher_precision_type_ids = get_higher_precision_chrono_type_ids(cudf::type_to_id<T>());
  // For each higher-precision type, down-cast to TypeParam and validate
  // that the values were floored.
  for (cudf::type_id higher_precision_type_id : higher_precision_type_ids) {
    auto chrono_src = make_exp_chrono_column(higher_precision_type_id);
    auto chrono_got = cudf::cast(chrono_src, dtype_exp);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*chrono_got, chrono_exp);
  }
}

// Specific test to ensure down-casting to days happens correctly
TYPED_TEST(CastChronosTyped, DownCastingToDaysFloorsValues)
{
  using T = TypeParam;
  using namespace cudf::test;

  auto dtype_src  = make_data_type<T>();
  auto chrono_src = make_exp_chrono_column(dtype_src.id());

  // Convert {timestamp|duration}_X => timestamp_D
  auto timestamp_dtype_out = make_data_type<cudf::timestamp_D>();
  auto timestamps_got      = cudf::cast(chrono_src, timestamp_dtype_out);
  auto timestamp_exp       = make_column<cudf::timestamp_D>(test_timestamps_D);

  validate_cast_result<cudf::timestamp_D, cudf::timestamp_D>(timestamp_exp, *timestamps_got);

  // Convert {timestamp|duration}_X => duration_D
  auto duration_dtype_out = make_data_type<cudf::duration_D>();
  auto duration_got       = cudf::cast(chrono_src, duration_dtype_out);
  auto duration_exp       = make_column<cudf::duration_D>(test_durations_D);

  validate_cast_result<cudf::duration_D, cudf::duration_D>(duration_exp, *duration_got);
}

struct CastToTimestamps : public cudf::test::BaseFixture {};

// Cast duration types to timestamps (as integral types can't be converted)
TEST_F(CastToTimestamps, AllValid)
{
  using namespace cudf::test;

  auto durations_D  = make_column<cudf::duration_D>(test_durations_D);
  auto durations_s  = make_column<cudf::duration_s>(test_durations_s);
  auto durations_ms = make_column<cudf::duration_ms>(test_durations_ms);
  auto durations_us = make_column<cudf::duration_us>(test_durations_us);
  auto durations_ns = make_column<cudf::duration_ns>(test_durations_ns);

  auto timestamps_D_got = cudf::cast(durations_D, cudf::data_type{cudf::type_id::TIMESTAMP_DAYS});
  auto timestamps_s_got =
    cudf::cast(durations_s, cudf::data_type{cudf::type_id::TIMESTAMP_SECONDS});
  auto timestamps_ms_got =
    cudf::cast(durations_ms, cudf::data_type{cudf::type_id::TIMESTAMP_MILLISECONDS});
  auto timestamps_us_got =
    cudf::cast(durations_us, cudf::data_type{cudf::type_id::TIMESTAMP_MICROSECONDS});
  auto timestamps_ns_got =
    cudf::cast(durations_ns, cudf::data_type{cudf::type_id::TIMESTAMP_NANOSECONDS});

  validate_cast_result<cudf::duration_D, cudf::timestamp_D>(durations_D, *timestamps_D_got);
  validate_cast_result<cudf::duration_s, cudf::timestamp_s>(durations_s, *timestamps_s_got);
  validate_cast_result<cudf::duration_ms, cudf::timestamp_ms>(durations_ms, *timestamps_ms_got);
  validate_cast_result<cudf::duration_us, cudf::timestamp_us>(durations_us, *timestamps_us_got);
  validate_cast_result<cudf::duration_ns, cudf::timestamp_ns>(durations_ns, *timestamps_ns_got);
}

struct CastFromTimestamps : public cudf::test::BaseFixture {};

// Convert timestamps to duration types
TEST_F(CastFromTimestamps, AllValid)
{
  using namespace cudf::test;

  auto timestamps_D  = make_column<cudf::timestamp_D>(test_timestamps_D);
  auto timestamps_s  = make_column<cudf::timestamp_s>(test_timestamps_s);
  auto timestamps_ms = make_column<cudf::timestamp_ms>(test_timestamps_ms);
  auto timestamps_us = make_column<cudf::timestamp_us>(test_timestamps_us);
  auto timestamps_ns = make_column<cudf::timestamp_ns>(test_timestamps_ns);

  auto duration_D_exp  = make_column<cudf::duration_D>(test_durations_D);
  auto duration_s_exp  = make_column<cudf::duration_s>(test_durations_s);
  auto duration_ms_exp = make_column<cudf::duration_us>(test_durations_ms);
  auto duration_us_exp = make_column<cudf::duration_ms>(test_durations_us);
  auto duration_ns_exp = make_column<cudf::duration_ns>(test_durations_ns);

  auto durations_D_got  = cudf::cast(timestamps_D, make_data_type<cudf::duration_D>());
  auto durations_s_got  = cudf::cast(timestamps_s, make_data_type<cudf::duration_s>());
  auto durations_ms_got = cudf::cast(timestamps_ms, make_data_type<cudf::duration_ms>());
  auto durations_us_got = cudf::cast(timestamps_us, make_data_type<cudf::duration_us>());
  auto durations_ns_got = cudf::cast(timestamps_ns, make_data_type<cudf::duration_ns>());

  validate_cast_result<cudf::duration_D, cudf::duration_D>(duration_D_exp, *durations_D_got);
  validate_cast_result<cudf::duration_s, cudf::duration_s>(duration_s_exp, *durations_s_got);
  validate_cast_result<cudf::duration_ms, cudf::duration_ms>(duration_ms_exp, *durations_ms_got);
  validate_cast_result<cudf::duration_us, cudf::duration_us>(duration_us_exp, *durations_us_got);
  validate_cast_result<cudf::duration_ns, cudf::duration_ns>(duration_ns_exp, *durations_ns_got);
}

TEST_F(CastFromTimestamps, WithNulls)
{
  using namespace cudf::test;

  auto timestamps_D  = make_column<cudf::timestamp_D>(test_timestamps_D, {true, false, true});
  auto timestamps_s  = make_column<cudf::timestamp_s>(test_timestamps_s, {true, false, true});
  auto timestamps_ms = make_column<cudf::timestamp_ms>(test_timestamps_ms, {true, false, true});
  auto timestamps_us = make_column<cudf::timestamp_us>(test_timestamps_us, {true, false, true});
  auto timestamps_ns = make_column<cudf::timestamp_ns>(test_timestamps_ns, {true, false, true});

  auto duration_D_exp  = make_column<cudf::duration_D>(test_durations_D, {true, false, true});
  auto duration_s_exp  = make_column<cudf::duration_s>(test_durations_s, {true, false, true});
  auto duration_ms_exp = make_column<cudf::duration_us>(test_durations_ms, {true, false, true});
  auto duration_us_exp = make_column<cudf::duration_ms>(test_durations_us, {true, false, true});
  auto duration_ns_exp = make_column<cudf::duration_ns>(test_durations_ns, {true, false, true});

  auto durations_D_got  = cudf::cast(timestamps_D, make_data_type<cudf::duration_D>());
  auto durations_s_got  = cudf::cast(timestamps_s, make_data_type<cudf::duration_s>());
  auto durations_ms_got = cudf::cast(timestamps_ms, make_data_type<cudf::duration_ms>());
  auto durations_us_got = cudf::cast(timestamps_us, make_data_type<cudf::duration_us>());
  auto durations_ns_got = cudf::cast(timestamps_ns, make_data_type<cudf::duration_ns>());

  validate_cast_result<cudf::duration_D, cudf::duration_D>(duration_D_exp, *durations_D_got);
  validate_cast_result<cudf::duration_s, cudf::duration_s>(duration_s_exp, *durations_s_got);
  validate_cast_result<cudf::duration_ms, cudf::duration_ms>(duration_ms_exp, *durations_ms_got);
  validate_cast_result<cudf::duration_us, cudf::duration_us>(duration_us_exp, *durations_us_got);
  validate_cast_result<cudf::duration_ns, cudf::duration_ns>(duration_ns_exp, *durations_ns_got);
}

template <typename T>
struct CastToDurations : public cudf::test::BaseFixture {};

TYPED_TEST_SUITE(CastToDurations, cudf::test::IntegralTypes);

TYPED_TEST(CastToDurations, AllValid)
{
  using T = TypeParam;
  using namespace cudf::test;

  auto durations_D  = make_column<T>(test_durations_D);
  auto durations_s  = make_column<T>(test_durations_s);
  auto durations_ms = make_column<T>(test_durations_ms);
  auto durations_us = make_column<T>(test_durations_us);
  auto durations_ns = make_column<T>(test_durations_ns);

  auto durations_D_got = cudf::cast(durations_D, cudf::data_type{cudf::type_id::DURATION_DAYS});
  auto durations_s_got = cudf::cast(durations_s, cudf::data_type{cudf::type_id::DURATION_SECONDS});
  auto durations_ms_got =
    cudf::cast(durations_ms, cudf::data_type{cudf::type_id::DURATION_MILLISECONDS});
  auto durations_us_got =
    cudf::cast(durations_us, cudf::data_type{cudf::type_id::DURATION_MICROSECONDS});
  auto durations_ns_got =
    cudf::cast(durations_ns, cudf::data_type{cudf::type_id::DURATION_NANOSECONDS});

  validate_cast_result<T, cudf::duration_D>(durations_D, *durations_D_got);
  validate_cast_result<T, cudf::duration_s>(durations_s, *durations_s_got);
  validate_cast_result<T, cudf::duration_ms>(durations_ms, *durations_ms_got);
  validate_cast_result<T, cudf::duration_us>(durations_us, *durations_us_got);
  validate_cast_result<T, cudf::duration_ns>(durations_ns, *durations_ns_got);
}

template <typename T>
struct CastFromDurations : public cudf::test::BaseFixture {};

TYPED_TEST_SUITE(CastFromDurations, cudf::test::NumericTypes);

TYPED_TEST(CastFromDurations, AllValid)
{
  using T = TypeParam;
  using namespace cudf::test;

  auto durations_D  = make_column<cudf::duration_D>(test_durations_D);
  auto durations_s  = make_column<cudf::duration_s>(test_durations_s);
  auto durations_ms = make_column<cudf::duration_ms>(test_durations_ms);
  auto durations_us = make_column<cudf::duration_us>(test_durations_us);
  auto durations_ns = make_column<cudf::duration_ns>(test_durations_ns);

  auto durations_D_exp  = make_column<T>(test_durations_D);
  auto durations_s_exp  = make_column<T>(test_durations_s);
  auto durations_ms_exp = make_column<T>(test_durations_ms);
  auto durations_us_exp = make_column<T>(test_durations_us);
  auto durations_ns_exp = make_column<T>(test_durations_ns);

  auto durations_D_got  = cudf::cast(durations_D, make_data_type<T>());
  auto durations_s_got  = cudf::cast(durations_s, make_data_type<T>());
  auto durations_ms_got = cudf::cast(durations_ms, make_data_type<T>());
  auto durations_us_got = cudf::cast(durations_us, make_data_type<T>());
  auto durations_ns_got = cudf::cast(durations_ns, make_data_type<T>());

  validate_cast_result<T, T>(durations_D_exp, *durations_D_got);
  validate_cast_result<T, T>(durations_s_exp, *durations_s_got);
  validate_cast_result<T, T>(durations_ms_exp, *durations_ms_got);
  validate_cast_result<T, T>(durations_us_exp, *durations_us_got);
  validate_cast_result<T, T>(durations_ns_exp, *durations_ns_got);
}

TYPED_TEST(CastFromDurations, WithNulls)
{
  using T = TypeParam;
  using namespace cudf::test;

  auto durations_D  = make_column<cudf::duration_D>(test_durations_D, {true, false, true});
  auto durations_s  = make_column<cudf::duration_s>(test_durations_s, {true, false, true});
  auto durations_ms = make_column<cudf::duration_ms>(test_durations_ms, {true, false, true});
  auto durations_us = make_column<cudf::duration_us>(test_durations_us, {true, false, true});
  auto durations_ns = make_column<cudf::duration_ns>(test_durations_ns, {true, false, true});

  auto durations_D_exp  = make_column<T>(test_durations_D, {true, false, true});
  auto durations_s_exp  = make_column<T>(test_durations_s, {true, false, true});
  auto durations_ms_exp = make_column<T>(test_durations_ms, {true, false, true});
  auto durations_us_exp = make_column<T>(test_durations_us, {true, false, true});
  auto durations_ns_exp = make_column<T>(test_durations_ns, {true, false, true});

  auto durations_D_got  = cudf::cast(durations_D, make_data_type<T>());
  auto durations_s_got  = cudf::cast(durations_s, make_data_type<T>());
  auto durations_ms_got = cudf::cast(durations_ms, make_data_type<T>());
  auto durations_us_got = cudf::cast(durations_us, make_data_type<T>());
  auto durations_ns_got = cudf::cast(durations_ns, make_data_type<T>());

  validate_cast_result<T, T>(durations_D_exp, *durations_D_got);
  validate_cast_result<T, T>(durations_s_exp, *durations_s_got);
  validate_cast_result<T, T>(durations_ms_exp, *durations_ms_got);
  validate_cast_result<T, T>(durations_us_exp, *durations_us_got);
  validate_cast_result<T, T>(durations_ns_exp, *durations_ns_got);
}

template <typename T>
inline auto make_fixed_point_data_type(int32_t scale)
{
  return cudf::data_type{cudf::type_to_id<T>(), scale};
}

struct FixedPointTestSingleType : public cudf::test::BaseFixture {};

template <typename T>
struct FixedPointTests : public cudf::test::BaseFixture {};

TYPED_TEST_SUITE(FixedPointTests, cudf::test::FixedPointTypes);

TYPED_TEST(FixedPointTests, CastToDouble)
{
  using namespace numeric;
  using decimalXX  = TypeParam;
  using RepType    = cudf::device_storage_type_t<decimalXX>;
  using fp_wrapper = cudf::test::fixed_point_column_wrapper<RepType>;
  using fw_wrapper = cudf::test::fixed_width_column_wrapper<double>;

  auto const input    = fp_wrapper{{1729, 17290, 172900, 1729000}, scale_type{-3}};
  auto const expected = fw_wrapper{1.729, 17.29, 172.9, 1729.0};
  auto const result   = cudf::cast(input, make_data_type<double>());

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(FixedPointTests, CastToDoubleLarge)
{
  using namespace numeric;
  using namespace cudf::test;
  using decimalXX  = TypeParam;
  using RepType    = cudf::device_storage_type_t<decimalXX>;
  using fp_wrapper = cudf::test::fixed_point_column_wrapper<RepType>;
  using fw_wrapper = cudf::test::fixed_width_column_wrapper<double>;

  auto begin =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return 10 * (i + 0.5); });
  auto begin2 = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i + 0.5; });
  auto const input    = fp_wrapper{begin, begin + 2000, scale_type{-1}};
  auto const expected = fw_wrapper(begin2, begin2 + 2000);
  auto const result   = cudf::cast(input, make_data_type<double>());

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(FixedPointTests, CastToInt32)
{
  using namespace numeric;
  using decimalXX  = TypeParam;
  using RepType    = cudf::device_storage_type_t<decimalXX>;
  using fp_wrapper = cudf::test::fixed_point_column_wrapper<RepType>;
  using fw_wrapper = cudf::test::fixed_width_column_wrapper<int32_t>;

  auto const input    = fp_wrapper{{1729, 17290, 172900, 1729000}, scale_type{-3}};
  auto const expected = fw_wrapper{1, 17, 172, 1729};
  auto const result   = cudf::cast(input, make_data_type<int32_t>());

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TEST_F(FixedPointTestSingleType, CastDecimal64ToInt32)
{
  using fp_wrapper = cudf::test::fixed_point_column_wrapper<int64_t>;
  using fw_wrapper = cudf::test::fixed_width_column_wrapper<int32_t>;

  auto const input    = fp_wrapper{{7246212000}, numeric::scale_type{-5}};
  auto const expected = fw_wrapper{72462};
  auto const result   = cudf::cast(input, make_data_type<int32_t>());

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(FixedPointTests, CastToIntLarge)
{
  using namespace numeric;
  using namespace cudf::test;
  using decimalXX  = TypeParam;
  using RepType    = cudf::device_storage_type_t<decimalXX>;
  using fp_wrapper = cudf::test::fixed_point_column_wrapper<RepType>;
  using fw_wrapper = cudf::test::fixed_width_column_wrapper<int32_t>;

  auto begin  = cuda::counting_iterator<RepType>{0};
  auto begin2 = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return 10 * i; });
  auto const input    = fp_wrapper{begin, begin + 2000, scale_type{1}};
  auto const expected = fw_wrapper(begin2, begin2 + 2000);
  auto const result   = cudf::cast(input, make_data_type<int32_t>());

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(FixedPointTests, CastFromDouble)
{
  using namespace numeric;
  using decimalXX  = TypeParam;
  using RepType    = cudf::device_storage_type_t<decimalXX>;
  using fp_wrapper = cudf::test::fixed_point_column_wrapper<RepType>;
  using fw_wrapper = cudf::test::fixed_width_column_wrapper<double>;

  auto const input    = fw_wrapper{1.729, 17.29, 172.9, 1729.0};
  auto const expected = fp_wrapper{{1729, 17290, 172900, 1729000}, scale_type{-3}};
  auto const result   = cudf::cast(input, make_fixed_point_data_type<decimalXX>(-3));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(FixedPointTests, CastFromDoubleWithNaNAndInf)
{
  using namespace numeric;
  using decimalXX  = TypeParam;
  using RepType    = cudf::device_storage_type_t<decimalXX>;
  using fp_wrapper = cudf::test::fixed_point_column_wrapper<RepType>;
  using fw_wrapper = cudf::test::fixed_width_column_wrapper<double>;

  auto const NaN  = std::numeric_limits<double>::quiet_NaN();
  auto const inf  = std::numeric_limits<double>::infinity();
  auto const null = 0;

  auto const input    = fw_wrapper{1.729, -inf, NaN, 172.9, -inf, NaN, inf, 1.23, inf};
  auto const expected = fp_wrapper{{1729, null, null, 172900, null, null, null, 1230, null},
                                   {true, false, false, true, false, false, false, true, false},
                                   scale_type{-3}};
  auto const result   = cudf::cast(input, make_fixed_point_data_type<decimalXX>(-3));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(FixedPointTests, CastFromDoubleLarge)
{
  using namespace numeric;
  using namespace cudf::test;
  using decimalXX  = TypeParam;
  using RepType    = cudf::device_storage_type_t<decimalXX>;
  using fp_wrapper = cudf::test::fixed_point_column_wrapper<RepType>;
  using fw_wrapper = cudf::test::fixed_width_column_wrapper<double>;

  auto begin = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i + 0.5; });
  auto begin2 =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return 10 * (i + 0.5); });
  auto const input    = fw_wrapper(begin, begin + 2000);
  auto const expected = fp_wrapper{begin2, begin2 + 2000, scale_type{-1}};
  auto const result   = cudf::cast(input, make_fixed_point_data_type<decimalXX>(-1));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(FixedPointTests, CastFromInt)
{
  using namespace numeric;
  using decimalXX  = TypeParam;
  using RepType    = cudf::device_storage_type_t<decimalXX>;
  using fp_wrapper = cudf::test::fixed_point_column_wrapper<RepType>;
  using fw_wrapper = cudf::test::fixed_width_column_wrapper<int32_t>;

  auto const input    = fw_wrapper{1729, 172, 17, 1};
  auto const expected = fp_wrapper{{17, 1, 0, 0}, scale_type{2}};
  auto const result   = cudf::cast(input, make_fixed_point_data_type<decimalXX>(2));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TEST_F(FixedPointTestSingleType, CastInt32ToDecimal64)
{
  using fp_wrapper = cudf::test::fixed_point_column_wrapper<int64_t>;
  using fw_wrapper = cudf::test::fixed_width_column_wrapper<int32_t>;

  auto const input    = fw_wrapper{-48938};
  auto const expected = fp_wrapper{{-4893800000LL}, numeric::scale_type{-5}};
  auto const result   = cudf::cast(input, make_fixed_point_data_type<numeric::decimal64>(-5));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(FixedPointTests, CastFromIntLarge)
{
  using namespace numeric;
  using namespace cudf::test;
  using decimalXX  = TypeParam;
  using RepType    = cudf::device_storage_type_t<decimalXX>;
  using fp_wrapper = cudf::test::fixed_point_column_wrapper<RepType>;
  using fw_wrapper = cudf::test::fixed_width_column_wrapper<int32_t>;

  auto begin  = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return 1000 * i; });
  auto begin2 = cuda::counting_iterator<RepType>{0};
  auto const input    = fw_wrapper(begin, begin + 2000);
  auto const expected = fp_wrapper{begin2, begin2 + 2000, scale_type{3}};
  auto const result   = cudf::cast(input, make_fixed_point_data_type<decimalXX>(3));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(FixedPointTests, FixedPointToFixedPointSameTypeidUp)
{
  using namespace numeric;
  using decimalXX  = TypeParam;
  using RepType    = cudf::device_storage_type_t<decimalXX>;
  using fp_wrapper = cudf::test::fixed_point_column_wrapper<RepType>;

  auto const input    = fp_wrapper{{1729, 17290, 172900, 1729000}, scale_type{-3}};
  auto const expected = fp_wrapper{{172, 1729, 17290, 172900}, scale_type{-2}};
  auto const result   = cudf::cast(input, make_fixed_point_data_type<decimalXX>(-2));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(FixedPointTests, FixedPointToFixedPointSameTypeidDown)
{
  using namespace numeric;
  using decimalXX  = TypeParam;
  using RepType    = cudf::device_storage_type_t<decimalXX>;
  using fp_wrapper = cudf::test::fixed_point_column_wrapper<RepType>;

  auto const input    = fp_wrapper{{1729, 17290, 172900, 1729000}, scale_type{-3}};
  auto const expected = fp_wrapper{{17290, 172900, 1729000, 17290000}, scale_type{-4}};
  auto const result   = cudf::cast(input, make_fixed_point_data_type<decimalXX>(-4));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(FixedPointTests, FixedPointToFixedPointSameTypeidUpPositive)
{
  using namespace numeric;
  using decimalXX  = TypeParam;
  using RepType    = cudf::device_storage_type_t<decimalXX>;
  using fp_wrapper = cudf::test::fixed_point_column_wrapper<RepType>;

  auto const input    = fp_wrapper{{1, 12, 123, 1234, 12345, 123456}, scale_type{1}};
  auto const expected = fp_wrapper{{0, 1, 12, 123, 1234, 12345}, scale_type{2}};
  auto const result   = cudf::cast(input, make_fixed_point_data_type<decimalXX>(2));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(FixedPointTests, FixedPointToFixedPointSameTypeidEmpty)
{
  using namespace numeric;
  using decimalXX  = TypeParam;
  using RepType    = cudf::device_storage_type_t<decimalXX>;
  using fp_wrapper = cudf::test::fixed_point_column_wrapper<RepType>;

  auto const input    = fp_wrapper{{}, scale_type{1}};
  auto const expected = fp_wrapper{{}, scale_type{2}};
  auto const result   = cudf::cast(input, make_fixed_point_data_type<decimalXX>(2));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(FixedPointTests, FixedPointToFixedPointSameTypeidDownPositive)
{
  using namespace numeric;
  using decimalXX  = TypeParam;
  using RepType    = cudf::device_storage_type_t<decimalXX>;
  using fp_wrapper = cudf::test::fixed_point_column_wrapper<RepType>;

  auto const input    = fp_wrapper{{0, 1, 12, 123, 1234}, scale_type{2}};
  auto const expected = fp_wrapper{{0, 1000, 12000, 123000, 1234000}, scale_type{-1}};
  auto const result   = cudf::cast(input, make_fixed_point_data_type<decimalXX>(-1));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(FixedPointTests, Decimal32ToDecimalXX)
{
  using namespace numeric;
  using decimalXX      = TypeParam;
  using RepTypeFrom    = int32_t;
  using RepTypeTo      = cudf::device_storage_type_t<decimalXX>;
  using fp_wrapperFrom = cudf::test::fixed_point_column_wrapper<RepTypeFrom>;
  using fp_wrapperTo   = cudf::test::fixed_point_column_wrapper<RepTypeTo>;

  auto const input    = fp_wrapperFrom{{1729, 17290, 172900, 1729000}, scale_type{-3}};
  auto const expected = fp_wrapperTo{{1729, 17290, 172900, 1729000}, scale_type{-3}};
  auto const result   = cudf::cast(input, make_fixed_point_data_type<decimalXX>(-3));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(FixedPointTests, Decimal64ToDecimalXX)
{
  using namespace numeric;
  using decimalXX      = TypeParam;
  using RepTypeFrom    = int64_t;
  using RepTypeTo      = cudf::device_storage_type_t<decimalXX>;
  using fp_wrapperFrom = cudf::test::fixed_point_column_wrapper<RepTypeFrom>;
  using fp_wrapperTo   = cudf::test::fixed_point_column_wrapper<RepTypeTo>;

  auto const input    = fp_wrapperFrom{{1729, 17290, 172900, 1729000}, scale_type{-3}};
  auto const expected = fp_wrapperTo{{1729, 17290, 172900, 1729000}, scale_type{-3}};
  auto const result   = cudf::cast(input, make_fixed_point_data_type<decimalXX>(-3));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(FixedPointTests, Decimal128ToDecimalXX)
{
  using namespace numeric;
  using decimalXX      = TypeParam;
  using RepTypeFrom    = __int128_t;
  using RepTypeTo      = cudf::device_storage_type_t<decimalXX>;
  using fp_wrapperFrom = cudf::test::fixed_point_column_wrapper<RepTypeFrom>;
  using fp_wrapperTo   = cudf::test::fixed_point_column_wrapper<RepTypeTo>;

  auto const input    = fp_wrapperFrom{{1729, 17290, 172900, 1729000}, scale_type{-3}};
  auto const expected = fp_wrapperTo{{1729, 17290, 172900, 1729000}, scale_type{-3}};
  auto const result   = cudf::cast(input, make_fixed_point_data_type<decimalXX>(-3));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(FixedPointTests, Decimal32ToDecimalXXWithSmallerScale)
{
  using namespace numeric;
  using decimalXX      = TypeParam;
  using RepTypeFrom    = int32_t;
  using RepTypeTo      = cudf::device_storage_type_t<decimalXX>;
  using fp_wrapperFrom = cudf::test::fixed_point_column_wrapper<RepTypeFrom>;
  using fp_wrapperTo   = cudf::test::fixed_point_column_wrapper<RepTypeTo>;

  auto const input    = fp_wrapperFrom{{1729, 17290, 172900, 1729000}, scale_type{-3}};
  auto const expected = fp_wrapperTo{{172900, 1729000, 17290000, 172900000}, scale_type{-5}};
  auto const result   = cudf::cast(input, make_fixed_point_data_type<decimalXX>(-5));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(FixedPointTests, Decimal64ToDecimalXXWithSmallerScale)
{
  using namespace numeric;
  using decimalXX      = TypeParam;
  using RepTypeFrom    = int64_t;
  using RepTypeTo      = cudf::device_storage_type_t<decimalXX>;
  using fp_wrapperFrom = cudf::test::fixed_point_column_wrapper<RepTypeFrom>;
  using fp_wrapperTo   = cudf::test::fixed_point_column_wrapper<RepTypeTo>;

  auto const input    = fp_wrapperFrom{{1729, 17290, 172900, 1729000}, scale_type{-3}};
  auto const expected = fp_wrapperTo{{172900, 1729000, 17290000, 172900000}, scale_type{-5}};
  auto const result   = cudf::cast(input, make_fixed_point_data_type<decimalXX>(-5));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(FixedPointTests, Decimal128ToDecimalXXWithSmallerScale)
{
  using namespace numeric;
  using decimalXX      = TypeParam;
  using RepTypeFrom    = __int128_t;
  using RepTypeTo      = cudf::device_storage_type_t<decimalXX>;
  using fp_wrapperFrom = cudf::test::fixed_point_column_wrapper<RepTypeFrom>;
  using fp_wrapperTo   = cudf::test::fixed_point_column_wrapper<RepTypeTo>;

  auto const input    = fp_wrapperFrom{{1729, 17290, 172900, 1729000}, scale_type{-3}};
  auto const expected = fp_wrapperTo{{172900, 1729000, 17290000, 172900000}, scale_type{-5}};
  auto const result   = cudf::cast(input, make_fixed_point_data_type<decimalXX>(-5));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(FixedPointTests, Decimal32ToDecimalXXWithLargerScale)
{
  using namespace numeric;
  using decimalXX      = TypeParam;
  using RepTypeFrom    = int32_t;
  using RepTypeTo      = cudf::device_storage_type_t<decimalXX>;
  using fp_wrapperFrom = cudf::test::fixed_point_column_wrapper<RepTypeFrom>;
  using fp_wrapperTo   = cudf::test::fixed_point_column_wrapper<RepTypeTo>;

  auto const input    = fp_wrapperFrom{{1729, 17290, 172900, 1729000}, scale_type{-3}};
  auto const expected = fp_wrapperTo{{1, 17, 172, 1729}, scale_type{0}};
  auto const result   = cudf::cast(input, make_fixed_point_data_type<decimalXX>(0));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(FixedPointTests, Decimal64ToDecimalXXWithLargerScale)
{
  using namespace numeric;
  using decimalXX      = TypeParam;
  using RepTypeFrom    = int64_t;
  using RepTypeTo      = cudf::device_storage_type_t<decimalXX>;
  using fp_wrapperFrom = cudf::test::fixed_point_column_wrapper<RepTypeFrom>;
  using fp_wrapperTo   = cudf::test::fixed_point_column_wrapper<RepTypeTo>;

  auto const input    = fp_wrapperFrom{{1729, 17290, 172900, 1729000}, scale_type{-3}};
  auto const expected = fp_wrapperTo{{1, 17, 172, 1729}, scale_type{0}};
  auto const result   = cudf::cast(input, make_fixed_point_data_type<decimalXX>(0));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TEST_F(FixedPointTestSingleType, AvoidOverflowDecimal32ToDecimal64)
{
  using namespace numeric;
  using fp_wrapper32 = cudf::test::fixed_point_column_wrapper<int32_t>;
  using fp_wrapper64 = cudf::test::fixed_point_column_wrapper<int64_t>;

  auto const input    = fp_wrapper32{{9999999}, scale_type{3}};
  auto const expected = fp_wrapper64{{9999999}, scale_type{3}};
  auto const result   = cudf::cast(input, make_fixed_point_data_type<decimal64>(3));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TEST_F(FixedPointTestSingleType, AvoidOverflowDecimal32ToInt64)
{
  using namespace numeric;
  using fp_wrapper32 = cudf::test::fixed_point_column_wrapper<int32_t>;
  using fw_wrapper64 = cudf::test::fixed_width_column_wrapper<int64_t>;

  auto const input    = fp_wrapper32{{9999999}, scale_type{3}};
  auto const expected = fw_wrapper64{{9999999000}};
  auto const result   = cudf::cast(input, cudf::data_type{cudf::type_id::INT64});

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(FixedPointTests, Decimal128ToDecimalXXWithLargerScale)
{
  using namespace numeric;
  using decimalXX      = TypeParam;
  using RepTypeFrom    = __int128_t;
  using RepTypeTo      = cudf::device_storage_type_t<decimalXX>;
  using fp_wrapperFrom = cudf::test::fixed_point_column_wrapper<RepTypeFrom>;
  using fp_wrapperTo   = cudf::test::fixed_point_column_wrapper<RepTypeTo>;

  auto const input    = fp_wrapperFrom{{1729, 17290, 172900, 1729000}, scale_type{-3}};
  auto const expected = fp_wrapperTo{{1, 17, 172, 1729}, scale_type{0}};
  auto const result   = cudf::cast(input, make_fixed_point_data_type<decimalXX>(0));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(FixedPointTests, ValidateCastRescalePrecision)
{
  using namespace numeric;
  using decimalXX  = TypeParam;
  using RepType    = cudf::device_storage_type_t<decimalXX>;
  using fp_wrapper = cudf::test::fixed_point_column_wrapper<RepType>;

  // This test is designed to protect against floating point conversion
  // introducing errors in fixed-point arithmetic. The rescaling that occurs
  // during casting to different scales should only use fixed-precision math.
  // Realistically, we are only able to show precision failures due to floating
  // conversion in a few very specific circumstances where dividing by specific
  // powers of 10 works against us.  Some examples: 10^23, 10^25, 10^26, 10^27,
  // 10^30, 10^32, 10^36. See https://godbolt.org/z/cP1MddP8P for a derivation.
  // For completeness and to ensure that we are not missing any other cases, we
  // test casting to/from all scales in the range of each decimal type. Values
  // that are powers of ten show this error more readily than non-powers of 10
  // because the rescaling factor is a power of 10, meaning that errors in
  // division are more visible.
  constexpr auto min_scale = -cuda::std::numeric_limits<RepType>::digits10;
  for (int input_scale = 0; input_scale >= min_scale; --input_scale) {
    for (int result_scale = 0; result_scale >= min_scale; --result_scale) {
      RepType input_value = 1;
      for (int k = 0; k > input_scale; --k) {
        input_value *= 10;
      }
      RepType result_value = 1;
      for (int k = 0; k > result_scale; --k) {
        result_value *= 10;
      }
      auto const input    = fp_wrapper{{input_value}, scale_type{input_scale}};
      auto const expected = fp_wrapper{{result_value}, scale_type{result_scale}};
      auto const result   = cudf::cast(input, make_fixed_point_data_type<decimalXX>(result_scale));
      CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
    }
  }
}

TYPED_TEST(FixedPointTests, Decimal32ToDecimalXXWithLargerScaleAndNullMask)
{
  using namespace numeric;
  using decimalXX      = TypeParam;
  using RepTypeFrom    = int32_t;
  using RepTypeTo      = cudf::device_storage_type_t<decimalXX>;
  using fp_wrapperFrom = cudf::test::fixed_point_column_wrapper<RepTypeFrom>;
  using fp_wrapperTo   = cudf::test::fixed_point_column_wrapper<RepTypeTo>;

  auto const vec = std::vector{1729, 17290, 172900, 1729000};
  auto const input =
    fp_wrapperFrom{vec.cbegin(), vec.cend(), {true, true, true, false}, scale_type{-3}};
  auto const expected = fp_wrapperTo{{1, 17, 172, 1729000}, {1, 1, 1, 0}, scale_type{0}};
  auto const result   = cudf::cast(input, make_fixed_point_data_type<decimalXX>(0));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(FixedPointTests, Decimal64ToDecimalXXWithLargerScaleAndNullMask)
{
  using namespace numeric;
  using decimalXX      = TypeParam;
  using RepTypeFrom    = int64_t;
  using RepTypeTo      = cudf::device_storage_type_t<decimalXX>;
  using fp_wrapperFrom = cudf::test::fixed_point_column_wrapper<RepTypeFrom>;
  using fp_wrapperTo   = cudf::test::fixed_point_column_wrapper<RepTypeTo>;

  auto const vec = std::vector{1729, 17290, 172900, 1729000};
  auto const input =
    fp_wrapperFrom{vec.cbegin(), vec.cend(), {true, true, true, false}, scale_type{-3}};
  auto const expected = fp_wrapperTo{{1, 17, 172, 1729000}, {1, 1, 1, 0}, scale_type{0}};
  auto const result   = cudf::cast(input, make_fixed_point_data_type<decimalXX>(0));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(FixedPointTests, Decimal128ToDecimalXXWithLargerScaleAndNullMask)
{
  using namespace numeric;
  using decimalXX      = TypeParam;
  using RepTypeFrom    = __int128_t;
  using RepTypeTo      = cudf::device_storage_type_t<decimalXX>;
  using fp_wrapperFrom = cudf::test::fixed_point_column_wrapper<RepTypeFrom>;
  using fp_wrapperTo   = cudf::test::fixed_point_column_wrapper<RepTypeTo>;

  auto const vec = std::vector{1729, 17290, 172900, 1729000};
  auto const input =
    fp_wrapperFrom{vec.cbegin(), vec.cend(), {true, true, true, false}, scale_type{-3}};
  auto const expected = fp_wrapperTo{{1, 17, 172, 1729000}, {1, 1, 1, 0}, scale_type{0}};
  auto const result   = cudf::cast(input, make_fixed_point_data_type<decimalXX>(0));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(FixedPointTests, DecimalRescaleOverflowAndNullMask)
{
  using namespace numeric;
  using decimalXX  = TypeParam;
  using RepType    = cudf::device_storage_type_t<decimalXX>;
  using fp_wrapper = cudf::test::fixed_point_column_wrapper<RepType>;

  auto const vec      = std::vector{1729, 17290, 172900, 1729000};
  auto const scale    = cuda::std::numeric_limits<RepType>::digits10 + 1;
  auto const input    = fp_wrapper{vec.cbegin(), vec.cend(), {1, 0, 0, 1}, scale_type{0}};
  auto const expected = fp_wrapper{{0, 0, 0, 0}, {1, 0, 0, 1}, scale_type{scale}};
  auto const result   = cudf::cast(input, make_fixed_point_data_type<decimalXX>(scale));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TEST_F(FixedPointTestSingleType, Int32ToInt64Convert)
{
  using namespace numeric;
  using fp_wrapperA = cudf::test::fixed_point_column_wrapper<int32_t>;
  using fp_wrapperB = cudf::test::fixed_point_column_wrapper<int64_t>;

  auto const input    = fp_wrapperB{{141230900000L}, scale_type{-10}};
  auto const expected = fp_wrapperA{{14123}, scale_type{-3}};
  auto const result   = cudf::cast(input, make_fixed_point_data_type<decimal32>(-3));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}
