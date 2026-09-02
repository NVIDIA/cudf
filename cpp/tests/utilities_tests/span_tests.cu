/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "io/utilities/hostdevice_vector.hpp"

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/cudf_gtest.hpp>
#include <cudf_test/testing_main.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/device_vector.hpp>

#include <cuda/std/span>
#include <cuda/stream>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cstddef>
#include <cstring>
#include <span>
#include <string>

using cudf::device_span;
using cudf::host_span;
using cudf::detail::device_2dspan;
using cudf::detail::host_2dspan;
using cudf::detail::hostdevice_2dvector;

template <typename T>
void expect_equivalent(host_span<T> a, host_span<T> b)
{
  EXPECT_EQ(a.size(), b.size());
  EXPECT_EQ(a.data(), b.data());
}

template <typename T>
void expect_equivalent(cudf::detail::hostdevice_span<T> a, cudf::detail::hostdevice_span<T> b)
{
  EXPECT_EQ(a.size(), b.size());
  EXPECT_EQ(a.host_ptr(), b.host_ptr());
}

template <typename Iterator1, typename T>
void expect_match(Iterator1 expected, size_t expected_size, host_span<T> input)
{
  EXPECT_EQ(expected_size, input.size());
  for (size_t i = 0; i < expected_size; i++) {
    EXPECT_EQ(*(expected + i), *(input.begin() + i));
  }
}

template <typename T>
void expect_match(std::string expected, host_span<T> input)
{
  return expect_match(expected.begin(), expected.size(), input);
}

template <typename T>
void expect_match(std::string expected, cudf::detail::hostdevice_span<T> input)
{
  return expect_match(expected.begin(), expected.size(), host_span<T>(input));
}

std::string const hello_world_message = "hello world";
std::vector<char> create_hello_world_message()
{
  return std::vector<char>(hello_world_message.begin(), hello_world_message.end());
}

class SpanTest : public cudf::test::BaseFixture {};

TEST(SpanTest, CanCreateFullSubspan)
{
  auto message            = create_hello_world_message();
  auto const message_span = host_span<char>(message.data(), message.size());

  expect_equivalent(message_span, message_span.subspan(0, message_span.size()));
}

TEST(SpanTest, CanTakeFirst)
{
  auto message            = create_hello_world_message();
  auto const message_span = host_span<char>(message.data(), message.size());

  expect_match("hello", message_span.first(5));
}

TEST(SpanTest, CanTakeLast)
{
  auto message            = create_hello_world_message();
  auto const message_span = host_span<char>(message.data(), message.size());

  expect_match("world", message_span.last(5));
}

TEST(SpanTest, CanTakeSubspanFull)
{
  auto message            = create_hello_world_message();
  auto const message_span = host_span<char>(message.data(), message.size());

  expect_match("hello world", message_span.subspan(0, 11));
}

TEST(SpanTest, CanTakeSubspanPartial)
{
  auto message            = create_hello_world_message();
  auto const message_span = host_span<char>(message.data(), message.size());

  expect_match("lo w", message_span.subspan(3, 4));
}

TEST(SpanTest, CanGetFront)
{
  auto message            = create_hello_world_message();
  auto const message_span = host_span<char>(message.data(), message.size());

  EXPECT_EQ('h', message_span.front());
}

TEST(SpanTest, CanGetBack)
{
  auto message            = create_hello_world_message();
  auto const message_span = host_span<char>(message.data(), message.size());

  EXPECT_EQ('d', message_span.back());
}

TEST(SpanTest, CanGetData)
{
  auto message            = create_hello_world_message();
  auto const message_span = host_span<char>(message.data(), message.size());

  EXPECT_EQ(message.data(), message_span.data());
}

TEST(SpanTest, CanDetermineEmptiness)
{
  auto message            = create_hello_world_message();
  auto const message_span = host_span<char>(message.data(), message.size());
  auto const empty_span   = host_span<char>();

  EXPECT_FALSE(message_span.empty());
  EXPECT_TRUE(empty_span.empty());
}

TEST(SpanTest, CanGetSize)
{
  auto message            = create_hello_world_message();
  auto const message_span = host_span<char>(message.data(), message.size());
  auto const empty_span   = host_span<char>();

  EXPECT_EQ(static_cast<size_t>(11), message_span.size());
  EXPECT_EQ(static_cast<size_t>(0), empty_span.size());
}

TEST(SpanTest, CanGetSizeBytes)
{
  auto doubles            = std::vector<double>({6, 3, 2});
  auto const doubles_span = host_span<double>(doubles.data(), doubles.size());
  auto const empty_span   = host_span<double>();

  EXPECT_EQ(static_cast<size_t>(24), doubles_span.size_bytes());
  EXPECT_EQ(static_cast<size_t>(0), empty_span.size_bytes());
}

TEST(SpanTest, CanCopySpan)
{
  auto message = create_hello_world_message();
  host_span<char> message_span_copy;

  {
    auto const message_span = host_span<char>(message.data(), message.size());

    message_span_copy = message_span;
  }

  EXPECT_EQ(message.data(), message_span_copy.data());
  EXPECT_EQ(message.size(), message_span_copy.size());
}

TEST(SpanTest, CanSubscriptRead)
{
  auto message            = create_hello_world_message();
  auto const message_span = host_span<char>(message.data(), message.size());

  EXPECT_EQ('o', message_span[4]);
}

TEST(SpanTest, CanSubscriptWrite)
{
  auto message            = create_hello_world_message();
  auto const message_span = host_span<char>(message.data(), message.size());

  message_span[4] = 'x';

  EXPECT_EQ('x', message_span[4]);
}

TEST(SpanTest, CanConstructFromHostContainers)
{
  auto std_vector = std::vector<int>(1);
  auto h_vector   = thrust::host_vector<int>(1);

  (void)host_span<int>(std_vector);
  (void)host_span<int>(h_vector);

  auto const std_vector_c = std_vector;
  auto const h_vector_c   = h_vector;

  (void)host_span<int const>(std_vector_c);
  (void)host_span<int const>(h_vector_c);
}

TEST(SpanTest, CanUseStdSpan)
{
  auto message            = create_hello_world_message();
  auto const message_span = host_span<char>(message.data(), message.size());

  std::span std_span = message_span;
  EXPECT_EQ(std_span.data(), message_span.data());
  EXPECT_EQ(std_span.size(), message_span.size());
}

CUDF_KERNEL void simple_device_kernel(device_span<bool> result) { result[0] = true; }

TEST(SpanTest, CanUseDeviceSpan)
{
  auto d_message = cudf::detail::make_zeroed_device_uvector_async<bool>(
    1, cudf::get_default_stream(), cudf::get_current_device_resource_ref());

  auto d_span = device_span<bool>(d_message.data(), d_message.size());

  simple_device_kernel<<<1, 1, 0, cudf::get_default_stream().get()>>>(d_span);

  ASSERT_TRUE(d_message.element(0, cudf::get_default_stream()));
}

TEST(SpanTest, CanUseCudaStdSpan)
{
  auto d_message = cudf::detail::make_zeroed_device_uvector_async<int>(
    1, cudf::get_default_stream(), cudf::get_current_device_resource_ref());

  auto const d_span = device_span<int const>(d_message.data(), d_message.size());

  cuda::std::span std_span = d_span;
  EXPECT_EQ(std_span.data(), d_span.data());
  EXPECT_EQ(std_span.size(), d_span.size());
}

class MdSpanTest : public cudf::test::BaseFixture {};

TEST(MdSpanTest, CanDetermineEmptiness)
{
  auto const vector            = hostdevice_2dvector<int>(1, 2, cudf::get_default_stream());
  auto const no_rows_vector    = hostdevice_2dvector<int>(0, 2, cudf::get_default_stream());
  auto const no_columns_vector = hostdevice_2dvector<int>(1, 0, cudf::get_default_stream());

  EXPECT_FALSE(host_2dspan<int const>{vector}.is_empty());
  EXPECT_FALSE(device_2dspan<int const>{vector}.is_empty());
  EXPECT_TRUE(host_2dspan<int const>{no_rows_vector}.is_empty());
  EXPECT_TRUE(device_2dspan<int const>{no_rows_vector}.is_empty());
  EXPECT_TRUE(host_2dspan<int const>{no_columns_vector}.is_empty());
  EXPECT_TRUE(device_2dspan<int const>{no_columns_vector}.is_empty());
}

CUDF_KERNEL void readwrite_kernel(device_2dspan<int> result)
{
  if (result[5][6] == 5) {
    result[5][6] *= 6;
  } else {
    result[5][6] = 5;
  }
}

TEST(MdSpanTest, DeviceReadWrite)
{
  auto vector = hostdevice_2dvector<int>(11, 23, cudf::get_default_stream());

  readwrite_kernel<<<1, 1, 0, cudf::get_default_stream().get()>>>(vector);
  readwrite_kernel<<<1, 1, 0, cudf::get_default_stream().get()>>>(vector);
  vector.device_to_host(cudf::get_default_stream());
  EXPECT_EQ(vector[5][6], 30);
}

TEST(MdSpanTest, HostReadWrite)
{
  auto vector = hostdevice_2dvector<int>(11, 23, cudf::get_default_stream());
  auto span   = host_2dspan<int>{vector};
  span[5][6]  = 5;
  if (span[5][6] == 5) { span[5][6] *= 6; }

  EXPECT_EQ(vector[5][6], 30);
}

TEST(MdSpanTest, CanGetSize)
{
  auto const vector = hostdevice_2dvector<int>(1, 2, cudf::get_default_stream());

  EXPECT_EQ(host_2dspan<int const>{vector}.size(), vector.size());
  EXPECT_EQ(device_2dspan<int const>{vector}.size(), vector.size());
}

TEST(MdSpanTest, CanGetCount)
{
  auto const vector = hostdevice_2dvector<int>(11, 23, cudf::get_default_stream());

  EXPECT_EQ(host_2dspan<int const>{vector}.count(), 11ul * 23);
  EXPECT_EQ(device_2dspan<int const>{vector}.count(), 11ul * 23);
}

auto get_test_hostdevice_vector()
{
  auto const msg = create_hello_world_message();
  auto v         = cudf::detail::hostdevice_vector<char>(msg.size(), cudf::get_default_stream());
  std::memcpy(v.host_ptr(), msg.data(), msg.size());

  return v;
}

TEST(HostDeviceSpanTest, CanCreateFullSubspan)
{
  auto message            = get_test_hostdevice_vector();
  auto const message_span = cudf::detail::hostdevice_span<char>{message};

  expect_equivalent(message_span.subspan(0, message_span.size()), message_span);
}

TEST(HostDeviceSpanTest, CanCreateHostSpan)
{
  auto message            = get_test_hostdevice_vector();
  auto const message_span = host_span<char>(message.host_ptr(), message.size());
  auto const hd_span      = cudf::detail::hostdevice_span<char>{message};

  expect_equivalent(message_span, cudf::host_span<char>(hd_span));
}

TEST(HostDeviceSpanTest, CanTakeSubspanFull)
{
  auto message            = get_test_hostdevice_vector();
  auto const message_span = cudf::detail::hostdevice_span<char>{message};

  expect_match("hello world", message_span.subspan(0, 11));
}

TEST(HostDeviceSpanTest, CanTakeSubspanPartial)
{
  auto message            = get_test_hostdevice_vector();
  auto const message_span = cudf::detail::hostdevice_span<char>{message};

  expect_match("lo w", message_span.subspan(3, 4));
}

TEST(HostDeviceSpanTest, CanGetData)
{
  auto message            = get_test_hostdevice_vector();
  auto const message_span = cudf::detail::hostdevice_span<char>{message};

  EXPECT_EQ(message.host_ptr(), message_span.host_ptr());
}

TEST(HostDeviceSpanTest, CanGetSize)
{
  auto message            = get_test_hostdevice_vector();
  auto const message_span = cudf::detail::hostdevice_span<char>{message};
  auto const empty_span   = cudf::detail::hostdevice_span<char>();

  EXPECT_EQ(static_cast<size_t>(11), message_span.size());
  EXPECT_EQ(static_cast<size_t>(0), empty_span.size());
}

TEST(HostDeviceSpanTest, CanGetSizeBytes)
{
  auto doubles = std::vector<double>({6, 3, 2});
  auto doubles_hdv =
    cudf::detail::hostdevice_vector<double>(doubles.size(), cudf::get_default_stream());
  std::memcpy(doubles_hdv.host_ptr(), doubles.data(), doubles.size() * sizeof(double));

  auto const doubles_span = cudf::detail::hostdevice_span<double>(doubles_hdv);
  auto const empty_span   = cudf::detail::hostdevice_span<double>();

  EXPECT_EQ(static_cast<size_t>(24), doubles_span.size_bytes());
  EXPECT_EQ(static_cast<size_t>(0), empty_span.size_bytes());
}

TEST(HostDeviceSpanTest, CanCopySpan)
{
  auto message = get_test_hostdevice_vector();
  cudf::detail::hostdevice_span<char> message_span_copy;

  {
    auto const message_span = cudf::detail::hostdevice_span<char>{message};

    message_span_copy = message_span;
  }

  EXPECT_EQ(message.host_ptr(), message_span_copy.host_ptr());
  EXPECT_EQ(message.device_ptr(), message_span_copy.device_ptr());
  EXPECT_EQ(message.size(), message_span_copy.size());
}

TEST(HostDeviceSpanTest, CanSendToDevice)
{
  auto original_message   = get_test_hostdevice_vector();
  cuda::stream_ref stream = cudf::get_default_stream();

  original_message.host_to_device_async(stream);

  std::string got_message(original_message.size(), '\0');
  cudaMemcpyAsync(got_message.data(),
                  original_message.device_ptr(),
                  original_message.size(),
                  cudaMemcpyDefault,
                  stream.get());
  stream.sync();

  EXPECT_EQ(got_message, hello_world_message);
}

CUDF_KERNEL void simple_device_char_kernel(device_span<char> result)
{
  char const* str = "world hello";
  for (int offset = 0; offset < result.size(); ++offset) {
    result.data()[offset] = str[offset];
  }
}

TEST(HostDeviceSpanTest, CanGetFromDevice)
{
  auto message = get_test_hostdevice_vector();
  message.host_to_device_async(cudf::get_default_stream());
  simple_device_char_kernel<<<1, 1, 0, cudf::get_default_stream().get()>>>(message);

  message.device_to_host(cudf::get_default_stream());
  expect_match("world hello", cudf::detail::hostdevice_span<char>(message));
}

CUDF_TEST_PROGRAM_MAIN()
