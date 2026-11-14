/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/memory_resource_utilities.hpp>

#include <cudf/detail/utilities/buffer_factories.hpp>
#include <cudf/detail/utilities/cuda_memcpy.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_device.hpp>

#include <cuda/stream>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <type_traits>
#include <vector>

namespace {

template <typename T>
class BufferFactoriesTest : public cudf::test::BaseFixture {
 protected:
  cuda::stream stream{cuda::device_ref{rmm::get_current_cuda_device().value()}};

  static std::vector<T> make_input(std::size_t size)
  {
    std::vector<T> input(size);
    for (std::size_t i = 0; i < size; ++i) {
      input[i] = static_cast<T>(static_cast<int>(i % 97) - 48);
    }
    return input;
  }

  template <typename Factory>
  void check_buffer(Factory factory, std::vector<T> const& expected)
  {
    cudf::test::memory_resource_test_harness harness;
    cudf::test::expect_api_uses_memory_resources(
      harness,
      factory,
      [](auto const& result) { return result.size() * sizeof(T); },
      [&](auto const& result) {
        static_assert(
          std::is_same_v<std::remove_cvref_t<decltype(result)>, cuda::device_buffer<T>>);
        ASSERT_EQ(result.size(), expected.size());
        std::vector<T> actual(result.size());
        if (!actual.empty()) {
          CUDF_CUDA_TRY(cudf::detail::memcpy_async(
            actual.data(), result.data(), actual.size() * sizeof(T), stream));
        }
        stream.sync();
        EXPECT_EQ(actual, expected);
      },
      {},
      stream);
  }
};

using BufferTypes = ::testing::Types<int8_t, int32_t, int64_t, float, double>;
TYPED_TEST_SUITE(BufferFactoriesTest, BufferTypes);

TYPED_TEST(BufferFactoriesTest, Zeroed)
{
  for (auto size : {0U, 1U, 257U, 65537U}) {
    SCOPED_TRACE(::testing::Message() << "size=" << size);
    this->check_buffer(
      [&](cudf::memory_resources mr) {
        return cudf::detail::make_zeroed_device_buffer<TypeParam>(size, this->stream, mr);
      },
      std::vector<TypeParam>(size, TypeParam{0}));
    SCOPED_TRACE(::testing::Message() << "size=" << size << ", async");
    this->check_buffer(
      [&](cudf::memory_resources mr) {
        return cudf::detail::make_zeroed_device_buffer_async<TypeParam>(size, this->stream, mr);
      },
      std::vector<TypeParam>(size, TypeParam{0}));
  }
}

TYPED_TEST(BufferFactoriesTest, ConstHostSpan)
{
  for (auto size : {0U, 1U, 257U, 65537U}) {
    auto const input = this->make_input(size);
    auto const span  = cudf::host_span<TypeParam const>{input};
    SCOPED_TRACE(::testing::Message() << "size=" << size);
    this->check_buffer(
      [&](cudf::memory_resources mr) {
        return cudf::detail::make_device_buffer(span, this->stream, mr);
      },
      input);
    SCOPED_TRACE(::testing::Message() << "size=" << size << ", async");
    this->check_buffer(
      [&](cudf::memory_resources mr) {
        return cudf::detail::make_device_buffer_async(span, this->stream, mr);
      },
      input);
  }
}

TYPED_TEST(BufferFactoriesTest, StdSpan)
{
  for (auto size : {0U, 1U, 257U, 65537U}) {
    auto const input = this->make_input(size);
    auto const span  = std::span<TypeParam const>{input};
    SCOPED_TRACE(::testing::Message() << "size=" << size);
    this->check_buffer(
      [&](cudf::memory_resources mr) {
        return cudf::detail::make_device_buffer(span, this->stream, mr);
      },
      input);
    SCOPED_TRACE(::testing::Message() << "size=" << size << ", async");
    this->check_buffer(
      [&](cudf::memory_resources mr) {
        return cudf::detail::make_device_buffer_async(span, this->stream, mr);
      },
      input);
    SCOPED_TRACE(::testing::Message() << "size=" << size << ", mutable");
    auto mutable_input = input;
    this->check_buffer(
      [&](cudf::memory_resources mr) {
        return cudf::detail::make_device_buffer(
          std::span<TypeParam>{mutable_input}, this->stream, mr);
      },
      input);
  }
}

TYPED_TEST(BufferFactoriesTest, MutableHostSpanAsync)
{
  for (auto size : {0U, 1U, 257U, 65537U}) {
    SCOPED_TRACE(::testing::Message() << "size=" << size);
    auto input = this->make_input(size);
    this->check_buffer(
      [&](cudf::memory_resources mr) {
        return cudf::detail::make_device_buffer_async(
          cudf::host_span<TypeParam>{input}, this->stream, mr);
      },
      input);
  }
}

TYPED_TEST(BufferFactoriesTest, StdVectorDeepCopy)
{
  for (auto size : {0U, 1U, 257U, 65537U}) {
    SCOPED_TRACE(::testing::Message() << "size=" << size);
    auto input          = this->make_input(size);
    auto const expected = input;
    this->check_buffer(
      [&](cudf::memory_resources mr) {
        auto result = cudf::detail::make_device_buffer(input, this->stream, mr);
        std::fill(input.begin(), input.end(), TypeParam{99});
        return result;
      },
      expected);
    SCOPED_TRACE(::testing::Message() << "size=" << size << ", async");
    input = expected;
    this->check_buffer(
      [&](cudf::memory_resources mr) {
        auto result = cudf::detail::make_device_buffer_async(input, this->stream, mr);
        // The asynchronous copy must finish before its host source is modified.
        this->stream.sync();
        std::fill(input.begin(), input.end(), TypeParam{99});
        return result;
      },
      expected);
  }
}

TYPED_TEST(BufferFactoriesTest, PinnedHostContainer)
{
  for (auto size : {0U, 1U, 257U, 65537U}) {
    auto const expected = this->make_input(size);
    auto input          = cudf::detail::make_pinned_vector_async<TypeParam>(size, this->stream);
    std::copy(expected.begin(), expected.end(), input.begin());
    ASSERT_TRUE(cudf::host_span<TypeParam const>{input}.is_device_accessible());
    SCOPED_TRACE(::testing::Message() << "size=" << size);
    this->check_buffer(
      [&](cudf::memory_resources mr) {
        return cudf::detail::make_device_buffer(input, this->stream, mr);
      },
      expected);
    SCOPED_TRACE(::testing::Message() << "size=" << size << ", async");
    this->check_buffer(
      [&](cudf::memory_resources mr) {
        return cudf::detail::make_device_buffer_async(input, this->stream, mr);
      },
      expected);
  }
}

}  // namespace
