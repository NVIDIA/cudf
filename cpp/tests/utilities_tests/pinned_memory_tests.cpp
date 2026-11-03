/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "io/utilities/hostdevice_vector.hpp"

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/memory_resource_utilities.hpp>
#include <cudf_test/table_utilities.hpp>
#include <cudf_test/testing_main.hpp>

#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/io/config_utils.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/pinned_memory.hpp>
#include <cudf/utilities/span.hpp>

#include <cuda/iterator>

using cudf::host_span;
using cudf::detail::host_2dspan;
using cudf::detail::hostdevice_2dvector;
using cudf::detail::hostdevice_span;
using cudf::detail::hostdevice_vector;

class PinnedMemoryTest : public cudf::test::BaseFixture {
  size_t prev_copy_threshold;
  size_t prev_alloc_threshold;

 public:
  PinnedMemoryTest()
    : prev_copy_threshold{cudf::get_kernel_pinned_copy_threshold()},
      prev_alloc_threshold{cudf::get_allocate_host_as_pinned_threshold()}
  {
  }
  ~PinnedMemoryTest() override
  {
    cudf::set_kernel_pinned_copy_threshold(prev_copy_threshold);
    cudf::set_allocate_host_as_pinned_threshold(prev_alloc_threshold);
  }
};

TEST_F(PinnedMemoryTest, MemoryResourceGetAndSet)
{
  // Global environment for temporary files
  auto const temp_env = static_cast<cudf::test::TempDirTestEnvironment*>(
    ::testing::AddGlobalTestEnvironment(new cudf::test::TempDirTestEnvironment));

  // pinned/pooled host memory resource
  cudf::test::pinned_pool mr{4 * 1024 * 1024};

  // set new resource
  auto last_mr = cudf::get_pinned_memory_resource();
  cudf::set_pinned_memory_resource(mr);

  constexpr int num_rows = 32 * 1024;
  auto valids =
    cudf::detail::make_counting_transform_iterator(0, [&](int index) { return index % 2; });
  auto values = cuda::counting_iterator<int>{0};

  cudf::test::fixed_width_column_wrapper<int> col(values, values + num_rows, valids);

  cudf::table_view expected({col});
  auto filepath = temp_env->get_temp_filepath("MemoryResourceGetAndSetTest.parquet");
  cudf::io::parquet_writer_options out_args =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, expected);
  cudf::io::write_parquet(out_args);

  cudf::io::parquet_reader_options const read_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
  auto const result = cudf::io::read_parquet(read_opts);
  CUDF_TEST_EXPECT_TABLES_EQUAL(*result.tbl, expected);

  // reset memory resource back
  cudf::set_pinned_memory_resource(last_mr);
}

TEST_F(PinnedMemoryTest, KernelCopyThresholdGetAndSet)
{
  cudf::set_kernel_pinned_copy_threshold(12345);
  EXPECT_EQ(cudf::get_kernel_pinned_copy_threshold(), 12345);
}

TEST_F(PinnedMemoryTest, HostAsPinnedThresholdGetAndSet)
{
  cudf::set_allocate_host_as_pinned_threshold(12345);
  EXPECT_EQ(cudf::get_allocate_host_as_pinned_threshold(), 12345);
}

TEST_F(PinnedMemoryTest, MakePinnedVector)
{
  cudf::set_allocate_host_as_pinned_threshold(0);

  // should always use pinned memory
  {
    auto const vec = cudf::detail::make_pinned_vector_async<char>(1, cudf::get_default_stream());
    EXPECT_TRUE(vec.get_allocator().is_device_accessible());
  }
}

TEST_F(PinnedMemoryTest, MakeHostVector)
{
  cudf::set_allocate_host_as_pinned_threshold(7);

  // allocate smaller than the threshold
  {
    auto const vec = cudf::detail::make_host_vector<int32_t>(1, cudf::get_default_stream());
    EXPECT_TRUE(vec.get_allocator().is_device_accessible());
  }

  // allocate the same size as the threshold
  {
    auto const vec = cudf::detail::make_host_vector<char>(7, cudf::get_default_stream());
    EXPECT_TRUE(vec.get_allocator().is_device_accessible());
  }

  // allocate larger than the threshold
  {
    auto const vec = cudf::detail::make_host_vector<int32_t>(2, cudf::get_default_stream());
    EXPECT_FALSE(vec.get_allocator().is_device_accessible());
  }
}

TEST_F(PinnedMemoryTest, HostSpan)
{
  auto test_ctors = [](auto&& vec) {
    auto const is_vec_device_accessible = vec.get_allocator().is_device_accessible();
    // Test conversion from a vector
    auto const span = host_span<int16_t>{vec};
    EXPECT_EQ(span.is_device_accessible(), is_vec_device_accessible);
    // Test conversion from host_span with different type
    auto const span_converted = host_span<int16_t const>{span};
    EXPECT_EQ(span_converted.is_device_accessible(), is_vec_device_accessible);
  };

  cudf::set_allocate_host_as_pinned_threshold(7);
  for (int i = 1; i < 10; i++) {
    // some iterations will use pinned memory, some will not
    test_ctors(cudf::detail::make_host_vector<int16_t>(i, cudf::get_default_stream()));
  }

  auto stream{cudf::get_default_stream()};

  // hostdevice vectors use pinned memory for the host side; test that host_span can be constructed
  // from a hostdevice_vector with correct device accessibility

  hostdevice_vector<int16_t> hd_vec(10, stream, cudf::get_current_device_resource_ref());
  auto const span = host_span<int16_t>{hd_vec};
  EXPECT_TRUE(span.is_device_accessible());

  // test host_view and operator[]
  {
    hostdevice_2dvector<int16_t> hd_2dvec(10, 10, stream, cudf::get_current_device_resource_ref());
    auto const span2d = hd_2dvec.host_view().flat_view();
    EXPECT_TRUE(span2d.is_device_accessible());

    auto const span2d_from_cast = host_2dspan<int16_t>{hd_2dvec};
    EXPECT_TRUE(span2d_from_cast.flat_view().is_device_accessible());

    auto const row_span = hd_2dvec[0];
    EXPECT_TRUE(row_span.is_device_accessible());
  }

  // test const versions of host_view and operator[]
  {
    hostdevice_2dvector<int16_t> const const_hd_2dvec(
      10, 10, stream, cudf::get_current_device_resource_ref());
    auto const const_span2d = const_hd_2dvec.host_view().flat_view();
    EXPECT_TRUE(const_span2d.is_device_accessible());

    auto const const_span2d_from_cast = host_2dspan<int16_t const>{const_hd_2dvec};
    EXPECT_TRUE(const_span2d_from_cast.flat_view().is_device_accessible());

    auto const const_row_span = const_hd_2dvec[0];
    EXPECT_TRUE(const_row_span.is_device_accessible());
  }

  // test hostdevice_span
  {
    hostdevice_span<int16_t> hd_span(hd_vec);
    EXPECT_TRUE(host_span<int16_t>{hd_span}.is_device_accessible());
  }
}

TEST_F(PinnedMemoryTest, HostDeviceVectorMemoryResources)
{
  auto const stream = cudf::test::get_default_stream();
  // With the integrated memory optimization, the device side aliases the pinned host buffer
  auto const device_bytes = [](std::size_t bytes) {
    return cudf::io::integrated_memory_optimization::is_enabled() ? 0 : bytes;
  };

  {
    auto harness = cudf::test::memory_resource_test_harness{};
    cudf::test::expect_api_uses_memory_resources(
      harness,
      [&](cudf::memory_resources mr) { return hostdevice_vector<int16_t>(10, stream, mr); },
      [&](auto const& result) { return device_bytes(result.size_bytes()); },
      [](auto const& result) { EXPECT_EQ(result.size(), 10); },
      {},
      stream);
  }

  {
    auto harness = cudf::test::memory_resource_test_harness{};
    cudf::test::expect_api_uses_memory_resources(
      harness,
      [&](cudf::memory_resources mr) { return hostdevice_2dvector<int16_t>(4, 5, stream, mr); },
      [&](auto const& result) { return device_bytes(result.size_bytes()); },
      [](auto const& result) {
        EXPECT_EQ(result.size().first, 4);
        EXPECT_EQ(result.size().second, 5);
      },
      {},
      stream);
  }
}
