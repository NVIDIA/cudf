/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/cudf_gtest.hpp>
#include <cudf_test/testing_main.hpp>

#include <cudf/binary/binary_column_factories.hpp>
#include <cudf/binary/binary_column_view.hpp>
#include <cudf/binary/binary_view.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/get_value.cuh>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/hashing.hpp>
#include <cudf/sorting.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>

#include <cuda_runtime_api.h>

#include <array>
#include <cstdint>
#include <type_traits>
#include <vector>

namespace {

struct BinaryTest : public cudf::test::BaseFixture {};

struct is_binary_dispatch {
  template <typename T>
  bool operator()() const
  {
    return std::is_same_v<T, cudf::binary_view>;
  }
};

rmm::device_buffer make_payload(std::vector<uint8_t> const& bytes,
                                rmm::cuda_stream_view stream)
{
  rmm::device_buffer result(bytes.size(), stream);
  if (not bytes.empty()) {
    CUDF_CUDA_TRY(cudaMemcpyAsync(
      result.data(), bytes.data(), bytes.size(), cudaMemcpyHostToDevice, stream.value()));
  }
  return result;
}

template <typename Offset>
std::unique_ptr<cudf::column> make_test_column(rmm::cuda_stream_view stream)
{
  auto offsets = cudf::test::fixed_width_column_wrapper<Offset>({0, 2, 2, 5}).release();
  auto payload = make_payload({0x00, 0xFF, 0x61, 0x62, 0x63}, stream);
  return cudf::make_binary_column(
    3, std::move(offsets), std::move(payload), 0, rmm::device_buffer{});
}

CUDF_KERNEL void inspect_binary(cudf::column_device_view const* input,
                                cudf::size_type* sizes,
                                uint8_t* first_bytes)
{
  auto const index = static_cast<cudf::size_type>(threadIdx.x);
  if (index < input->size()) {
    auto const value   = input->element<cudf::binary_view>(index);
    sizes[index]       = value.size_bytes();
    first_bytes[index] = value.empty() ? uint8_t{0} : value[0];
  }
}

TEST_F(BinaryTest, BinaryViewUsesUnsignedLexicographicOrdering)
{
  std::array<uint8_t, 2> low{0x00, 0xFF};
  std::array<uint8_t, 2> high{0x80, 0x00};
  std::array<uint8_t, 3> extension{0x00, 0xFF, 0x00};

  auto const low_view       = cudf::binary_view{low.data(), low.size()};
  auto const high_view      = cudf::binary_view{high.data(), high.size()};
  auto const extension_view = cudf::binary_view{extension.data(), extension.size()};

  EXPECT_LT(low_view, high_view);
  EXPECT_LT(low_view, extension_view);
  EXPECT_EQ(low_view, (cudf::binary_view{low.data(), low.size()}));
}

TEST_F(BinaryTest, TypeDispatchesToBinaryView)
{
  static_assert(cudf::type_to_id<cudf::binary_view>() == cudf::type_id::BINARY);
  EXPECT_TRUE(
    cudf::type_dispatcher(cudf::data_type{cudf::type_id::BINARY}, is_binary_dispatch{}));
  EXPECT_EQ(cudf::type_to_name(cudf::data_type{cudf::type_id::BINARY}), "cudf::binary_view");
}

TEST_F(BinaryTest, TypeTraitsDescribeVariableWidthLeaf)
{
  auto const type = cudf::data_type{cudf::type_id::BINARY};

  EXPECT_TRUE(cudf::is_compound(type));
  EXPECT_FALSE(cudf::is_nested(type));
  EXPECT_FALSE(cudf::is_fixed_width(type));
  EXPECT_TRUE(cudf::is_equality_comparable(type));
  EXPECT_TRUE(cudf::is_relationally_comparable(type));
}

TEST_F(BinaryTest, FactoryAcceptsInt32Offsets)
{
  auto const stream = cudf::get_default_stream();
  auto column       = make_test_column<int32_t>(stream);
  auto view         = cudf::binary_column_view{column->view()};

  EXPECT_EQ(column->type().id(), cudf::type_id::BINARY);
  EXPECT_EQ(column->size(), 3);
  EXPECT_EQ(column->num_children(), 1);
  EXPECT_EQ(view.offsets().type().id(), cudf::type_id::INT32);
  EXPECT_EQ(view.bytes_size(stream), 5);
}

TEST_F(BinaryTest, FactoryAcceptsInt64Offsets)
{
  auto const stream = cudf::get_default_stream();
  auto column       = make_test_column<int64_t>(stream);
  auto view         = cudf::binary_column_view{column->view()};

  EXPECT_EQ(view.offsets().type().id(), cudf::type_id::INT64);
  EXPECT_EQ(view.bytes_size(stream), 5);
}

TEST_F(BinaryTest, EmptyColumnHasCanonicalLayout)
{
  auto column = cudf::make_empty_binary_column();

  EXPECT_EQ(column->type().id(), cudf::type_id::BINARY);
  EXPECT_EQ(column->size(), 0);
  EXPECT_EQ(column->num_children(), 0);
  EXPECT_EQ(cudf::binary_column_view{column->view()}.bytes_size(cudf::get_default_stream()), 0);
}

TEST_F(BinaryTest, FactoryRejectsInvalidOffsetsType)
{
  auto offsets = cudf::test::fixed_width_column_wrapper<uint32_t>({0, 1}).release();
  auto payload = make_payload({0x01}, cudf::get_default_stream());

  EXPECT_THROW(cudf::make_binary_column(
                 1, std::move(offsets), std::move(payload), 0, rmm::device_buffer{}),
               cudf::logic_error);
}

TEST_F(BinaryTest, DeviceViewAccessesRows)
{
  auto const stream = cudf::get_default_stream();
  auto column       = make_test_column<int64_t>(stream);
  auto device_view  = cudf::column_device_view::create(column->view(), stream);
  rmm::device_uvector<cudf::size_type> sizes(column->size(), stream);
  rmm::device_uvector<uint8_t> first_bytes(column->size(), stream);

  inspect_binary<<<1, column->size(), 0, stream.value()>>>(
    device_view.get(), sizes.data(), first_bytes.data());
  CUDF_CUDA_TRY(cudaGetLastError());

  auto const host_sizes = cudf::detail::make_std_vector_async(sizes, stream);
  auto const host_first = cudf::detail::make_std_vector_async(first_bytes, stream);
  stream.synchronize();

  EXPECT_EQ(host_sizes, (std::vector<cudf::size_type>{2, 0, 3}));
  EXPECT_EQ(host_first, (std::vector<uint8_t>{0x00, 0x00, 0x61}));
}

TEST_F(BinaryTest, OwningCopyNormalizesSlicedOffsets)
{
  auto const stream = cudf::get_default_stream();
  auto input        = make_test_column<int64_t>(stream);
  auto const sliced = cudf::slice(input->view(), {1, 3}, stream).front();
  auto copied       = std::make_unique<cudf::column>(sliced, stream);
  auto copied_view  = cudf::binary_column_view{copied->view()};

  EXPECT_EQ(copied->size(), 2);
  EXPECT_EQ(copied_view.bytes_size(stream), 3);
  EXPECT_EQ(cudf::detail::get_value<int64_t>(copied_view.offsets(), 0, stream), 0);
  EXPECT_EQ(cudf::detail::get_value<int64_t>(copied_view.offsets(), 2, stream), 3);

  auto device_view = cudf::column_device_view::create(copied->view(), stream);
  rmm::device_uvector<cudf::size_type> sizes(copied->size(), stream);
  rmm::device_uvector<uint8_t> first_bytes(copied->size(), stream);
  inspect_binary<<<1, copied->size(), 0, stream.value()>>>(
    device_view.get(), sizes.data(), first_bytes.data());
  CUDF_CUDA_TRY(cudaGetLastError());

  auto const host_sizes = cudf::detail::make_std_vector_async(sizes, stream);
  auto const host_first = cudf::detail::make_std_vector_async(first_bytes, stream);
  stream.synchronize();

  EXPECT_EQ(host_sizes, (std::vector<cudf::size_type>{0, 3}));
  EXPECT_EQ(host_first, (std::vector<uint8_t>{0x00, 0x61}));
}

TEST_F(BinaryTest, GatherReordersBinaryRows)
{
  auto const stream = cudf::get_default_stream();
  auto input        = make_test_column<int32_t>(stream);
  auto gather_map   = cudf::test::fixed_width_column_wrapper<cudf::size_type>({2, 0, 1});
  auto result = cudf::gather(cudf::table_view{{input->view()}},
                             gather_map,
                             cudf::out_of_bounds_policy::DONT_CHECK,
                             stream);
  auto output = std::move(result->release().front());

  auto device_view = cudf::column_device_view::create(output->view(), stream);
  rmm::device_uvector<cudf::size_type> sizes(output->size(), stream);
  rmm::device_uvector<uint8_t> first_bytes(output->size(), stream);
  inspect_binary<<<1, output->size(), 0, stream.value()>>>(
    device_view.get(), sizes.data(), first_bytes.data());
  CUDF_CUDA_TRY(cudaGetLastError());

  auto const host_sizes = cudf::detail::make_std_vector_async(sizes, stream);
  auto const host_first = cudf::detail::make_std_vector_async(first_bytes, stream);
  stream.synchronize();

  EXPECT_EQ(host_sizes, (std::vector<cudf::size_type>{3, 2, 0}));
  EXPECT_EQ(host_first, (std::vector<uint8_t>{0x61, 0x00, 0x00}));
}

TEST_F(BinaryTest, GatherNullifiesOutOfBoundsRows)
{
  auto const stream = cudf::get_default_stream();
  auto input        = make_test_column<int32_t>(stream);
  auto gather_map   = cudf::test::fixed_width_column_wrapper<cudf::size_type>({0, 99, 2});
  auto result = cudf::gather(cudf::table_view{{input->view()}},
                             gather_map,
                             cudf::out_of_bounds_policy::NULLIFY,
                             stream);
  auto output = std::move(result->release().front());

  EXPECT_EQ(output->null_count(), 1);
  EXPECT_EQ(cudf::binary_column_view{output->view()}.bytes_size(stream), 5);
}

TEST_F(BinaryTest, ConcatenateSupportsSlicedColumns)
{
  auto const stream = cudf::get_default_stream();
  auto input        = make_test_column<int32_t>(stream);
  auto const sliced = cudf::slice(input->view(), {1, 3}, stream).front();
  auto output =
    cudf::concatenate(std::vector<cudf::column_view>{sliced, input->view()}, stream);

  auto device_view = cudf::column_device_view::create(output->view(), stream);
  rmm::device_uvector<cudf::size_type> sizes(output->size(), stream);
  rmm::device_uvector<uint8_t> first_bytes(output->size(), stream);
  inspect_binary<<<1, output->size(), 0, stream.value()>>>(
    device_view.get(), sizes.data(), first_bytes.data());
  CUDF_CUDA_TRY(cudaGetLastError());

  auto const host_sizes = cudf::detail::make_std_vector_async(sizes, stream);
  stream.synchronize();

  EXPECT_EQ(host_sizes, (std::vector<cudf::size_type>{0, 3, 2, 0, 3}));
  EXPECT_EQ(cudf::binary_column_view{output->view()}.bytes_size(stream), 8);
}

TEST_F(BinaryTest, ScatterMovesBinaryRows)
{
  auto const stream = cudf::get_default_stream();
  auto target       = make_test_column<int32_t>(stream);
  auto const source = cudf::slice(target->view(), {2, 3}, stream).front();
  auto scatter_map  = cudf::test::fixed_width_column_wrapper<cudf::size_type>({1});
  auto result = cudf::scatter(cudf::table_view{{source}},
                              scatter_map,
                              cudf::table_view{{target->view()}},
                              stream);
  auto output = std::move(result->release().front());

  auto device_view = cudf::column_device_view::create(output->view(), stream);
  rmm::device_uvector<cudf::size_type> sizes(output->size(), stream);
  rmm::device_uvector<uint8_t> first_bytes(output->size(), stream);
  inspect_binary<<<1, output->size(), 0, stream.value()>>>(
    device_view.get(), sizes.data(), first_bytes.data());
  CUDF_CUDA_TRY(cudaGetLastError());

  auto const host_sizes = cudf::detail::make_std_vector_async(sizes, stream);
  stream.synchronize();

  EXPECT_EQ(host_sizes, (std::vector<cudf::size_type>{2, 3, 3}));
  EXPECT_EQ(cudf::binary_column_view{output->view()}.bytes_size(stream), 8);
}

TEST_F(BinaryTest, HashesBinaryPayloadBytes)
{
  auto const stream = cudf::get_default_stream();
  auto offsets      = cudf::test::fixed_width_column_wrapper<int32_t>({0, 2, 4, 4}).release();
  auto payload      = make_payload({0x00, 0xFF, 0x00, 0xFF}, stream);
  auto input        = cudf::make_binary_column(
    3, std::move(offsets), std::move(payload), 0, rmm::device_buffer{});

  auto hashes = cudf::hashing::murmurhash3_x86_32(cudf::table_view{{input->view()}}, 0, stream);
  auto const [host_hashes, validity] = cudf::test::to_host<uint32_t>(hashes->view(), stream);

  EXPECT_EQ(host_hashes[0], host_hashes[1]);
  EXPECT_NE(host_hashes[0], host_hashes[2]);
}

TEST_F(BinaryTest, SortsByUnsignedByteOrder)
{
  auto const stream = cudf::get_default_stream();
  auto input        = make_test_column<int32_t>(stream);

  auto order = cudf::sorted_order(cudf::table_view{{input->view()}}, {}, {}, stream);
  auto const [host_order, validity] =
    cudf::test::to_host<cudf::size_type>(order->view(), stream);

  EXPECT_EQ(host_order, (std::vector<cudf::size_type>{1, 0, 2}));
}

TEST_F(BinaryTest, DistinctUsesBinaryEquality)
{
  auto const stream = cudf::get_default_stream();
  auto offsets      = cudf::test::fixed_width_column_wrapper<int32_t>({0, 2, 4, 4}).release();
  auto payload      = make_payload({0x00, 0xFF, 0x00, 0xFF}, stream);
  auto input        = cudf::make_binary_column(
    3, std::move(offsets), std::move(payload), 0, rmm::device_buffer{});

  auto result = cudf::distinct(cudf::table_view{{input->view()}},
                               {0},
                               cudf::duplicate_keep_option::KEEP_FIRST,
                               cudf::null_equality::EQUAL,
                               cudf::nan_equality::ALL_EQUAL,
                               stream);

  EXPECT_EQ(result->num_rows(), 2);
  EXPECT_EQ(result->get_column(0).type().id(), cudf::type_id::BINARY);
}

}  // namespace

CUDF_TEST_PROGRAM_MAIN()
