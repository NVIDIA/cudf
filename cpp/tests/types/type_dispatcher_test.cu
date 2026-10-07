/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/cudf_gtest.hpp>
#include <cudf_test/testing_main.hpp>
#include <cudf_test/type_list_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/device_uvector.hpp>

struct DispatcherTest : public cudf::test::BaseFixture {};

template <typename T>
struct TypedDispatcherTest : public DispatcherTest {};

TYPED_TEST_SUITE(TypedDispatcherTest, cudf::test::AllTypes);

namespace {
template <typename Expected>
struct type_tester {
  template <typename Dispatched>
  bool operator()()
  {
    return std::is_same_v<Expected, Dispatched>;
  }
};
}  // namespace

TYPED_TEST(TypedDispatcherTest, TypeToId)
{
  EXPECT_TRUE(cudf::type_dispatcher(cudf::data_type{cudf::type_to_id<TypeParam>()},
                                    type_tester<TypeParam>{}));
  EXPECT_TRUE(cudf::type_dispatcher(cudf::data_type{cudf::type_to_id<TypeParam const>()},
                                    type_tester<TypeParam>{}));
  EXPECT_TRUE(cudf::type_dispatcher(cudf::data_type{cudf::type_to_id<TypeParam volatile>()},
                                    type_tester<TypeParam>{}));
  EXPECT_TRUE(cudf::type_dispatcher(cudf::data_type{cudf::type_to_id<TypeParam const volatile>()},
                                    type_tester<TypeParam>{}));
}

namespace {
struct verify_dispatched_type {
  cudf::type_id expected_id;

  template <typename T>
  __host__ __device__ bool operator()(cudf::type_id id)
  {
    return id == expected_id;
  }
};

CUDF_KERNEL void dispatch_test_kernel(cudf::type_id id, bool* d_result)
{
  if (0 == threadIdx.x + blockIdx.x * blockDim.x)
    *d_result = cudf::type_dispatcher(cudf::data_type{id}, verify_dispatched_type{id}, id);
}
}  // namespace

TYPED_TEST(TypedDispatcherTest, DeviceDispatch)
{
  auto result = cudf::detail::make_zeroed_device_uvector<bool>(
    1, cudf::get_default_stream(), cudf::get_current_device_resource_ref());
  dispatch_test_kernel<<<1, 1, 0, cudf::get_default_stream().get()>>>(cudf::type_to_id<TypeParam>(),
                                                                      result.data());
  CUDF_CUDA_TRY(cudaDeviceSynchronize());
  EXPECT_EQ(true, result.front_element(cudf::get_default_stream()));
}

struct IdDispatcherTest : public DispatcherTest,
                          public testing::WithParamInterface<cudf::type_id> {};

INSTANTIATE_TEST_CASE_P(TestAllIds, IdDispatcherTest, testing::ValuesIn(cudf::test::all_type_ids));

TEST_P(IdDispatcherTest, IdToType)
{
  auto t = GetParam();
  EXPECT_TRUE(cudf::type_dispatcher(cudf::data_type{t}, verify_dispatched_type{t}, t));
}

template <typename T>
struct TypedDoubleDispatcherTest : public DispatcherTest {};

TYPED_TEST_SUITE(TypedDoubleDispatcherTest, cudf::test::AllTypes);

namespace {
template <typename Expected1, typename Expected2>
struct two_type_tester {
  template <typename Dispatched1, typename Dispatched2>
  bool operator()()
  {
    return std::is_same_v<Expected1, Dispatched1> && std::is_same_v<Expected2, Dispatched2>;
  }
};
}  // namespace

TYPED_TEST(TypedDoubleDispatcherTest, TypeToId)
{
  EXPECT_TRUE(cudf::double_type_dispatcher(cudf::data_type{cudf::type_to_id<TypeParam>()},
                                           cudf::data_type{cudf::type_to_id<TypeParam>()},
                                           two_type_tester<TypeParam, TypeParam>{}));
}

namespace {
struct verify_double_dispatched_type {
  cudf::type_id expected_id1;
  cudf::type_id expected_id2;

  template <typename T1, typename T2>
  __host__ __device__ bool operator()(cudf::type_id id1, cudf::type_id id2)
  {
    return id1 == expected_id1 && id2 == expected_id2;
  }
};

CUDF_KERNEL void double_dispatch_test_kernel(cudf::type_id id1, cudf::type_id id2, bool* d_result)
{
  if (0 == threadIdx.x + blockIdx.x * blockDim.x)
    *d_result = cudf::double_type_dispatcher(cudf::data_type{id1},
                                             cudf::data_type{id2},
                                             verify_double_dispatched_type{id1, id2},
                                             id1,
                                             id2);
}
}  // namespace

TYPED_TEST(TypedDoubleDispatcherTest, DeviceDoubleDispatch)
{
  auto result = cudf::detail::make_zeroed_device_uvector<bool>(
    1, cudf::get_default_stream(), cudf::get_current_device_resource_ref());
  double_dispatch_test_kernel<<<1, 1, 0, cudf::get_default_stream().get()>>>(
    cudf::type_to_id<TypeParam>(), cudf::type_to_id<TypeParam>(), result.data());
  CUDF_CUDA_TRY(cudaDeviceSynchronize());
  EXPECT_EQ(true, result.front_element(cudf::get_default_stream()));
}

struct IdDoubleDispatcherTest : public DispatcherTest,
                                public testing::WithParamInterface<cudf::type_id> {};

INSTANTIATE_TEST_CASE_P(TestAllIds,
                        IdDoubleDispatcherTest,
                        testing::ValuesIn(cudf::test::all_type_ids));

TEST_P(IdDoubleDispatcherTest, IdToType)
{
  // Test double-dispatch of all types using the same type for both dispatches
  auto t = GetParam();
  EXPECT_TRUE(cudf::double_type_dispatcher(
    cudf::data_type{t}, cudf::data_type{t}, verify_double_dispatched_type{t, t}, t, t));
}

struct IdFixedDoubleDispatcherTest : public DispatcherTest,
                                     public testing::WithParamInterface<cudf::type_id> {};

INSTANTIATE_TEST_CASE_P(TestAllIds,
                        IdFixedDoubleDispatcherTest,
                        testing::ValuesIn(cudf::test::all_type_ids));

TEST_P(IdFixedDoubleDispatcherTest, IdToType)
{
  // Test double-dispatch of all types against one fixed type, in each direction
  auto t = GetParam();
  EXPECT_TRUE(
    cudf::double_type_dispatcher(cudf::data_type{t},
                                 cudf::data_type{cudf::type_to_id<float>()},
                                 verify_double_dispatched_type{t, cudf::type_to_id<float>()},
                                 t,
                                 cudf::type_to_id<float>()));
  EXPECT_TRUE(
    cudf::double_type_dispatcher(cudf::data_type{cudf::type_to_id<float>()},
                                 cudf::data_type{t},
                                 verify_double_dispatched_type{cudf::type_to_id<float>(), t},
                                 cudf::type_to_id<float>(),
                                 t));
}

CUDF_TEST_PROGRAM_MAIN()
