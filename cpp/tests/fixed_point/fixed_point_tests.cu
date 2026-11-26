/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/detail/operators/logic.cuh>
#include <cudf/detail/utilities/buffer_factories.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/exec_policy.hpp>

#include <cuda/buffer>
#include <cuda/iterator>
#include <cuda/std/optional>
#include <thrust/logical.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/transform.h>

#include <algorithm>
#include <limits>
#include <numeric>
#include <vector>

using namespace numeric;

struct FixedPointTest : public cudf::test::BaseFixture {};

template <typename T>
struct FixedPointTestAllReps : public cudf::test::BaseFixture {};

using RepresentationTypes = ::testing::Types<int32_t, int64_t, __int128_t>;

TYPED_TEST_SUITE(FixedPointTestAllReps, RepresentationTypes);

TYPED_TEST(FixedPointTestAllReps, DecimalXXThrust)
{
  using decimalXX = fixed_point<TypeParam, Radix::BASE_10>;

  std::vector<decimalXX> vec1(1000);
  std::vector<int32_t> vec2(1000);

  std::iota(std::begin(vec1), std::end(vec1), decimalXX{0, scale_type{-2}});
  std::iota(std::begin(vec2), std::end(vec2), 0);

  auto const res1 =
    thrust::reduce(std::cbegin(vec1), std::cend(vec1), decimalXX{0, scale_type{-2}});

  auto const res2 = std::accumulate(std::cbegin(vec2), std::cend(vec2), 0);

  EXPECT_EQ(static_cast<int32_t>(res1), res2);

  std::vector<int32_t> vec3(vec1.size());

  thrust::transform(std::cbegin(vec1), std::cend(vec1), std::begin(vec3), [](auto const& e) {
    return static_cast<int32_t>(e);
  });

  EXPECT_EQ(vec2, vec3);
}

namespace {

struct decimal32_decimal64_null_logic {
  __device__ bool operator()(int i) const
  {
    // Each operand takes the values false, true and null, with different decimal scales.
    auto const lhs = i / 3 == 2 ? cuda::std::optional<decimal32>{}
                                : cuda::std::optional{decimal32{i / 3, scale_type{-2}}};
    auto const rhs = i % 3 == 2 ? cuda::std::optional<decimal64>{}
                                : cuda::std::optional{decimal64{i % 3, scale_type{1}}};
    using cuda::std::nullopt;
    cuda::std::optional<bool> const expected_and[] = {
      false, false, false, false, true, nullopt, false, nullopt, nullopt};
    cuda::std::optional<bool> const expected_or[] = {
      false, true, nullopt, true, true, true, nullopt, true, nullopt};

    return cudf::detail::ops::null_logical_and(lhs, rhs) == expected_and[i] &&
           cudf::detail::ops::null_logical_and(rhs, lhs) == expected_and[i] &&
           cudf::detail::ops::null_logical_or(lhs, rhs) == expected_or[i] &&
           cudf::detail::ops::null_logical_or(rhs, lhs) == expected_or[i];
  }
};

struct cast_to_int32_fn {
  using decimal32 = fixed_point<int32_t, Radix::BASE_10>;
  int32_t __host__ __device__ operator()(decimal32 fp) { return static_cast<int32_t>(fp); }
};
}  // namespace

TEST_F(FixedPointTest, ImplicitNullLogicCast)
{
  EXPECT_TRUE(thrust::all_of(rmm::exec_policy_nosync(cudf::get_default_stream()),
                             cuda::counting_iterator<int>{0},
                             cuda::counting_iterator<int>{9},
                             decimal32_decimal64_null_logic{}));
}

TEST_F(FixedPointTest, DecimalXXThrustOnDevice)
{
  using decimal32 = fixed_point<int32_t, Radix::BASE_10>;

  std::vector<decimal32> vec1(1000, decimal32{1, scale_type{-2}});
  auto d_vec1 = cudf::detail::make_device_buffer(
    vec1, cudf::get_default_stream(), cudf::get_current_device_resource_ref());

  auto const sum = thrust::reduce(rmm::exec_policy_nosync(cudf::get_default_stream()),
                                  std::cbegin(d_vec1),
                                  std::cend(d_vec1),
                                  decimal32{0, scale_type{-2}});

  EXPECT_EQ(static_cast<int32_t>(sum), 1000);

  // TODO: Once nvbugs/1990211 is fixed (ExclusiveSum initial_value = 0 bug)
  //       change inclusive scan to run on device (avoid copying to host)
  thrust::inclusive_scan(std::cbegin(vec1), std::cend(vec1), std::begin(vec1));

  d_vec1 = cudf::detail::make_device_buffer(
    vec1, cudf::get_default_stream(), cudf::get_current_device_resource_ref());

  std::vector<int32_t> vec2(1000);
  std::iota(std::begin(vec2), std::end(vec2), 1);

  auto const res1 = thrust::reduce(rmm::exec_policy_nosync(cudf::get_default_stream()),
                                   std::cbegin(d_vec1),
                                   std::cend(d_vec1),
                                   decimal32{0, scale_type{-2}});

  auto const res2 = std::accumulate(std::cbegin(vec2), std::cend(vec2), 0);

  EXPECT_EQ(static_cast<int32_t>(res1), res2);

  cuda::device_buffer<int32_t> d_vec3(
    cudf::get_default_stream(), cudf::get_current_device_resource_ref(), 1000, cuda::no_init);

  thrust::transform(rmm::exec_policy_nosync(cudf::get_default_stream()),
                    std::cbegin(d_vec1),
                    std::cend(d_vec1),
                    std::begin(d_vec3),
                    cast_to_int32_fn{});

  auto vec3 = cudf::detail::make_std_vector(d_vec3, cudf::get_default_stream());

  EXPECT_EQ(vec2, vec3);
}
