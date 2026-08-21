/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/cudf_gtest.hpp>
#include <cudf_test/type_list_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_device_view.cuh>
#include <cudf/strings/string_view.cuh>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <thrust/sequence.h>

#include <random>

template <typename T>
struct TypedScalarDeviceViewTest : public cudf::test::BaseFixture {};

TYPED_TEST_SUITE(TypedScalarDeviceViewTest, cudf::test::FixedWidthTypesWithoutFixedPoint);

template <typename T>
struct FixedPointScalarDeviceViewTest : public cudf::test::BaseFixture {};

TYPED_TEST_SUITE(FixedPointScalarDeviceViewTest, cudf::test::FixedPointTypes);

template <typename ScalarDeviceViewType>
CUDF_KERNEL void test_set_value(ScalarDeviceViewType s, ScalarDeviceViewType s1)
{
  s1.set_value(s.value());
  s1.set_valid(true);
}

template <typename ScalarDeviceViewType>
CUDF_KERNEL void test_value(ScalarDeviceViewType s, ScalarDeviceViewType s1, bool* result)
{
  *result = (s.value() == s1.value());
}

TYPED_TEST(TypedScalarDeviceViewTest, Value)
{
  TypeParam value  = cudf::test::make_type_param_scalar<TypeParam>(7);
  TypeParam value1 = cudf::test::make_type_param_scalar<TypeParam>(11);
  cudf::scalar_type_t<TypeParam> s(value);
  cudf::scalar_type_t<TypeParam> s1{value1};

  auto scalar_device_view  = cudf::get_scalar_device_view(s);
  auto scalar_device_view1 = cudf::get_scalar_device_view(s1);
  cudf::detail::device_scalar<bool> result{cudf::get_default_stream()};

  test_set_value<<<1, 1, 0, cudf::get_default_stream().value()>>>(scalar_device_view,
                                                                  scalar_device_view1);
  CUDF_CHECK_CUDA(0);

  EXPECT_EQ(s1.value(), value);
  EXPECT_TRUE(s1.is_valid());

  test_value<<<1, 1, 0, cudf::get_default_stream().value()>>>(
    scalar_device_view, scalar_device_view1, result.data());
  CUDF_CHECK_CUDA(0);

  EXPECT_TRUE(result.value(cudf::get_default_stream()));
}

template <typename ScalarDeviceViewType, typename T>
CUDF_KERNEL void test_fixed_point_value(ScalarDeviceViewType s, T expected, bool* result)
{
  auto const actual = s.value();
  *result           = actual.value() == expected.value() and actual.scale() == expected.scale();
}

TYPED_TEST(FixedPointScalarDeviceViewTest, Value)
{
  using rep_type = typename TypeParam::rep;

  auto constexpr value = rep_type{12'345};
  auto constexpr scale = cudf::numeric::scale_type{-2};
  auto const expected  = TypeParam{cudf::numeric::scaled_integer<rep_type>{value, scale}};
  cudf::fixed_point_scalar<TypeParam> s{value, scale};

  auto scalar_device_view = cudf::get_scalar_device_view(s);
  cudf::detail::device_scalar<bool> result{cudf::get_default_stream()};

  test_fixed_point_value<<<1, 1, 0, cudf::get_default_stream().value()>>>(
    scalar_device_view, expected, result.data());
  CUDF_CHECK_CUDA(0);

  EXPECT_TRUE(result.value(cudf::get_default_stream()));
}

template <typename ScalarDeviceViewType>
CUDF_KERNEL void test_set_fixed_point_rep(ScalarDeviceViewType s,
                                          typename ScalarDeviceViewType::rep_type value)
{
  s.set_value(value);
}

TYPED_TEST(FixedPointScalarDeviceViewTest, SetRepresentation)
{
  using rep_type = typename TypeParam::rep;

  auto constexpr initial_value = rep_type{0};
  auto constexpr value         = rep_type{12'345};
  auto constexpr scale         = cudf::numeric::scale_type{-2};
  cudf::fixed_point_scalar<TypeParam> s{initial_value, scale};

  auto scalar_device_view = cudf::get_scalar_device_view(s);
  test_set_fixed_point_rep<<<1, 1, 0, cudf::get_default_stream().value()>>>(scalar_device_view,
                                                                            value);
  CUDF_CHECK_CUDA(0);

  EXPECT_EQ(s.value(), value);
}

TYPED_TEST(FixedPointScalarDeviceViewTest, SetValue)
{
  using rep_type = typename TypeParam::rep;

  auto constexpr source_rep   = rep_type{12'345};
  auto constexpr initial_rep  = rep_type{0};
  auto constexpr source_scale = cudf::numeric::scale_type{-2};
  auto constexpr target_scale = cudf::numeric::scale_type{-3};
  auto const source_value =
    TypeParam{cudf::numeric::scaled_integer<rep_type>{source_rep, source_scale}};
  auto const expected_rep = source_value.rescaled(target_scale).value();
  cudf::fixed_point_scalar<TypeParam> source{source_rep, source_scale};
  cudf::fixed_point_scalar<TypeParam> same_scale_target{initial_rep, source_scale};
  cudf::fixed_point_scalar<TypeParam> target{initial_rep, target_scale};

  auto source_device_view            = cudf::get_scalar_device_view(source);
  auto same_scale_target_device_view = cudf::get_scalar_device_view(same_scale_target);
  auto target_device_view            = cudf::get_scalar_device_view(target);
  test_set_value<<<1, 1, 0, cudf::get_default_stream().value()>>>(source_device_view,
                                                                  same_scale_target_device_view);
  CUDF_CHECK_CUDA(0);

  test_set_value<<<1, 1, 0, cudf::get_default_stream().value()>>>(source_device_view,
                                                                  target_device_view);
  CUDF_CHECK_CUDA(0);

  EXPECT_EQ(same_scale_target.value(), source_rep);
  EXPECT_EQ(target.value(), expected_rep);
}

template <typename ScalarDeviceViewType>
CUDF_KERNEL void test_null(ScalarDeviceViewType s, bool* result)
{
  *result = s.is_valid();
}

TYPED_TEST(TypedScalarDeviceViewTest, ConstructNull)
{
  TypeParam value = cudf::test::make_type_param_scalar<TypeParam>(5);
  cudf::scalar_type_t<TypeParam> s(value, false);
  auto scalar_device_view = cudf::get_scalar_device_view(s);
  cudf::detail::device_scalar<bool> result{cudf::get_default_stream()};

  test_null<<<1, 1, 0, cudf::get_default_stream().value()>>>(scalar_device_view, result.data());
  CUDF_CHECK_CUDA(0);

  EXPECT_FALSE(result.value(cudf::get_default_stream()));
}

template <typename ScalarDeviceViewType>
CUDF_KERNEL void test_setnull(ScalarDeviceViewType s)
{
  s.set_valid(false);
}

TYPED_TEST(TypedScalarDeviceViewTest, SetNull)
{
  TypeParam value = cudf::test::make_type_param_scalar<TypeParam>(5);
  cudf::scalar_type_t<TypeParam> s{value};
  auto scalar_device_view = cudf::get_scalar_device_view(s);
  s.set_valid_async(true);
  EXPECT_TRUE(s.is_valid());

  test_setnull<<<1, 1, 0, cudf::get_default_stream().value()>>>(scalar_device_view);
  CUDF_CHECK_CUDA(0);

  EXPECT_FALSE(s.is_valid());
}

struct StringScalarDeviceViewTest : public cudf::test::BaseFixture {};

CUDF_KERNEL void test_string_value(cudf::string_scalar_device_view s,
                                   char const* value,
                                   cudf::size_type size,
                                   bool* result)
{
  *result = (s.value() == cudf::string_view(value, size));
}

TEST_F(StringScalarDeviceViewTest, Value)
{
  std::string value("test string");
  cudf::string_scalar s(value);

  auto scalar_device_view = cudf::get_scalar_device_view(s);
  cudf::detail::device_scalar<bool> result{cudf::get_default_stream()};
  auto value_v = cudf::detail::make_device_uvector(cudf::host_span<char const>{value},
                                                   cudf::get_default_stream(),
                                                   cudf::get_current_device_resource_ref());

  test_string_value<<<1, 1, 0, cudf::get_default_stream().value()>>>(
    scalar_device_view, value_v.data(), value.size(), result.data());
  CUDF_CHECK_CUDA(0);

  EXPECT_TRUE(result.value(cudf::get_default_stream()));
}
