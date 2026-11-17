/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/testing_main.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/detail/iterator.cuh>
#include <cudf/filling.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/traits.hpp>

#include <cuda/iterator>

auto all_valid   = [](cudf::size_type row) { return true; };
auto odd_valid   = [](cudf::size_type row) { return row % 2 != 0; };
auto all_invalid = [](cudf::size_type row) { return false; };

template <typename T>
class FillTypedTestFixture : public cudf::test::BaseFixture {
 public:
  static constexpr cudf::size_type column_size{1000};

  template <typename BitInitializerType = decltype(all_valid)>
  void test(cudf::size_type begin,
            cudf::size_type end,
            T value,
            bool value_is_valid                     = true,
            BitInitializerType destination_validity = all_valid)
  {
    static_assert(cudf::is_fixed_width<T>(), "this code assumes fixed-width types.");

    cudf::size_type size{FillTypedTestFixture<T>::column_size};

    cudf::test::fixed_width_column_wrapper<T, int32_t> destination(
      cuda::counting_iterator<int32_t>{0},
      cuda::counting_iterator<int32_t>{0} + size,
      cudf::detail::make_counting_transform_iterator(0, destination_validity));

    std::unique_ptr<cudf::scalar> p_val{nullptr};
    cudf::data_type type{cudf::type_to_id<T>()};
    if (cudf::is_numeric<T>()) {
      p_val = cudf::make_numeric_scalar(type);
    } else if (cudf::is_timestamp<T>()) {
      p_val = cudf::make_timestamp_scalar(type);
    } else if (cudf::is_duration<T>()) {
      p_val = cudf::make_duration_scalar(type);
    } else {
      ASSERT_TRUE(false);  // should not be reached
    }
    using ScalarType = cudf::scalar_type_t<T>;
    static_cast<ScalarType*>(p_val.get())->set_value(value);
    static_cast<ScalarType*>(p_val.get())->set_valid_async(value_is_valid);

    auto expected_elements =
      cudf::detail::make_counting_transform_iterator(0, [begin, end, value](auto i) {
        return (i >= begin && i < end) ? value : cudf::test::make_type_param_scalar<T>(i);
      });
    cudf::test::fixed_width_column_wrapper<T> expected(
      expected_elements,
      expected_elements + size,
      cudf::detail::make_counting_transform_iterator(
        0, [begin, end, value_is_valid, destination_validity](auto i) {
          return (i >= begin && i < end) ? value_is_valid : destination_validity(i);
        }));

    // test out-of-place version first

    auto p_ret = cudf::fill(destination, begin, end, *p_val);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*p_ret, expected);

    // test in-place version second

    cudf::mutable_column_view mutable_view{destination};
    EXPECT_NO_THROW(cudf::fill_in_place(mutable_view, begin, end, *p_val));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(mutable_view, expected);
  }
};

TYPED_TEST_SUITE(FillTypedTestFixture, cudf::test::FixedWidthTypesWithoutFixedPoint);

TYPED_TEST(FillTypedTestFixture, SetSingle)
{
  using T = TypeParam;

  cudf::size_type index{9};
  T value = cudf::test::make_type_param_scalar<TypeParam>(1);

  // First set it as valid
  this->test(index, index + 1, value, true);

  // next set it as invalid
  this->test(index, index + 1, value, false);
}

TYPED_TEST(FillTypedTestFixture, SetAll)
{
  using T = TypeParam;

  cudf::size_type size{FillTypedTestFixture<T>::column_size};

  T value = cudf::test::make_type_param_scalar<TypeParam>(1);

  // First set it as valid
  this->test(0, size, value, true);

  // next set it as invalid
  this->test(0, size, value, false);
}

TYPED_TEST(FillTypedTestFixture, SetRange)
{
  using T = TypeParam;

  cudf::size_type begin{99};
  cudf::size_type end{299};
  T value = cudf::test::make_type_param_scalar<TypeParam>(1);

  // First set it as valid
  this->test(begin, end, value, true);

  // Next set it as invalid
  this->test(begin, end, value, false);
}

TYPED_TEST(FillTypedTestFixture, SetRangeNullCount)
{
  using T = TypeParam;

  cudf::size_type size{FillTypedTestFixture<T>::column_size};

  cudf::size_type begin{10};
  cudf::size_type end{50};
  T value = cudf::test::make_type_param_scalar<TypeParam>(1);

  // First set it as valid value
  this->test(begin, end, value, true, odd_valid);

  // Next set it as invalid
  this->test(begin, end, value, false, odd_valid);

  // All invalid column should have some valid
  this->test(begin, end, value, true, all_invalid);

  // All should be invalid
  this->test(begin, end, value, false, all_invalid);

  // All should be valid
  this->test(0, size, value, true, odd_valid);
}

class FillStringTestFixture : public cudf::test::BaseFixture {
 public:
  static constexpr cudf::size_type column_size{100};

  template <typename BitInitializerType = decltype(all_valid)>
  void test(cudf::size_type begin,
            cudf::size_type end,
            std::string value,
            bool value_is_valid                     = true,
            BitInitializerType destination_validity = all_valid)
  {
    cudf::size_type size{FillStringTestFixture::column_size};

    auto destination_elements = cudf::detail::make_counting_transform_iterator(
      0, [](auto i) { return "#" + std::to_string(i); });
    auto destination = cudf::test::strings_column_wrapper(
      destination_elements,
      destination_elements + size,
      cudf::detail::make_counting_transform_iterator(0, destination_validity));

    auto p_val       = cudf::make_string_scalar(value);
    using ScalarType = cudf::scalar_type_t<cudf::string_view>;
    static_cast<ScalarType*>(p_val.get())->set_valid_async(value_is_valid);

    auto p_chars   = value.c_str();
    auto num_chars = value.length();
    auto expected_elements =
      cudf::detail::make_counting_transform_iterator(0, [begin, end, p_chars, num_chars](auto i) {
        return (i >= begin && i < end) ? std::string(p_chars, num_chars) : "#" + std::to_string(i);
      });
    auto expected = cudf::test::strings_column_wrapper(
      expected_elements,
      expected_elements + size,
      cudf::detail::make_counting_transform_iterator(
        0, [begin, end, value_is_valid, destination_validity](auto i) {
          return (i >= begin && i < end) ? value_is_valid : destination_validity(i);
        }));

    auto p_ret = cudf::fill(destination, begin, end, *p_val);
    CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(*p_ret, expected);
  }
};

TEST_F(FillStringTestFixture, SetSingle)
{
  cudf::size_type size{FillStringTestFixture::column_size};

  cudf::size_type index{9};
  auto value = "#" + std::to_string(size * 2);

  // First set it as valid
  this->test(index, index + 1, value, true);

  // next set it as invalid
  this->test(index, index + 1, value, false);
}

TEST_F(FillStringTestFixture, SetAll)
{
  cudf::size_type size{FillStringTestFixture::column_size};

  auto value = "#" + std::to_string(size * 2);

  // First set it as valid
  this->test(0, size, value, true);

  // next set it as invalid
  this->test(0, size, value, false);
}

TEST_F(FillStringTestFixture, SetRange)
{
  cudf::size_type size{FillStringTestFixture::column_size};

  cudf::size_type begin{9};
  cudf::size_type end{99};
  auto value = "#" + std::to_string(size * 2);

  // First set it as valid
  this->test(begin, end, value, true);

  // Next set it as invalid
  this->test(begin, end, value, false);
}

TEST_F(FillStringTestFixture, SetRangeNullCount)
{
  cudf::size_type size{FillStringTestFixture::column_size};

  cudf::size_type begin{10};
  cudf::size_type end{50};
  auto value = "#" + std::to_string(size * 2);

  // First set it as valid value
  this->test(begin, end, value, true, odd_valid);

  // Next set it as invalid
  this->test(begin, end, value, false, odd_valid);

  // All invalid column should have some valid
  this->test(begin, end, value, true, all_invalid);

  // All should be invalid
  this->test(begin, end, value, false, all_invalid);

  // All should be valid
  this->test(0, size, value, true, odd_valid);
}

class FillErrorTestFixture : public cudf::test::BaseFixture {};

TEST_F(FillErrorTestFixture, InvalidInplaceCall)
{
  auto p_val_int   = cudf::make_numeric_scalar(cudf::data_type(cudf::type_id::INT32));
  using T_int      = cudf::id_to_type<cudf::type_id::INT32>;
  using ScalarType = cudf::scalar_type_t<T_int>;
  static_cast<ScalarType*>(p_val_int.get())->set_value(5);
  static_cast<ScalarType*>(p_val_int.get())->set_valid_async(false);

  auto destination = cudf::test::fixed_width_column_wrapper<int32_t>(
    cuda::counting_iterator<int32_t>{0}, cuda::counting_iterator<int32_t>{0} + 100);

  auto destination_view = cudf::mutable_column_view{destination};
  EXPECT_THROW(cudf::fill_in_place(destination_view, 0, 100, *p_val_int), cudf::logic_error);

  auto p_val_str = cudf::make_string_scalar("five");

  std::vector<std::string> strings{"", "this", "is", "a", "column", "of", "strings"};
  auto destination_string = cudf::test::strings_column_wrapper(strings.begin(), strings.end());

  cudf::mutable_column_view destination_view_string{destination_string};
  EXPECT_THROW(cudf::fill_in_place(destination_view_string, 0, 100, *p_val_str), cudf::logic_error);
}

TEST_F(FillErrorTestFixture, InvalidRange)
{
  auto p_val       = cudf::make_numeric_scalar(cudf::data_type(cudf::type_id::INT32));
  using T          = cudf::id_to_type<cudf::type_id::INT32>;
  using ScalarType = cudf::scalar_type_t<T>;
  static_cast<ScalarType*>(p_val.get())->set_value(5);

  auto destination =
    cudf::test::fixed_width_column_wrapper<int32_t>(cuda::counting_iterator<int32_t>{0},
                                                    cuda::counting_iterator<int32_t>{0} + 100,
                                                    cuda::make_constant_iterator(true));

  cudf::mutable_column_view destination_view{destination};

  // empty range == no-op, this is valid
  EXPECT_NO_THROW(cudf::fill_in_place(destination_view, 0, 0, *p_val));
  EXPECT_NO_THROW(auto p_ret = cudf::fill(destination, 0, 0, *p_val));

  // out_begin is negative
  EXPECT_THROW(cudf::fill_in_place(destination_view, -10, 0, *p_val), cudf::logic_error);
  EXPECT_THROW(auto p_ret = cudf::fill(destination, -10, 0, *p_val), cudf::logic_error);

  // out_begin > out_end
  EXPECT_THROW(cudf::fill_in_place(destination_view, 10, 5, *p_val), cudf::logic_error);
  EXPECT_THROW(auto p_ret = cudf::fill(destination, 10, 5, *p_val), cudf::logic_error);

  // out_begin > destination.size()
  EXPECT_THROW(cudf::fill_in_place(destination_view, 101, 100, *p_val), cudf::logic_error);
  EXPECT_THROW(auto p_ret = cudf::fill(destination, 101, 100, *p_val), cudf::logic_error);

  // out_end > destination.size()
  EXPECT_THROW(cudf::fill_in_place(destination_view, 99, 101, *p_val), cudf::logic_error);
  EXPECT_THROW(auto p_ret = cudf::fill(destination, 99, 101, *p_val), cudf::logic_error);

  // Empty Column
  destination      = cudf::test::fixed_width_column_wrapper<int32_t>{};
  destination_view = destination;

  // empty column, this is valid
  EXPECT_NO_THROW(cudf::fill_in_place(destination_view, 0, destination_view.size(), *p_val));
  EXPECT_NO_THROW(auto p_ret = cudf::fill(destination, 0, destination_view.size(), *p_val));
}

TEST_F(FillErrorTestFixture, DTypeMismatch)
{
  cudf::size_type size{100};

  auto p_val       = cudf::make_numeric_scalar(cudf::data_type(cudf::type_id::INT32));
  using T          = cudf::id_to_type<cudf::type_id::INT32>;
  using ScalarType = cudf::scalar_type_t<T>;
  static_cast<ScalarType*>(p_val.get())->set_value(5);

  auto float_iter = cudf::detail::make_counting_transform_iterator(
    0, [](int32_t i) { return static_cast<float>(i); });
  auto destination = cudf::test::fixed_width_column_wrapper<float>(float_iter, float_iter + size);

  auto destination_view = cudf::mutable_column_view{destination};

  EXPECT_THROW(cudf::fill_in_place(destination_view, 0, 10, *p_val), cudf::data_type_error);
  EXPECT_THROW(auto p_ret = cudf::fill(destination, 0, 10, *p_val), cudf::data_type_error);
}

template <typename T>
class FixedPointAllReps : public cudf::test::BaseFixture {};

TYPED_TEST_SUITE(FixedPointAllReps, cudf::test::FixedPointTypes);

TYPED_TEST(FixedPointAllReps, OutOfPlaceFill)
{
  using namespace numeric;
  using decimalXX  = TypeParam;
  using RepType    = cudf::device_storage_type_t<decimalXX>;
  using fp_wrapper = cudf::test::fixed_point_column_wrapper<RepType>;

  for (auto const i : {0, -1, -2, -3, -4}) {
    auto const scale    = scale_type{i};
    auto const column   = fp_wrapper{{4104, 42, 1729, 55}, scale};
    auto const expected = fp_wrapper{{42, 42, 42, 42}, scale};
    auto const scalar   = cudf::make_fixed_point_scalar<decimalXX>(42, scale);

    auto const result = cudf::fill(column, 0, 4, *scalar);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expected);
  }
}

TYPED_TEST(FixedPointAllReps, InPlaceFill)
{
  using namespace numeric;
  using decimalXX  = TypeParam;
  using RepType    = cudf::device_storage_type_t<decimalXX>;
  using fp_wrapper = cudf::test::fixed_point_column_wrapper<RepType>;

  for (auto const i : {0, -1, -2, -3, -4}) {
    auto const scale    = scale_type{i};
    auto column         = fp_wrapper{{4104, 42, 1729, 55}, scale};
    auto const expected = fp_wrapper{{42, 42, 42, 42}, scale};
    auto const scalar   = cudf::make_fixed_point_scalar<decimalXX>(42, scale);

    auto mut_column = cudf::mutable_column_view{column};
    cudf::fill_in_place(mut_column, 0, 4, *scalar);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(column, expected);
  }
}

CUDF_TEST_PROGRAM_MAIN()
