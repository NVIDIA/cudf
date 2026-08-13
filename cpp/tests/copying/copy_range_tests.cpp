/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/dictionary/encode.hpp>

#include <cuda/iterator>
#include <thrust/iterator/transform_iterator.h>

#include <stdexcept>

auto all_valid  = [](cudf::size_type row) { return true; };
auto even_valid = [](cudf::size_type row) { return (row % 2 == 0); };

template <typename T>
class CopyRangeTypedTestFixture : public cudf::test::BaseFixture {
 public:
  static constexpr cudf::size_type column_size{1000};

  void test(cudf::column_view const& source,
            cudf::column_view const& expected,
            cudf::mutable_column_view& target,
            cudf::size_type source_begin,
            cudf::size_type source_end,
            cudf::size_type target_begin)
  {
    static_assert(cudf::is_fixed_width<T>(), "this code assumes fixed-width types.");

    // test the out-of-place version first

    cudf::column_view const immutable_view{target};
    auto p_ret = cudf::copy_range(source, immutable_view, source_begin, source_end, target_begin);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*p_ret, expected);

    // test the in-place version second

    EXPECT_NO_THROW(
      cudf::copy_range_in_place(source, target, source_begin, source_end, target_begin));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(target, expected);
  }
};

TYPED_TEST_SUITE(CopyRangeTypedTestFixture, cudf::test::FixedWidthTypesWithoutFixedPoint);

TYPED_TEST(CopyRangeTypedTestFixture, CopyWithNulls)
{
  using T = TypeParam;

  cudf::size_type size{CopyRangeTypedTestFixture<T>::column_size};
  cudf::size_type source_begin{9};
  cudf::size_type source_end{size - 50};
  cudf::size_type target_begin{30};
  auto target_end = target_begin + (source_end - source_begin);
  auto row_diff   = source_begin - target_begin;

  cudf::test::fixed_width_column_wrapper<T, int32_t> target(
    cuda::counting_iterator<int32_t>{0},
    cuda::counting_iterator<int32_t>{0} + size,
    cudf::detail::make_counting_transform_iterator(0, all_valid));

  auto source_elements =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i * 2; });
  cudf::test::fixed_width_column_wrapper<T, typename decltype(source_elements)::value_type> source(
    source_elements,
    source_elements + size,
    cudf::detail::make_counting_transform_iterator(0, even_valid));

  auto expected_elements =
    cudf::detail::make_counting_transform_iterator(0, [target_begin, target_end, row_diff](auto i) {
      return ((i >= target_begin) && (i < target_end)) ? (i + row_diff) * 2 : i;
    });
  cudf::test::fixed_width_column_wrapper<T, typename decltype(expected_elements)::value_type>
    expected(expected_elements,
             expected_elements + size,
             cudf::detail::make_counting_transform_iterator(
               0, [target_begin, target_end, row_diff](auto i) {
                 return ((i >= target_begin) && (i < target_end)) ? even_valid(i + row_diff)
                                                                  : all_valid(i);
               }));

  cudf::mutable_column_view target_view{target};
  this->test(source, expected, target_view, source_begin, source_end, target_begin);
}

TYPED_TEST(CopyRangeTypedTestFixture, CopyNoNulls)
{
  using T = TypeParam;

  cudf::size_type size{CopyRangeTypedTestFixture<T>::column_size};
  cudf::size_type source_begin{9};
  cudf::size_type source_end{size - 50};
  cudf::size_type target_begin{30};
  auto target_end = target_begin + (source_end - source_begin);
  auto row_diff   = source_begin - target_begin;

  cudf::test::fixed_width_column_wrapper<T, int32_t> target(
    cuda::counting_iterator<int32_t>{0}, cuda::counting_iterator<int32_t>{0} + size);

  auto source_elements =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i * 2; });
  cudf::test::fixed_width_column_wrapper<T, typename decltype(source_elements)::value_type> source(
    source_elements, source_elements + size);

  auto expected_elements =
    cudf::detail::make_counting_transform_iterator(0, [target_begin, target_end, row_diff](auto i) {
      return ((i >= target_begin) && (i < target_end)) ? (i + row_diff) * 2 : i;
    });
  cudf::test::fixed_width_column_wrapper<T, typename decltype(expected_elements)::value_type>
    expected(expected_elements, expected_elements + size);

  cudf::mutable_column_view target_view{target};
  this->test(source, expected, target_view, source_begin, source_end, target_begin);
}

TYPED_TEST(CopyRangeTypedTestFixture, CopyWithNullsNonzeroOffset)
{
  using T = TypeParam;

  cudf::size_type size{CopyRangeTypedTestFixture<T>::column_size};
  cudf::size_type source_offset{27};
  cudf::size_type source_begin{9};
  cudf::size_type source_end{50};
  cudf::size_type target_offset{58};
  cudf::size_type target_begin{30};
  auto target_end = target_begin + (source_end - source_begin);
  auto row_diff   = (source_offset + source_begin) - (target_offset + target_begin);

  cudf::test::fixed_width_column_wrapper<T, int32_t> target(
    cuda::counting_iterator<int32_t>{0},
    cuda::counting_iterator<int32_t>{0} + size,
    cudf::detail::make_counting_transform_iterator(0, all_valid));

  cudf::mutable_column_view tmp = target;
  cudf::mutable_column_view target_slice(tmp.type(),
                                         tmp.size() - target_offset,
                                         tmp.head<T>(),
                                         tmp.null_mask(),
                                         tmp.null_count(),
                                         target_offset);

  auto source_elements =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i * 2; });
  cudf::test::fixed_width_column_wrapper<T, typename decltype(source_elements)::value_type> source(
    source_elements,
    source_elements + size,
    cudf::detail::make_counting_transform_iterator(0, even_valid));

  auto source_slice = cudf::slice(source, std::vector<cudf::size_type>{source_offset, size})[0];

  auto expected_elements = cudf::detail::make_counting_transform_iterator(
    0, [target_offset, target_begin, target_end, row_diff](auto i) {
      return ((i >= target_offset + target_begin) && (i < target_offset + target_end))
               ? (i + row_diff) * 2
               : i;
    });
  cudf::test::fixed_width_column_wrapper<T, typename decltype(expected_elements)::value_type>
    expected(expected_elements,
             expected_elements + size,
             cudf::detail::make_counting_transform_iterator(
               0, [target_offset, target_begin, target_end, row_diff](auto i) {
                 return ((i >= target_offset + target_begin) && (i < target_offset + target_end))
                          ? even_valid(i + row_diff)
                          : all_valid(i);
               }));

  auto expected_slice = cudf::slice(expected, std::vector<cudf::size_type>{target_offset, size})[0];

  this->test(source_slice, expected_slice, target_slice, source_begin, source_end, target_begin);
}

class CopyRangeTestFixture : public cudf::test::BaseFixture {};

TEST_F(CopyRangeTestFixture, CopyWithNullsString)
{
  cudf::size_type size{100};
  cudf::size_type source_begin{9};
  cudf::size_type source_end{50};
  cudf::size_type target_begin{30};
  auto target_end = target_begin + (source_end - source_begin);
  auto row_diff   = source_begin - target_begin;

  auto target_elements = cudf::detail::make_counting_transform_iterator(
    0, [](auto i) { return "#" + std::to_string(i); });
  auto target = cudf::test::strings_column_wrapper(
    target_elements,
    target_elements + size,
    cudf::detail::make_counting_transform_iterator(0, all_valid));

  auto source_elements = cudf::detail::make_counting_transform_iterator(
    0, [](auto i) { return "#" + std::to_string(i * 2); });
  auto source = cudf::test::strings_column_wrapper(
    source_elements,
    source_elements + size,
    cudf::detail::make_counting_transform_iterator(0, even_valid));

  auto expected_elements =
    cudf::detail::make_counting_transform_iterator(0, [target_begin, target_end, row_diff](auto i) {
      auto num = std::to_string(((i >= target_begin) && (i < target_end)) ? (i + row_diff) * 2 : i);
      return "#" + num;
    });
  auto expected = cudf::test::strings_column_wrapper(
    expected_elements,
    expected_elements + size,
    cudf::detail::make_counting_transform_iterator(0, [target_begin, target_end, row_diff](auto i) {
      return ((i >= target_begin) && (i < target_end)) ? even_valid(i + row_diff) : all_valid(i);
    }));

  auto p_ret = cudf::copy_range(source, target, source_begin, source_end, target_begin);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*p_ret, expected);
}

TEST_F(CopyRangeTestFixture, CopyWithTargetNullsString)
{
  auto target =
    cudf::test::strings_column_wrapper({"a", "b", "", "d", "", "é"}, {1, 1, 0, 1, 1, 1});
  auto source   = cudf::test::strings_column_wrapper({"A", "B", "C", "D", "E", "F"});
  auto result   = cudf::copy_range(source, target, 1, 5, 1);
  auto expected = cudf::test::strings_column_wrapper({"a", "B", "C", "D", "E", "é"});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(result->view(), expected);
}

TEST_F(CopyRangeTestFixture, CopyNoNullsString)
{
  cudf::size_type size{100};
  cudf::size_type source_begin{9};
  cudf::size_type source_end{50};
  cudf::size_type target_begin{30};
  auto target_end = target_begin + (source_end - source_begin);
  auto row_diff   = source_begin - target_begin;

  auto target_elements = cudf::detail::make_counting_transform_iterator(
    0, [](auto i) { return "#" + std::to_string(i); });
  auto target = cudf::test::strings_column_wrapper(target_elements, target_elements + size);

  auto source_elements = cudf::detail::make_counting_transform_iterator(
    0, [](auto i) { return "#" + std::to_string(i * 2); });
  auto source = cudf::test::strings_column_wrapper(source_elements, source_elements + size);

  auto expected_elements =
    cudf::detail::make_counting_transform_iterator(0, [target_begin, target_end, row_diff](auto i) {
      auto num = std::to_string(((i >= target_begin) && (i < target_end)) ? (i + row_diff) * 2 : i);
      return "#" + num;
    });
  auto expected = cudf::test::strings_column_wrapper(expected_elements, expected_elements + size);

  auto p_ret = cudf::copy_range(source, target, source_begin, source_end, target_begin);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*p_ret, expected);
}

TEST_F(CopyRangeTestFixture, CopyWithNullsNonzeroOffsetString)
{
  cudf::size_type size{200};
  cudf::size_type source_offset{27};
  cudf::size_type source_begin{9};
  cudf::size_type source_end{50};
  cudf::size_type target_offset{58};
  cudf::size_type target_begin{30};
  auto target_end = target_begin + (source_end - source_begin);
  auto row_diff   = (source_offset + source_begin) - (target_offset + target_begin);

  auto target_elements = cudf::detail::make_counting_transform_iterator(
    0, [](auto i) { return "#" + std::to_string(i); });
  auto target = cudf::test::strings_column_wrapper(
    target_elements,
    target_elements + size,
    cudf::detail::make_counting_transform_iterator(0, all_valid));

  auto target_slice = cudf::slice(target, std::vector<cudf::size_type>{target_offset, size})[0];

  auto source_elements = cudf::detail::make_counting_transform_iterator(
    0, [](auto i) { return "#" + std::to_string(i * 2); });
  auto source = cudf::test::strings_column_wrapper(
    source_elements,
    source_elements + size,
    cudf::detail::make_counting_transform_iterator(0, even_valid));

  auto source_slice = cudf::slice(source, std::vector<cudf::size_type>{source_offset, size})[0];

  auto expected_elements = cudf::detail::make_counting_transform_iterator(
    0, [target_offset, target_begin, target_end, row_diff](auto i) {
      auto num =
        std::to_string(((i >= target_offset + target_begin) && (i < target_offset + target_end))
                         ? (i + row_diff) * 2
                         : i);
      return "#" + num;
    });
  auto expected = cudf::test::strings_column_wrapper(
    expected_elements,
    expected_elements + size,
    cudf::detail::make_counting_transform_iterator(
      0, [target_offset, target_begin, target_end, row_diff](auto i) {
        return ((i >= target_offset + target_begin) && (i < target_offset + target_end))
                 ? even_valid(i + row_diff)
                 : all_valid(i);
      }));

  auto expected_slice = cudf::slice(expected, std::vector<cudf::size_type>{target_offset, size})[0];

  auto p_ret = cudf::copy_range(source_slice, target_slice, source_begin, source_end, target_begin);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*p_ret, expected_slice);
}

TEST_F(CopyRangeTestFixture, CopyDictionary)
{
  cudf::size_type source_begin{1};
  cudf::size_type source_end{5};
  cudf::size_type target_begin{3};

  std::vector<std::string> source_elements({"e", "b", "a", "c", "c", "e", "b", "c", "e", "b"});
  std::vector<std::string> target_elements({"a", "e", "d", "c", "c", "b", "b", "a", "f", "f"});
  std::vector<std::string> expected_elements(target_elements);
  std::copy(source_elements.begin() + source_begin,
            source_elements.begin() + source_end,
            expected_elements.begin() + target_begin);

  {
    auto source = cudf::dictionary::encode(
      cudf::test::strings_column_wrapper(source_elements.begin(), source_elements.end()));
    auto target = cudf::dictionary::encode(
      cudf::test::strings_column_wrapper(target_elements.begin(), target_elements.end()));

    auto result =
      cudf::copy_range(source->view(), target->view(), source_begin, source_end, target_begin);
    auto decoded = cudf::dictionary::decode(cudf::dictionary_column_view(result->view()));

    auto expected =
      cudf::test::strings_column_wrapper(expected_elements.begin(), expected_elements.end());
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*decoded, expected);
  }

  auto source_validity = thrust::make_transform_iterator(
    cuda::counting_iterator<cudf::size_type>{0}, [](auto i) { return i != 3; });
  auto target_validity = thrust::make_transform_iterator(
    cuda::counting_iterator<cudf::size_type>{0}, [](auto i) { return i != 3 && i != 9; });
  auto expected_validity = thrust::make_transform_iterator(
    cuda::counting_iterator<cudf::size_type>{0}, [](auto i) { return i != 5 && i != 9; });
  {
    auto source = cudf::dictionary::encode(cudf::test::strings_column_wrapper(
      source_elements.begin(), source_elements.end(), source_validity));
    auto target = cudf::dictionary::encode(cudf::test::strings_column_wrapper(
      target_elements.begin(), target_elements.end(), target_validity));

    auto result =
      cudf::copy_range(source->view(), target->view(), source_begin, source_end, target_begin);
    auto decoded = cudf::dictionary::decode(cudf::dictionary_column_view(result->view()));

    auto expected = cudf::test::strings_column_wrapper(
      expected_elements.begin(), expected_elements.end(), expected_validity);

    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*decoded, expected);
  }
}

class CopyRangeErrorTestFixture : public cudf::test::BaseFixture {};

TEST_F(CopyRangeErrorTestFixture, InvalidInplaceCall)
{
  cudf::size_type size{100};

  auto target = cudf::test::fixed_width_column_wrapper<int32_t>(
    cuda::counting_iterator<int32_t>{0}, cuda::counting_iterator<int32_t>{0} + size);

  auto source = cudf::test::fixed_width_column_wrapper<int32_t>(
    cuda::counting_iterator<int32_t>{0},
    cuda::counting_iterator<int32_t>{0} + size,
    cudf::detail::make_counting_transform_iterator(0, even_valid));

  cudf::mutable_column_view target_view{target};
  // source has null values but target is not nullable.
  EXPECT_THROW(cudf::copy_range_in_place(source, target_view, 0, size, 0), std::invalid_argument);

  std::vector<std::string> strings{"", "this", "is", "a", "column", "of", "strings"};
  auto target_string = cudf::test::strings_column_wrapper(strings.begin(), strings.end());
  auto source_string = cudf::test::strings_column_wrapper(strings.begin(), strings.end());

  cudf::mutable_column_view target_view_string{target_string};
  EXPECT_THROW(cudf::copy_range_in_place(source_string, target_view_string, 0, size, 0),
               cudf::data_type_error);
}

TEST_F(CopyRangeErrorTestFixture, InvalidRange)
{
  cudf::size_type size{100};

  auto target = cudf::test::fixed_width_column_wrapper<int32_t>(
    cuda::counting_iterator<int32_t>{0}, cuda::counting_iterator<int32_t>{0} + size);

  auto source = cudf::test::fixed_width_column_wrapper<int32_t>(
    cuda::counting_iterator<int32_t>{0}, cuda::counting_iterator<int32_t>{0} + size);

  cudf::mutable_column_view target_view{target};
  cudf::column_view source_view{source};

  // empty_range == no-op, this is valid
  EXPECT_NO_THROW(cudf::copy_range_in_place(source, target_view, 0, 0, 0));
  EXPECT_NO_THROW(auto p_ret = cudf::copy_range(source, target, 0, 0, 0));

  // source_begin is negative
  EXPECT_THROW(cudf::copy_range_in_place(source, target_view, -1, size, 0), std::out_of_range);
  EXPECT_THROW(auto p_ret = cudf::copy_range(source, target, -1, size, 0), std::out_of_range);

  // source_begin > source_end
  EXPECT_THROW(cudf::copy_range_in_place(source, target_view, 10, 5, 0), std::out_of_range);
  EXPECT_THROW(auto p_ret = cudf::copy_range(source, target, 10, 5, 0), std::out_of_range);

  // source_begin >= source.size()
  EXPECT_THROW(cudf::copy_range_in_place(source, target_view, 101, 100, 0), std::out_of_range);
  EXPECT_THROW(auto p_ret = cudf::copy_range(source, target, 101, 100, 0), std::out_of_range);

  // source_end > source.size()
  EXPECT_THROW(cudf::copy_range_in_place(source, target_view, 99, 101, 0), std::out_of_range);
  EXPECT_THROW(auto p_ret = cudf::copy_range(source, target, 99, 101, 0), std::out_of_range);

  // target_begin < 0
  EXPECT_THROW(cudf::copy_range_in_place(source, target_view, 50, 100, -5), std::out_of_range);
  EXPECT_THROW(auto p_ret = cudf::copy_range(source, target, 50, 100, -5), std::out_of_range);

  // target_begin >= target.size()
  EXPECT_THROW(cudf::copy_range_in_place(source, target_view, 50, 100, 100), std::out_of_range);
  EXPECT_THROW(auto p_ret = cudf::copy_range(source, target, 50, 100, 100), std::out_of_range);

  // target_begin + (source_end - source_begin) > target.size()
  EXPECT_THROW(cudf::copy_range_in_place(source, target_view, 50, 100, 80), std::out_of_range);
  EXPECT_THROW(auto p_ret = cudf::copy_range(source, target, 50, 100, 80), std::out_of_range);

  // Empty column
  target      = cudf::test::fixed_width_column_wrapper<int32_t>{};
  source      = cudf::test::fixed_width_column_wrapper<int32_t>{};
  target_view = target;
  source_view = source;

  // empty column == no-op, this is valid
  EXPECT_NO_THROW(cudf::copy_range_in_place(source_view, target_view, 0, source_view.size(), 0));
  EXPECT_NO_THROW(auto p_ret = cudf::copy_range(source_view, target, 0, source_view.size(), 0));
}

TEST_F(CopyRangeErrorTestFixture, DTypeMismatch)
{
  cudf::size_type size{100};

  auto target = cudf::test::fixed_width_column_wrapper<int32_t>(
    cuda::counting_iterator<int32_t>{0}, cuda::counting_iterator<int32_t>{0} + size);

  auto float_iter = cudf::detail::make_counting_transform_iterator(
    0, [](int32_t i) { return static_cast<float>(i); });
  auto source = cudf::test::fixed_width_column_wrapper<float>(float_iter, float_iter + size);

  cudf::mutable_column_view target_view{target};

  EXPECT_THROW(cudf::copy_range_in_place(source, target_view, 0, 100, 0), cudf::data_type_error);
  EXPECT_THROW(cudf::copy_range(source, target, 0, 100, 0), cudf::data_type_error);

  auto dict_target = cudf::dictionary::encode(target);
  auto dict_source = cudf::dictionary::encode(source);
  EXPECT_THROW(cudf::copy_range(dict_source->view(), dict_target->view(), 0, 100, 0),
               cudf::data_type_error);
}

template <typename T>
struct FixedPointTypesCopyRange : public cudf::test::BaseFixture {};

TYPED_TEST_SUITE(FixedPointTypesCopyRange, cudf::test::FixedPointTypes);

TYPED_TEST(FixedPointTypesCopyRange, FixedPointSimple)
{
  using namespace numeric;
  using decimalXX  = TypeParam;
  using RepType    = cudf::device_storage_type_t<decimalXX>;
  using fp_wrapper = cudf::test::fixed_point_column_wrapper<RepType>;

  auto const source   = fp_wrapper{{110, 220, 330, 440, 550, 660}, scale_type{-2}};
  auto const target   = fp_wrapper{{0, 0, 0, 0, 0, 0}, scale_type{-2}};
  auto const expected = fp_wrapper{{0, 220, 330, 440, 0, 0}, scale_type{-2}};
  auto const result   = cudf::copy_range(source, target, 1, 4, 1);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(FixedPointTypesCopyRange, FixedPointLarge)
{
  using namespace numeric;
  using decimalXX  = TypeParam;
  using RepType    = cudf::device_storage_type_t<decimalXX>;
  using fp_wrapper = cudf::test::fixed_point_column_wrapper<RepType>;

  auto s = cuda::counting_iterator{-1000};
  auto t = cuda::make_constant_iterator(0);
  auto e =
    cudf::detail::make_counting_transform_iterator(500, [](int i) { return i < 1000 ? i : 0; });

  auto const source   = fp_wrapper{s, s + 2000, scale_type{-1}};
  auto const target   = fp_wrapper{t, t + 2000, scale_type{-1}};
  auto const expected = fp_wrapper{e, e + 2000, scale_type{-1}};
  auto const result   = cudf::copy_range(source, target, 1500, 2000, 0);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(FixedPointTypesCopyRange, FixedPointScaleMismatch)
{
  using namespace numeric;
  using decimalXX  = TypeParam;
  using RepType    = cudf::device_storage_type_t<decimalXX>;
  using fp_wrapper = cudf::test::fixed_point_column_wrapper<RepType>;

  auto const source = fp_wrapper{{110, 220, 330, 440, 550, 660}, scale_type{-2}};
  auto const target = fp_wrapper{{0, 0, 0, 0, 0, 0}, scale_type{-3}};

  EXPECT_THROW(cudf::copy_range(source, target, 1, 4, 1), cudf::data_type_error);
}
