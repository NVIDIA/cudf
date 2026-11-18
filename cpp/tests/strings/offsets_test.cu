/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>

#include <cudf/strings/detail/strings_children.cuh>

#include <cuda/iterator>

#include <cstdlib>
#include <limits>
#include <optional>
#include <string>

struct StringsOffsetsTest : cudf::test::BaseFixtureWithHarness {
  static std::optional<std::string> saved(char const* name)
  {
    auto const value = std::getenv(name);
    return value ? std::optional<std::string>{value} : std::nullopt;
  }

  static void restore(char const* name, std::optional<std::string> const& value)
  {
    if (value) {
      setenv(name, value->c_str(), 1);
    } else {
      unsetenv(name);
    }
  }

  std::optional<std::string> enabled{saved("LIBCUDF_LARGE_STRINGS_ENABLED")};
  std::optional<std::string> threshold{saved("LIBCUDF_LARGE_STRINGS_THRESHOLD")};

  void SetUp() override
  {
    ASSERT_EQ(setenv("LIBCUDF_LARGE_STRINGS_ENABLED", "1", 1), 0);
    ASSERT_EQ(setenv("LIBCUDF_LARGE_STRINGS_THRESHOLD", "0", 1), 0);
  }

  ~StringsOffsetsTest() override
  {
    restore("LIBCUDF_LARGE_STRINGS_ENABLED", enabled);
    restore("LIBCUDF_LARGE_STRINGS_THRESHOLD", threshold);
  }
};

template <typename Iterator>
struct size_function {
  explicit size_function(Iterator input) : begin{input} {}
  __device__ cudf::size_type operator()(cudf::size_type i) const { return begin[i]; }
  Iterator begin;
};

struct gathered_size_function {
  cudf::size_type const* offsets;
  cudf::size_type const* gather_map;

  __device__ cudf::size_type operator()(cudf::size_type i) const
  {
    auto const row = gather_map[i];
    return offsets[row + 1] - offsets[row];
  }
};

TEST_F(StringsOffsetsTest, KnownTotal)
{
  cudf::test::fixed_width_column_wrapper<cudf::size_type> sizes{2, 0, 3, 1};
  auto const view = cudf::column_view{sizes};
  auto result     = cudf::strings::detail::make_offsets_child_column(
    size_function(view.begin<cudf::size_type>()), view.size(), 6, stream(), resources());
  cudf::test::fixed_width_column_wrapper<cudf::size_type> expected{0, 2, 2, 5, 6};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TEST_F(StringsOffsetsTest, GatheredSizes)
{
  // Nonzero input offsets model a slice; the permutation includes an empty string.
  cudf::test::fixed_width_column_wrapper<cudf::size_type> offsets{11, 13, 13, 16, 17};
  cudf::test::fixed_width_column_wrapper<cudf::size_type> gather_map{2, 0, 3, 1};
  auto const map_view = cudf::column_view{gather_map};
  auto const size_at  = gathered_size_function{cudf::column_view{offsets}.data<cudf::size_type>(),
                                              map_view.data<cudf::size_type>()};
  auto result         = cudf::strings::detail::make_offsets_child_column(
    size_at, map_view.size(), 6, stream(), resources());
  cudf::test::fixed_width_column_wrapper<cudf::size_type> expected{0, 3, 5, 6, 6};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());

  // Existing callers can still use the iterator overload to compute the total.
  auto const sizes = cuda::transform_iterator(cuda::counting_iterator<cudf::size_type>{0}, size_at);
  auto [iterator_result, total] = cudf::strings::detail::make_offsets_child_column(
    sizes, sizes + map_view.size(), stream(), resources());
  EXPECT_EQ(total, 6);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, iterator_result->view());
}

TEST_F(StringsOffsetsTest, ZeroTotal)
{
  auto const begin = cuda::make_constant_iterator<cudf::size_type>(0);
  for (auto const rows : {0, 5}) {
    auto result = cudf::strings::detail::make_offsets_child_column(
      size_function(begin), rows, 0, stream(), resources());
    cudf::test::fixed_width_column_wrapper<cudf::size_type> expected(begin, begin + rows + 1);
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
  }
}

TEST_F(StringsOffsetsTest, TotalExceedsInt32)
{
  auto constexpr max = std::numeric_limits<cudf::size_type>::max();
  cudf::test::fixed_width_column_wrapper<cudf::size_type> sizes{max, 1};
  auto const view = cudf::column_view{sizes};
  auto result =
    cudf::strings::detail::make_offsets_child_column(size_function(view.begin<cudf::size_type>()),
                                                     view.size(),
                                                     int64_t{max} + 1,
                                                     stream(),
                                                     resources());
  cudf::test::fixed_width_column_wrapper<int64_t> expected{
    int64_t{0}, int64_t{max}, int64_t{max} + 1};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TEST_F(StringsOffsetsTest, LargeStringsDisabled)
{
  ASSERT_EQ(setenv("LIBCUDF_LARGE_STRINGS_ENABLED", "0", 1), 0);
  auto constexpr max = std::numeric_limits<cudf::size_type>::max();
  auto const begin   = cuda::make_constant_iterator<cudf::size_type>(max);
  EXPECT_THROW(cudf::strings::detail::make_offsets_child_column(
                 size_function(begin), 1, max, stream(), resources()),
               std::overflow_error);
}

TEST_F(StringsOffsetsTest, NegativeTotal)
{
  auto const begin = cuda::make_constant_iterator<cudf::size_type>(0);
  EXPECT_THROW(cudf::strings::detail::make_offsets_child_column(
                 size_function(begin), 0, -1, stream(), resources()),
               cudf::logic_error);
}

TEST_F(StringsOffsetsTest, TooManyOffsets)
{
  auto const begin = cuda::make_constant_iterator<cudf::size_type>(0);
  EXPECT_THROW(
    cudf::strings::detail::make_offsets_child_column(
      size_function(begin), std::numeric_limits<cudf::size_type>::max(), 0, stream(), resources()),
    std::overflow_error);
}
