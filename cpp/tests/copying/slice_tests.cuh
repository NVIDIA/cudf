/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>

#include <cuda/iterator>

template <typename T, typename InputIterator>
cudf::test::fixed_width_column_wrapper<T> create_fixed_columns(cudf::size_type start,
                                                               cudf::size_type size,
                                                               InputIterator valids)
{
  auto iter = cudf::detail::make_counting_transform_iterator(start, [](auto i) { return T(i); });

  return cudf::test::fixed_width_column_wrapper<T>(iter, iter + size, valids);
}

template <typename T, typename InputIterator>
cudf::table create_fixed_table(cudf::size_type num_cols,
                               cudf::size_type start,
                               cudf::size_type col_size,
                               InputIterator valids)
{
  std::vector<std::unique_ptr<cudf::column>> cols;
  for (int idx = 0; idx < num_cols; idx++) {
    cudf::test::fixed_width_column_wrapper<T> wrap =
      create_fixed_columns<T>(start + (idx * num_cols), col_size, valids);
    cols.push_back(wrap.release());
  }
  return cudf::table(std::move(cols));
}

template <typename T>
std::vector<cudf::test::fixed_width_column_wrapper<T>> create_expected_columns(
  std::vector<cudf::size_type> const& indices, bool nullable)
{
  std::vector<cudf::test::fixed_width_column_wrapper<T>> result = {};

  for (unsigned long index = 0; index < indices.size(); index += 2) {
    auto iter =
      cudf::detail::make_counting_transform_iterator(indices[index], [](auto i) { return T(i); });
    if (not nullable) {
      result.push_back(cudf::test::fixed_width_column_wrapper<T>(
        iter, iter + (indices[index + 1] - indices[index])));
    } else {
      auto valids = cudf::detail::make_counting_transform_iterator(
        indices[index], [](auto i) { return i % 2 == 0; });
      result.push_back(cudf::test::fixed_width_column_wrapper<T>(
        iter, iter + (indices[index + 1] - indices[index]), valids));
    }
  }

  return result;
}

template <typename T>
std::vector<cudf::test::fixed_width_column_wrapper<T>> create_expected_columns(
  std::vector<cudf::size_type> const& indices, std::vector<bool> const& validity)
{
  std::vector<cudf::test::fixed_width_column_wrapper<T>> result = {};

  for (unsigned long index = 0; index < indices.size(); index += 2) {
    auto iter =
      cudf::detail::make_counting_transform_iterator(indices[index], [](auto i) { return T(i); });
    result.push_back(cudf::test::fixed_width_column_wrapper<T>(
      iter, iter + (indices[index + 1] - indices[index]), validity.begin() + indices[index]));
  }

  return result;
}

template <typename T, typename ElementIter>
std::vector<cudf::test::fixed_width_column_wrapper<T>> create_expected_columns(
  std::vector<cudf::size_type> const& indices, ElementIter begin, bool nullable)
{
  std::vector<cudf::test::fixed_width_column_wrapper<T>> result = {};

  for (unsigned long index = 0; index < indices.size(); index += 2) {
    auto iter = begin + indices[index];

    if (not nullable) {
      result.push_back(cudf::test::fixed_width_column_wrapper<T>(
        iter, iter + (indices[index + 1] - indices[index])));
    } else {
      auto valids = cudf::detail::make_counting_transform_iterator(
        indices[index], [](auto i) { return i % 2 == 0; });
      result.push_back(cudf::test::fixed_width_column_wrapper<T>(
        iter, iter + (indices[index + 1] - indices[index]), valids));
    }
  }

  return result;
}

template <typename T, typename ElementIter>
std::vector<cudf::test::fixed_width_column_wrapper<T>> create_expected_columns(
  std::vector<cudf::size_type> const& indices, ElementIter begin, std::vector<bool> const& validity)
{
  std::vector<cudf::test::fixed_width_column_wrapper<T>> result = {};

  for (unsigned long index = 0; index < indices.size(); index += 2) {
    auto iter = begin + indices[index];
    result.push_back(cudf::test::fixed_width_column_wrapper<T>(
      iter, iter + (indices[index + 1] - indices[index]), validity.begin() + indices[index]));
  }

  return result;
}

template <typename T>
std::vector<cudf::table> create_expected_tables(cudf::size_type num_cols,
                                                std::vector<cudf::size_type> const& indices,
                                                bool nullable)
{
  std::vector<cudf::table> result;

  for (unsigned long index = 0; index < indices.size(); index += 2) {
    std::vector<std::unique_ptr<cudf::column>> cols = {};

    for (int idx = 0; idx < num_cols; idx++) {
      auto iter = cudf::detail::make_counting_transform_iterator(indices[index] + (idx * num_cols),
                                                                 [](auto i) { return T(i); });

      if (not nullable) {
        cudf::test::fixed_width_column_wrapper<T> wrap(
          iter, iter + (indices[index + 1] - indices[index]));
        cols.push_back(wrap.release());
      } else {
        auto valids = cudf::detail::make_counting_transform_iterator(
          indices[index], [](auto i) { return i % 2 == 0; });
        cudf::test::fixed_width_column_wrapper<T> wrap(
          iter, iter + (indices[index + 1] - indices[index]), valids);
        cols.push_back(wrap.release());
      }
    }

    result.emplace_back(std::move(cols));
  }

  return result;
}

inline std::vector<cudf::test::strings_column_wrapper> create_expected_string_columns(
  std::vector<std::string> const& strings,
  std::vector<cudf::size_type> const& indices,
  bool nullable)
{
  std::vector<cudf::test::strings_column_wrapper> result = {};

  for (unsigned long index = 0; index < indices.size(); index += 2) {
    if (not nullable) {
      result.emplace_back(strings.begin() + indices[index], strings.begin() + indices[index + 1]);
    } else {
      auto valids = cudf::detail::make_counting_transform_iterator(
        indices[index], [](auto i) { return i % 2 == 0; });
      result.emplace_back(
        strings.begin() + indices[index], strings.begin() + indices[index + 1], valids);
    }
  }

  return result;
}

inline std::vector<cudf::test::strings_column_wrapper> create_expected_string_columns(
  std::vector<std::string> const& strings,
  std::vector<cudf::size_type> const& indices,
  std::vector<bool> const& validity)
{
  std::vector<cudf::test::strings_column_wrapper> result = {};

  for (unsigned long index = 0; index < indices.size(); index += 2) {
    result.emplace_back(strings.begin() + indices[index],
                        strings.begin() + indices[index + 1],
                        validity.begin() + indices[index]);
  }

  return result;
}

inline std::vector<cudf::table> create_expected_string_tables(
  std::vector<std::vector<std::string>> const strings,
  std::vector<cudf::size_type> const& indices,
  bool nullable)
{
  std::vector<cudf::table> result = {};

  for (unsigned long index = 0; index < indices.size(); index += 2) {
    std::vector<std::unique_ptr<cudf::column>> cols = {};

    for (int idx = 0; idx < 2; idx++) {
      if (not nullable) {
        cudf::test::strings_column_wrapper wrap(strings[idx].begin() + indices[index],
                                                strings[idx].begin() + indices[index + 1]);
        cols.push_back(wrap.release());
      } else {
        auto valids = cudf::detail::make_counting_transform_iterator(
          indices[index], [](auto i) { return i % 2 == 0; });
        cudf::test::strings_column_wrapper wrap(
          strings[idx].begin() + indices[index], strings[idx].begin() + indices[index + 1], valids);
        cols.push_back(wrap.release());
      }
    }

    result.emplace_back(std::move(cols));
  }

  return result;
}

inline std::unique_ptr<cudf::column> make_long_offsets_string_column()
{
  // manually specified long offsets, but < 2B chars
  auto const num_chars = 1024;
  std::vector<int8_t> chars(num_chars);
  auto iter = cuda::counting_iterator<cudf::size_type>{0};
  std::transform(iter, iter + num_chars, chars.begin(), [](cudf::size_type i) {
    return static_cast<int8_t>('a' + (i % 26));
  });
  rmm::device_buffer d_chars(num_chars, cudf::get_default_stream());
  cudf::detail::cuda_memcpy(
    cudf::device_span<int8_t>{static_cast<int8_t*>(d_chars.data()), d_chars.size()},
    cudf::host_span<int8_t const>{chars.data(), chars.size()},
    cudf::get_default_stream());

  cudf::test::fixed_width_column_wrapper<int64_t> long_offsets{
    0, 20, 40, 60, 80, 100, 500, 600, 700, 1000, 1010};
  auto str = cudf::make_strings_column(10, long_offsets.release(), std::move(d_chars), 0, {});
  cudf::strings_column_view scv(*str);
  CUDF_EXPECTS(scv.offsets().type().id() == cudf::type_id::INT64, "Unexpected short offset type");

  return str;
}

inline std::unique_ptr<cudf::column> make_long_offsets_and_chars_string_column()
{
  rmm::device_buffer d_chars{size_t{3} * 1024 * 1024 * 1024, cudf::get_default_stream()};

  int8_t* charp         = reinterpret_cast<int8_t*>(d_chars.data());
  auto const block_size = 100 * 1024 * 1024;
  // memset a few blocks to known values, leave the rest uninitialized
  int64_t const block_a = 0;
  cudaMemsetAsync(charp + block_a, 'a', block_size, cudf::get_default_stream().get());
  // first 100 MB
  int64_t const block_b = block_size;
  cudaMemsetAsync(charp + block_b, 'b', block_size, cudf::get_default_stream().get());
  // second 100 MB
  int64_t const block_c = d_chars.size() - (block_size * 2);
  cudaMemsetAsync(charp + block_c, 'c', block_size, cudf::get_default_stream().get());
  // second-to-last 100 MB
  int64_t const block_d = d_chars.size() - block_size;
  cudaMemsetAsync(charp + block_d, 'd', block_size, cudf::get_default_stream().get());
  // last 100 MB

  // choose some rows that span various boundaries of the blocks
  cudf::test::fixed_width_column_wrapper<int64_t> long_offsets{
    int64_t{0},
    block_a + (block_size / 4),
    block_a + (block_size / 2),
    block_b - 16,
    block_b + 16,
    block_b + (block_size / 2),
    block_c + 100,
    block_c + (block_size / 2),
    block_d - 1000,
    block_d + (block_size / 2),
    static_cast<int64_t>(d_chars.size())};
  auto str = cudf::make_strings_column(10, long_offsets.release(), std::move(d_chars), 0, {});
  cudf::strings_column_view scv(*str);
  CUDF_EXPECTS(scv.offsets().type().id() == cudf::type_id::INT64, "Unexpected short offset type");

  return str;
}
