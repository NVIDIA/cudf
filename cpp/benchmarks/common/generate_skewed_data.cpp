/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "generate_skewed_data.hpp"

#include <benchmarks/common/generate_input.hpp>

#include <cudf_test/column_wrapper.hpp>

#include <cudf/binaryop.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/filling.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/error.hpp>

#include <algorithm>
#include <array>
#include <format>
#include <string>
#include <string_view>
#include <vector>

namespace {
/**
 * @brief Template strings for skewed string benchmarks.
 *
 * Based on the strings used by `create_string_column`, with the first entry ending in
 * `skewed_string_target_substring`.
 */
constexpr auto template_strings = std::to_array<std::string_view>({
  "123abc 0987 5W43",  // matches pattern;
  "012345 6789 0123",  // the rest do not match
  "abc 4567890 DEFG",
  "01234abcdefghijk",
  "sksksksksksksksk",
  "AbcéDEFGHIJKLMN",
  "9876543210,abcde",
  "                ",
  "123 édf 4567890",
  "1111111111111111",
});

constexpr cudf::size_type matching_template_length{
  static_cast<cudf::size_type>(template_strings[0].size())};

constexpr bool templates_have_matching_length()
{
  return std::ranges::all_of(
    template_strings, [](std::string_view s) { return s.size() == matching_template_length; });
}
static_assert(templates_have_matching_length(),
              "all elements in template_strings must have the same length");

/**
 * @brief Repeat a string until it reaches the given width, then take `str[0:width]`.
 */
std::string repeat_to_width(std::string_view str, std::size_t width)
{
  CUDF_EXPECTS(!str.empty(), "template string must be non-empty");
  CUDF_EXPECTS(str.size() == matching_template_length,
               "template string must be the same length as the other template strings");
  CUDF_EXPECTS(width % str.size() == 0,
               "width must be a multiple of the template length so that this function does not "
               "accidentally truncate a UTF-8 sequence");
  std::string result;
  result.reserve(width);
  while (result.size() < width) {
    result += str.substr(0, std::min(width - result.size(), str.size()));
  }
  return result;
}

/**
 * @brief Build a template column by repeating each string to `template_width`.
 */
std::unique_ptr<cudf::column> make_template_column(cudf::size_type template_width)
{
  std::vector<std::string> repeated;
  repeated.reserve(template_strings.size());
  std::ranges::transform(
    template_strings, std::back_inserter(repeated), [template_width](std::string_view str) {
      return repeat_to_width(str, template_width);
    });
  return cudf::test::strings_column_wrapper(repeated.begin(), repeated.end()).release();
}

/**
 * @brief Build a gather map that selects template strings with the given hit rate.
 *
 * This follows the same gather/scatter pattern as `create_string_column`.
 */
std::unique_ptr<cudf::column> make_gather_map(cudf::size_type num_rows,
                                              cudf::size_type num_templates,
                                              int32_t hit_rate)
{
  auto const num_matches = (static_cast<int64_t>(num_rows) * hit_rate) / 100;

  data_profile gather_profile = data_profile_builder().cardinality(0).no_validity().distribution(
    cudf::type_id::INT32, distribution_id::UNIFORM, 1, num_templates - 1);
  auto gather_table =
    create_random_table({cudf::type_id::INT32}, row_count{num_rows}, gather_profile);

  if (num_matches > 0) {  // guard against division by zero
    auto const zero_scalar  = cudf::numeric_scalar<int32_t>(0);
    auto const scatter_data = cudf::sequence(
      num_matches, zero_scalar, cudf::numeric_scalar<int32_t>(num_rows / num_matches));
    gather_table = cudf::scatter({zero_scalar}, scatter_data->view(), gather_table->view());
  }

  auto columns = gather_table->release();
  return std::move(columns.front());
}

std::unique_ptr<cudf::column> gather_templates(cudf::column_view const& templates,
                                               cudf::column_view const& gather_map)
{
  auto table = cudf::gather(cudf::table_view({templates}), gather_map);
  return std::move(table->release().front());
}

/**
 * @brief Build a mask indicating which rows should use the cropped short string length.
 */
std::unique_ptr<cudf::column> make_short_row_mask(cudf::size_type num_rows,
                                                  cudf::size_type short_string_pct)
{
  auto const num_short =
    static_cast<cudf::size_type>((static_cast<int64_t>(num_rows) * short_string_pct) / 100);
  auto type_table = create_random_table({cudf::type_id::INT32},
                                        row_count{num_rows},
                                        data_profile_builder().no_validity().distribution(
                                          cudf::type_id::INT32, distribution_id::UNIFORM, 1, 1));

  if (num_short > 0) {  // guard against division by zero
    auto const zero_scalar = cudf::numeric_scalar<int32_t>(0);
    auto const scatter_data =
      cudf::sequence(num_short, zero_scalar, cudf::numeric_scalar<int32_t>(num_rows / num_short));
    type_table = cudf::scatter({zero_scalar}, scatter_data->view(), type_table->view());
  }

  auto const zero_col = cudf::make_column_from_scalar(cudf::numeric_scalar<int32_t>(0), num_rows);
  return cudf::binary_operation(type_table->view().column(0),
                                zero_col->view(),
                                cudf::binary_operator::EQUAL,
                                cudf::data_type{cudf::type_id::BOOL8});
}
}  // namespace

std::unique_ptr<cudf::column> create_skewed_string_column(cudf::size_type num_rows,
                                                          cudf::size_type short_length,
                                                          cudf::size_type long_tail_length,
                                                          int32_t short_string_pct,
                                                          int32_t hit_rate)
{
  CUDF_EXPECTS(num_rows >= 0, "num_rows must be at least 0");
  CUDF_EXPECTS(short_length >= 16, "short_length must be at least 16");
  CUDF_EXPECTS(long_tail_length >= 1024, "long tail length must be at least 1024");
  CUDF_EXPECTS(short_string_pct >= 0 && short_string_pct <= 100,
               "short_string_pct must be in the range [0, 100]");
  CUDF_EXPECTS(hit_rate >= 0 && hit_rate <= 100, "hit_rate must be in the range [0, 100]");
  CUDF_EXPECTS(long_tail_length > short_length,
               "long_tail_length must be greater than short_length");
  CUDF_EXPECTS(short_length % matching_template_length == 0,
               std::format("short_length must be a multiple of the template length ({} bytes)",
                           matching_template_length));
  CUDF_EXPECTS(long_tail_length % matching_template_length == 0,
               std::format("long_tail_length must be a multiple of the template length ({} bytes)",
                           matching_template_length));

  auto const short_templates = make_template_column(short_length);
  auto const long_templates  = make_template_column(long_tail_length);
  auto const gather_map =
    make_gather_map(num_rows, static_cast<cudf::size_type>(template_strings.size()), hit_rate);
  auto const short_col     = gather_templates(short_templates->view(), gather_map->view());
  auto const long_col      = gather_templates(long_templates->view(), gather_map->view());
  auto const is_short_mask = make_short_row_mask(num_rows, short_string_pct);

  return cudf::copy_if_else(short_col->view(), long_col->view(), is_short_mask->view());
}
