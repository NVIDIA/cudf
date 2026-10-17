/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "data_generation.hpp"

#include <cudf/copying.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <algorithm>
#include <cmath>
#include <functional>
#include <sstream>
#include <stdexcept>

namespace {

// Names of the writer settings, for the descriptions of failures

std::string to_string(cudf::io::column_encoding encoding)
{
  using E = cudf::io::column_encoding;
  switch (encoding) {
    case E::USE_DEFAULT: return "default";
    case E::DICTIONARY: return "DICTIONARY";
    case E::PLAIN: return "PLAIN";
    case E::DELTA_BINARY_PACKED: return "DELTA_BINARY_PACKED";
    case E::DELTA_LENGTH_BYTE_ARRAY: return "DELTA_LENGTH_BYTE_ARRAY";
    case E::DELTA_BYTE_ARRAY: return "DELTA_BYTE_ARRAY";
    case E::BYTE_STREAM_SPLIT: return "BYTE_STREAM_SPLIT";
    default: return "encoding " + std::to_string(static_cast<int>(encoding));
  }
}

std::string to_string(cudf::io::compression_type compression)
{
  using C = cudf::io::compression_type;
  switch (compression) {
    case C::NONE: return "NONE";
    case C::SNAPPY: return "SNAPPY";
    case C::GZIP: return "GZIP";
    case C::LZ4: return "LZ4";
    case C::ZSTD: return "ZSTD";
    default: return "compression " + std::to_string(static_cast<int>(compression));
  }
}

std::string to_string(cudf::io::dictionary_policy policy)
{
  using D = cudf::io::dictionary_policy;
  switch (policy) {
    case D::NEVER: return "NEVER";
    case D::ADAPTIVE: return "ADAPTIVE";
    case D::ALWAYS: return "ALWAYS";
    default: return "policy " + std::to_string(static_cast<int>(policy));
  }
}

std::string to_string(cudf::io::statistics_freq statistics)
{
  using S = cudf::io::statistics_freq;
  switch (statistics) {
    case S::STATISTICS_NONE: return "NONE";
    case S::STATISTICS_ROWGROUP: return "ROWGROUP";
    case S::STATISTICS_PAGE: return "PAGE";
    case S::STATISTICS_COLUMN: return "COLUMN";
    default: return "statistics " + std::to_string(static_cast<int>(statistics));
  }
}

std::string to_string(std::vector<cudf::type_id> const& types)
{
  std::string result;
  for (auto const type : types) {
    result += (result.empty() ? "" : ",") + cudf::type_to_name(cudf::data_type{type});
  }
  return result;
}

}  // namespace

table_settings pick_table_settings(table_recipe const& recipe, std::mt19937& rng)
{
  CUDF_EXPECTS(std::ranges::all_of(recipe.num_rows, [](auto num_rows) { return num_rows > 0; }),
               "The row counts of a table recipe must be positive",
               std::invalid_argument);
  CUDF_EXPECTS(recipe.max_columns > 0 and recipe.max_list_depth > 0 and recipe.max_struct_depth > 0,
               "The maxima of a table recipe must be at least one",
               std::invalid_argument);
  table_settings result;
  result.num_rows     = pick(rng, recipe.num_rows);
  auto const num_cols = std::uniform_int_distribution<int>(1, recipe.max_columns)(rng);
  for (int i = 0; i < num_cols; ++i) {
    result.types.push_back(pick(rng, recipe.column_types));
  }

  auto& profile               = result.profile;
  auto const null_probability = pick(rng, recipe.null_probabilities);
  profile.set_null_probability(null_probability);
  auto const cardinality = pick(rng, recipe.cardinalities);
  profile.set_cardinality(cardinality);
  auto const run_length = pick(rng, recipe.run_lengths);
  profile.set_avg_run_length(run_length);
  auto const string_length = pick(rng, recipe.string_lengths);
  profile.set_distribution_params(cudf::type_id::STRING, distribution_id::NORMAL, 0, string_length);

  auto const list_depth = std::uniform_int_distribution<int>(1, recipe.max_list_depth)(rng);
  profile.set_list_depth(list_depth);
  auto const list_type = pick(rng, recipe.list_types);
  profile.set_list_type(list_type);
  // Use shorter lists if the table would otherwise have more than `max_list_elements` leaf
  // elements, estimating the length of the lists at each level as a third of `list_length`
  auto list_length       = pick(rng, recipe.list_lengths);
  auto const is_too_long = [&](int length) {
    return result.num_rows * std::pow(length / 3.0, list_depth) > recipe.max_list_elements;
  };
  auto shorter_lengths = recipe.list_lengths;
  std::ranges::sort(shorter_lengths, std::greater{});
  for (auto const length : shorter_lengths) {
    if (length < list_length and is_too_long(list_length)) { list_length = length; }
  }
  profile.set_distribution_params(cudf::type_id::LIST, distribution_id::GEOMETRIC, 0, list_length);

  auto const struct_depth = coin(rng, recipe.deep_struct_probability)
                              ? recipe.deep_struct_depth
                              : std::uniform_int_distribution<int>(1, recipe.max_struct_depth)(rng);
  profile.set_struct_depth(struct_depth);
  std::vector<cudf::type_id> struct_types;
  auto const first_member = pick(rng, recipe.column_types);
  if (first_member != cudf::type_id::LIST and first_member != cudf::type_id::STRUCT) {
    struct_types.push_back(first_member);
  }
  struct_types.insert(
    struct_types.end(), recipe.struct_member_types.begin(), recipe.struct_member_types.end());
  profile.set_struct_types(struct_types);

  std::ostringstream description;
  description << "rows=" << result.num_rows << " types=" << to_string(result.types)
              << " null_probability=" << null_probability.value_or(-1)
              << " cardinality=" << cardinality << " run_length=" << run_length
              << " string_length=" << string_length << " list_depth=" << list_depth
              << " list_type=" << to_string(std::vector<cudf::type_id>{list_type})
              << " list_length=" << list_length << " struct_depth=" << struct_depth
              << " struct_types=" << to_string(struct_types);
  result.description = description.str();
  return result;
}

std::map<cudf::type_id, std::vector<cudf::io::column_encoding>> supported_column_encodings()
{
  using E = cudf::io::column_encoding;
  // Integers (including timestamps and 32- and 64-bit decimals), floating point values and 128-bit
  // decimals, and strings
  auto const integers = std::vector<E>{
    E::USE_DEFAULT, E::PLAIN, E::DICTIONARY, E::DELTA_BINARY_PACKED, E::BYTE_STREAM_SPLIT};
  auto const floats_and_decimal128 =
    std::vector<E>{E::USE_DEFAULT, E::PLAIN, E::DICTIONARY, E::BYTE_STREAM_SPLIT};
  auto const byte_arrays = std::vector<E>{
    E::USE_DEFAULT, E::PLAIN, E::DICTIONARY, E::DELTA_LENGTH_BYTE_ARRAY, E::DELTA_BYTE_ARRAY};
  return {{cudf::type_id::BOOL8, {E::USE_DEFAULT, E::PLAIN}},
          {cudf::type_id::INT8, integers},
          {cudf::type_id::INT16, integers},
          {cudf::type_id::INT32, integers},
          {cudf::type_id::INT64, integers},
          {cudf::type_id::UINT8, integers},
          {cudf::type_id::UINT16, integers},
          {cudf::type_id::UINT32, integers},
          {cudf::type_id::UINT64, integers},
          {cudf::type_id::TIMESTAMP_MILLISECONDS, integers},
          {cudf::type_id::TIMESTAMP_MICROSECONDS, integers},
          {cudf::type_id::DECIMAL32, integers},
          {cudf::type_id::DECIMAL64, integers},
          {cudf::type_id::FLOAT32, floats_and_decimal128},
          {cudf::type_id::FLOAT64, floats_and_decimal128},
          {cudf::type_id::DECIMAL128, floats_and_decimal128},
          {cudf::type_id::STRING, byte_arrays}};
}

writer_settings pick_writer_settings(writer_recipe const& recipe,
                                     cudf::table_view const& table,
                                     std::mt19937& rng)
{
  writer_settings result{cudf::io::table_input_metadata{table}};
  result.dictionary_policy      = pick(rng, recipe.dictionary_policies);
  result.v2_headers             = coin(rng, recipe.v2_headers_probability);
  result.compression            = pick(rng, recipe.compressions);
  result.max_page_bytes         = pick(rng, recipe.max_page_bytes);
  result.max_page_rows          = pick(rng, recipe.max_page_rows);
  result.max_page_fragment_rows = pick(rng, recipe.max_page_fragment_rows);
  result.row_group_rows         = pick(rng, recipe.row_group_rows);
  result.row_group_bytes        = pick(rng, recipe.row_group_bytes);
  result.max_dictionary_bytes   = pick(rng, recipe.max_dictionary_bytes);
  result.statistics             = pick(rng, recipe.statistics);
  result.num_files =
    table.num_rows() >= recipe.min_split_rows and coin(rng, recipe.split_probability)
      ? pick(rng, recipe.split_file_counts)
      : 1;

  std::ostringstream description;
  description << "encodings=";
  for (cudf::size_type i = 0; i < table.num_columns(); ++i) {
    auto const encodings = recipe.column_encodings.find(table.column(i).type().id());
    auto const encoding  = encodings == recipe.column_encodings.end()
                             ? cudf::io::column_encoding::USE_DEFAULT
                             : pick(rng, encodings->second);
    result.metadata.column_metadata[i].set_encoding(encoding);
    description << (i > 0 ? "," : "") << to_string(encoding);
  }
  description << " dictionary_policy=" << to_string(result.dictionary_policy)
              << " v2_headers=" << result.v2_headers
              << " compression=" << to_string(result.compression)
              << " max_page_bytes=" << result.max_page_bytes
              << " max_page_rows=" << result.max_page_rows
              << " max_page_fragment_rows=" << result.max_page_fragment_rows
              << " row_group_rows=" << result.row_group_rows
              << " row_group_bytes=" << result.row_group_bytes
              << " max_dictionary_bytes=" << result.max_dictionary_bytes
              << " statistics=" << to_string(result.statistics) << " files=" << result.num_files;
  result.description = description.str();
  return result;
}

void write_parquet_files(cudf::table_view const& table,
                         writer_settings const& settings,
                         std::vector<std::string> const& paths)
{
  auto const num_rows  = table.num_rows();
  auto const num_files = static_cast<cudf::size_type>(paths.size());
  auto const file_rows = (num_rows + num_files - 1) / num_files;
  for (cudf::size_type f = 0; f < num_files; ++f) {
    auto const begin = std::min(f * file_rows, num_rows);
    auto const end   = std::min(begin + file_rows, num_rows);
    cudf::io::write_parquet(cudf::io::parquet_writer_options::builder(
                              cudf::io::sink_info{paths[f]}, cudf::slice(table, {begin, end})[0])
                              .metadata(settings.metadata)
                              .dictionary_policy(settings.dictionary_policy)
                              .write_v2_headers(settings.v2_headers)
                              .compression(settings.compression)
                              .max_page_size_bytes(settings.max_page_bytes)
                              .max_page_size_rows(settings.max_page_rows)
                              .max_page_fragment_size(settings.max_page_fragment_rows)
                              .row_group_size_rows(settings.row_group_rows)
                              .row_group_size_bytes(settings.row_group_bytes)
                              .max_dictionary_size(settings.max_dictionary_bytes)
                              .stats_level(settings.statistics)
                              .build());
  }
}

bool is_writer_limit(std::exception const& e)
{
  if (dynamic_cast<std::overflow_error const*>(&e) != nullptr) { return true; }
  std::string const message = e.what();
  // A row larger than a page, or the encoder's PAGE_SIZE_OVERFLOW (0x1) and
  // COLUMN_CHUNK_SIZE_OVERFLOW (0x2) errors for pages or column chunks too large for its buffers
  return message.find("exceeds the maximum Parquet page size") != std::string::npos or
         message.ends_with("code(s) 0x1") or message.ends_with("code(s) 0x2") or
         message.ends_with("code(s) 0x3");
}
