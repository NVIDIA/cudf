/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "hybrid_scan_common.hpp"

#include "tests/io/parquet_common.hpp"

#include <cudf_test/column_wrapper.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/io/parquet_io_utils.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/traits.hpp>

#include <rmm/device_buffer.hpp>

#include <cuda/iterator>

#include <algorithm>
#include <format>
#include <iterator>
#include <string>
#include <tuple>
#include <type_traits>

namespace {

/**
 * @brief Creates a strings column with a constant stringified value between 0 and 9999
 *
 * @param value String value between 0 and 9999
 * @return Strings column wrapper
 */
cudf::test::strings_column_wrapper constant_strings(cudf::size_type value)
{
  CUDF_EXPECTS(value >= 0 && value <= 9999, "String value must be between 0000 and 9999");

  auto elements = thrust::make_transform_iterator(cuda::make_constant_iterator(value),
                                                  [](auto i) { return std::format("{:04d}", i); });
  return cudf::test::strings_column_wrapper(elements, elements + num_ordered_rows);
}

/**
 * @brief Fail for types other than duration or timestamp
 */
template <typename T, CUDF_ENABLE_IF(not cudf::is_chrono<T>())>
cudf::test::fixed_width_column_wrapper<T> descending_low_cardinality()
{
  static_assert(
    cudf::is_chrono<T>(),
    "Use testdata::descending<T>() to generate descending values for non-temporal types");
}

/**
 * @brief Creates a duration column wrapper with low cardinality descending values
 *
 * @tparam T Duration type
 * @return Column wrapper
 */
template <typename T, CUDF_ENABLE_IF(cudf::is_duration<T>())>
cudf::test::fixed_width_column_wrapper<T> descending_low_cardinality()
{
  auto elements = cudf::detail::make_counting_transform_iterator(
    0, [](auto i) { return T((num_ordered_rows - i) / 100); });
  return cudf::test::fixed_width_column_wrapper<T>(elements, elements + num_ordered_rows);
}

/**
 * @brief Creates a timestamp column wrapper with low cardinality descending values
 *
 * @tparam T Timestamp type
 * @return Column wrapper
 */
template <typename T, CUDF_ENABLE_IF(cudf::is_timestamp<T>())>
cudf::test::fixed_width_column_wrapper<T> descending_low_cardinality()
{
  auto elements = cudf::detail::make_counting_transform_iterator(
    0, [](auto i) { return T(typename T::duration((num_ordered_rows - i) / 100)); });
  return cudf::test::fixed_width_column_wrapper<T>(elements, elements + num_ordered_rows);
}

}  // namespace

std::unique_ptr<cudf::column> make_list_str_column(std::mt19937& gen,
                                                   bool is_str_nullable,
                                                   bool is_list_nullable)
{
  auto constexpr num_rows        = num_ordered_rows;
  auto constexpr string_per_row  = 3;
  auto constexpr num_string_rows = num_rows * string_per_row;

  std::vector<std::string> strings{
    "abc", "x", "bananas", "gpu", "minty", "backspace", "", "cayenne", "turbine", "soft"};
  std::uniform_int_distribution<int> uni(0, strings.size() - 1);
  auto string_iter = cudf::detail::make_counting_transform_iterator(
    0, [&](cudf::size_type idx) { return strings[uni(gen)]; });

  std::bernoulli_distribution bn(0.7f);
  auto string_valids = cudf::detail::make_counting_transform_iterator(
    0, [&](int index) { return is_str_nullable ? bn(gen) : true; });
  cudf::test::strings_column_wrapper string_col{
    string_iter, string_iter + num_string_rows, string_valids};

  auto offset_iter = cudf::detail::make_counting_transform_iterator(
    0, [](cudf::size_type idx) { return idx * string_per_row; });
  cudf::test::fixed_width_column_wrapper<cudf::size_type> offsets(offset_iter,
                                                                  offset_iter + num_rows + 1);

  auto list_valids =
    cudf::detail::make_counting_transform_iterator(0, [&](int index) { return index % 100; });
  auto [null_mask, null_count] = [&]() {
    if (is_list_nullable) {
      return cudf::test::detail::make_null_mask(list_valids, list_valids + num_rows);
    } else {
      return std::make_pair(rmm::device_buffer{}, 0);
    }
  }();
  return cudf::make_lists_column(
    num_rows, offsets.release(), string_col.release(), null_count, std::move(null_mask));
}

multifile_inputs::multifile_inputs(cudf::io::source_info const& source_info)
  : datasources{cudf::io::make_datasources(source_info)}
{
  datasource_refs.reserve(datasources.size());
  footer_buffers.reserve(datasources.size());
  footer_byte_spans.reserve(datasources.size());

  for (auto const& datasource : datasources) {
    datasource_refs.emplace_back(*datasource);
    footer_buffers.emplace_back(cudf::io::parquet::fetch_footer_to_host(datasource_refs.back()));
    footer_byte_spans.emplace_back(*footer_buffers.back());
  }
}

cudf::io::source_info build_source_info(std::vector<std::vector<char>> const& file_buffers)
{
  std::vector<cudf::host_span<char const>> spans;
  spans.reserve(file_buffers.size());
  for (auto const& buf : file_buffers) {
    spans.emplace_back(buf.data(), buf.size());
  }
  return cudf::io::source_info(cudf::host_span<cudf::host_span<char const>>{spans});
}

void setup_page_indexes(cudf::io::parquet::experimental::hybrid_scan_multifile const& reader,
                        multifile_inputs const& inputs)
{
  auto const page_index_byte_ranges = reader.page_index_byte_ranges();
  std::vector<cudf::host_span<uint8_t const>> page_index_byte_spans;
  page_index_byte_spans.reserve(page_index_byte_ranges.size());

  auto const page_index_buffers = cudf::io::parquet::fetch_page_indexes_to_host(
    cudf::host_span<std::reference_wrapper<cudf::io::datasource> const>{inputs.datasource_refs},
    cudf::host_span<cudf::io::parquet::byte_range_info const>{page_index_byte_ranges});
  std::transform(page_index_buffers.begin(),
                 page_index_buffers.end(),
                 std::back_inserter(page_index_byte_spans),
                 [](auto const& buffer) { return cudf::host_span<uint8_t const>{*buffer}; });

  reader.setup_page_indexes(
    cudf::host_span<cudf::host_span<uint8_t const> const>{page_index_byte_spans});
}

std::vector<std::vector<cudf::io::text::byte_range_info>> group_byte_ranges_by_source(
  std::pair<std::vector<cudf::io::text::byte_range_info>, std::vector<cudf::size_type>> const&
    byte_ranges_and_source_map,
  std::size_t num_sources)
{
  auto const& [byte_ranges, source_map] = byte_ranges_and_source_map;
  CUDF_EXPECTS(byte_ranges.size() == source_map.size(), "Invalid source map size");

  auto byte_ranges_per_source =
    std::vector<std::vector<cudf::io::text::byte_range_info>>(num_sources);
  std::for_each(byte_ranges.begin(),
                byte_ranges.end(),
                [&, range_index = std::size_t{0}](auto const& range) mutable {
                  auto const source_index = source_map[range_index++];
                  CUDF_EXPECTS(source_index >= 0 and static_cast<std::size_t>(source_index) <
                                                       byte_ranges_per_source.size(),
                               "Invalid byte range source index");
                  byte_ranges_per_source[source_index].push_back(range);
                });
  return byte_ranges_per_source;
}

multisource_device_data fetch_multisource_device_data(
  multifile_inputs const& inputs,
  std::pair<std::vector<cudf::io::text::byte_range_info>, std::vector<cudf::size_type>> const&
    byte_ranges_and_source_map,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  auto const byte_ranges_per_source =
    group_byte_ranges_by_source(byte_ranges_and_source_map, inputs.datasources.size());
  auto [buffers, per_source_spans, tasks] = cudf::io::parquet::fetch_byte_ranges_to_device_async(
    inputs.datasource_refs,
    cudf::host_span<std::vector<cudf::io::text::byte_range_info> const>{byte_ranges_per_source},
    stream,
    mr);
  tasks.get();

  auto flat_spans = std::vector<cudf::device_span<uint8_t const>>{};
  for (auto const& source_spans : per_source_spans) {
    flat_spans.insert(flat_spans.end(), source_spans.begin(), source_spans.end());
  }

  return {std::move(buffers), std::move(per_source_spans), std::move(flat_spans)};
}

std::unique_ptr<cudf::table> concatenate_tables(std::vector<std::unique_ptr<cudf::table>>&& tables,
                                                rmm::cuda_stream_view stream,
                                                rmm::device_async_resource_ref mr)
{
  if (tables.size() == 1) { return std::move(tables[0]); }

  auto table_views = std::vector<cudf::table_view>{};
  table_views.reserve(tables.size());
  std::transform(
    tables.begin(), tables.end(), std::back_inserter(table_views), [](auto const& tbl) {
      return tbl->view();
    });
  return cudf::concatenate(table_views, stream, mr);
}

namespace {

// Shared implementation templated over the reader; the single-file and multi-file public helpers
// below forward to it.
template <typename ReaderType, typename InputType>
auto filter_row_groups_with_dictionaries_impl(InputType& inputs,
                                              ReaderType const& reader,
                                              cudf::io::parquet_reader_options const& options,
                                              rmm::cuda_stream_view stream,
                                              rmm::device_async_resource_ref mr)
{
  reader.reset_column_selection();
  auto const row_group_indices = reader.all_row_groups(options);

  if constexpr (std::is_same_v<ReaderType,
                               cudf::io::parquet::experimental::hybrid_scan_multifile>) {
    auto const dict_pages = reader.dictionary_pages_byte_ranges(row_group_indices, options);
    CUDF_EXPECTS(dict_pages.first.size() > 0, "No dictionary page byte ranges found");

    auto const dict_page_ranges_per_source =
      group_byte_ranges_by_source(dict_pages, inputs.datasources.size());
    [[maybe_unused]] auto [dict_page_buffers, dict_page_data_per_source, task] =
      cudf::io::parquet::fetch_byte_ranges_to_device_async(
        inputs.datasource_refs, dict_page_ranges_per_source, stream, mr);
    task.get();

    std::vector<cudf::device_span<uint8_t const>> dict_page_data;
    for (auto const& source_dict_pages : dict_page_data_per_source) {
      dict_page_data.insert(
        dict_page_data.end(), source_dict_pages.begin(), source_dict_pages.end());
    }

    return reader.filter_row_groups_with_dictionary_pages(
      dict_page_data, row_group_indices, options, stream);
  } else {
    auto const dict_page_byte_ranges =
      reader.secondary_filters_byte_ranges(row_group_indices, options).second;
    CUDF_EXPECTS(dict_page_byte_ranges.size() > 0, "No dictionary page byte ranges found");

    [[maybe_unused]] auto [dict_page_buffers, dict_page_data, dict_page_tasks] =
      cudf::io::parquet::fetch_byte_ranges_to_device_async(
        inputs, dict_page_byte_ranges, stream, mr);
    dict_page_tasks.get();

    return reader.filter_row_groups_with_dictionary_pages(
      dict_page_data, row_group_indices, options, stream);
  }
}

}  // namespace

std::vector<cudf::size_type> filter_row_groups_with_dictionaries(
  cudf::io::datasource& datasource,
  cudf::io::parquet::experimental::hybrid_scan_reader const& reader,
  cudf::io::parquet_reader_options const& options,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  return filter_row_groups_with_dictionaries_impl(datasource, reader, options, stream, mr);
}

std::vector<std::vector<cudf::size_type>> filter_row_groups_with_dictionaries(
  multifile_inputs const& inputs,
  cudf::io::parquet::experimental::hybrid_scan_multifile const& reader,
  cudf::io::parquet_reader_options const& options,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  return filter_row_groups_with_dictionaries_impl(inputs, reader, options, stream, mr);
}

template <typename T, size_t NumTableConcats, bool IsConstantStrings, bool IsNullable>
std::pair<std::unique_ptr<cudf::table>, std::vector<char>> create_parquet_with_stats(
  cudf::size_type str_col_value,
  cudf::io::compression_type compression,
  std::vector<std::string> column_names,
  std::vector<cudf::size_type> column_order,
  rmm::cuda_stream_view stream)
{
  static_assert(NumTableConcats >= 1, "Concatenated table must contain at least one table");

  CUDF_EXPECTS(column_names.size() == column_order.size(),
               "Column names and column order must have the same size");
  CUDF_EXPECTS(column_order.size() == 3, "Column order must include all three test columns");

  auto col0 = testdata::ascending<T>();
  auto col1 = []() {
    if constexpr (cudf::is_chrono<T>()) {
      return descending_low_cardinality<T>();
    } else {
      return testdata::descending<T>();
    }
  }();

  auto col2 = [&]() {
    if constexpr (IsConstantStrings) {
      return constant_strings(str_col_value);  // constant stringified value
    } else {
      return testdata::ascending<cudf::string_view>();  // ascending strings
    }
  }();

  // Output table view
  auto output = table_view{{col0, col1, col2}};

  // Add nullmasks to the columns if specified
  std::vector<std::unique_ptr<cudf::column>> columns;
  if constexpr (IsNullable) {
    std::mt19937 gen(0xc0ffee);
    std::bernoulli_distribution bn(0.7f);
    auto valids =
      cudf::detail::make_counting_transform_iterator(0, [&](int index) { return bn(gen); });
    auto const num_rows = static_cast<cudf::column_view>(col0).size();

    auto const make_null_mask = [stream](auto begin, auto end) {
      auto [null_mask, null_count] = cudf::test::detail::make_null_mask_vector(begin, end);
      auto d_mask                  = rmm::device_buffer{
        null_mask.data(), cudf::bitmask_allocation_size_bytes(cudf::distance(begin, end)), stream};
      return std::pair{std::move(d_mask), null_count};
    };

    columns.emplace_back(col0.release());
    auto [nullmask, nullcount] = make_null_mask(valids, valids + num_rows);
    columns.back()->set_null_mask(std::move(nullmask), nullcount);

    columns.emplace_back(col1.release());
    std::tie(nullmask, nullcount) = make_null_mask(valids + num_rows, valids + 2 * num_rows);
    columns.back()->set_null_mask(std::move(nullmask), nullcount);

    columns.emplace_back(col2.release());
    std::tie(nullmask, nullcount) = make_null_mask(valids + 2 * num_rows, valids + 3 * num_rows);
    columns.back()->set_null_mask(std::move(nullmask), nullcount);

    // Purge non-empty nulls from the strings column only
    columns.back() = cudf::purge_nonempty_nulls(columns.back()->view(), stream);

    // Update the output table view with the nullable columns
    output = table_view{{columns[0]->view(), columns[1]->view(), columns[2]->view()}};
  }

  // Reorder the base [col0, col1, col2] columns into the requested physical order, naming them in
  // that new order.
  std::vector<cudf::column_view> reordered_columns;
  reordered_columns.reserve(column_order.size());
  for (auto const col_idx : column_order) {
    reordered_columns.emplace_back(output.column(col_idx));
  }
  output = table_view{reordered_columns};

  auto table = cudf::concatenate(std::vector<table_view>(NumTableConcats, output), stream);
  output     = table->view();
  cudf::io::table_input_metadata output_metadata(output);
  for (std::size_t i = 0; i < column_names.size(); ++i) {
    output_metadata.column_metadata[i].set_name(column_names[i]);
  }

  std::vector<char> buffer;
  cudf::io::parquet_writer_options out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{&buffer}, output)
      .metadata(std::move(output_metadata))
      .row_group_size_rows(page_size_for_ordered_tests)
      .max_page_size_rows(page_size_for_ordered_tests / 5)
      .compression(compression)
      .dictionary_policy(cudf::io::dictionary_policy::ALWAYS)
      .stats_level(cudf::io::statistics_freq::STATISTICS_COLUMN);

  if constexpr (NumTableConcats > 1) {
    out_opts.set_row_group_size_rows(num_ordered_rows);
    out_opts.set_max_page_size_rows(page_size_for_ordered_tests);
  }

  cudf::io::write_parquet(out_opts, stream);

  return std::pair{std::move(table), std::move(buffer)};
}

#define INSTANTIATE_CREATE_PARQUET_WITH_STATS(T, NUM_CONCATS, CONSTANT_STRINGS, NULLABLE) \
  template std::pair<std::unique_ptr<cudf::table>, std::vector<char>>                     \
  create_parquet_with_stats<T, NUM_CONCATS, CONSTANT_STRINGS, NULLABLE>(                  \
    cudf::size_type,                                                                      \
    cudf::io::compression_type,                                                           \
    std::vector<std::string>,                                                             \
    std::vector<cudf::size_type>,                                                         \
    rmm::cuda_stream_view)

#define INSTANTIATE_CREATE_PARQUET_WITH_STATS_DICT(T)       \
  INSTANTIATE_CREATE_PARQUET_WITH_STATS(T, 1, true, false); \
  INSTANTIATE_CREATE_PARQUET_WITH_STATS(T, 1, true, true)

INSTANTIATE_CREATE_PARQUET_WITH_STATS(uint32_t, 4, true, false);
INSTANTIATE_CREATE_PARQUET_WITH_STATS(cudf::timestamp_ms, 2, true, false);
INSTANTIATE_CREATE_PARQUET_WITH_STATS(cudf::duration_ms, 2, true, false);

INSTANTIATE_CREATE_PARQUET_WITH_STATS(uint8_t, 1, false, false);
INSTANTIATE_CREATE_PARQUET_WITH_STATS(uint16_t, 1, false, false);
INSTANTIATE_CREATE_PARQUET_WITH_STATS(uint32_t, 1, false, false);
INSTANTIATE_CREATE_PARQUET_WITH_STATS(uint64_t, 1, false, false);
INSTANTIATE_CREATE_PARQUET_WITH_STATS(cudf::duration_ms, 1, false, false);

INSTANTIATE_CREATE_PARQUET_WITH_STATS(uint8_t, 2, false, false);
INSTANTIATE_CREATE_PARQUET_WITH_STATS(uint16_t, 2, false, false);
INSTANTIATE_CREATE_PARQUET_WITH_STATS(uint32_t, 2, false, false);
INSTANTIATE_CREATE_PARQUET_WITH_STATS(uint64_t, 2, false, false);

INSTANTIATE_CREATE_PARQUET_WITH_STATS_DICT(int8_t);
INSTANTIATE_CREATE_PARQUET_WITH_STATS_DICT(int16_t);
INSTANTIATE_CREATE_PARQUET_WITH_STATS_DICT(int32_t);
INSTANTIATE_CREATE_PARQUET_WITH_STATS_DICT(int64_t);
INSTANTIATE_CREATE_PARQUET_WITH_STATS_DICT(uint8_t);
INSTANTIATE_CREATE_PARQUET_WITH_STATS_DICT(uint16_t);
INSTANTIATE_CREATE_PARQUET_WITH_STATS_DICT(uint32_t);
INSTANTIATE_CREATE_PARQUET_WITH_STATS_DICT(uint64_t);
INSTANTIATE_CREATE_PARQUET_WITH_STATS_DICT(float);
INSTANTIATE_CREATE_PARQUET_WITH_STATS_DICT(double);
INSTANTIATE_CREATE_PARQUET_WITH_STATS_DICT(cudf::timestamp_D);
INSTANTIATE_CREATE_PARQUET_WITH_STATS_DICT(cudf::timestamp_ms);
INSTANTIATE_CREATE_PARQUET_WITH_STATS_DICT(cudf::timestamp_us);
INSTANTIATE_CREATE_PARQUET_WITH_STATS_DICT(cudf::timestamp_ns);
INSTANTIATE_CREATE_PARQUET_WITH_STATS_DICT(cudf::duration_ms);
INSTANTIATE_CREATE_PARQUET_WITH_STATS_DICT(cudf::duration_us);
INSTANTIATE_CREATE_PARQUET_WITH_STATS_DICT(cudf::duration_ns);
INSTANTIATE_CREATE_PARQUET_WITH_STATS_DICT(cudf::string_view);

#undef INSTANTIATE_CREATE_PARQUET_WITH_STATS_DICT
#undef INSTANTIATE_CREATE_PARQUET_WITH_STATS
