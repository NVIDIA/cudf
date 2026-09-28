/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "io/orc/aggregate_orc_metadata.hpp"

#include "io/utilities/row_selection.hpp"

#include <algorithm>
#include <functional>
#include <numeric>
#include <utility>

namespace cudf::io::orc::detail {

column_hierarchy::column_hierarchy(nesting_map child_map) : children{std::move(child_map)}
{
  // Sort columns by nesting levels
  std::function<void(size_type, int32_t)> levelize = [&](size_type id, int32_t level) {
    if (std::cmp_equal(levels.size(), level)) levels.emplace_back();

    levels[level].push_back({id, static_cast<int32_t>(children[id].size())});

    for (auto child_id : children[id]) {
      levelize(child_id, level + 1);
    }
  };

  std::for_each(
    children[0].cbegin(), children[0].cend(), [&](auto col_id) { levelize(col_id, 0); });
}

namespace {

/**
 * @brief Goes up to the root to include the column with the given id and its parents.
 */
void update_parent_mapping(std::map<size_type, std::vector<size_type>>& selected_columns,
                           metadata const& metadata,
                           size_type id)
{
  auto current_id = id;
  while (metadata.column_has_parent(current_id)) {
    auto parent_id = metadata.parent_id(current_id);
    if (std::find(selected_columns[parent_id].cbegin(),
                  selected_columns[parent_id].cend(),
                  current_id) == selected_columns[parent_id].end()) {
      selected_columns[parent_id].push_back(current_id);
    }
    current_id = parent_id;
  }
}

/**
 * @brief Adds all columns nested under the column with the given id to the nesting map.
 */
void add_nested_columns(std::map<size_type, std::vector<size_type>>& selected_columns,
                        std::vector<SchemaType> const& types,
                        size_type id)
{
  for (auto child_id : types[id].subtypes) {
    if (std::find(selected_columns[id].cbegin(), selected_columns[id].cend(), child_id) ==
        selected_columns[id].end()) {
      selected_columns[id].push_back(child_id);
    }
    add_nested_columns(selected_columns, types, child_id);
  }
}

/**
 * @brief Adds the column with the given id to the mapping
 *
 * All nested columns and direct ancestors of column `id` are included.
 * Columns that are not on the direct path are excluded, which may result in pruning.
 */
void add_column_to_mapping(std::map<size_type, std::vector<size_type>>& selected_columns,
                           metadata const& metadata,
                           size_type id)
{
  update_parent_mapping(selected_columns, metadata, id);
  add_nested_columns(selected_columns, metadata.ff.types, id);
}

/**
 * @brief Create a metadata object from each element in the source vector
 */
auto metadatas_from_sources(std::vector<std::unique_ptr<datasource>> const& sources,
                            cuda::stream_ref stream)
{
  std::vector<metadata> metadatas;
  metadatas.reserve(sources.size());
  std::transform(
    sources.cbegin(), sources.cend(), std::back_inserter(metadatas), [stream](auto const& source) {
      return metadata(source.get(), stream);
    });
  return metadatas;
}

}  // namespace

int64_t aggregate_orc_metadata::calc_num_rows() const
{
  return std::accumulate(
    per_file_metadata.begin(), per_file_metadata.end(), 0l, [](auto const& sum, auto const& pfm) {
      return sum + pfm.get_total_rows();
    });
}

size_type aggregate_orc_metadata::calc_num_stripes() const
{
  return std::accumulate(
    per_file_metadata.begin(), per_file_metadata.end(), 0, [](auto const& sum, auto const& pfm) {
      return sum + pfm.get_num_stripes();
    });
}

aggregate_orc_metadata::aggregate_orc_metadata(
  std::vector<std::unique_ptr<datasource>> const& sources, cuda::stream_ref stream)
  : per_file_metadata(metadatas_from_sources(sources, stream)),
    num_rows(calc_num_rows()),
    num_stripes(calc_num_stripes())
{
  // Verify that the input files have the same number of columns,
  // as well as matching types, compression, and names
  for (auto const& pfm : per_file_metadata) {
    CUDF_EXPECTS(per_file_metadata[0].get_num_columns() == pfm.get_num_columns(),
                 "All sources must have the same number of columns");
    CUDF_EXPECTS(per_file_metadata[0].ps.compression == pfm.ps.compression,
                 "All sources must have the same compression type");

    // Check the types, column names, and decimal scale
    for (size_t i = 0; i < pfm.ff.types.size(); i++) {
      CUDF_EXPECTS(pfm.ff.types[i].kind == per_file_metadata[0].ff.types[i].kind,
                   "Column types across all input sources must be the same");
      CUDF_EXPECTS(std::equal(pfm.ff.types[i].fieldNames.begin(),
                              pfm.ff.types[i].fieldNames.end(),
                              per_file_metadata[0].ff.types[i].fieldNames.begin()),
                   "All source column names must be the same");
      CUDF_EXPECTS(
        pfm.ff.types[i].scale.value_or(0) == per_file_metadata[0].ff.types[i].scale.value_or(0),
        "All scale values must be the same");
    }
  }
}

std::tuple<int64_t, int64_t, std::vector<metadata::orc_stripe_info>>
aggregate_orc_metadata::select_stripes(
  std::vector<std::vector<size_type>> const& user_specified_stripes,
  int64_t skip_rows,
  std::optional<size_type> const& num_read_rows,
  cuda::stream_ref stream)
{
  CUDF_EXPECTS((skip_rows == 0 and not num_read_rows.has_value()) or user_specified_stripes.empty(),
               "Can't use both the row selection and the stripe selection");

  auto [rows_to_skip, rows_to_read] = [&]() {
    if (not user_specified_stripes.empty()) { return std::pair<int64_t, int64_t>{0, 0}; }
    return cudf::io::detail::skip_rows_num_rows_from_options(
      skip_rows, num_read_rows, get_num_rows());
  }();

  struct stripe_source_mapping {
    int source_idx;
    std::vector<metadata::orc_stripe_info> stripe_info;
  };

  std::vector<stripe_source_mapping> selected_stripes_mapping;

  if (!user_specified_stripes.empty()) {
    CUDF_EXPECTS(user_specified_stripes.size() == per_file_metadata.size(),
                 "Must specify stripes for each source");

    // Each vector entry represents a source file; each nested vector represents the
    // user_defined_stripes to get from that source file
    for (size_t src_file_idx = 0; src_file_idx < user_specified_stripes.size(); ++src_file_idx) {
      std::vector<metadata::orc_stripe_info> stripe_infos;
      stripe_infos.reserve(user_specified_stripes[src_file_idx].size());

      // Coalesce stripe info at the source file later since that makes downstream processing much
      // easier in impl::read
      for (auto const& stripe_idx : user_specified_stripes[src_file_idx]) {
        CUDF_EXPECTS(
          stripe_idx >= 0 and stripe_idx < static_cast<decltype(stripe_idx)>(
                                             per_file_metadata[src_file_idx].ff.stripes.size()),
          "Invalid stripe index");
        stripe_infos.push_back({&per_file_metadata[src_file_idx].ff.stripes[stripe_idx],
                                nullptr,
                                static_cast<int>(src_file_idx)});

        auto const stripe_rows =
          per_file_metadata[src_file_idx].ff.stripes[stripe_idx].numberOfRows;
        CUDF_EXPECTS(stripe_rows < static_cast<uint64_t>(std::numeric_limits<size_type>::max()),
                     "The number of rows in one stripe exceeds the column size limit.",
                     std::overflow_error);
        rows_to_read += static_cast<int64_t>(stripe_rows);
      }
      selected_stripes_mapping.emplace_back(
        stripe_source_mapping{static_cast<int>(src_file_idx), std::move(stripe_infos)});
    }
  } else {
    int64_t count            = 0;
    int64_t stripe_skip_rows = 0;
    // Iterate all source files, each source file has correlating metadata
    for (size_t src_file_idx = 0;
         src_file_idx < per_file_metadata.size() && count < rows_to_skip + rows_to_read;
         ++src_file_idx) {
      std::vector<metadata::orc_stripe_info> stripe_infos;
      stripe_infos.reserve(per_file_metadata[src_file_idx].ff.stripes.size());

      for (size_t stripe_idx = 0; stripe_idx < per_file_metadata[src_file_idx].ff.stripes.size() &&
                                  count < rows_to_skip + rows_to_read;
           ++stripe_idx) {
        auto const stripe_rows =
          per_file_metadata[src_file_idx].ff.stripes[stripe_idx].numberOfRows;
        CUDF_EXPECTS(stripe_rows < static_cast<uint64_t>(std::numeric_limits<size_type>::max()),
                     "The number of rows in one stripe exceeds the column size limit.",
                     std::overflow_error);
        count += static_cast<int64_t>(stripe_rows);

        if (count > rows_to_skip || count == 0) {
          stripe_infos.push_back({&per_file_metadata[src_file_idx].ff.stripes[stripe_idx],
                                  nullptr,
                                  static_cast<int>(src_file_idx)});
        } else {
          stripe_skip_rows = count;
        }
      }

      selected_stripes_mapping.emplace_back(
        stripe_source_mapping{static_cast<int>(src_file_idx), std::move(stripe_infos)});
    }
    // Need to remove skipped rows from the stripes which are not selected.
    rows_to_skip -= stripe_skip_rows;
  }

  std::vector<metadata::orc_stripe_info> output;

  // Read each stripe's stripefooter metadata
  for (auto& mapping : selected_stripes_mapping) {
    // Resize to all stripe_info for the source level
    per_file_metadata[mapping.source_idx].stripefooters.resize(mapping.stripe_info.size());

    for (size_t i = 0; i < mapping.stripe_info.size(); i++) {
      auto const stripe    = mapping.stripe_info[i].stripe_info;
      auto const file_size = per_file_metadata[mapping.source_idx].source->size();
      CUDF_EXPECTS(stripe->offset <= file_size,
                   "Invalid stripe information: offset exceeds file size",
                   std::out_of_range);
      auto remaining = file_size - stripe->offset;
      CUDF_EXPECTS(stripe->indexLength <= remaining,
                   "Invalid stripe information: indexLength exceeds file size",
                   std::out_of_range);
      remaining -= stripe->indexLength;
      CUDF_EXPECTS(stripe->dataLength <= remaining,
                   "Invalid stripe information: dataLength exceeds file size",
                   std::out_of_range);
      remaining -= stripe->dataLength;
      CUDF_EXPECTS(stripe->footerLength <= remaining,
                   "Invalid stripe information: footerLength exceeds file size",
                   std::out_of_range);
      auto const sf_comp_offset = stripe->offset + stripe->indexLength + stripe->dataLength;
      auto const sf_comp_length = stripe->footerLength;
      auto const buffer =
        per_file_metadata[mapping.source_idx].source->host_read(sf_comp_offset, sf_comp_length);
      auto sf_data = per_file_metadata[mapping.source_idx].decompressor->decompress_blocks(
        {buffer->data(), buffer->size()});
      protobuf_reader(sf_data.data(), sf_data.size())
        .read(per_file_metadata[mapping.source_idx].stripefooters[i]);
      auto const& stripe_footer = per_file_metadata[mapping.source_idx].stripefooters[i];
      auto const num_types      = per_file_metadata[mapping.source_idx].ff.types.size();
      CUDF_EXPECTS(stripe_footer.columns.size() >= num_types,
                   "Invalid ColumnEncoding field in a stripe footer.",
                   std::out_of_range);
      mapping.stripe_info[i].stripe_footer =
        &per_file_metadata[mapping.source_idx].stripefooters[i];
      if (stripe->indexLength == 0) { row_grp_idx_present = false; }
    }

    output.insert(output.end(),
                  std::make_move_iterator(mapping.stripe_info.begin()),
                  std::make_move_iterator(mapping.stripe_info.end()));
  }

  return {rows_to_skip, rows_to_read, std::move(output)};
}

column_hierarchy aggregate_orc_metadata::select_columns(
  std::optional<std::vector<std::string>> const& column_paths) const
{
  auto const& pfm = per_file_metadata[0];

  column_hierarchy::nesting_map selected_columns;
  if (not column_paths.has_value()) {
    for (auto const& col_id : pfm.ff.types[0].subtypes) {
      add_column_to_mapping(selected_columns, pfm, col_id);
    }
  } else {
    for (auto const& path : column_paths.value()) {
      bool name_found = false;
      for (auto col_id = 1; col_id < pfm.get_num_columns(); ++col_id) {
        if (pfm.column_path(col_id) == path) {
          name_found = true;
          add_column_to_mapping(selected_columns, pfm, col_id);
          break;
        }
      }
      CUDF_EXPECTS(name_found, "Unknown column name: " + std::string(path));
    }
  }
  return column_hierarchy{std::move(selected_columns)};
}

}  // namespace cudf::io::orc::detail
