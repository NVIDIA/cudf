/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "orc.hpp"

#include <map>
#include <optional>
#include <vector>

namespace cudf::io::orc::detail {

/**
 * @brief Describes a column hierarchy, which may exclude some input columns.
 */
struct column_hierarchy {
  // Maps column IDs to the IDs of their children columns
  using nesting_map = std::map<size_type, std::vector<size_type>>;
  // Children IDs of each column
  nesting_map children;
  // Each element contains column at the given nesting level
  std::vector<std::vector<orc_column_meta>> levels;

  explicit column_hierarchy(nesting_map child_map);
  [[nodiscard]] auto num_levels() const { return levels.size(); }
};

/**
 * @brief In order to support multiple input files/buffers we need to gather
 * the metadata across all of those input(s). This class provides a place
 * to aggregate that metadata from all the files.
 */
class aggregate_orc_metadata {
  /**
   * @brief Sums up the number of rows of each source
   */
  [[nodiscard]] int64_t calc_num_rows() const;

  /**
   * @brief Sums up the number of stripes of each source
   */
  [[nodiscard]] size_type calc_num_stripes() const;

 public:
  std::vector<metadata> per_file_metadata;
  int64_t const num_rows;
  size_type const num_stripes;
  bool row_grp_idx_present{true};

  aggregate_orc_metadata(std::vector<std::unique_ptr<datasource>> const& sources,
                         cuda::stream_ref stream);

  aggregate_orc_metadata(aggregate_orc_metadata const&)            = delete;
  aggregate_orc_metadata& operator=(aggregate_orc_metadata const&) = delete;
  aggregate_orc_metadata(aggregate_orc_metadata&&)                 = delete;
  aggregate_orc_metadata& operator=(aggregate_orc_metadata&&)      = delete;

  [[nodiscard]] auto get_col_type(int col_idx) const
  {
    return per_file_metadata[0].ff.types[col_idx];
  }

  [[nodiscard]] auto get_num_rows() const { return num_rows; }

  [[nodiscard]] auto get_num_cols() const { return per_file_metadata[0].get_num_columns(); }

  [[nodiscard]] auto get_num_stripes() const { return num_stripes; }

  [[nodiscard]] auto const& get_types() const { return per_file_metadata[0].ff.types; }

  [[nodiscard]] size_type get_row_index_stride() const
  {
    CUDF_EXPECTS(per_file_metadata[0].ff.rowIndexStride <= std::numeric_limits<size_type>::max(),
                 "Row index stride exceeds size_type max");
    return per_file_metadata[0].ff.rowIndexStride;
  }

  [[nodiscard]] auto is_row_grp_idx_present() const { return row_grp_idx_present; }

  /**
   * @brief Returns the name of the given column from the given source.
   */
  [[nodiscard]] std::string const& column_name(int const source_idx, int const column_id) const
  {
    CUDF_EXPECTS(std::cmp_less_equal(source_idx, per_file_metadata.size()),
                 "Out of range source_idx provided");
    return per_file_metadata[source_idx].column_name(column_id);
  }

  /**
   * @brief Returns the full name of the given column from the given source.
   *
   * Full name includes ancestor columns' names.
   */
  [[nodiscard]] std::string const& column_path(int const source_idx, int const column_id) const
  {
    CUDF_EXPECTS(std::cmp_less_equal(source_idx, per_file_metadata.size()),
                 "Out of range source_idx provided");
    return per_file_metadata[source_idx].column_path(column_id);
  }

  /**
   * @brief Selects the stripes to read, based on the row/stripe selection parameters.
   *
   * Stripes are potentially selected from multiple files.
   *
   * Upon parsing stripes' information, the number of skip rows and reading rows are also updated
   * to be matched with the actual numbers for reading stripes from data sources.
   *
   * @param user_specified_stripes The specified stripe indices to read
   * @param skip_rows Number of rows to skip from reading
   * @param num_read_rows Number of rows to read
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @return A tuple of the corrected skip_rows and num_rows values along with a vector of
   *         stripes' metadata such as footer, data information, and source index
   */
  [[nodiscard]] std::tuple<int64_t, int64_t, std::vector<metadata::orc_stripe_info>> select_stripes(
    std::vector<std::vector<size_type>> const& user_specified_stripes,
    int64_t skip_rows,
    std::optional<size_type> const& num_read_rows,
    cuda::stream_ref stream);

  /**
   * @brief Filters ORC file to a selection of columns, based on their paths in the file.
   *
   * Paths are in format "grandparent_col.parent_col.child_col", where the root ORC column is
   * omitted to match the cuDF table hierarchy.
   *
   * @param column_paths List of full column names (i.e. paths) to select from the ORC file;
   * `nullopt` if user did not select columns to read
   * @return Columns hierarchy - lists of children columns and sorted columns in each nesting level
   */
  [[nodiscard]] column_hierarchy select_columns(
    std::optional<std::vector<std::string>> const& column_paths) const;
};

}  // namespace cudf::io::orc::detail
