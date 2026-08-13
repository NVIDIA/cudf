/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/column/column_view.hpp>
#include <cudf/lists/lists_column_view.hpp>

/**
 * @file
 * @brief tdigest data APIs
 */

namespace CUDF_EXPORT cudf {
//! Tdigest interfaces
namespace tdigest {
/**
 * @addtogroup tdigest
 * @{
 */

/**
 * @brief Given a column_view containing tdigest data, an instance of this class
 * provides a wrapper on the compound column for tdigest operations.
 *
 * A tdigest is a "compressed" set of input scalars represented as a sorted
 * set of centroids (https://arxiv.org/pdf/1902.04023).
 * This data can be queried for quantile information. Each row in a tdigest
 * column represents an entire tdigest.
 *
 * The column has the following structure:
 *
 * struct {
 *   // centroids for the digest
 *   list {
 *    struct {
 *      double    // mean
 *      double    // weight
 *    }
 *   }
 *   // these are from the input stream, not the centroids. they are used
 *   // during the percentile_approx computation near the beginning or
 *   // end of the quantiles
 *   double       // min
 *   double       // max
 * }
 */
class tdigest_column_view : private column_view {
 public:
  tdigest_column_view(column_view const&);  ///< Construct tdigest_column_view from a column_view
  tdigest_column_view(tdigest_column_view&&)      = default;  ///< Move constructor
  tdigest_column_view(tdigest_column_view const&) = default;  ///< Copy constructor
  ~tdigest_column_view() override                 = default;
  /**
   * @brief Copy assignment operator
   *
   * @return this object after copying the contents of the other object (copy)
   */
  tdigest_column_view& operator=(tdigest_column_view const&) = default;
  /**
   * @brief Move assignment operator
   *
   * @return this object after moving the contents of the other object (transfer ownership)
   */
  tdigest_column_view& operator=(tdigest_column_view&&) = default;

  using column_view::size;
  using offset_iterator = size_type const*;  ///< Iterator over offsets

  // mean and weight column indices within tdigest inner struct columns
  static constexpr size_type mean_column_index{0};    ///< Mean column index
  static constexpr size_type weight_column_index{1};  ///< Weight column index

  // min and max column indices within tdigest outer struct columns
  static constexpr size_type centroid_column_index{0};  ///< Centroid column index
  static constexpr size_type min_column_index{1};       ///< Min column index
  static constexpr size_type max_column_index{2};       ///< Max column index

  /**
   * @brief Returns the parent column.
   *
   * @return The parent column
   */
  [[nodiscard]] column_view parent() const;

  /**
   * @brief Returns the column of centroids
   *
   * @return The list column of centroids
   */
  [[nodiscard]] lists_column_view centroids() const;

  /**
   * @brief Returns the internal column of mean values
   *
   * @return The internal column of mean values
   */
  [[nodiscard]] column_view means() const;

  /**
   * @brief Returns the internal column of weight values
   *
   * @return The internal column of weight values
   */
  [[nodiscard]] column_view weights() const;

  /**
   * @brief Returns the first min value for the column. Each row corresponds
   * to the minimum value for the accompanying digest.
   *
   * @return const pointer to the first min value for the column
   */
  [[nodiscard]] double const* min_begin() const;

  /**
   * @brief Returns the first max value for the column. Each row corresponds
   * to the maximum value for the accompanying digest.
   *
   * @return const pointer to the first max value for the column
   */
  [[nodiscard]] double const* max_begin() const;
};

/** @} */  // end of group
}  // namespace tdigest
}  // namespace CUDF_EXPORT cudf
