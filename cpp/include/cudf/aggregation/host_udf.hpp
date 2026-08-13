/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/aggregation.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

#include <functional>
#include <optional>

/**
 * @file host_udf.hpp
 * @brief Declare the base class for host-side user-defined function (`HOST_UDF`) and example of
 * subclass implementation.
 */

namespace CUDF_EXPORT cudf {
/**
 * @addtogroup aggregation_factories
 * @{
 */

/**
 * @brief The fundamental interface for host-based UDF implementation.
 *
 * This class declares the functions `do_hash`, `is_equal`, and `clone` that must be defined in
 * the users' UDF implementation. These functions are required for libcudf aggregation framework
 * to perform its operations.
 */
class host_udf_base {
  // Declare constructor private to prevent the users from deriving from this class.
 private:
  host_udf_base() = default;  ///< Default constructor

  // Only allow deriving from the structs below.
  friend struct reduce_host_udf;
  friend struct segmented_reduce_host_udf;
  friend struct groupby_host_udf;

 public:
  virtual ~host_udf_base() = default;  ///< Default destructor

  /**
   * @brief Computes hash value of the instance.
   *
   * Overriding this function is optional when the derived class has data members such that
   * each instance needs to be differentiated from each other.
   *
   * @return The hash value of the instance
   */
  [[nodiscard]] virtual std::size_t do_hash() const
  {
    return std::hash<int>{}(static_cast<int>(aggregation::Kind::HOST_UDF));
  }

  /**
   * @brief Compares two instances of the derived class for equality.
   * @param other The other instance to compare with
   * @return True if the two instances are equal
   */
  [[nodiscard]] virtual bool is_equal(host_udf_base const& other) const = 0;

  /**
   * @brief Clones the instance.
   *
   * The instances of the derived class should be lightweight for efficient cloning.
   *
   * @return A new instance cloned from this one
   */
  [[nodiscard]] virtual std::unique_ptr<host_udf_base> clone() const = 0;
};

/**
 * @brief The interface for host-based UDF implementation for reduction contexts.
 *
 * An implementation of host-based UDF for reduction needs to be derived from this class.
 * In addition to implementing the virtual functions declared in the base class `host_udf_base`,
 * such derived classes must also define the `operator()` function to perform reduction
 * operations.
 *
 * Example:
 * @code{.cpp}
 * struct my_udf_aggregation : cudf::reduce_host_udf {
 *   my_udf_aggregation() = default;
 *
 *   [[nodiscard]] std::unique_ptr<scalar> operator()(
 *     column_view const& input,
 *     data_type output_dtype,
 *     std::optional<std::reference_wrapper<scalar const>> init,
 *     rmm::cuda_stream_view stream,
 *     rmm::device_async_resource_ref mr) const override
 *   {
 *     // Perform reduction computation using the input data and return the reduction result.
 *     // This is where the actual reduction logic is implemented.
 *   }
 *
 *   [[nodiscard]] bool is_equal(host_udf_base const& other) const override
 *   {
 *     // Check if the other object is also instance of this class.
 *     // If there are internal state variables, they may need to be checked for equality as well.
 *     return dynamic_cast<my_udf_aggregation const*>(&other) != nullptr;
 *   }
 *
 *   [[nodiscard]] std::unique_ptr<host_udf_base> clone() const override
 *   {
 *     return std::make_unique<my_udf_aggregation>();
 *   }
 * };
 * @endcode
 */
struct reduce_host_udf : host_udf_base {
  /**
   * @brief Perform reduction operations.
   *
   * @param input The input column for reduction
   * @param output_dtype The data type for the final output scalar
   * @param init The initial value of the reduction
   * @param stream The CUDA stream to use for any kernel launches
   * @param mr Device memory resource to use for any allocations
   * @return The output result of the aggregation
   */
  [[nodiscard]] virtual std::unique_ptr<scalar> operator()(
    column_view const& input,
    data_type output_dtype,
    std::optional<std::reference_wrapper<scalar const>> init,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) const = 0;
};

/**
 * @brief The interface for host-based UDF implementation for segmented reduction context.
 *
 * An implementation of host-based UDF for segmented reduction needs to be derived from this class.
 * In addition to implementing the virtual functions declared in the base class `host_udf_base`,
 * such derived class must also define the `operator()` function to perform segmented reduction.
 *
 * Example:
 * @code{.cpp}
 * struct my_udf_aggregation : cudf::segmented_reduce_host_udf {
 *   my_udf_aggregation() = default;
 *
 *   [[nodiscard]] std::unique_ptr<column> operator()(
 *     column_view const& input,
 *     device_span<size_type const> offsets,
 *     data_type output_dtype,
 *     null_policy null_handling,
 *     std::optional<std::reference_wrapper<scalar const>> init,
 *     rmm::cuda_stream_view stream,
 *     rmm::device_async_resource_ref mr) const override
 *   {
 *     // Perform computation using the input data and return the result.
 *     // This is where the actual segmented reduction logic is implemented.
 *   }
 *
 *   [[nodiscard]] bool is_equal(host_udf_base const& other) const override
 *   {
 *     // Check if the other object is also instance of this class.
 *     // If there are internal state variables, they may need to be checked for equality as well.
 *     return dynamic_cast<my_udf_aggregation const*>(&other) != nullptr;
 *   }
 *
 *   [[nodiscard]] std::unique_ptr<host_udf_base> clone() const override
 *   {
 *     return std::make_unique<my_udf_aggregation>();
 *   }
 * };
 * @endcode
 */
struct segmented_reduce_host_udf : host_udf_base {
  /**
   * @brief Perform segmented reduction operations.
   *
   * @param input The input column for reduction
   * @param offsets A list of offsets defining the segments for reduction
   * @param output_dtype The data type for the final output column
   * @param null_handling If `INCLUDE` then the reduction result is valid only if all elements in
   *        the segment are valid, and if `EXCLUDE` then the reduction result is valid if any
   *        element in the segment is valid
   * @param init The initial value of the reduction
   * @param stream The CUDA stream to use for any kernel launches
   * @param mr Device memory resource to use for any allocations
   * @return The output result of the aggregation
   */
  [[nodiscard]] virtual std::unique_ptr<column> operator()(
    column_view const& input,
    device_span<size_type const> offsets,
    data_type output_dtype,
    null_policy null_handling,
    std::optional<std::reference_wrapper<scalar const>> init,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) const = 0;
};

// Forward declaration.
namespace groupby ::detail {
struct aggregate_result_functor;
}

/**
 * @brief The interface for host-based UDF implementation for groupby aggregation context.
 *
 * An implementation of host-based UDF for groupby needs to be derived from this class.
 * In addition to implementing the virtual functions declared in the base class `host_udf_base`,
 * such a derived class must also define the functions `get_empty_output()` to return result when
 * the input is empty, and ``operator()`` to perform its groupby operations.
 *
 * During execution, the derived class can access internal data provided by the libcudf groupby
 * framework through a set of ``get*`` accessors, as well as calling other built-in groupby
 * aggregations through the ``compute_aggregation`` function.
 *
 * @note The derived class can only perform sort-based groupby aggregations. Hash-based groupby
 * aggregations require more complex data structure and is not yet supported.
 *
 * Example:
 * @code{.cpp}
 * struct my_udf_aggregation : cudf::groupby_host_udf {
 *   my_udf_aggregation() = default;
 *
 *   [[nodiscard]] std::unique_ptr<column> get_empty_output(
 *     rmm::cuda_stream_view stream,
 *     rmm::device_async_resource_ref mr) const override
 *   {
 *     // Return a column corresponding to the result when the input values column is empty.
 *   }
 *
 *   [[nodiscard]] std::unique_ptr<column> operator()(
 *     rmm::cuda_stream_view stream,
 *     rmm::device_async_resource_ref mr) const override
 *   {
 *     // Perform UDF computation using the input data and return the result.
 *   }
 *
 *   [[nodiscard]] bool is_equal(host_udf_base const& other) const override
 *   {
 *     // Check if the other object is also instance of this class.
 *     // If there are internal state variables, they may need to be checked for equality as well.
 *     return dynamic_cast<my_udf_aggregation const*>(&other) != nullptr;
 *   }
 *
 *   [[nodiscard]] std::unique_ptr<host_udf_base> clone() const override
 *   {
 *     return std::make_unique<my_udf_aggregation>();
 *   }
 * };
 * @endcode
 */
struct groupby_host_udf : host_udf_base {
  /**
   * @brief Get the output when the input values column is empty.
   *
   * This is called in libcudf when the input values column is empty. In such situations libcudf
   * tries to generate the output directly without unnecessarily evaluating the intermediate data.
   *
   * @param stream The CUDA stream to use for any kernel launches
   * @param mr Device memory resource to use for any allocations
   * @return The output result of the aggregation when the input values column is empty
   */
  [[nodiscard]] virtual std::unique_ptr<column> get_empty_output(
    rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr) const = 0;

  /**
   * @brief Perform the main groupby computation for the host-based UDF.
   *
   * @param stream The CUDA stream to use for any kernel launches
   * @param mr Device memory resource to use for any allocations
   * @return The output result of the aggregation
   */
  [[nodiscard]] virtual std::unique_ptr<column> operator()(
    rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr) const = 0;

 private:
  // Allow the struct `aggregate_result_functor` to set its private callback variables.
  friend struct groupby::detail::aggregate_result_functor;

  /**
   * @brief Callback to access the input values column.
   */
  std::function<column_view(void)> callback_input_values;

  /**
   * @brief Callback to access the input values grouped according to the input keys for which the
   * values within each group maintain their original order.
   */
  std::function<column_view(void)> callback_grouped_values;

  /**
   * @brief Callback to access the input values grouped according to the input keys and sorted
   * within each group.
   */
  std::function<column_view(void)> callback_sorted_grouped_values;

  /**
   * @brief Callback to access the number of groups (i.e., number of distinct keys).
   */
  std::function<size_type(void)> callback_num_groups;

  /**
   * @brief Callback to access the offsets separating groups.
   */
  std::function<device_span<size_type const>(void)> callback_group_offsets;

  /**
   * @brief Callback to access the group labels (which is also the same as group indices).
   */
  std::function<device_span<size_type const>(void)> callback_group_labels;

  /**
   * @brief Callback to access the result from other groupby aggregations.
   */
  std::function<column_view(std::unique_ptr<aggregation>)> callback_compute_aggregation;

 protected:
  /**
   * @brief Access the input values column.
   *
   * @return The input values column.
   */
  [[nodiscard]] column_view get_input_values() const
  {
    CUDF_EXPECTS(callback_input_values, "Uninitialized callback_input_values.");
    return callback_input_values();
  }

  /**
   * @brief Access the input values grouped according to the input keys for which the values
   * within each group maintain their original order.
   *
   * @return The grouped values column.
   */
  [[nodiscard]] column_view get_grouped_values() const
  {
    CUDF_EXPECTS(callback_grouped_values, "Uninitialized callback_grouped_values.");
    return callback_grouped_values();
  }

  /**
   * @brief Access the input values grouped according to the input keys and sorted within each
   * group.
   *
   * @return The sorted grouped values column.
   */
  [[nodiscard]] column_view get_sorted_grouped_values() const
  {
    CUDF_EXPECTS(callback_sorted_grouped_values, "Uninitialized callback_sorted_grouped_values.");
    return callback_sorted_grouped_values();
  }

  /**
   * @brief Access the number of groups (i.e., number of distinct keys).
   *
   * @return The number of groups.
   */
  [[nodiscard]] size_type get_num_groups() const
  {
    CUDF_EXPECTS(callback_num_groups, "Uninitialized callback_num_groups.");
    return callback_num_groups();
  }

  /**
   * @brief Access the offsets separating groups.
   *
   * @return The array of group offsets.
   */
  [[nodiscard]] device_span<size_type const> get_group_offsets() const
  {
    CUDF_EXPECTS(callback_group_offsets, "Uninitialized callback_group_offsets.");
    return callback_group_offsets();
  }

  /**
   * @brief Access the group labels (which is also the same as group indices).
   *
   * @return The array of group labels.
   */
  [[nodiscard]] device_span<size_type const> get_group_labels() const
  {
    CUDF_EXPECTS(callback_group_labels, "Uninitialized callback_group_labels.");
    return callback_group_labels();
  }

  /**
   * @brief Compute a built-in groupby aggregation and access its result.
   *
   * This allows the derived class to call any other built-in groupby aggregations on the same input
   * values column and access the output for its operations.
   *
   * @param other_agg An arbitrary built-in groupby aggregation
   * @return A `column_view` object corresponding to the output result of the given aggregation
   */
  [[nodiscard]] column_view compute_aggregation(std::unique_ptr<aggregation> other_agg) const
  {
    CUDF_EXPECTS(callback_compute_aggregation, "Uninitialized callback for computing aggregation.");
    return callback_compute_aggregation(std::move(other_agg));
  }
};

/** @} */  // end of group
}  // namespace CUDF_EXPORT cudf
