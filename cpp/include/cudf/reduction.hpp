/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/aggregation.hpp>
#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/column/scalar_column_view.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/types.hpp>
#include <cudf/udf.hpp>
#include <cudf/user_data.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

/**
 * @file
 * @brief APIs for computing reductions and scans over columns.
 */

namespace CUDF_EXPORT cudf {
/**
 * @addtogroup aggregation_reduction
 * @{
 */

/**
 *  @brief Enum to describe scan operation type
 */
enum class scan_type : bool { INCLUSIVE, EXCLUSIVE };

// clang-format off
/**
 * @brief  Computes the reduction of the values in all rows of a column.
 *
 * This function does not detect overflows in reductions except for the `SUM_OVERFLOW`
 * aggregation. When `output_type` does not match the `col.type()`, their values may be promoted to
 * `int64_t` or `double` for computing aggregations and then cast to `output_type` before returning.
 *
 * The `SUM_OVERFLOW` aggregation is a special case that detects integer
 * overflow during summation of signed integer or decimal values and returns a struct
 * containing both the sum result and an overflow flag. On overflow the sum value
 * is unspecified; the boolean flag is the source of truth.
 *
 * Only `min` and `max` ops are supported for reduction of non-arithmetic
 * types (e.g. timestamp or string).
 *
 * Any null values are skipped for the operation.
 * If the reduction fails, the output scalar returns with `%is_valid()==false`.
 *
 * For empty or all-null input, the result is generally an invalid scalar except for specific
 * aggregations where the aggregation has a well-defined output.
 *
 * If the input column is an arithmetic type, the `output_type` can be any arithmetic
 * type. If the input column is a non-arithmetic type (e.g. timestamp or string)
 * the `output_type` must match the `col.type()`. If the reduction type is `any` or
 * `all`, the `output_type` must be type BOOL8.
 *
 * | Aggregation | Output Type | Init Value | Empty Input | Comments |
 * | :---------: | ----------- | :--------: | ----------- | -------- |
 * | SUM/PRODUCT | output_type | yes | NA | Input accumulated into output_type variable |
 * | SUM_OVERFLOW | STRUCT{col.type,BOOL8} | yes | {null,false} | {sum, overflow_flag}, input must be signed integer or decimal |
 * | SUM_OF_SQUARES | output_type | no | NA | Input accumulated into output_type variable |
 * | MIN/MAX | col.type | yes | NA | Supports arithmetic, timestamp, duration, string types only |
 * | ANY/ALL | BOOL8 | yes | True for ALL only | Checks for non-zero elements |
 * | MEAN/VARIANCE/STD | FLOAT32/FLOAT64 | no | NA | output_type must be a float type |
 * | MEDIAN/QUANTILE | output_type | no | NA | Exact value if output_type is FLOAT64. See @ref cudf::quantile |
 * | NUNIQUE | output_type | no | 1 if all-nulls | May process null rows |
 * | NTH_ELEMENT | col.type | no | NA |  |
 * | BITWISE_AGG | col.type | no | NA | Supports only integral types |
 * | HISTOGRAM/MERGE_HISTOGRAM | LIST of col.type | no | empty list returned |  |
 * | COLLECT_LIST/COLLECT_SET | LIST of col.type | no | empty list returned |  |
 * | TDIGEST/MERGE_TDIGEST | STRUCT | no | empty struct returned | tdigest scalar is returned |
 * | HOST_UDF | output_type | yes | NA | Custom UDF could ignore output_type |
 *
 * The NA in the table indicates an output scalar with `%is_valid()==false`
 *
 * @throw std::invalid_argument if reduction is called for non-arithmetic output
 * type and operator other than `min` and `max`.
 * @throw std::invalid_argument if input column data type is not convertible to `output_type`.
 * @throw std::invalid_argument if `min` or `max` reduction is called and the
 * output type does not match the input column data type.
 * @throw std::invalid_argument if `any` or `all` reduction is called and the output type is not BOOL8.
 * @throw std::invalid_argument if `mean`, `var`, or `std` reduction is called and
 * the `output_type` is not floating point.
 * @throw std::invalid_argument if `sum_overflow` reduction is called and the
 * input column type is not a signed integer or decimal, or the `output_type` is not `STRUCT`.
 *
 * @param col Input column view
 * @param agg Aggregation operator applied by the reduction
 * @param output_type The output scalar type
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned scalar's device memory
 * @returns Output scalar with reduce result
 */
std::unique_ptr<scalar> reduce(
  column_view const& col,
  reduce_aggregation const& agg,
  data_type output_type,
  cuda::stream_ref stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());
// clang-format on

/**
 * @brief  Computes the reduction of the values in all rows of a column with an initial value
 *
 * Only `sum`, `product`, `min`, `max`, `any`, `all`, and `sum_overflow` reductions are
 * supported. For `sum_overflow`, the initial value is added to the sum and overflow
 * detection is performed throughout the entire computation.
 *
 * @see cudf::reduce(column_view const&,reduce_aggregation
 * const&,data_type,cuda::stream_ref,rmm::device_async_resource_ref) for more details
 *
 * @throw std::invalid_argument if reduction is not `sum`, `product`, `min`, `max`, `any`, `all`,
 * or `sum_overflow` and `init` is specified.
 *
 * @param col Input column view
 * @param agg Aggregation operator applied by the reduction
 * @param output_type The output scalar type
 * @param init The initial value of the reduction
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned scalar's device memory
 * @returns Output scalar with reduce result
 */
std::unique_ptr<scalar> reduce(
  column_view const& col,
  reduce_aggregation const& agg,
  data_type output_type,
  std::optional<std::reference_wrapper<scalar const>> init,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief  Compute reduction of each segment in the input column
 *
 * This function does not detect overflows in reductions. When `output_type`
 * does not match the `segmented_values.type()`, their values may be promoted to
 * `int64_t` or `double` for computing aggregations and then cast to
 * `output_type` before returning.
 *
 * Null values are treated as identities during reduction.
 *
 * If the segment is empty, the row corresponding to the result of the
 * segment is null.
 *
 * If any index in `offsets` is out of bound of `segmented_values`, the behavior
 * is undefined.
 *
 * If the input column has arithmetic type, `output_type` can be any arithmetic
 * type. If the input column has non-arithmetic type, e.g. timestamp, the same
 * output type must be specified.
 *
 * If input is not empty, the result is always nullable.
 *
 * @throw cudf::logic_error if reduction is called for non-arithmetic output
 * type and operator other than `min` and `max`.
 * @throw cudf::logic_error if input column data type is not convertible to
 * `output_type` type.
 * @throw cudf::logic_error if `min` or `max` reduction is called and the
 * `output_type` does not match the input column data type.
 * @throw cudf::logic_error if `any` or `all` reduction is called and the
 * `output_type` is not BOOL8.
 *
 * @param segmented_values Column view of segmented inputs
 * @param offsets Each segment's offset of `segmented_values`. A list of offsets with size
 * `num_segments + 1`. The size of `i`th segment is `offsets[i+1] - offsets[i]`.
 * @param agg Aggregation operator applied by the reduction
 * @param output_type  The output column type
 * @param null_handling If `INCLUDE`, the reduction is valid if all elements in a segment are valid,
 * otherwise null. If `EXCLUDE`, the reduction is valid if any element in the segment is valid,
 * otherwise null.
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned scalar's device memory
 * @returns Output column with results of segmented reduction
 */
std::unique_ptr<column> segmented_reduce(
  column_view const& segmented_values,
  device_span<size_type const> offsets,
  segmented_reduce_aggregation const& agg,
  data_type output_type,
  null_policy null_handling,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief  Compute reduction of each segment in the input column with an initial value. Only SUM,
 * PRODUCT, MIN, MAX, ANY, and ALL aggregations are supported.
 *
 * @param segmented_values Column view of segmented inputs
 * @param offsets Each segment's offset of `segmented_values`. A list of offsets with size
 * `num_segments + 1`. The size of `i`th segment is `offsets[i+1] - offsets[i]`.
 * @param agg Aggregation operator applied by the reduction
 * @param output_type  The output column type
 * @param null_handling If `INCLUDE`, the reduction is valid if all elements in a segment are valid,
 * otherwise null. If `EXCLUDE`, the reduction is valid if any element in the segment is valid,
 * otherwise null.
 * @param init The initial value of the reduction
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned scalar's device memory
 * @returns Output column with results of segmented reduction.
 */
std::unique_ptr<column> segmented_reduce(
  column_view const& segmented_values,
  device_span<size_type const> offsets,
  segmented_reduce_aggregation const& agg,
  data_type output_type,
  null_policy null_handling,
  std::optional<std::reference_wrapper<scalar const>> init,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief  Computes the scan of a column.
 *
 * The null values are skipped for the operation, and if an input element
 * at `i` is null, then the output element at `i` will also be null.
 *
 * @throws cudf::logic_error if column datatype is not numeric type.
 *
 * @param[in] input The input column view for the scan
 * @param[in] agg unique_ptr to aggregation operator applied by the scan
 * @param[in] inclusive The flag for applying an inclusive scan if scan_type::INCLUSIVE, an
 * exclusive scan if scan_type::EXCLUSIVE.
 * @param[in] null_handling Exclude null values when computing the result if null_policy::EXCLUDE.
 * Include nulls if null_policy::INCLUDE. Any operation with a null results in a null.
 * @param[in] stream CUDA stream used for device memory operations and kernel launches
 * @param[in] mr Device memory resource used to allocate the returned scalar's device memory
 * @returns Scanned output column
 */
std::unique_ptr<column> scan(
  column_view const& input,
  scan_aggregation const& agg,
  scan_type inclusive,
  null_policy null_handling         = null_policy::EXCLUDE,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Determines the minimum and maximum values of a column.
 *
 *
 * @param col column to compute minmax
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return A std::pair of scalars with the first scalar being the minimum value and the second
 * scalar being the maximum value of the input column.
 */
std::pair<std::unique_ptr<scalar>, std::unique_ptr<scalar>> minmax(
  column_view const& col,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Reduction namespace
 */
namespace reduction {
/**
 * @brief Indicate if a reduction is supported for a source datatype.
 *
 * @param source The source data type.
 * @param kind The reduction aggregation.
 * @returns true if the reduction is supported.
 */
bool is_valid_aggregation(data_type source, aggregation::Kind kind);
}  // namespace reduction

/**
 * @brief Non-owning columns, broadcast scalars and opaque device pointers.
 *
 * Keep referenced device storage alive until stream work completes.
 */
using reduce_input = std::variant<column_view, scalar_column_view, user_data>;

/**
 * @brief Compiled callbacks defining a reduction; each returns @c cudf::errc.
 *
 * Parameters are ordered as follows; each ellipsis expands fields in specification order.
 *
 * | Callback | Parameters |
 * |----------|------------|
 * | init | state pointers... |
 * | filter | row index, bool* selected, input values... |
 * | update | row index, state pointers..., input values... |
 * | merge | destination state pointers..., source state values... |
 * | finalize | output pointers..., state values... |
 *
 * Initialize every independent state to its identity. Merge must be associative and preserve
 * that identity; update/merge order is unspecified. An empty filter symbol selects all rows.
 * A false selection or filter error skips update. Row indices refer to the input views.
 * Every segmented callback also receives a leading zero-based segment ID, including for empty
 * segments. Initialization and merging can each run multiple times for the same segment.
 *
 * Inputs follow transform's decoded types, including dictionary keys and @c cudf::string_view.
 * State and outputs support fixed-width types and strings. Keep referenced string characters
 * alive until stream work completes; returned string columns own a copy.
 *
 * With @c null_aware::YES, inputs, state and outputs use @c cuda::std::optional<T>, with mutable
 * parameters passed as pointers to optionals. Empty optionals represent nulls. Callbacks define
 * null-state identities and output validity. With @c null_aware::NO, use plain values; rows with
 * any null column/scalar input are skipped before filtering. Indices, the filter's bool pointer
 * and @c void* user-data inputs are never optional. Filter and update share one set of row loads.
 *
 * Fragment bytes and symbol names are non-owning and must outlive the call. See @ref lto_fragment.
 */
struct reduce_udf {
  lto_fragments fragments{};    ///< Device-code fragments
  std::string_view init{};      ///< Initializes independent state
  std::string_view update{};    ///< Incorporates a selected row
  std::string_view merge{};     ///< Combines partial states
  std::string_view finalize{};  ///< Writes output values
  std::string_view filter{};    ///< Optional row predicate
};

/**
 * @brief Reduces rows into one single-row column per requested output, in specification order.
 *
 * Column lengths must match; scalars and user data broadcast. With no column input there are
 * zero rows. Empty inputs and selections finalize the identity state; callbacks determine
 * values and validity. Input/state/output types and callback signatures follow @ref reduce_udf.
 *
 * @throws std::invalid_argument for mismatched input lengths or unsupported state/output types
 * @throws cudf::evaluation_error for callback failures
 * @param udf Compiled fragments and callback symbols
 * @param is_null_aware Whether values use optional wrappers
 * @param inputs Inputs in callback order
 * @param state_types State fields in callback order
 * @param outputs Returned types in finalization order
 * @param stream Execution stream
 * @param mr Output and temporary memory resources
 * @return One single-row column per output
 */
std::vector<std::unique_ptr<column>> reduce_lto(
  reduce_udf const& udf,
  null_aware is_null_aware,
  std::span<reduce_input const> inputs,
  std::span<data_type const> state_types,
  std::span<data_type const> outputs,
  cuda::stream_ref stream   = cudf::get_default_stream(),
  cudf::memory_resources mr = cudf::get_current_device_resource_ref());

/**
 * @brief Reduces each segment, returning independent single-row columns in segment/output order.
 *
 * Segment i covers [offsets[i], offsets[i+1]); equal offsets describe an empty segment.
 * Offsets must contain at least one entry, be nonnegative/nondecreasing and stay within input
 * bounds. The first offset may be nonzero and the last need not cover every input row.
 *
 * Results use result[segment * outputs.size() + output]. Zero segments return an empty vector;
 * empty or fully filtered segments finalize the identity. Other contracts follow @ref reduce_lto,
 * with a leading segment ID on every callback. With no column inputs, offsets define row bounds.
 *
 * @throws std::invalid_argument for mismatched lengths, unsupported state/output types,
 * empty/oversized offsets, or a negative maximum-length hint
 * @throws cudf::evaluation_error for callback failures
 * @param udf Compiled fragments and callback symbols
 * @param is_null_aware Whether values use optional wrappers
 * @param inputs Inputs in callback order
 * @param offsets Device segment boundaries, with num_segments + 1 entries
 * @param state_types State fields in callback order
 * @param outputs Returned types in finalization order
 * @param max_segment_size Upper bound on every segment length, or std::nullopt when unknown
 * @param stream Execution stream
 * @param mr Output and temporary memory resources
 * @return One single-row column per segment/output pair
 */
std::vector<std::unique_ptr<column>> segmented_reduce_lto(
  reduce_udf const& udf,
  null_aware is_null_aware,
  std::span<reduce_input const> inputs,
  device_span<size_type const> offsets,
  std::span<data_type const> state_types,
  std::span<data_type const> outputs,
  std::optional<size_type> max_segment_size,
  cuda::stream_ref stream   = cudf::get_default_stream(),
  cudf::memory_resources mr = cudf::get_current_device_resource_ref());

/** @} */  // end of group

}  // namespace CUDF_EXPORT cudf
