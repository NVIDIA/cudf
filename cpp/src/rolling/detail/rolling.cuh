/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "lead_lag_nested.cuh"
#include "nth_element.cuh"
#include "reductions/nested_types_extrema_utils.cuh"
#include "rolling.hpp"
#include "rolling_collect_list.cuh"
#include "rolling_operators.cuh"

#include <cudf/aggregation.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/detail/aggregation/aggregation.cuh>
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/detail/device_scalar.hpp>
#include <cudf/detail/gather.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/unary.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/dictionary/dictionary_factories.hpp>
#include <cudf/lists/detail/stream_compaction.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <memory>

namespace cudf {

namespace detail {

static std::unique_ptr<column> empty_output_for_rolling_aggregation(column_view const& input,
                                                                    rolling_aggregation const& agg)
{
  // TODO:
  //  Ideally, for UDF aggregations, the returned column would match
  //  the agg's return type. It currently returns empty_like(input), because:
  //    1. This preserves prior behavior for empty input columns.
  //    2. There is insufficient information to construct nested return columns.
  //       `cudf::make_udf_aggregation()` expresses the return type as a `data_type`
  //        which cannot express recursively nested types (e.g. `STRUCT<LIST<INT32>>`.)
  //    3. In any case, UDFs that return nested types are not currently supported.
  //  Constructing a more accurate return type for UDFs will be taken up
  //  at a later date.
  return agg.kind == aggregation::CUDA || agg.kind == aggregation::PTX
           ? empty_like(input)
           : cudf::detail::dispatch_type_and_aggregation(
               input.type(), agg.kind, agg_specific_empty_output{}, input, agg);
}

/**
 * @brief Rolling window specific functor for preprocessing aggregations.
 *
 * The purpose of this functor is to preprocess incoming aggregation/type pairs and
 * potentially transform them into other aggregation/type pairs. Typically when this
 * happens, rolling_postprocessor will perform some postprocessing step.
 *
 * An example of this would be applying a MIN aggregation to strings. This cannot be done
 * directly in the rolling operation, so instead the following happens:
 *
 * - The rolling_preprocessor transforms the incoming MIN/string pair to
 *   an ARGMIN/int pair.
 * - The ARGMIN/int has the rolling operation applied to it, generating a list of indices
 *   that can then be used as a gather map.
 * - rolling_postprocessor then takes this gather map and performs a final
 *   gather() on the input string data to generate the final output.
 *
 * Another example is COLLECT_LIST. COLLECT_LIST is odd in that it doesn't go through the
 * normal gpu rolling kernel at all. It has a completely custom implementation. So the
 * following happens:
 *
 * - The rolling_preprocessor transforms the COLLECT_LIST aggregation into nothing,
 *   since no actual rolling window operation will be performed.
 * - rolling_postprocessor calls the specialized rolling_collect_list()
 *   function to generate the final output.
 *
 * This functor has a templated operator() that is specialized for aggregation types
 * that require special preprocessing for rolling window operations.
 */
struct rolling_preprocessor {
  // Default case: return clone of the aggregation
  template <aggregation::Kind k>
  std::vector<std::unique_ptr<aggregation>> operator()(data_type, aggregation const& agg) const
  {
    std::vector<std::unique_ptr<aggregation>> aggs;
    aggs.push_back(agg.clone());
    return aggs;
  }
};

// Helper for MIN/MAX preprocessing - strings/structs use ARG variants
template <typename MakeArgAggFn, typename MakeDirectAggFn>
inline std::vector<std::unique_ptr<aggregation>> preprocess_minmax(data_type col_type,
                                                                   MakeArgAggFn make_arg_agg,
                                                                   MakeDirectAggFn make_direct_agg)
{
  std::vector<std::unique_ptr<aggregation>> aggs;
  aggs.push_back(cudf::is_compound(col_type) ? make_arg_agg() : make_direct_agg());
  return aggs;
}

// MIN/MAX aggregations with strings/structs are processed in 2 passes. The first pass performs
// the rolling operation on a ARG(MIN/MAX) aggregation to generate indices instead of values.
// Then a second pass uses those indices to gather the final strings/structs.
template <>
inline std::vector<std::unique_ptr<aggregation>> rolling_preprocessor::operator()<aggregation::MIN>(
  data_type col_type, aggregation const&) const
{
  return preprocess_minmax(col_type, make_argmin_aggregation<>, make_min_aggregation<>);
}

template <>
inline std::vector<std::unique_ptr<aggregation>> rolling_preprocessor::operator()<aggregation::MAX>(
  data_type col_type, aggregation const&) const
{
  return preprocess_minmax(col_type, make_argmax_aggregation<>, make_max_aggregation<>);
}

// COLLECT_LIST, COLLECT_SET, and NTH_ELEMENT aggregations do not perform a rolling operation.
// They get processed entirely in the postprocessing step.
template <>
inline std::vector<std::unique_ptr<aggregation>>
rolling_preprocessor::operator()<aggregation::COLLECT_LIST>(data_type, aggregation const&) const
{
  return {};
}

template <>
inline std::vector<std::unique_ptr<aggregation>>
rolling_preprocessor::operator()<aggregation::COLLECT_SET>(data_type, aggregation const&) const
{
  return {};
}

// Specialization for STD aggregation
// STD aggregations depend on VARIANCE aggregation. Each element is applied
// with square-root in the postprocessing step.
template <>
inline std::vector<std::unique_ptr<aggregation>> rolling_preprocessor::operator()<aggregation::STD>(
  data_type, aggregation const& agg) const
{
  // Dynamic cast needed due to virtual inheritance
  auto const& std_agg = dynamic_cast<cudf::detail::std_aggregation const&>(agg);
  std::vector<std::unique_ptr<aggregation>> aggs;
  aggs.push_back(make_variance_aggregation(std_agg._ddof));
  return aggs;
}

// LEAD/LAG have custom behaviors for non fixed-width types - no rolling op, just postprocess.
template <>
inline std::vector<std::unique_ptr<aggregation>>
rolling_preprocessor::operator()<aggregation::LEAD>(data_type col_type,
                                                    aggregation const& agg) const
{
  if (!cudf::is_fixed_width(col_type) && !cudf::is_dictionary(col_type)) { return {}; }
  std::vector<std::unique_ptr<aggregation>> aggs;
  aggs.push_back(agg.clone());
  return aggs;
}

template <>
inline std::vector<std::unique_ptr<aggregation>> rolling_preprocessor::operator()<aggregation::LAG>(
  data_type col_type, aggregation const& agg) const
{
  if (!cudf::is_fixed_width(col_type) && !cudf::is_dictionary(col_type)) { return {}; }
  std::vector<std::unique_ptr<aggregation>> aggs;
  aggs.push_back(agg.clone());
  return aggs;
}

template <>
inline std::vector<std::unique_ptr<aggregation>>
rolling_preprocessor::operator()<aggregation::NTH_ELEMENT>(data_type, aggregation const&) const
{
  return {};
}

/**
 * @brief Functor for rolling window postprocessing.
 *
 * This functor holds all parameters needed for postprocessing and provides
 * constrained operator() overloads for different aggregation kinds.
 * The default case passes through the intermediate result unchanged.
 *
 * @tparam PrecedingWindowIterator Type of iterator for preceding window bounds
 * @tparam FollowingWindowIterator Type of iterator for following window bounds
 */
template <typename PrecedingWindowIterator, typename FollowingWindowIterator>
struct rolling_postprocessor {
  column_view const& input;
  column_view const& default_outputs;
  data_type result_type;
  PrecedingWindowIterator preceding_window_begin;
  FollowingWindowIterator following_window_begin;
  int min_periods;
  rmm::cuda_stream_view stream;
  rmm::device_async_resource_ref mr;

  // Default case: pass through the intermediate result unchanged
  template <aggregation::Kind k>
  std::unique_ptr<column> operator()(aggregation const&,
                                     std::unique_ptr<column>& intermediate) const
  {
    return std::move(intermediate);
  }

  template <aggregation::Kind k>
    requires(k == aggregation::MIN || k == aggregation::MAX)
  std::unique_ptr<column> operator()(aggregation const&,
                                     std::unique_ptr<column>& intermediate) const
  {
    if (cudf::is_compound(result_type)) {
      // The rows that represent null elements will have negative values in gather map,
      // and that's why nullify_out_of_bounds/ignore_out_of_bounds is true.
      auto output_table = detail::gather(table_view{{input}},
                                         intermediate->view(),
                                         cudf::out_of_bounds_policy::NULLIFY,
                                         negative_index_policy::NOT_ALLOWED,
                                         stream,
                                         mr);
      return std::make_unique<cudf::column>(std::move(output_table->get_column(0)));
    } else {
      return std::move(intermediate);
    }
  }

  template <aggregation::Kind k>
    requires(k == aggregation::COLLECT_LIST)
  std::unique_ptr<column> operator()(aggregation const& agg, std::unique_ptr<column>&) const
  {
    auto const& collect_agg = dynamic_cast<cudf::detail::collect_list_aggregation const&>(agg);
    return rolling_collect_list(input,
                                default_outputs,
                                preceding_window_begin,
                                following_window_begin,
                                min_periods,
                                collect_agg._null_handling,
                                stream,
                                mr);
  }

  template <aggregation::Kind k>
    requires(k == aggregation::COLLECT_SET)
  std::unique_ptr<column> operator()(aggregation const& agg, std::unique_ptr<column>&) const
  {
    auto const& collect_agg   = dynamic_cast<cudf::detail::collect_set_aggregation const&>(agg);
    auto const collected_list = rolling_collect_list(input,
                                                     default_outputs,
                                                     preceding_window_begin,
                                                     following_window_begin,
                                                     min_periods,
                                                     collect_agg._null_handling,
                                                     stream,
                                                     cudf::get_current_device_resource_ref());

    return lists::detail::distinct(lists_column_view{collected_list->view()},
                                   collect_agg._nulls_equal,
                                   collect_agg._nans_equal,
                                   duplicate_keep_option::KEEP_ANY,
                                   stream,
                                   mr);
  }

  template <aggregation::Kind k>
    requires(k == aggregation::STD)
  std::unique_ptr<column> operator()(aggregation const&,
                                     std::unique_ptr<column>& intermediate) const
  {
    return detail::unary_operation(intermediate->view(), unary_operator::SQRT, stream, mr);
  }

  template <aggregation::Kind k>
    requires(k == aggregation::LEAD || k == aggregation::LAG)
  std::unique_ptr<column> operator()(aggregation const& agg,
                                     std::unique_ptr<column>& intermediate) const
  {
    auto const& lead_lag_agg = dynamic_cast<cudf::detail::lead_lag_aggregation const&>(agg);
    if (cudf::is_dictionary(result_type)) {
      // 'intermediate' is the column of indices for the output dictionary
      auto keys =  // copy the keys
        std::make_unique<cudf::column>(cudf::dictionary_column_view(input).keys(), stream, mr);
      return cudf::make_dictionary_column(std::move(keys), std::move(intermediate), stream, mr);
    }
    if (cudf::is_compound(result_type)) {
      return cudf::detail::compute_lead_lag_for_nested<PrecedingWindowIterator,
                                                       FollowingWindowIterator>(
        lead_lag_agg.kind,
        input,
        default_outputs,
        preceding_window_begin,
        following_window_begin,
        lead_lag_agg.row_offset,
        stream,
        mr);
    }
    return std::move(intermediate);
  }

  template <aggregation::Kind k>
    requires(k == aggregation::NTH_ELEMENT)
  std::unique_ptr<column> operator()(aggregation const& agg, std::unique_ptr<column>&) const
  {
    auto const& nth_agg = dynamic_cast<cudf::detail::nth_element_aggregation const&>(agg);
    return nth_agg._null_handling == null_policy::EXCLUDE
             ? rolling::nth_element<null_policy::EXCLUDE>(nth_agg._n,
                                                          input,
                                                          preceding_window_begin,
                                                          following_window_begin,
                                                          min_periods,
                                                          stream,
                                                          mr)
             : rolling::nth_element<null_policy::INCLUDE>(nth_agg._n,
                                                          input,
                                                          preceding_window_begin,
                                                          following_window_begin,
                                                          min_periods,
                                                          stream,
                                                          mr);
  }
};

/**
 * @brief Computes the rolling window function
 *
 * @tparam OutputType Datatype of `output`
 * @tparam block_size CUDA block size for the kernel
 * @tparam has_nulls true if the input column has nulls
 * @tparam DeviceRollingOperator An operator that performs a single windowing operation
 * @tparam PrecedingWindowIterator iterator type (inferred)
 * @tparam FollowingWindowIterator iterator type (inferred)
 * @param[in] input Input column device view
 * @param[in] default_outputs A column of per-row default values to be returned instead
 *            of nulls for certain aggregation types.
 * @param[out] output Output column device view
 * @param[out] output_valid_count Output count of valid values
 * @param[in] device_operator The operator used to perform a single window operation
 * @param[in] preceding_window_begin Rolling window size iterator, accumulates from
 *            in_col[i-preceding_window + 1] to in_col[i] inclusive
 * @param[in] following_window_begin Rolling window size iterator in the forward
 *            direction, accumulates from in_col[i] to in_col[i+following_window] inclusive
 */
template <typename OutputType,
          int block_size,
          typename DeviceRollingOperator,
          typename PrecedingWindowIterator,
          typename FollowingWindowIterator>
__launch_bounds__(block_size) CUDF_KERNEL
  void gpu_rolling(column_device_view input,
                   bool has_nulls,
                   column_device_view default_outputs,
                   mutable_column_device_view output,
                   size_type* __restrict__ output_valid_count,
                   DeviceRollingOperator device_operator,
                   PrecedingWindowIterator preceding_window_begin,
                   FollowingWindowIterator following_window_begin)
{
  thread_index_type i            = blockIdx.x * block_size + threadIdx.x;
  thread_index_type const stride = block_size * gridDim.x;

  size_type warp_valid_count{0};

  auto const num_rows = input.size();
  auto active_threads = __ballot_sync(0xffff'ffffu, i < num_rows);
  while (i < num_rows) {
    // The caller is required to provide window bounds that will
    // result in indexing that is in-bounds for the column. Therefore all
    // of these calculations cannot overflow and we can do everything
    // in size_type arithmetic. Moreover, we require that start <=
    // end, i.e., the window is never "reversed" though it may be empty.
    auto const preceding_window = preceding_window_begin[i];
    auto const following_window = following_window_begin[i];

    size_type const start = i - preceding_window + 1;
    size_type const end   = i + following_window + 1;

    // aggregate
    // TODO: We should explore using shared memory to avoid redundant loads.
    //       This might require separating the kernel into a special version
    //       for dynamic and static sizes.

    bool const output_is_valid = device_operator.template operator()<OutputType>(
      input, has_nulls, default_outputs, output, start, end, i);

    // set the mask
    cudf::bitmask_type const result_mask{__ballot_sync(active_threads, output_is_valid)};

    // only one thread writes the mask
    if (0 == threadIdx.x % cudf::detail::warp_size) {
      output.set_mask_word(cudf::word_index(i), result_mask);
      warp_valid_count += __popc(result_mask);
    }

    // process next element
    i += stride;
    active_threads = __ballot_sync(active_threads, i < num_rows);
  }

  // sum the valid counts across the whole block
  size_type block_valid_count =
    cudf::detail::single_lane_block_sum_reduce<block_size, 0>(warp_valid_count);

  if (threadIdx.x == 0) { atomicAdd(output_valid_count, block_valid_count); }
}

/**
 * @brief Type/aggregation dispatched functor for launching the gpu rolling window
 *        kernel.
 */
template <typename InputType>
struct rolling_window_launcher {
  template <aggregation::Kind op,
            typename PrecedingWindowIterator,
            typename FollowingWindowIterator>
  std::unique_ptr<column> operator()(column_view const& input,
                                     column_view const& default_outputs,
                                     PrecedingWindowIterator preceding_window_begin,
                                     FollowingWindowIterator following_window_begin,
                                     int min_periods,
                                     [[maybe_unused]] rolling_aggregation const& agg,
                                     rmm::cuda_stream_view stream,
                                     rmm::device_async_resource_ref mr)
    requires(corresponding_rolling_operator<InputType, op>::type::is_supported())
  {
    auto const do_rolling = [&](auto const& device_op) {
      // indices are used with dictionary aggregations; the element method always returns size_type
      auto out_type = cudf::is_dictionary(input.type()) ? data_type{type_to_id<size_type>()}
                                                        : target_type(input.type(), op);
      using OutType =
        typename std::conditional_t<cudf::is_dictionary<InputType>(),
                                    size_type,
                                    device_storage_type_t<target_type_t<InputType, op>>>;

      auto output =
        make_fixed_width_column(out_type, input.size(), mask_state::UNINITIALIZED, stream, mr);

      auto const d_inp_ptr         = column_device_view::create(input, stream);
      auto const d_default_out_ptr = column_device_view::create(default_outputs, stream);
      auto const d_out_ptr = mutable_column_device_view::create(output->mutable_view(), stream);
      auto d_valid_count =
        cudf::detail::device_scalar<size_type>{0, stream, cudf::get_current_device_resource_ref()};

      auto constexpr block_size = 256;
      auto const grid           = cudf::detail::grid_1d(input.size(), block_size);

      gpu_rolling<OutType, block_size>
        <<<grid.num_blocks, block_size, 0, stream.value()>>>(*d_inp_ptr,
                                                             input.has_nulls(),
                                                             *d_default_out_ptr,
                                                             *d_out_ptr,
                                                             d_valid_count.data(),
                                                             device_op,
                                                             preceding_window_begin,
                                                             following_window_begin);
      CUDF_CUDA_TRY(cudaGetLastError());

      auto const valid_count = d_valid_count.value(stream);
      output->set_null_count(output->size() - valid_count);

      return output;
    };  // end do_rolling

    auto constexpr is_arg_minmax =
      op == aggregation::Kind::ARGMIN || op == aggregation::Kind::ARGMAX;

    if constexpr (is_arg_minmax && std::is_same_v<InputType, cudf::struct_view>) {
      // Using comp_generator to create a LESS operator for finding ARGMIN/ARGMAX of structs.
      auto const comp_generator =
        cudf::reduction::detail::arg_minmax_binop_generator::create<op>(input, stream);
      auto const device_op =
        create_rolling_operator<InputType, op>{}(min_periods, comp_generator.binop());
      return do_rolling(device_op);
    } else {  // all the remaining rolling operations
      auto const device_op = create_rolling_operator<InputType, op>{}(min_periods, agg);
      return do_rolling(device_op);
    }
  }

  template <aggregation::Kind op,
            typename PrecedingWindowIterator,
            typename FollowingWindowIterator>
  std::unique_ptr<column> operator()(column_view const&,
                                     column_view const&,
                                     PrecedingWindowIterator,
                                     FollowingWindowIterator,
                                     int,
                                     rolling_aggregation const&,
                                     rmm::cuda_stream_view,
                                     rmm::device_async_resource_ref)
    requires(!corresponding_rolling_operator<InputType, op>::type::is_supported())
  {
    CUDF_FAIL("Invalid aggregation type/pair");
  }
};

/**
 * @brief Functor for performing the high level rolling logic.
 *
 * This does 3 basic things:
 *
 * - It calls the preprocess step on incoming aggregation/type pairs
 * - It calls the aggregation-dispatched gpu-rolling operation
 * - It calls the final postprocess step
 */
struct dispatch_rolling {
  template <typename InputType, typename PrecedingWindowIterator, typename FollowingWindowIterator>
  std::unique_ptr<column> operator()(column_view const& input,
                                     column_view const& default_outputs,
                                     PrecedingWindowIterator preceding_window_begin,
                                     FollowingWindowIterator following_window_begin,
                                     size_type min_periods,
                                     rolling_aggregation const& agg,
                                     rmm::cuda_stream_view stream,
                                     rmm::device_async_resource_ref mr)
  {
    // do any preprocessing of aggregations (eg, MIN -> ARGMIN, COLLECT_LIST -> nothing)
    auto preprocessed_aggs =
      cudf::detail::aggregation_dispatcher(agg.kind, rolling_preprocessor{}, input.type(), agg);
    CUDF_EXPECTS(preprocessed_aggs.size() <= 1,
                 "Encountered a non-trivial rolling aggregation result");

    // perform the rolling window if we produced an aggregation to use
    auto intermediate = preprocessed_aggs.size() > 0
                          ? aggregation_dispatcher(
                              dynamic_cast<rolling_aggregation const&>(*preprocessed_aggs[0]).kind,
                              rolling_window_launcher<InputType>{},
                              input,
                              default_outputs,
                              preceding_window_begin,
                              following_window_begin,
                              min_periods,
                              dynamic_cast<rolling_aggregation const&>(*preprocessed_aggs[0]),
                              stream,
                              mr)
                          : nullptr;

    // finalize.
    auto const result_type   = target_type(input.type(), agg.kind);
    auto const postprocessor = rolling_postprocessor{input,
                                                     default_outputs,
                                                     result_type,
                                                     preceding_window_begin,
                                                     following_window_begin,
                                                     min_periods,
                                                     stream,
                                                     mr};
    return cudf::detail::aggregation_dispatcher(agg.kind, postprocessor, agg, intermediate);
  }
};

/**
 * @copydoc cudf::rolling_window(column_view const& input,
 *                               PrecedingWindowIterator preceding_window_begin,
 *                               FollowingWindowIterator following_window_begin,
 *                               size_type min_periods,
 *                               rolling_aggregation const& agg,
 *                               rmm::device_async_resource_ref mr)
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
template <typename PrecedingWindowIterator, typename FollowingWindowIterator>
std::unique_ptr<column> rolling_window(column_view const& input,
                                       column_view const& default_outputs,
                                       PrecedingWindowIterator preceding_window_begin,
                                       FollowingWindowIterator following_window_begin,
                                       size_type min_periods,
                                       rolling_aggregation const& agg,
                                       rmm::cuda_stream_view stream,
                                       rmm::device_async_resource_ref mr)
{
  static_assert(warp_size == cudf::detail::size_in_bits<cudf::bitmask_type>(),
                "bitmask_type size does not match CUDA warp size");

  if (input.is_empty()) { return cudf::detail::empty_output_for_rolling_aggregation(input, agg); }

  if (cudf::is_dictionary(input.type())) {
    CUDF_EXPECTS(agg.kind == aggregation::COUNT_ALL || agg.kind == aggregation::COUNT_VALID ||
                   agg.kind == aggregation::ROW_NUMBER || agg.kind == aggregation::MIN ||
                   agg.kind == aggregation::MAX || agg.kind == aggregation::LEAD ||
                   agg.kind == aggregation::LAG,
                 "Invalid aggregation for dictionary column");
  }

  if (agg.kind != aggregation::LEAD && agg.kind != aggregation::LAG) {
    CUDF_EXPECTS(default_outputs.is_empty(),
                 "Only LEAD/LAG window functions support default values.");
  }

  min_periods = std::max(min_periods, 0);

  return cudf::type_dispatcher(input.type(),
                               dispatch_rolling{},
                               input,
                               default_outputs,
                               preceding_window_begin,
                               following_window_begin,
                               min_periods,
                               agg,
                               stream,
                               mr);
}

}  // namespace detail

}  // namespace cudf
